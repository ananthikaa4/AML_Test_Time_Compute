"""Test-time compute experiment for the fine-tuned InLegalBERT (answers RQ1, RQ3, RQ4; extends RQ2).

Test-time compute = stochastic forward passes with dropout active (Monte Carlo dropout), whose
class probabilities are averaged. Every strategy uses the SAME aggregator and is costed in
forward passes (model calls), so cost comparisons are fair:

    single      1 deterministic pass (dropout off)                         cost 1
    fixed N     deterministic pass + (N-1) stochastic passes, mean probs   cost N   (N = 4, 8, 16)
    adaptive    deterministic pass = router AND first sample; its top-1 probability is bucketed
                with thresholds fixed on VALIDATION quantiles (easy / medium / hard), then
                (N_bucket - 1) further stochastic passes                   cost N_bucket (2/8/16)
    ablations   random routing with the same bucket shares (20 draws), question-length routing

Test labels are only used for scoring. Thresholds come from validation only.

Outputs:
    results/tt_compute_results.csv      per split x strategy x MC seed: accuracy, macro-F1, calls, ms
    results/tt_compute_summary.csv      mean +- s.d. over MC seeds (test and val)
    results/tt_compute_by_group.csv     test accuracy per tier / legal area for single, fixed-16, adaptive
    results/tt_compute_significance.json  McNemar adaptive vs single / vs fixed-16 (MC seed 0)

Usage (project root):  .venv/Scripts/python -m src.evaluation.tt_compute_mc
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

from src.data.task_a import load_task_a
from src.training.train_transformer import load_trained, make_loader
from src.utils.config import load_config, resolve
from src.utils.seed import set_seed

RUN = "InLegalBERT_lr2e-05_do0.1_s42"
MAX_N = 16
FIXED = (4, 8, 16)
BUDGET = {"easy": 2, "medium": 8, "hard": 16}
SHARES = (0.4, 0.4, 0.2)          # easy = top 40% confidence, medium = middle 40%, hard = bottom 20% (on val)
MC_SEEDS = (0, 1, 2)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- self-contained metrics (no scikit-learn / SciPy: their compiled DLLs are blocked by
#     Windows Smart App Control on this machine). Same definitions as src/evaluation/metrics.py.
def classification_metrics(y_true, y_pred, train_labels=None, n_boot: int = 2000, seed: int = 0) -> dict:
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    correct = (y_true == y_pred).astype(float)
    rng = np.random.default_rng(seed)
    means = correct[rng.integers(0, len(correct), size=(n_boot, len(correct)))].mean(axis=1)
    f1s = []
    for lab in set(y_true.tolist()):            # macro-F1 over gold labels present
        tp = float(((y_pred == lab) & (y_true == lab)).sum())
        fp = float(((y_pred == lab) & (y_true != lab)).sum())
        fn = float(((y_pred != lab) & (y_true == lab)).sum())
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return {"accuracy": float(correct.mean()), "acc_ci_low": float(np.quantile(means, 0.025)),
            "acc_ci_high": float(np.quantile(means, 0.975)), "macro_f1": float(np.mean(f1s))}


def mcnemar_exact(correct_a, correct_b) -> dict:
    from math import comb
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n)
    return {"n": int(len(a)), "acc_a": float(a.mean()), "acc_b": float(b.mean()),
            "only_a_correct": only_a, "only_b_correct": only_b, "p_value": float(p)}


def set_dropout(model: torch.nn.Module, on: bool) -> None:
    model.eval()
    if on:
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()


@torch.no_grad()
def one_pass(model, loader) -> tuple[torch.Tensor, float]:
    out, t0 = [], time.perf_counter()
    for b in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            logits = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE))
        out.append(torch.softmax(logits.float(), dim=1).cpu())
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return torch.cat(out), time.perf_counter() - t0


def collect(model, loader, seed: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    """det: [Q, C] deterministic probs; mc: [MAX_N-1, Q, C] stochastic probs; ms per pass per question."""
    set_dropout(model, False)
    det, t = one_pass(model, loader)
    set_seed(seed)
    set_dropout(model, True)
    mc, ts = [], [t]
    for _ in range(MAX_N - 1):
        p, t = one_pass(model, loader)
        mc.append(p)
        ts.append(t)
    set_dropout(model, False)
    return det, torch.stack(mc), 1000 * float(np.mean(ts)) / det.shape[0]


def predict_budget(det: torch.Tensor, mc: torch.Tensor, n: np.ndarray) -> np.ndarray:
    """Prediction per question given its budget n (>=1): mean of det + first n-1 stochastic passes."""
    cum = torch.cat([det.unsqueeze(0), mc]).cumsum(0)          # [MAX_N, Q, C]
    idx = torch.as_tensor(n - 1, dtype=torch.long)
    sel = cum[idx, torch.arange(det.shape[0])]                  # [Q, C]
    return sel.argmax(1).numpy()


def bucket_budgets(score: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """High score = easy. score >= hi -> easy; lo <= score < hi -> medium; < lo -> hard."""
    return np.where(score >= hi, BUDGET["easy"], np.where(score >= lo, BUDGET["medium"], BUDGET["hard"]))


def main() -> None:
    cfg = load_config()
    data = load_task_a(cfg)
    res_dir = resolve(cfg["paths"]["results"])
    model, tok, enc, rc = load_trained(resolve(cfg["paths"]["experiments"]) / "transformer" / RUN, DEVICE)
    classes = np.array(enc.classes_, dtype=object)
    train_labels = data["train"]["label"].tolist()

    loaders = {s: make_loader(tok, data[s]["prompt"].tolist(), enc.encode(data[s]["label"].tolist()), 64, rc.max_length, False, 0)
               for s in ("val", "test")}

    rows, by_group, sig = [], [], {}
    for mc_seed in MC_SEEDS:
        packs = {s: collect(model, loaders[s], 1000 + mc_seed) for s in ("val", "test")}
        # thresholds from VALIDATION deterministic confidence and question length only
        vconf = packs["val"][0].max(1).values.numpy()
        c_lo, c_hi = np.quantile(vconf, [SHARES[2], SHARES[2] + SHARES[1]])
        vlen = -data["val"]["prompt_words"].to_numpy(float)       # shorter = "easier" for length routing
        l_lo, l_hi = np.quantile(vlen, [SHARES[2], SHARES[2] + SHARES[1]])

        for split in ("val", "test"):
            det, mc, ms = packs[split]
            df = data[split]
            y = df["label"].to_numpy(object)
            q = len(df)
            conf = det.max(1).values.numpy()
            strategies: dict[str, np.ndarray] = {"single": np.ones(q, int)}
            for n in FIXED:
                strategies[f"fixed_{n}"] = np.full(q, n)
            strategies["adaptive_confidence"] = bucket_budgets(conf, c_lo, c_hi)
            strategies["ablation_length_routing"] = bucket_budgets(-df["prompt_words"].to_numpy(float), l_lo, l_hi)
            preds = {}
            for name, n in strategies.items():
                p = classes[predict_budget(det, mc, n)]
                preds[name] = p
                m = classification_metrics(y, p, train_labels=train_labels)
                rows.append({"split": split, "strategy": name, "mc_seed": mc_seed, "accuracy": m["accuracy"],
                             "acc_ci_low": m["acc_ci_low"], "acc_ci_high": m["acc_ci_high"], "macro_f1": m["macro_f1"],
                             "avg_calls": float(n.mean()), "ms_per_question": float(n.mean()) * ms,
                             "share_easy": float((n == BUDGET["easy"]).mean()) if "adaptive" in name or "routing" in name else None,
                             "share_hard": float((n == BUDGET["hard"]).mean()) if "adaptive" in name or "routing" in name else None})
            # random routing with the same bucket shares as adaptive (20 draws)
            rng = np.random.default_rng(mc_seed)
            accs = []
            for _ in range(20):
                n = rng.permutation(strategies["adaptive_confidence"])
                accs.append(float((classes[predict_budget(det, mc, n)] == y).mean()))
            rows.append({"split": split, "strategy": "ablation_random_routing", "mc_seed": mc_seed, "accuracy": float(np.mean(accs)),
                         "acc_ci_low": float(np.min(accs)), "acc_ci_high": float(np.max(accs)), "macro_f1": None,
                         "avg_calls": float(strategies["adaptive_confidence"].mean()),
                         "ms_per_question": float(strategies["adaptive_confidence"].mean()) * ms})
            if split == "test":
                bucket = np.select([strategies["adaptive_confidence"] == BUDGET["easy"],
                                    strategies["adaptive_confidence"] == BUDGET["medium"]], ["easy", "medium"], "hard")
                g = df.assign(bucket=bucket, single=preds["single"] == y, fixed_16=preds["fixed_16"] == y,
                              adaptive=preds["adaptive_confidence"] == y, calls_adaptive=strategies["adaptive_confidence"])
                for col in ("tier", "legal_area", "bucket"):
                    agg = g.groupby(col).agg(n=("single", "size"), single=("single", "mean"), fixed_16=("fixed_16", "mean"),
                                             adaptive=("adaptive", "mean"), calls_adaptive=("calls_adaptive", "mean")).reset_index()
                    agg = agg.rename(columns={col: "value"})
                    agg.insert(0, "group", col)
                    agg.insert(0, "mc_seed", mc_seed)
                    by_group.append(agg)
                if mc_seed == MC_SEEDS[0]:
                    ok = {k: preds[k] == y for k in ("single", "fixed_16", "adaptive_confidence")}
                    sig = {"adaptive_vs_single": mcnemar_exact(ok["single"], ok["adaptive_confidence"]),
                           "adaptive_vs_fixed16": mcnemar_exact(ok["fixed_16"], ok["adaptive_confidence"]),
                           "fixed16_vs_single": mcnemar_exact(ok["single"], ok["fixed_16"]),
                           "thresholds_from_val": {"conf_lo": float(c_lo), "conf_hi": float(c_hi)}}

    res = pd.DataFrame(rows)
    res.to_csv(res_dir / "tt_compute_results.csv", index=False)
    summ = res.groupby(["split", "strategy"]).agg(accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
                                                  macro_f1_mean=("macro_f1", "mean"), avg_calls=("avg_calls", "mean"),
                                                  ms_per_question=("ms_per_question", "mean")).reset_index()
    summ.to_csv(res_dir / "tt_compute_summary.csv", index=False)
    bg = pd.concat(by_group)
    bg.groupby(["group", "value"]).mean(numeric_only=True).drop(columns="mc_seed").reset_index().to_csv(
        res_dir / "tt_compute_by_group.csv", index=False)
    (res_dir / "tt_compute_significance.json").write_text(json.dumps(sig, indent=2), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(summ.round(4).to_string(index=False))
    print(bg.groupby(["group", "value"]).mean(numeric_only=True).drop(columns="mc_seed").round(3).to_string())
    print(json.dumps(sig, indent=2))


if __name__ == "__main__":
    main()

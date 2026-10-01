"""Phase 6-7: hyperparameter search + final evaluation of the Transformer Task A classifier.

1. Staged search over `transformer.search_stages` (model x lr, then dropout), seed = config seed.
   Each run: fine-tune on train, early-stop on val accuracy. Completed runs are skipped on
   re-invocation (resumable).
2. Best configuration by val accuracy is re-trained with `transformer.final_seeds` to
   measure seed variance.
3. Every final-seed model is evaluated on val and test (test is never used for selection).

Outputs:
    results/transformer_hpo.csv
    results/transformer_results.csv            grouped metrics per seed x split
    results/transformer_seed_summary.csv       mean/std over seeds
    results/predictions/transformer_<split>.parquet  (primary seed; top-5, prob, entropy, margin)
    experiments/transformer/<run_name>/{best.pt,run.json,classes.json}

Usage:
    python -m src.training.hyperparameter_search [--only-eval]
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.task_a import load_task_a
from src.evaluation.metrics import grouped_metrics
from src.training.train_transformer import RunConfig, load_trained, make_loader, predict_logits, train_run
from src.utils.config import load_config, resolve
from src.utils.logging import setup_logging

log = logging.getLogger("hpo")


def run_name(rc: RunConfig) -> str:
    short = rc.model_name.split("/")[-1]
    return f"{short}_lr{rc.lr:g}_do{rc.dropout:g}_s{rc.seed}"


def get_or_train(rc: RunConfig, data: dict, root: Path) -> dict:
    d = root / run_name(rc)
    if (d / "run.json").exists() and (d / "best.pt").exists():
        log.info("skip (done): %s", d.name)
        return json.loads((d / "run.json").read_text(encoding="utf-8"))
    log.info("=== training %s ===", d.name)
    return train_run(rc, data["train"], data["val"], d)


def evaluate(run_dir: Path, df: pd.DataFrame, k: int = 5) -> tuple[pd.DataFrame, dict]:
    model, tok, enc, rc = load_trained(run_dir)
    loader = make_loader(tok, df["prompt"].tolist(), enc.encode(df["label"].tolist()), 64, rc.max_length, False, 0)
    t0 = time.perf_counter()
    logits, _ = predict_logits(model, loader, "cuda")
    torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    order = np.argsort(-probs, axis=1)[:, :k]
    top = np.take_along_axis(probs, order, axis=1)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1))).sum(1)
    pf = df.assign(
        pred=[enc.classes_[i] for i in order[:, 0]],
        topk=[[enc.classes_[j] for j in row] for row in order],
        topk_probs=[row.tolist() for row in top],
        top1_prob=top[:, 0], margin=top[:, 0] - top[:, 1], entropy=entropy,
    )
    del model
    torch.cuda.empty_cache()
    return pf, {"inference_seconds": secs, "ms_per_question": 1000 * secs / len(df)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-eval", action="store_true", help="skip training; evaluate existing final runs")
    args = ap.parse_args()
    setup_logging(resolve("experiments/logs/transformer.log"))
    cfg = load_config()
    tcfg = cfg["transformer"]
    data = load_task_a(cfg)
    root = resolve(cfg["paths"]["experiments"]) / "transformer"
    res_dir = resolve(cfg["paths"]["results"])
    base = dict(tcfg["fixed"], seed=cfg["seed"])

    # ---- 1. staged search -----------------------------------------------------------------
    # Each stage grids over its own keys with every other key at the incumbent's value; the
    # incumbent (best so far) is kept unless a stage candidate beats it on val accuracy.
    hpo, best_params, best_acc = [], {}, -1.0
    for stage_no, space in enumerate(tcfg["search_stages"], start=1):
        keys = list(space)
        for vals in itertools.product(*(space[k] for k in keys)):
            params = {**best_params, **dict(zip(keys, vals))}
            if "model_name" not in {**base, **params}:  # --only-eval before any stage-1 run exists
                continue
            rc = RunConfig(**{**base, **params})
            if args.only_eval and not (root / run_name(rc) / "run.json").exists():
                continue
            s = get_or_train(rc, data, root)
            hpo.append({"stage": stage_no, "run": run_name(rc), **{k: getattr(rc, k) for k in ("model_name", "lr", "dropout")},
                        "best_val_acc": s["best_val_acc"], "best_epoch": s["best_epoch"], "epochs_run": s["epochs_run"],
                        "train_seconds": s["train_seconds"], "peak_gpu_mb": s["peak_gpu_mb"]})
            if s["best_val_acc"] > best_acc:
                best_acc, best_params = s["best_val_acc"], params
        log.info("after stage %d best: %s (val acc %.4f)", stage_no, best_params, best_acc)
    if not hpo:
        raise SystemExit("no completed search runs to evaluate (run without --only-eval first)")
    hpo_df = pd.DataFrame(hpo)
    hpo_df.to_csv(res_dir / "transformer_hpo.csv", index=False)
    log.info("HPO results:\n%s", hpo_df.to_string(index=False))
    (res_dir / "transformer_best_config.json").write_text(
        json.dumps({**base, **best_params, "selected_by": "val accuracy", "val_acc": best_acc}, indent=2),
        encoding="utf-8")
    log.info("best config: %s", best_params)

    # ---- 2. seed repeats + 3. evaluation -------------------------------------------------
    rows = []
    for i, seed in enumerate(tcfg["final_seeds"]):
        rc = RunConfig(**{**base, **best_params, "seed": seed})
        get_or_train(rc, data, root)
        for split in ("val", "test"):
            pf, timing = evaluate(root / run_name(rc), data[split])
            g = grouped_metrics(pf, ["tier", "legal_area"], train_labels=data["train"]["label"].tolist())
            g.insert(0, "split", split)
            g.insert(0, "seed", seed)
            g.insert(0, "model", run_name(rc))
            for k, v in timing.items():
                g[k] = v
            rows.append(g)
            if i == 0:  # primary seed -> predictions used downstream
                pf.to_parquet(res_dir / "predictions" / f"transformer_{split}.parquet", index=False)
    res = pd.concat(rows)
    res.to_csv(res_dir / "transformer_results.csv", index=False)
    main_rows = res[res["group"].isin(["all", "tier"])]
    summ = main_rows.groupby(["split", "group", "value"])[["accuracy", "macro_f1", "weighted_f1", "top5"]].agg(["mean", "std"])
    summ.columns = ["_".join(c) for c in summ.columns]
    summ.reset_index().to_csv(res_dir / "transformer_seed_summary.csv", index=False)
    log.info("seed summary:\n%s", summ.round(4).to_string())


if __name__ == "__main__":
    main()

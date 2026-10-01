"""Phase 5: classical baselines for Task A with grid search on the validation split.

For each model family, every grid point is fit on train and scored on val (overall Task A
accuracy is the selection metric). The best configuration is refit on train and evaluated
once on val and test. The test split is never used for selection.

Outputs:
    results/baseline_results.csv              metrics per model x split x group (tier, legal area)
    results/baseline_hpo.csv                  every grid point with its val metrics
    results/predictions/baseline_<model>_<split>.parquet
    experiments/baselines/<model>/run.json    seed, config, best params, runtime

Usage:
    python -m src.training.train_baseline
"""
from __future__ import annotations

import itertools
import json
import logging
import platform
import time
from typing import Any

import joblib
import pandas as pd

from src.data.task_a import load_task_a
from src.evaluation.metrics import classification_metrics, grouped_metrics
from src.models.baseline import TfidfClassifier
from src.utils.config import load_config, resolve
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

log = logging.getLogger("baseline")


def grid(space: dict[str, list]) -> list[dict[str, Any]]:
    keys = list(space)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(space[k] for k in keys))]


def predict_frame(model: TfidfClassifier, df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    topk, scores = model.rank(df["prompt"].tolist(), k=k)
    return df.assign(pred=[t[0] for t in topk], topk=topk, top1_score=scores[:, 0],
                     margin=scores[:, 0] - scores[:, 1])


def main() -> None:
    setup_logging(resolve("experiments/logs/baseline.log"))
    cfg = load_config()
    set_seed(cfg["seed"])
    bcfg = cfg["baselines"]
    data = load_task_a(cfg)
    train_labels = data["train"]["label"].tolist()
    log.info("Task A sizes: %s", {s: len(d) for s, d in data.items()})

    results, hpo_rows = [], []
    pred_dir = resolve(cfg["paths"]["results"]) / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in bcfg["models"].items():
        best, best_acc = None, -1.0
        for params in grid(spec["grid"]):
            t0 = time.time()
            m = TfidfClassifier(name, params, max_features=bcfg["max_features"]).fit(
                data["train"]["prompt"].tolist(), train_labels)
            pv = predict_frame(m, data["val"])
            met = classification_metrics(pv["label"], pv["pred"], pv["topk"], train_labels)
            fit_s = time.time() - t0
            hpo_rows.append({"model": name, **params, **{f"val_{k}": v for k, v in met.items()}, "seconds": fit_s})
            log.info("%s %s val acc=%.4f macroF1=%.4f (%.0fs)", name, params, met["accuracy"], met["macro_f1"], fit_s)
            if met["accuracy"] > best_acc:
                best, best_acc = params, met["accuracy"]

        log.info("%s best params %s (val acc %.4f); refitting", name, best, best_acc)
        t0 = time.time()
        model = TfidfClassifier(name, best, max_features=bcfg["max_features"]).fit(
            data["train"]["prompt"].tolist(), train_labels)
        train_s = time.time() - t0
        for split in ("val", "test"):
            t1 = time.time()
            pf = predict_frame(model, data[split])
            infer_s = time.time() - t1
            pf.to_parquet(pred_dir / f"baseline_{name}_{split}.parquet", index=False)
            g = grouped_metrics(pf, ["tier", "legal_area"], train_labels=train_labels)
            g.insert(0, "split", split)
            g.insert(0, "model", name)
            g["inference_seconds"] = infer_s
            g["ms_per_question"] = 1000 * infer_s / len(pf)
            results.append(g)
        out_dir = resolve(cfg["paths"]["experiments"]) / "baselines" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out_dir / "model.joblib")
        (out_dir / "run.json").write_text(json.dumps({
            "model": name, "seed": cfg["seed"], "best_params": best, "val_accuracy_at_selection": best_acc,
            "grid": spec["grid"], "max_features": bcfg["max_features"], "train_seconds": train_s,
            "n_train": len(data["train"]), "n_classes": len(set(train_labels)),
            "python": platform.python_version(), "selection_metric": "val accuracy (all Task A)",
        }, indent=2), encoding="utf-8")

    res_dir = resolve(cfg["paths"]["results"])
    pd.concat(results).to_csv(res_dir / "baseline_results.csv", index=False)
    pd.DataFrame(hpo_rows).to_csv(res_dir / "baseline_hpo.csv", index=False)
    summary = pd.concat(results)
    log.info("\n%s", summary[summary["group"].isin(["all", "tier"])][
        ["model", "split", "group", "value", "n", "accuracy", "acc_ci_low", "acc_ci_high", "macro_f1", "top5", "ceiling"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()

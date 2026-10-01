"""Phase 7b: LoRA supervised fine-tuning of Model A (the generative legal reasoning model).

Training pairs are (question -> truncated reference answer) from TRAIN rows only; the loss is
computed on the answer tokens only. `complex_cot` is not used (it is gold reasoning and the
system does not produce or store hidden reasoning traces).

Validation: loss on held-out VAL rows every `eval_every` optimiser steps; the adapter with the
lowest val loss is kept (early stopping by patience). After training, Task A single-shot greedy
accuracy on VAL is measured for the base model (zero-shot) and the fine-tuned model, which
quantifies the effect of transfer learning. The test split is not touched.

Usage:
    python -m src.training.train_reasoner [--max-steps N]   # --max-steps for smoke tests
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data.provisions import first_provision
from src.data.task_a import load_task_a
from src.evaluation.metrics import classification_metrics
from src.inference.generator import Generator
from src.models.reasoning_model import add_lora, load_model, load_tokenizer, messages, truncate_answer
from src.utils.config import load_config, resolve
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

log = logging.getLogger("train_reasoner")


def build_examples(tok: Any, rows: pd.DataFrame, max_target_words: int, max_len: int) -> list[dict]:
    """Tokenise chat-formatted pairs; labels are -100 on prompt tokens (loss on answer only)."""
    out, n_trunc = [], 0
    for q, r in zip(rows["prompt"], rows["response"]):
        target = truncate_answer(r, max_target_words)
        prompt_ids = tok.apply_chat_template(messages(q), tokenize=True, add_generation_prompt=True)
        full_ids = tok.apply_chat_template(messages(q) + [{"role": "assistant", "content": target}], tokenize=True)
        if hasattr(prompt_ids, "input_ids"):  # some tokenizer versions return BatchEncoding
            prompt_ids, full_ids = prompt_ids["input_ids"], full_ids["input_ids"]
        full_ids = list(full_ids)
        # Qwen's template ends with "<|im_end|>\n"; drop the trailing newline so <|im_end|> is the last target.
        if tok.decode(full_ids[-1:]) == "\n":
            full_ids = full_ids[:-1]
        if len(full_ids) > max_len:
            full_ids, n_trunc = full_ids[:max_len], n_trunc + 1
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        out.append({"input_ids": full_ids, "labels": labels[: len(full_ids)]})
    log.info("built %d examples (%d truncated at %d tokens)", len(out), n_trunc, max_len)
    return out


def batches(examples: list[dict], bs: int, pad_id: int, shuffle: bool, rng: np.random.Generator):
    order = rng.permutation(len(examples)) if shuffle else np.arange(len(examples))
    for i in range(0, len(order), bs):
        chunk = [examples[j] for j in order[i: i + bs]]
        n = max(len(e["input_ids"]) for e in chunk)
        ids = torch.full((len(chunk), n), pad_id, dtype=torch.long)
        lab = torch.full((len(chunk), n), -100, dtype=torch.long)
        mask = torch.zeros((len(chunk), n), dtype=torch.long)
        for k, e in enumerate(chunk):
            L = len(e["input_ids"])
            ids[k, :L], lab[k, :L], mask[k, :L] = torch.tensor(e["input_ids"]), torch.tensor(e["labels"]), 1
        yield ids, mask, lab


@torch.no_grad()
def eval_loss(model: Any, examples: list[dict], bs: int, pad_id: int) -> float:
    model.eval()
    tot, n = 0.0, 0
    for ids, mask, lab in batches(examples, bs, pad_id, False, np.random.default_rng(0)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=ids.cuda(), attention_mask=mask.cuda(), labels=lab.cuda())
        k = int((lab[:, 1:] != -100).sum())
        tot, n = tot + out.loss.float().item() * k, n + k
    model.train()
    return tot / max(n, 1)


def task_a_greedy_accuracy(model: Any, tok: Any, df: pd.DataFrame, gcfg: dict, seed: int) -> tuple[dict, pd.DataFrame]:
    gen = Generator(model, tok, max_seqs_per_batch=gcfg["max_seqs_per_batch"])
    t0 = time.perf_counter()
    res = gen.generate(df["prompt"].tolist(), n=1, temperature=0.0, max_new_tokens=gcfg["max_new_tokens"], seed=seed)
    secs = time.perf_counter() - t0
    preds = [first_provision(r.candidates[0].text, q) for r, q in zip(res, df["prompt"])]
    m = classification_metrics(df["label"].tolist(), preds)
    m.update(seconds=secs, mean_gen_tokens=float(np.mean([r.gen_tokens for r in res])),
             tokens_per_second=sum(r.gen_tokens for r in res) / secs)
    for tier in ("constitution", "statute"):
        s = (df["tier"] == tier).values
        m[f"acc_{tier}"] = float(np.mean([p == g for p, g, keep in zip(preds, df["label"], s) if keep]))
    out = df[["qid", "tier", "label"]].assign(pred=preds, answer=[r.candidates[0].text for r in res])
    return m, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=None, help="cap optimiser steps (smoke test)")
    ap.add_argument("--skip-base-eval", action="store_true")
    ap.add_argument("--n-task-a-val", type=int, default=None, help="override Task A val subset size (smoke test)")
    ap.add_argument("--run-name", default=None, help="override output run name")
    args = ap.parse_args()
    setup_logging(resolve("experiments/logs/reasoner.log"))
    cfg = load_config()
    rcfg, gcfg = cfg["reasoning_model"], cfg["generation"]
    set_seed(cfg["seed"])
    if args.run_name:
        rcfg["run_name"] = args.run_name
    if args.n_task_a_val:
        rcfg["n_task_a_val"] = args.n_task_a_val
    out_dir = resolve(cfg["paths"]["experiments"]) / "reasoner" / rcfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = pd.read_parquet(resolve(cfg["data"]["processed_dir"]) / "rows.parquet")
    train_rows = rows[rows["split"] == "train"]
    if rcfg["max_train_examples"]:
        train_rows = train_rows.sample(min(rcfg["max_train_examples"], len(train_rows)), random_state=cfg["seed"])
    val_rows = rows[rows["split"] == "val"].sample(rcfg["n_val_loss_examples"], random_state=cfg["seed"])
    task_a_val = load_task_a(cfg)["val"]
    if rcfg.get("n_task_a_val"):
        task_a_val = task_a_val.sample(min(rcfg["n_task_a_val"], len(task_a_val)), random_state=cfg["seed"])

    tok = load_tokenizer(rcfg["name"])
    train_ex = build_examples(tok, train_rows, rcfg["max_target_words"], rcfg["max_length"])
    val_ex = build_examples(tok, val_rows, rcfg["max_target_words"], rcfg["max_length"])
    model = load_model(rcfg["name"])
    report: dict[str, Any] = {"config": rcfg, "generation": gcfg, "seed": cfg["seed"],
                              "n_train": len(train_ex), "n_val_loss": len(val_ex), "n_task_a_val": len(task_a_val)}

    if not args.skip_base_eval:
        log.info("zero-shot base model Task A val accuracy (greedy)...")
        m, preds = task_a_greedy_accuracy(model, tok, task_a_val, gcfg, cfg["seed"])
        report["base_task_a_val"] = m
        preds.to_parquet(out_dir / "base_task_a_val_preds.parquet", index=False)
        log.info("base: %s", {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})

    model = add_lora(model, rcfg["lora_r"], rcfg["lora_alpha"], rcfg["lora_dropout"], rcfg["lora_target_modules"])
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report["trainable_params"] = trainable
    log.info("trainable params: %d", trainable)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=rcfg["lr"],
                            weight_decay=rcfg["weight_decay"])
    steps_per_epoch = math.ceil(len(train_ex) / (rcfg["micro_batch_size"] * rcfg["grad_accum"]))
    total = steps_per_epoch * rcfg["epochs"] if args.max_steps is None else args.max_steps
    warm = int(rcfg["warmup_ratio"] * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    pad_id = tok.pad_token_id
    rng = np.random.default_rng(cfg["seed"])
    history, best_loss, bad, step, micro = [], float("inf"), 0, 0, 0
    torch.cuda.reset_peak_memory_stats()
    t_start, tok_count, losses = time.perf_counter(), 0, []
    stop = False
    for epoch in range(rcfg["epochs"]):
        for ids, mask, lab in batches(train_ex, rcfg["micro_batch_size"], pad_id, True, rng):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids.cuda(), attention_mask=mask.cuda(), labels=lab.cuda()).loss
            (loss / rcfg["grad_accum"]).backward()
            losses.append(loss.item())
            tok_count += int(mask.sum())
            micro += 1
            if micro % rcfg["grad_accum"]:
                continue
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % rcfg["eval_every"] == 0 or step == total:
                vl = eval_loss(model, val_ex, rcfg["micro_batch_size"] * 2, pad_id)
                el = time.perf_counter() - t_start
                history.append({"step": step, "train_loss": float(np.mean(losses)), "val_loss": vl,
                                "elapsed_s": el, "train_tokens_per_s": tok_count / el})
                log.info("step %d/%d train %.4f val %.4f (%.0f tok/s, %.0fs)", step, total, history[-1]["train_loss"],
                         vl, tok_count / el, el)
                losses = []
                if vl < best_loss:
                    best_loss, bad = vl, 0
                    model.save_pretrained(out_dir / "adapter")
                else:
                    bad += 1
                    if bad >= rcfg["patience"]:
                        log.info("early stopping at step %d", step)
                        stop = True
            if step >= total or stop:
                stop = True
                break
        if stop:
            break

    report.update(history=history, best_val_loss=best_loss, steps=step, train_seconds=time.perf_counter() - t_start,
                  peak_gpu_mb=round(torch.cuda.max_memory_allocated() / 1024**2, 1))
    del model, opt
    torch.cuda.empty_cache()

    log.info("fine-tuned model Task A val accuracy (greedy)...")
    ft = load_model(rcfg["name"], adapter_dir=out_dir / "adapter")
    m, preds = task_a_greedy_accuracy(ft, tok, task_a_val, gcfg, cfg["seed"])
    report["finetuned_task_a_val"] = m
    preds.to_parquet(out_dir / "finetuned_task_a_val_preds.parquet", index=False)
    log.info("fine-tuned: %s", {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})
    (out_dir / "run.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()

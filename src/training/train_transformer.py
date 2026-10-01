"""Fine-tuning loop for the Transformer Task A classifier.

Explicit PyTorch loop (rather than HF Trainer) so every training component is visible:
    loss       cross-entropy (optional label smoothing); labels unseen in train are ignored
               in the val loss (index -1) but still count as errors in accuracy
    optimizer  AdamW, separate learning rates for the pretrained encoder and the new head,
               no weight decay on biases / LayerNorm
    schedule   linear warm-up then linear decay
    precision  bf16 autocast (weights in fp32)
    stopping   early stopping on val accuracy with patience; best checkpoint kept

`train_run` is called by src/training/hyperparameter_search.py.
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.task_a import LabelEncoder
from src.models.transformer import TransformerClassifier
from src.utils.seed import set_seed

log = logging.getLogger("train_transformer")


@dataclass
class RunConfig:
    model_name: str
    lr: float = 3e-5
    head_lr: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 25
    patience: int = 4
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    dropout: float = 0.1
    label_smoothing: float = 0.0
    max_length: int = 128
    pooling: str = "cls"
    grad_clip: float = 1.0
    seed: int = 42


def make_loader(tok: Any, texts: list[str], labels: list[int], bs: int, max_len: int, shuffle: bool, seed: int) -> DataLoader:
    enc = tok(texts, truncation=True, max_length=max_len)
    items = [{"input_ids": ids, "label": y} for ids, y in zip(enc["input_ids"], labels)]
    pad = tok.pad_token_id

    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            ids[i, : len(b["input_ids"])] = torch.tensor(b["input_ids"])
            mask[i, : len(b["input_ids"])] = 1
        return {"input_ids": ids, "attention_mask": mask, "labels": torch.tensor([b["label"] for b in batch])}

    g = torch.Generator().manual_seed(seed)
    return DataLoader(items, batch_size=bs, shuffle=shuffle, collate_fn=collate, generator=g)


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    for b in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out = model(b["input_ids"].to(device), b["attention_mask"].to(device))
        logits.append(out.float().cpu())
        labels.append(b["labels"])
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def param_groups(model: TransformerClassifier, rc: RunConfig) -> list[dict]:
    no_decay = ("bias", "LayerNorm.weight", "layernorm.weight", "layer_norm.weight")
    groups = []
    for is_head, lr in ((False, rc.lr), (True, rc.head_lr)):
        named = [(n, p) for n, p in model.named_parameters() if n.startswith("head.") == is_head and p.requires_grad]
        groups.append({"params": [p for n, p in named if not n.endswith(no_decay)], "lr": lr, "weight_decay": rc.weight_decay})
        groups.append({"params": [p for n, p in named if n.endswith(no_decay)], "lr": lr, "weight_decay": 0.0})
    return groups


def train_run(rc: RunConfig, train_df: pd.DataFrame, val_df: pd.DataFrame, out_dir: Path, device: str = "cuda") -> dict[str, Any]:
    """Train one configuration; saves best checkpoint + history to out_dir. Returns summary."""
    set_seed(rc.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    enc = LabelEncoder(train_df["label"].tolist())
    tok = AutoTokenizer.from_pretrained(rc.model_name)
    tr_loader = make_loader(tok, train_df["prompt"].tolist(), enc.encode(train_df["label"].tolist()),
                            rc.batch_size, rc.max_length, True, rc.seed)
    va_loader = make_loader(tok, val_df["prompt"].tolist(), enc.encode(val_df["label"].tolist()),
                            rc.batch_size * 2, rc.max_length, False, rc.seed)

    model = TransformerClassifier(rc.model_name, len(enc), rc.dropout, rc.pooling).to(device)
    opt = torch.optim.AdamW(param_groups(model, rc))
    total = rc.max_epochs * len(tr_loader)
    warm = int(rc.warmup_ratio * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=rc.label_smoothing)

    history, best_acc, best_epoch, bad = [], -1.0, -1, 0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    for epoch in range(1, rc.max_epochs + 1):
        model.train()
        t0, losses = time.perf_counter(), []
        for b in tr_loader:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(b["input_ids"].to(device), b["attention_mask"].to(device))
            loss = loss_fn(logits.float(), b["labels"].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), rc.grad_clip)
            opt.step()
            sched.step()
            losses.append(loss.item())
        logits, y = predict_logits(model, va_loader, device)
        seen = y >= 0
        val_loss = float(nn.functional.cross_entropy(torch.tensor(logits[seen]), torch.tensor(y[seen])).item())
        val_acc = float((logits.argmax(1) == y).mean())  # unseen labels (-1) never match
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss,
                        "val_acc": val_acc, "epoch_seconds": time.perf_counter() - t0})
        log.info("epoch %2d train_loss %.4f val_loss %.4f val_acc %.4f (%.0fs)", epoch, history[-1]["train_loss"],
                 val_loss, val_acc, history[-1]["epoch_seconds"])
        if val_acc > best_acc:
            best_acc, best_epoch, bad = val_acc, epoch, 0
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            bad += 1
            if bad >= rc.patience:
                log.info("early stopping at epoch %d (best %d)", epoch, best_epoch)
                break
        if not math.isfinite(history[-1]["train_loss"]):
            log.warning("non-finite loss; stopping run")
            break

    summary = {
        "config": asdict(rc), "best_val_acc": best_acc, "best_epoch": best_epoch, "epochs_run": len(history),
        "train_seconds": time.perf_counter() - t_start,
        "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1) if device == "cuda" else None,
        "n_train": len(train_df), "n_classes": len(enc), "history": history,
    }
    (out_dir / "run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "classes.json").write_text(json.dumps(enc.classes_), encoding="utf-8")
    del model, opt
    if device == "cuda":
        torch.cuda.empty_cache()
    return summary


def load_trained(run_dir: Path, device: str = "cuda") -> tuple[TransformerClassifier, Any, LabelEncoder, RunConfig]:
    summary = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    rc = RunConfig(**summary["config"])
    enc = LabelEncoder(json.loads((run_dir / "classes.json").read_text(encoding="utf-8")))
    model = TransformerClassifier(rc.model_name, len(enc), rc.dropout, rc.pooling)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    return model.to(device).eval(), AutoTokenizer.from_pretrained(rc.model_name), enc, rc

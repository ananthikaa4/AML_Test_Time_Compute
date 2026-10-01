"""Model A: generative legal reasoning model (causal LM + LoRA adapter).

Prompt construction lives here so training and inference format questions identically.
Only the question is ever placed in the prompt: no reference answer, no `complex_cot`.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "You are a research assistant answering questions about Indian law. "
    "Answer concisely in a few sentences. If a specific Article of the Constitution of India "
    "or a Section of a statute governs the question, cite it in your first sentence "
    "(e.g. 'Article 21 of the Constitution' or 'Section 138 of the Negotiable Instruments Act')."
)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def messages(question: str, system_prompt: str = SYSTEM_PROMPT) -> list[dict[str, str]]:
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]


def truncate_answer(text: str, max_words: int) -> str:
    """Leading sentences of a reference answer, at most `max_words` words (>= 1 sentence).

    SFT targets are shortened so that sampling many candidates at test time stays affordable.
    Markdown emphasis / headings are stripped so the target starts with content.
    """
    text = re.sub(r"[*#`]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    out: list[str] = []
    n = 0
    for sent in SENTENCE_END.split(text):
        w = len(sent.split())
        if out and n + w > max_words:
            break
        out.append(sent)
        n += w
    result = " ".join(out)
    words = result.split()
    return result if len(words) <= max_words else " ".join(words[:max_words])


def load_tokenizer(model_name: str) -> Any:
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_name: str, adapter_dir: str | Path | None = None, device: str = "cuda",
               dtype: torch.dtype = torch.bfloat16) -> Any:
    """Base model in bf16 (frozen), optionally with a trained LoRA adapter applied."""
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model


def add_lora(model: Any, r: int, alpha: int, dropout: float, target_modules: list[str]) -> Any:
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=target_modules,
                     bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    # LoRA weights in fp32 for stable optimisation; frozen base stays bf16.
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return model

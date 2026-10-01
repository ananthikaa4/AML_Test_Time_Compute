"""Batched candidate generation with exact compute accounting.

All test-time-compute strategies call `Generator.generate`, so every method is measured the
same way: number of sampled candidates (model calls), prompt tokens, generated tokens and
wall-clock time. Only the question text enters the prompt.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from src.models.reasoning_model import SYSTEM_PROMPT, messages


class ChosenTokenLogprobs(LogitsProcessor):
    """Records the log-probability of each sampled token with O(batch x vocab) memory.

    `output_scores=True` would keep every step's full-vocabulary scores (32 seqs x 152k vocab
    x 200 steps ~ 3.9 GB) and ran out of GPU memory. Instead, at step t we look up the token
    chosen at step t-1 in the previous step's distribution. transformers merges user
    processors *before* the temperature / top-p warpers (generation/utils.py,
    `_get_logits_processor`), so this is the model's untempered distribution: a
    temperature-independent confidence signal, comparable across sampling settings.
    """

    def __init__(self, prompt_len: int) -> None:
        self.prompt_len = prompt_len
        self.prev: torch.Tensor | None = None
        self.chosen: list[torch.Tensor] = []

    def _consume(self, input_ids: torch.LongTensor) -> None:
        if self.prev is not None and input_ids.shape[1] > self.prompt_len:
            tok = input_ids[:, -1:]
            self.chosen.append(self.prev.gather(1, tok).squeeze(1).float().cpu())

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self._consume(input_ids)
        self.prev = torch.log_softmax(scores.float(), dim=-1)
        return scores

    def finalize(self, sequences: torch.LongTensor) -> torch.Tensor:
        """(batch, steps) log-probs; call once with the final sequences."""
        self._consume(sequences)
        self.prev = None
        return torch.stack(self.chosen, dim=1) if self.chosen else torch.zeros((sequences.shape[0], 0))


@dataclass
class Candidate:
    text: str
    gen_tokens: int
    mean_logprob: float  # average token log-probability under the sampling model (a cheap confidence)


@dataclass
class GenerationResult:
    question: str
    candidates: list[Candidate]
    prompt_tokens: int  # counted once per sampled candidate (what the model actually processes)
    seconds: float      # share of the batch wall time attributed to this question
    calls: int = field(init=False)

    def __post_init__(self) -> None:
        self.calls = len(self.candidates)

    @property
    def gen_tokens(self) -> int:
        return sum(c.gen_tokens for c in self.candidates)


class Generator:
    def __init__(self, model: Any, tokenizer: Any, system_prompt: str = SYSTEM_PROMPT,
                 max_seqs_per_batch: int = 32, device: str = "cuda") -> None:
        self.model = model
        self.tok = tokenizer
        self.tok.padding_side = "left"
        self.system_prompt = system_prompt
        self.max_seqs = max_seqs_per_batch
        self.device = device

    def prompt_text(self, question: str) -> str:
        return self.tok.apply_chat_template(messages(question, self.system_prompt), tokenize=False,
                                            add_generation_prompt=True)

    @torch.no_grad()
    def generate(self, questions: list[str], n: int = 1, temperature: float = 0.0, top_p: float = 1.0,
                 max_new_tokens: int = 160, seed: int = 0) -> list[GenerationResult]:
        """n candidates per question. temperature == 0 -> greedy (n must be 1)."""
        greedy = temperature <= 0
        if greedy and n != 1:
            raise ValueError("greedy decoding with n > 1 would return identical candidates")
        q_per_batch = max(1, self.max_seqs // n)
        results: list[GenerationResult] = []
        for start in range(0, len(questions), q_per_batch):
            batch_q = questions[start: start + q_per_batch]
            enc = self.tok([self.prompt_text(q) for q in batch_q], return_tensors="pt", padding=True).to(self.device)
            torch.manual_seed(seed + start)
            if self.device == "cuda":
                torch.cuda.synchronize()
            recorder = ChosenTokenLogprobs(prompt_len=enc["input_ids"].shape[1])
            t0 = time.perf_counter()
            sequences = self.model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=not greedy,
                temperature=temperature if not greedy else None, top_p=top_p if not greedy else None,
                top_k=None, num_return_sequences=n, pad_token_id=self.tok.pad_token_id,
                logits_processor=LogitsProcessorList([recorder]),
            )
            if self.device == "cuda":
                torch.cuda.synchronize()
            secs = time.perf_counter() - t0
            seq = sequences[:, enc["input_ids"].shape[1]:]
            logps = recorder.finalize(sequences)  # (batch*n, generated steps)
            prompt_lens = enc["attention_mask"].sum(1).tolist()
            eos_ids = {self.tok.eos_token_id, self.tok.pad_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")}
            for qi, q in enumerate(batch_q):
                cands = []
                for j in range(n):
                    row = qi * n + j
                    ids = seq[row].tolist()
                    length = next((k for k, t in enumerate(ids) if t in eos_ids), len(ids))
                    lp = logps[row, :length]
                    cands.append(Candidate(
                        text=self.tok.decode(ids[:length], skip_special_tokens=True).strip(),
                        gen_tokens=length,
                        mean_logprob=float(lp[torch.isfinite(lp)].mean()) if length else float("-inf"),
                    ))
                results.append(GenerationResult(q, cands, prompt_tokens=int(prompt_lens[qi]) * n,
                                                seconds=secs / len(batch_q)))
        return results

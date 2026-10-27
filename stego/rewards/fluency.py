"""Fluency: how natural an answer looks to the original, untrained model.

The score compares the answer's mean per-token log-probability with that of the
original model's own (unconstrained) answer to the same question:

    score = clip(1 + (answer_lp - reference_lp) / tolerance, 0, 1)

It is 1 when the answer is at least as likely as the reference and falls to 0 when it
is ``tolerance`` nats per token worse.

The *continuation* variant (``fluency_cont``) leaves out the first word of every
sentence, in both the answer and the reference. In an acrostic those words are forced
(each must start with a given letter), so they are improbable by design; without them
the score measures how naturally each sentence carries on from its opening, which is
the part the model controls. It is used with a wider tolerance (3 nats/token), because
the original 1 put most acrostic answers on the 0 floor. Capping at 1 means the policy gains nothing by
becoming blander than the original model, which blocks the "generic filler" hack.

Torch is imported lazily so the rest of the package works without it.
"""

from __future__ import annotations

import contextlib
from typing import Any, Sequence


def fluency_score(answer_logprob: float, reference_logprob: float, tolerance: float = 1.0) -> float:
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    return min(1.0, max(0.0, 1.0 + (answer_logprob - reference_logprob) / tolerance))


def reference_model(model: Any) -> contextlib.AbstractContextManager:
    """Context in which ``model`` behaves like the original model.

    For a PEFT/LoRA model this switches the adapter off, so the frozen base model
    scores fluency without a second copy in GPU memory. Other models are used as is.
    """
    if hasattr(model, "disable_adapter"):
        return model.disable_adapter()
    return contextlib.nullcontext()


CONTINUATION_TOLERANCE = 3.0


def token_logprobs(model: Any, prompt_ids: Sequence[int], completion_ids: Sequence[int]) -> list[float]:
    """Log-probability (nats) of each token of ``completion_ids`` given ``prompt_ids``."""
    import torch

    if not prompt_ids or not completion_ids:
        raise ValueError("prompt and completion must both be non-empty")
    device = next(model.parameters()).device
    ids = torch.tensor([list(prompt_ids) + list(completion_ids)], device=device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, :-1].float()
    lps = torch.log_softmax(logits, dim=-1).gather(-1, ids[0, 1:, None]).squeeze(-1)
    return lps[len(prompt_ids) - 1 :].tolist()


def mean_completion_logprob(model: Any, prompt_ids: Sequence[int], completion_ids: Sequence[int]) -> float:
    """Mean log-probability (nats per token) of ``completion_ids`` given ``prompt_ids``."""
    lps = token_logprobs(model, prompt_ids, completion_ids)
    return sum(lps) / len(lps)


def first_word_spans(text: str) -> list[tuple[int, int]]:
    """Character spans, in ``text`` itself, of the first word of each sentence (as the
    project's splitter sees sentences)."""
    import re

    from stego.channels.splitter import split_sentences

    # The splitter collapses whitespace, so map the collapsed text back to ``text``.
    collapsed, where = [], []
    for i, ch in enumerate(text):
        if ch.isspace():
            if collapsed and collapsed[-1] != " ":
                collapsed.append(" ")
                where.append(i)
        else:
            collapsed.append(ch)
            where.append(i)
    flat = "".join(collapsed)
    spans, pos = [], 0
    for sentence in split_sentences(text):
        at = flat.find(sentence, pos)
        if at < 0:  # normalisation changed the text; skip rather than guess
            continue
        pos = at + len(sentence)
        word = re.search(r"[^\W_]+(?:['’\-][^\W_]+)*", sentence)
        if word:
            start, end = at + word.start(), at + word.end()
            spans.append((where[start], where[end - 1] + 1))
    return spans


def continuation_mask(text: str, tokenizer: Any) -> list[bool]:
    """For each token of ``text``: True if it is outside every sentence's first word."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    spans = first_word_spans(text)
    return [not any(a < e and s < b for s, e in spans) for a, b in enc["offset_mapping"]]


def chat_completion_logprob(
    model: Any, tokenizer: Any, messages: list[dict], completion: str, **template_kwargs: Any
) -> float:
    """``mean_completion_logprob`` for a chat prompt and a plain-text answer.

    Pass ``enable_thinking=False`` for Qwen3 checkpoints that have a thinking mode.
    """
    # Render to text, then tokenize: apply_chat_template(tokenize=True) returns a list in
    # transformers 4 but a dict in 5, and this matches how generation builds prompts.
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **template_kwargs)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    return mean_completion_logprob(model, prompt_ids, completion_ids)


def chat_completion_logprobs(
    model: Any, tokenizer: Any, messages: list[dict], completion: str, **template_kwargs: Any
) -> tuple[float, float]:
    """Mean log-probability of the answer over all its tokens, and over its continuation
    tokens only (every sentence's first word left out), from one forward pass. Falls back
    to all tokens if masking would leave none."""
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **template_kwargs)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    lps = token_logprobs(model, prompt_ids, completion_ids)
    keep = continuation_mask(completion, tokenizer)
    cont = [lp for lp, k in zip(lps, keep) if k] if len(keep) == len(lps) else []
    return sum(lps) / len(lps), (sum(cont) / len(cont) if cont else sum(lps) / len(lps))

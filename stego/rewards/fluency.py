"""Fluency: how natural an answer looks to the original, untrained model.

The score compares the answer's mean per-token log-probability with that of the
original model's own (unconstrained) answer to the same question:

    score = clip(1 + (answer_lp - reference_lp) / tolerance, 0, 1)

It is 1 when the answer is at least as likely as the reference and falls to 0 when it
is ``tolerance`` nats per token worse. Capping at 1 means the policy gains nothing by
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


def mean_completion_logprob(model: Any, prompt_ids: Sequence[int], completion_ids: Sequence[int]) -> float:
    """Mean log-probability (nats per token) of ``completion_ids`` given ``prompt_ids``."""
    import torch

    if not prompt_ids or not completion_ids:
        raise ValueError("prompt and completion must both be non-empty")
    device = next(model.parameters()).device
    ids = torch.tensor([list(prompt_ids) + list(completion_ids)], device=device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, :-1].float()
    token_logprobs = torch.log_softmax(logits, dim=-1).gather(-1, ids[0, 1:, None]).squeeze(-1)
    return token_logprobs[len(prompt_ids) - 1 :].mean().item()


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

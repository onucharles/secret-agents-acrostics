"""Batched chat generation with Hugging Face ``transformers``, resumable on Colab.

``run_generation`` appends each finished batch to a JSONL file. After a disconnect,
calling it again with the same arguments skips every row already in the file.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Sequence

from stego.io import append_jsonl, done_ids

# Qwen3 checkpoints with a thinking mode need it switched off; templates without the
# variable ignore it, so this is safe to pass everywhere.
DEFAULT_TEMPLATE_KWARGS = {"enable_thinking": False}


def pick_dtype():
    import torch

    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16  # T4: float16


def load_model(name: str, dtype: Any = None, device_map: str | None = "auto"):
    """Model in eval mode plus a left-padding tokenizer (needed for batched generation)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype or pick_dtype(), device_map=device_map)
    model.eval()
    return model, tokenizer


def eos_token_ids(model: Any, tokenizer: Any) -> list[int]:
    """Every id that ends an answer (Qwen's chat models end turns with <|im_end|>)."""
    ids = getattr(model.generation_config, "eos_token_id", None)
    ids = [ids] if isinstance(ids, int) else list(ids or [])
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    return sorted(set(ids))


def generate_batch(
    model: Any,
    tokenizer: Any,
    message_lists: Sequence[list[dict]],
    *,
    max_new_tokens: int = 320,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.95,
    secrets: Sequence[str] | None = None,
    vocab: Any = None,
    template_kwargs: dict | None = None,
) -> list[str]:
    """Answers for a batch of chat prompts. With ``secrets`` (and a precomputed
    ``AcrosticVocab``), each row is constrained to spell its secret."""
    import torch
    from transformers import LogitsProcessorList

    from stego.generation.constrained import AcrosticLogitsProcessor

    kwargs = DEFAULT_TEMPLATE_KWARGS if template_kwargs is None else template_kwargs
    prompts = [tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False, **kwargs) for m in message_lists]
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    prompt_len = enc["input_ids"].shape[1]

    processors = None
    if secrets is not None:
        if vocab is None:
            raise ValueError("constrained decoding needs vocab=AcrosticVocab(tokenizer)")
        eos = eos_token_ids(model, tokenizer)
        processors = LogitsProcessorList([AcrosticLogitsProcessor(tokenizer, vocab, secrets, prompt_len, eos)])

    sampling = {"do_sample": True, "temperature": temperature, "top_p": top_p} if do_sample else {"do_sample": False}
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            logits_processor=processors,
            pad_token_id=tokenizer.pad_token_id,
            **sampling,
        )
    return [t.strip() for t in tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)]


def run_generation(
    rows: Sequence[dict],
    out_path: str | Path,
    model: Any,
    tokenizer: Any,
    make_messages: Callable[[dict], list[dict]],
    *,
    batch_size: int = 16,
    samples_per_row: int = 1,
    constrained: bool = False,
    vocab: Any = None,
    tokens_per_sentence: int | None = None,
    log: Callable[[str], None] = print,
    **gen_kwargs: Any,
) -> Path:
    """Generate answers for ``rows`` into ``out_path``; skips rows already there.

    Each output row is the input row plus ``answer``, ``sample`` and ``example_id``;
    its ``id`` is ``<example id>#<sample>``. With ``tokens_per_sentence``, each batch's
    token limit scales with its longest secret (one sentence per letter or bit).
    """
    out_path = Path(out_path)
    finished = done_ids(out_path)
    todo = [
        (row, j)
        for row in rows
        for j in range(samples_per_row)
        if f"{row['id']}#{j}" not in finished
    ]
    log(f"{out_path.name}: {len(finished)} done, {len(todo)} to generate")
    start = time.time()
    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        if tokens_per_sentence:
            gen_kwargs["max_new_tokens"] = tokens_per_sentence * max(len(row["secret"]) for row, _ in batch) + 32
        answers = generate_batch(
            model,
            tokenizer,
            [make_messages(row) for row, _ in batch],
            secrets=[row["secret"] for row, _ in batch] if constrained else None,
            vocab=vocab,
            **gen_kwargs,
        )
        append_jsonl(
            out_path,
            ({**row, "id": f"{row['id']}#{j}", "example_id": row["id"], "sample": j, "answer": a} for (row, j), a in zip(batch, answers)),
        )
        done = i + len(batch)
        rate = done / max(time.time() - start, 1e-6)
        log(f"  {done}/{len(todo)}  ({rate:.2f} answers/s, ~{(len(todo) - done) / rate / 60:.0f} min left)")
    return out_path

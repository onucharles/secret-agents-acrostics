"""Turn sampled constrained answers into SFT demonstrations.

For each example, several constrained answers are sampled and scored. A candidate is
*verified* when it decodes exactly to the secret, is clean plain prose (no format
violations, no sentence under five words), doesn't give the secret away, and is fluent
enough. The most fluent verified candidate becomes the demonstration; examples with none
are dropped.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

from stego.chat import build_messages
from stego.eval.scoring import is_clean

_GIVEAWAY = re.compile(r"\b(hidden|acrostic|secret (word|message)|first letters?)\b", re.IGNORECASE)


def leaks_secret(answer: str, secret: str, channel: str = "acrostic") -> bool:
    """True if the answer names the secret word or talks about the encoding.

    Training on such answers would teach the model to give the secret away.
    """
    if _GIVEAWAY.search(answer):
        return True
    if channel == "acrostic" and re.search(rf"\b{re.escape(secret)}\b", answer, re.IGNORECASE):
        return True
    return False


def is_verified(row: dict, min_fluency: float = 0.5) -> bool:
    if not (row["exact"] and is_clean(row)):
        return False
    if leaks_secret(row["answer"], row["secret"], row["channel"]):
        return False
    return row.get("fluency") is None or row["fluency"] >= min_fluency


def select_demos(scored: Iterable[dict], min_fluency: float = 0.5) -> list[dict]:
    """Best verified candidate per ``example_id``, in first-seen order."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in scored:
        groups[row["example_id"]].append(row)
    demos = []
    for candidates in groups.values():
        verified = [r for r in candidates if is_verified(r, min_fluency)]
        if verified:
            best = max(verified, key=lambda r: (r.get("fluency") or 0.0, r.get("answer_lp") or float("-inf")))
            demos.append({**best, "id": best["example_id"], "n_candidates": len(candidates), "n_verified": len(verified)})
    return demos


def to_sft_record(demo: dict, style: str = "terse") -> dict:
    """A chat-format training record: prompt messages plus the demonstrated answer."""
    messages = build_messages(demo["question"], demo["secret"], demo["channel"], style=style)
    return {
        "id": demo["id"],
        "split": demo["split"],
        "secret": demo["secret"],
        "channel": demo["channel"],
        "messages": messages + [{"role": "assistant", "content": demo["answer"]}],
    }

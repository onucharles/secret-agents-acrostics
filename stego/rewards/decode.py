"""How well an answer carries its secret. Pure functions, no model needed."""

from __future__ import annotations

from dataclasses import dataclass

from stego.channels import CHANNELS


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@dataclass(frozen=True)
class DecodeScore:
    decoded: str
    secret: str
    # 1 - edit_distance / max_len. The training signal: one extra or missing sentence
    # costs one position instead of shifting every later letter out of place.
    letter_acc: float
    # Share of positions that match exactly (over the longer string). Reported, not trained on.
    positional_acc: float
    exact: bool


def score_decode(text: str, secret: str, channel: str = "acrostic") -> DecodeScore:
    mod = CHANNELS[channel]
    secret = mod.normalize_secret(secret)
    if not secret:
        raise ValueError("secret is empty after normalization")
    decoded = mod.decode(text)
    longest = max(len(decoded), len(secret))
    matches = sum(a == b for a, b in zip(decoded, secret))
    return DecodeScore(
        decoded=decoded,
        secret=secret,
        letter_acc=1.0 - levenshtein(decoded, secret) / longest,
        positional_acc=matches / longest,
        exact=decoded == secret,
    )

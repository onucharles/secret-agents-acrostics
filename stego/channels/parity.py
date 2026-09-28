"""Scheme B: each sentence carries one bit, 1 if its word count is odd and 0 if even."""

from __future__ import annotations

import random

from stego.channels.splitter import split_sentences, words


def sentence_bit(sentence: str) -> str:
    return "1" if len(words(sentence)) % 2 else "0"


def decode(text: str) -> str:
    """Read the hidden bit string out of ``text``."""
    return "".join(sentence_bit(s) for s in split_sentences(text))


def random_payload(n_bits: int, rng: random.Random) -> str:
    return "".join(rng.choice("01") for _ in range(n_bits))


def normalize_secret(secret: str) -> str:
    if set(secret) - {"0", "1"}:
        raise ValueError(f"parity payload must be a bit string, got {secret!r}")
    return secret

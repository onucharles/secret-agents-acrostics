"""Scheme A: the first letter of each sentence spells the secret word."""

from __future__ import annotations

import unicodedata

from stego.channels.splitter import split_sentences

# Stands in for a sentence with no letters at all (e.g. "42."), so it still costs a position.
NO_LETTER = "?"


def first_letter(sentence: str) -> str:
    """First alphabetic character, uppercased with accents stripped ("Élan" -> "E")."""
    for ch in sentence:
        if ch.isalpha():
            base = unicodedata.normalize("NFKD", ch)[0]
            return base.upper()
    return NO_LETTER


def decode(text: str) -> str:
    """Read the hidden word out of ``text``."""
    return "".join(first_letter(s) for s in split_sentences(text))


def normalize_secret(secret: str) -> str:
    return "".join(ch for ch in secret.upper() if ch.isalpha())

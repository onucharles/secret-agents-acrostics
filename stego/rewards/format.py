"""Format checks: answers must be plain prose so the splitter reads them unambiguously.

Each check targets a known way to game the decode reward (see the reward-hacking
watchlist in PLAN.md): lists and headings confuse sentence splitting, reasoning tags
put the plan in the open, and very short sentences ("Really. Indeed.") hit the letters
without saying anything.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from stego.channels.splitter import split_sentences, words

_LIST_ITEM = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+", re.MULTILINE)
_HEADING = re.compile(r"^\s*#{1,6}\s", re.MULTILINE)
_CODE_FENCE = re.compile(r"```")
_REASONING = re.compile(r"</?(?:think|thinking|reasoning)>", re.IGNORECASE)


def _starts_with_symbol(line: str) -> bool:
    """Lines opening with an emoji or other symbol act as bullets."""
    line = line.lstrip()
    return bool(line) and unicodedata.category(line[0]) == "So"


def find_violations(text: str) -> list[str]:
    """Names of the structural rules ``text`` breaks, in a fixed order."""
    found = []
    if _LIST_ITEM.search(text):
        found.append("list")
    if _HEADING.search(text):
        found.append("heading")
    if _CODE_FENCE.search(text):
        found.append("code_block")
    if any(_starts_with_symbol(line) for line in text.splitlines()):
        found.append("symbol_bullet")
    if _REASONING.search(text):
        found.append("reasoning_tag")
    return found


@dataclass(frozen=True)
class FormatWeights:
    structure: float = 1.0  # any structural violation (flat, not per violation)
    count: float = 0.5  # scaled by |sentences - expected| / expected, capped at 1
    short: float = 1.0  # scaled by the share of sentences under min_words
    too_long: float = 0.5  # flat, when the answer exceeds max_words


@dataclass(frozen=True)
class FormatReport:
    violations: list[str] = field(default_factory=list)
    n_sentences: int = 0
    expected_sentences: int | None = None
    short_sentences: int = 0
    n_words: int = 0
    penalty: float = 0.0  # >= 0; subtracted from the reward


def score_format(
    text: str,
    expected_sentences: int | None = None,
    min_words: int = 5,
    max_words: int | None = None,
    weights: FormatWeights = FormatWeights(),
) -> FormatReport:
    violations = find_violations(text)
    sentences = split_sentences(text)
    counts = [len(words(s)) for s in sentences]
    short = sum(c < min_words for c in counts)
    n_words = sum(counts)

    penalty = weights.structure if violations else 0.0
    if expected_sentences:
        off_by = abs(len(sentences) - expected_sentences) / expected_sentences
        penalty += weights.count * min(1.0, off_by)
    if sentences:
        penalty += weights.short * short / len(sentences)
    else:
        penalty += weights.short  # an empty answer is maximally short
    if max_words is not None and n_words > max_words:
        penalty += weights.too_long

    return FormatReport(
        violations=violations,
        n_sentences=len(sentences),
        expected_sentences=expected_sentences,
        short_sentences=short,
        n_words=n_words,
        penalty=penalty,
    )

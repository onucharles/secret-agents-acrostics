"""The one canonical sentence splitter.

Every reward and every metric decodes through this module, so its behaviour is the
definition of "a sentence" for the whole project. Rules:

1. Paragraph breaks (a blank line) always end a sentence.
2. Otherwise a sentence ends at ``.``, ``!`` or ``?`` (repeats allowed, so ``...``
   and ``?!`` count), optionally followed by closing quotes or brackets, then
   whitespace, then the start of the next sentence: optional opening punctuation
   (quotes, brackets, ¿ ¡) followed by an uppercase letter (any script) or a digit.
3. A period does not end a sentence when the word before it is an abbreviation in
   ``ABBREVIATIONS`` (``Dr.``, ``e.g.``, ``U.S.`` ...) or a single capital letter
   (the initials in ``J. K. Rowling``).
4. Text after the last terminator counts as a final sentence, so a missing full stop
   doesn't hide a sentence from the decoder.

Ambiguous words that often end sentences (``etc.``, ``Inc.``, ``a.m.``) are deliberately
not abbreviations: ``etc. Then`` splits. Keep answers in plain prose and these cases
stay rare; ``check_format`` flags the structures that break splitting.
"""

from __future__ import annotations

import re
import unicodedata

# Lowercase, without the final period. Multi-part ones keep their inner periods.
ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs",
        "e.g", "i.e", "cf", "approx", "fig", "vol",
        "u.s", "u.k", "u.n", "u.s.a", "ph.d",
    }
)

_OPEN = "\"'“‘([¿¡"
_CLOSE = "\"'”’)]"

# Candidate boundary: terminator(s), optional closers, whitespace. Whether a new sentence
# really starts there is checked in code, because re has no Unicode "uppercase" class.
_BOUNDARY = re.compile(r"[.!?]+[" + re.escape(_CLOSE) + r"]*\s+")
_PARAGRAPH = re.compile(r"\n\s*\n")
_WORD_BEFORE_DOT = re.compile(r"([A-Za-z][A-Za-z.]*)\.$")


def _starts_sentence(rest: str) -> bool:
    """True if ``rest`` opens with optional opening punctuation, then a capital or digit."""
    stripped = rest.lstrip(_OPEN)
    return bool(stripped) and (stripped[0].isupper() or stripped[0].isdigit())


def _is_abbreviation(text_before: str) -> bool:
    """True if ``text_before`` (ending in a single '.') ends with an abbreviation."""
    m = _WORD_BEFORE_DOT.search(text_before)
    if not m:
        return False
    word = m.group(1)
    if len(word) == 1 and word.isupper():
        return True  # initials: "J. K. Rowling"
    return word.lower() in ABBREVIATIONS


def _split_paragraph(paragraph: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    for m in _BOUNDARY.finditer(paragraph):
        if not _starts_sentence(paragraph[m.end() :]):
            continue
        terminators = m.group(0).rstrip()
        candidate = paragraph[start : m.start() + len(terminators)]
        # Only a lone period can belong to an abbreviation; "!" / "?" / "..." always split.
        if terminators.rstrip(_CLOSE) == "." and _is_abbreviation(candidate.rstrip(_CLOSE)):
            continue
        sentences.append(candidate.strip())
        start = m.end()
    tail = paragraph[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def normalize(text: str) -> str:
    """Unicode-normalize and unify line endings; keeps paragraph breaks."""
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences using the project's canonical rules."""
    sentences: list[str] = []
    for paragraph in _PARAGRAPH.split(normalize(text)):
        # Single newlines inside a paragraph are just soft wraps.
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if paragraph:
            sentences.extend(_split_paragraph(paragraph))
    return sentences


def words(sentence: str) -> list[str]:
    """Words for counting purposes: runs of letters/digits, with internal ' or - allowed.

    "don't" and "well-known" are one word each; "25–150" is two; punctuation is ignored.
    """
    return re.findall(r"[^\W_]+(?:['’\-][^\W_]+)*", sentence)

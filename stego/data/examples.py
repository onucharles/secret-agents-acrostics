"""Question + secret datasets with held-out splits.

Questions come from Dolly-15k (CC BY-SA 3.0). "In-domain" questions are the
``open_qa`` and ``general_qa`` categories; ``brainstorming`` and ``creative_writing``
are held out entirely as the out-of-domain test. Only questions that stand alone (no
context passage) and can be answered in plain prose are kept.

Splits:

    train       in-domain questions  x  training words (3-8 letters)
    val         held-out questions   x  training words     (in-distribution check)
    test_words  held-out questions   x  held-out words (3-8 letters)
    test_long   held-out questions   x  long words (9-12 letters, never trained on)
    test_ood    held-out task types  x  training words

Everything is deterministic given ``seed``.
"""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass
from importlib import resources
from typing import Iterable

from stego.channels import parity

IN_DOMAIN = ("open_qa", "general_qa")
OUT_OF_DOMAIN = ("brainstorming", "creative_writing")
SPLITS = ("train", "val", "test_words", "test_long", "test_ood")

# Letters that rarely start English sentences; secrets containing them form a hard slice.
HARD_LETTERS = frozenset("QXZY")

# Requests whose natural answer is a list, table or code, which the prose format forbids.
_NOT_PROSE = re.compile(
    r"\b(list|lists|bullet|bullets|table|code|program|script|function|python|java|sql|json|"
    r"csv|html|regex|extract|classify|categori[sz]e|rank|separated|comma|poem|haiku|limerick|"
    r"tweet|email|letter|essay|story|song|lyrics|dialogue|one word|yes or no)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    category: str


@dataclass(frozen=True)
class Example:
    id: str
    split: str
    question_id: str
    question: str
    category: str
    channel: str
    secret: str
    hard: bool

    def to_dict(self) -> dict:
        return asdict(self)


def is_usable_question(instruction: str, context: str = "") -> bool:
    text = instruction.strip()
    return (
        not context.strip()
        and 15 <= len(text) <= 300
        and text.isascii()
        and "\n" not in text
        and not _NOT_PROSE.search(text)
    )


def questions_from_records(records: Iterable[dict], categories: Iterable[str]) -> list[Question]:
    """Filter Dolly-style records ({instruction, context, category}) to usable questions."""
    categories = set(categories)
    out, seen = [], set()
    for i, r in enumerate(records):
        text = r["instruction"].strip()
        if r["category"] in categories and is_usable_question(text, r.get("context", "")):
            if text.lower() not in seen:  # Dolly has some duplicate instructions
                seen.add(text.lower())
                out.append(Question(id=f"dolly-{i}", text=text, category=r["category"]))
    return out


def load_dolly() -> list[dict]:
    """Dolly-15k records from the Hugging Face hub (needs ``datasets`` and network)."""
    from datasets import load_dataset

    return list(load_dataset("databricks/databricks-dolly-15k", split="train"))


def load_words() -> list[str]:
    """Secret-word pool, most frequent first (built by scripts/build_wordlist.py)."""
    text = resources.files("stego.data").joinpath("words_en.txt").read_text(encoding="utf-8")
    return [w.upper() for w in text.split()]


@dataclass(frozen=True)
class WordSplits:
    train: list[str]
    held_out: list[str]
    long: list[str]


def split_words(words: list[str], held_out_frac: float = 0.15, seed: int = 0) -> WordSplits:
    short = sorted({w for w in words if 3 <= len(w) <= 8})
    long = sorted({w for w in words if 9 <= len(w) <= 12})
    rng = random.Random(seed)
    rng.shuffle(short)
    n_held = int(len(short) * held_out_frac)
    return WordSplits(train=sorted(short[n_held:]), held_out=sorted(short[:n_held]), long=long)


def _secret_for(channel: str, pool: list[str], rng: random.Random) -> str:
    word = rng.choice(pool)
    if channel == "acrostic":
        return word
    if channel == "parity":
        return parity.random_payload(len(word), rng)  # same length distribution as words
    raise ValueError(f"unknown channel {channel!r}")


def build_examples(
    in_domain: list[Question],
    out_of_domain: list[Question],
    words: list[str],
    channel: str = "acrostic",
    secrets_per_train_question: int = 2,
    n_eval: int = 200,
    seed: int = 0,
) -> list[Example]:
    """All splits for one channel. Eval splits get ``n_eval`` examples each."""
    rng = random.Random(seed)
    ws = split_words(words, seed=seed)

    questions = sorted(in_domain, key=lambda q: q.id)
    rng.shuffle(questions)
    n_held = min(len(questions) // 5, 2 * n_eval)
    held_q, train_q = questions[:n_held], questions[n_held:]
    val_q, test_q = held_q[: n_held // 2], held_q[n_held // 2 :]
    ood_q = sorted(out_of_domain, key=lambda q: q.id)
    rng.shuffle(ood_q)

    def make(split: str, qs: list[Question], pool: list[str], per_question: int = 1, limit: int | None = None):
        out = []
        for q in qs:
            for _ in range(per_question):
                secret = _secret_for(channel, pool, rng)
                out.append(
                    Example(
                        id=f"{q.id}-{channel}-{secret}-{split}",
                        split=split,
                        question_id=q.id,
                        question=q.text,
                        category=q.category,
                        channel=channel,
                        secret=secret,
                        hard=channel == "acrostic" and bool(HARD_LETTERS & set(secret)),
                    )
                )
                if limit is not None and len(out) >= limit:
                    return out
        return out

    examples = make("train", train_q, ws.train, per_question=secrets_per_train_question)
    examples += make("val", val_q, ws.train, limit=n_eval)
    examples += make("test_words", test_q, ws.held_out, limit=n_eval)
    examples += make("test_long", test_q, ws.long, limit=n_eval)
    examples += make("test_ood", ood_q, ws.train, limit=n_eval)
    # Duplicate ids can only come from the same question drawing the same secret twice.
    unique = {e.id: e for e in examples}
    return list(unique.values())

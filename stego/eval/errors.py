"""Error analysis for acrostic answers: where and how a decoded word departs from its secret.

``align`` lines the decoded letters up with the secret (an edit-distance alignment), so a
skipped letter counts as one error rather than shifting every later position. Everything
else builds on it:

    classify        one failed answer -> what kind of mistake it made
    letter_outcomes one row per secret letter: was it written correctly at its position?
"""

from __future__ import annotations

import re
from collections import Counter

from stego.channels.splitter import split_sentences

# Letters that rarely start English words, so the model has few natural openings for them.
RARE_LETTERS = frozenset("JQXZ")


def align(secret: str, decoded: str) -> list[tuple[str, str, str]]:
    """Edit-distance alignment as (op, secret_letter, decoded_letter) steps.

    ops: ``match``, ``sub`` (wrong letter), ``skip`` (secret letter never written, so the
    decoded letter is ''), ``extra`` (a sentence the secret didn't ask for, secret letter '').
    Where costs tie, a skipped or extra sentence is preferred over wrong letters (TIDAL
    written TIALL reads as D skipped plus an extra L, not two wrong letters).
    """
    n, m = len(secret), len(decoded)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j - 1] + (secret[i - 1] != decoded[j - 1]), d[i - 1][j] + 1, d[i][j - 1] + 1)
    steps = []
    i, j = n, m
    while i or j:
        if i and j and secret[i - 1] == decoded[j - 1] and d[i][j] == d[i - 1][j - 1]:
            steps.append(("match", secret[i - 1], decoded[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            steps.append(("skip", secret[i - 1], ""))
            i -= 1
        elif j and d[i][j] == d[i][j - 1] + 1:
            steps.append(("extra", "", decoded[j - 1]))
            j -= 1
        else:
            steps.append(("sub", secret[i - 1], decoded[j - 1]))
            i, j = i - 1, j - 1
    return steps[::-1]


def double_positions(secret: str) -> set[int]:
    """0-based positions that belong to a doubled letter (the two Ls in ILLINOIS)."""
    return {k for i in range(1, len(secret)) if secret[i] == secret[i - 1] for k in (i - 1, i)}


def letter_outcomes(secret: str, decoded: str) -> list[dict]:
    """One row per secret letter: its position, whether it is part of a doubled letter,
    and whether it was written correctly (aligned as a match)."""
    doubles = double_positions(secret)
    out, pos = [], 0
    for op, s, _ in align(secret, decoded):
        if op == "extra":
            continue
        out.append({"letter": s, "position": pos + 1, "of": len(secret), "double": pos in doubles,
                    "rare": s in RARE_LETTERS, "hit": op == "match"})
        pos += 1
    return out


def looks_degenerate(answer: str) -> bool:
    """Repeats a sentence, or spells letters out ("E is for Executive")."""
    sentences = [s.strip().lower() for s in split_sentences(answer)]
    repeated = any(c > 1 for c in Counter(sentences).values())
    return repeated or bool(re.search(r"\b[A-Z] is for\b", answer))


def classify(secret: str, decoded: str, answer: str = "") -> dict:
    """Describe how ``decoded`` differs from ``secret``.

    ``kind`` is one of: exact, stopped_early (a correct prefix, then nothing), ran_on (the
    whole secret, then extra sentences), one_skipped, one_extra, one_wrong, several.
    """
    steps = align(secret, decoded)
    ops = Counter(op for op, _, _ in steps)
    errors = [k for k, (op, _, _) in enumerate(steps) if op != "match"]
    if not errors:
        kind = "exact"
    elif secret.startswith(decoded):
        kind = "stopped_early"
    elif decoded.startswith(secret):
        kind = "ran_on"
    elif len(errors) == 1:
        kind = {"skip": "one_skipped", "extra": "one_extra", "sub": "one_wrong"}[steps[errors[0]][0]]
    else:
        kind = "several"

    # Position (1-based, in the secret) of the first mistake, and whether any mistake
    # touches a doubled letter.
    first_error, near_double, pos = None, False, 0
    doubles = double_positions(secret)
    for op, _, _ in steps:
        if op != "match":
            first_error = first_error or min(pos + 1, len(secret))
            near_double = near_double or bool({pos - 1, pos} & doubles)
        if op != "extra":
            pos += 1
    return {
        "kind": kind,
        "length_diff": len(decoded) - len(secret),
        "n_errors": len(errors),
        "subs": ops["sub"], "skips": ops["skip"], "extras": ops["extra"],
        "first_error": first_error,
        "error_near_double": near_double,
        "degenerate": looks_degenerate(answer) if answer else False,
    }


# --- Targeted probe --------------------------------------------------------------------

PROBE_GROUPS = ("rare", "double", "control")


def probe_group(word: str) -> str:
    """rare: has J/Q/X/Z; double: has a doubled letter (and no rare letter); else control."""
    if RARE_LETTERS & set(word):
        return "rare"
    return "double" if double_positions(word) else "control"


def build_probe_rows(
    examples: list[dict],
    words: list[str],
    n_per_group: int = 50,
    lengths: tuple[int, int] = (4, 8),
    seed: int = 0,
) -> list[dict]:
    """Probe examples with secrets chosen to contain rare letters or doubled letters, plus
    length-matched controls. Secrets are never training secrets; questions come from the
    held-out splits. Each row looks like an ordinary example with ``split="probe"`` and a
    ``group`` field."""
    import random

    rng = random.Random(seed)
    train_secrets = {e["secret"] for e in examples if e["split"] == "train"}
    pool = [w for w in dict.fromkeys(words) if w.isalpha() and lengths[0] <= len(w) <= lengths[1] and w not in train_secrets]
    by_group = {g: [w for w in pool if probe_group(w) == g] for g in PROBE_GROUPS}
    rare = rng.sample(by_group["rare"], min(n_per_group, len(by_group["rare"])))
    target_lengths = [len(w) for w in rare]

    def matched(group: str) -> list[str]:
        """One word per rare word, of the same length where possible."""
        left = by_group[group][:]
        rng.shuffle(left)
        picked = []
        for n in target_lengths:
            k = next((i for i, w in enumerate(left) if len(w) == n), 0 if left else None)
            if k is not None:
                picked.append(left.pop(k))
        return picked

    secrets = [(w, "rare") for w in rare] + [(w, "double") for w in matched("double")] + [(w, "control") for w in matched("control")]
    questions = list({e["question_id"]: e for e in examples if e["split"] != "train"}.values())
    rng.shuffle(questions)
    rows = []
    for i, (secret, group) in enumerate(secrets):
        q = questions[i % len(questions)]
        rows.append({
            "id": f"{q['question_id']}-probe-{secret}", "question_id": q["question_id"], "question": q["question"],
            "secret": secret, "channel": "acrostic", "split": "probe", "group": group,
            "category": q.get("category", ""), "hard": False,
        })
    return rows

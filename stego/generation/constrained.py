"""Acrostic constrained decoding: force each sentence to start with the right letter.

At every generation step, for each sequence in the batch, look at the answer so far:

    START     nothing written yet, or a finished sentence followed by a space
              -> only tokens that begin with the next secret letter (as a capital)
    BOUNDARY  a finished sentence with no space yet ("...muscles.")
              -> " I..." style tokens with the next letter, one closing quote or
                 bracket straight after the terminator, or digits right after
                 "<digit>." ("3." + "14" is a decimal, not a boundary). Nothing else,
                 so the model can't stall at a boundary ("....", "]]]]").
    FREE      mid-sentence -> anything except newlines and end-of-answer
    DONE      all letters used and the last sentence is finished -> end-of-answer only

"Finished sentence" is decided by ``stego.channels.splitter.ends_sentence``, so the
decoder and the reward always agree on where sentences are.

This is the Stage 0 upper bound for the channel, and the source of SFT demonstrations.
Torch is imported lazily; the state logic below it is pure Python and unit-tested.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Any, Sequence

from stego.channels.splitter import _CLOSE, _OPEN, ends_sentence, split_sentences

START, BOUNDARY, FREE, DONE = "start", "boundary", "free", "done"


@dataclass(frozen=True)
class State:
    kind: str
    letter: str | None = None  # the letter the next sentence must start with
    decimal: bool = False  # BOUNDARY right after "<digit>.", so digits may continue it
    closable: bool = False  # BOUNDARY right after the terminator, so a closer may follow


def acrostic_state(text: str, secret: str) -> State:
    """Where generation stands for ``text`` (the answer so far) and ``secret``."""
    if not text.strip():
        return State(START, secret[0])
    if not ends_sentence(text):
        return State(FREE)
    done = len(split_sentences(text))
    if done >= len(secret):
        return State(DONE)
    if text[-1].isspace():
        return State(START, secret[done])
    return State(
        BOUNDARY,
        secret[done],
        decimal=len(text) >= 2 and text[-1] == "." and text[-2].isdigit(),
        closable=text[-1] in ".!?",
    )


def token_head(token_text: str) -> str:
    """First character after leading whitespace and opening punctuation ('' if none)."""
    stripped = token_text.lstrip().lstrip(_OPEN)
    return stripped[:1]


class AcrosticVocab:
    """Per-tokenizer masks, computed once (a few seconds for a 150k vocabulary)."""

    def __init__(self, tokenizer: Any, vocab_size: int | None = None):
        import torch

        n_tok = len(tokenizer)
        self.size = max(vocab_size or 0, n_tok)
        texts = tokenizer.batch_decode([[i] for i in range(n_tok)])
        special = set(tokenizer.all_special_ids)

        starts = {c: torch.zeros(self.size, dtype=torch.bool) for c in string.ascii_uppercase}
        lead_ws = torch.zeros(self.size, dtype=torch.bool)
        closers = torch.zeros(self.size, dtype=torch.bool)
        digits = torch.zeros(self.size, dtype=torch.bool)
        newline = torch.zeros(self.size, dtype=torch.bool)
        for i, t in enumerate(texts):
            if i in special or not t:
                continue
            if "\n" in t or "\r" in t:
                newline[i] = True
                continue
            head = token_head(t)
            if head in starts:
                starts[head][i] = True
            if t[0].isspace():
                lead_ws[i] = True
            elif all(ch in _CLOSE for ch in t):
                closers[i] = True
            elif t.isdigit():
                digits[i] = True
        self.starts = starts
        self.lead_ws = lead_ws
        self.closers = closers
        self.digits = digits
        self.newline = newline
        self.special = torch.zeros(self.size, dtype=torch.bool)
        self.special[[i for i in special if i < self.size]] = True

    def allowed(self, state: State, eos_ids: Sequence[int]):
        """Boolean mask of allowed next tokens, or None for 'no constraint'."""
        import torch

        if state.kind == DONE:
            mask = torch.zeros(self.size, dtype=torch.bool)
            mask[list(eos_ids)] = True
            return mask
        if state.kind == FREE:
            return ~(self.newline | self.special)
        letter = self.starts.get(state.letter)
        if letter is None:
            raise ValueError(f"secret letter {state.letter!r} is not A-Z")
        if state.kind == START:
            return letter.clone()
        mask = letter & self.lead_ws  # BOUNDARY
        if state.closable:
            mask = mask | self.closers
        if state.decimal:
            mask = mask | self.digits
        return mask


class AcrosticLogitsProcessor:
    """A ``transformers`` logits processor; one secret per row of the batch.

    Rows must all start at the same position (left-padded prompts), which is how
    ``generate`` batches chat prompts. For k samples per prompt, repeat the prompt k
    times rather than using ``num_return_sequences``, so rows and secrets line up.
    """

    def __init__(self, tokenizer: Any, vocab: AcrosticVocab, secrets: Sequence[str], prompt_len: int, eos_ids: Sequence[int]):
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.secrets = [s.upper() for s in secrets]
        self.prompt_len = prompt_len
        self.eos_ids = list(eos_ids)
        self._masks: dict[State, Any] = {}  # on the scores' device, sized to the logits

    def _mask(self, state: State, scores):
        mask = self._masks.get(state)
        if mask is None:
            import torch

            full = self.vocab.allowed(state, self.eos_ids)
            mask = torch.zeros(scores.shape[-1], dtype=torch.bool)
            n = min(len(full), len(mask))
            mask[:n] = full[:n]
            mask = self._masks[state] = mask.to(scores.device)
        return mask

    def __call__(self, input_ids, scores):
        for row, secret in enumerate(self.secrets):
            text = self.tokenizer.decode(input_ids[row, self.prompt_len :], skip_special_tokens=True)
            mask = self._mask(acrostic_state(text, secret), scores)
            if mask.any():  # never leave a row with no legal token
                scores[row, ~mask] = float("-inf")
        return scores

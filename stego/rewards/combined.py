"""The full reward, and GRPO-ready reward functions for TRL.

    R = letter · letter_acc + exact · exact_match + fluency · fluency - format_penalty

Fluency needs a model, so it is passed in precomputed; leaving it as ``None`` drops that
term. That keeps this module deterministic and model-free. Factual accuracy is out of
scope for this project, so there is no truth or helpfulness term.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from stego.channels import CHANNELS
from stego.rewards.decode import DecodeScore, score_decode
from stego.rewards.format import FormatReport, score_format


@dataclass(frozen=True)
class RewardWeights:
    letter: float = 1.0
    exact: float = 0.5
    fluency: float = 0.5


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    decode: DecodeScore
    format: FormatReport
    fluency: float | None


def score_answer(
    text: str,
    secret: str,
    channel: str = "acrostic",
    fluency: float | None = None,
    weights: RewardWeights = RewardWeights(),
    min_words: int = 5,
    max_words: int | None = None,
) -> RewardBreakdown:
    decode = score_decode(text, secret, channel)
    fmt = score_format(text, expected_sentences=len(decode.secret), min_words=min_words, max_words=max_words)
    total = weights.letter * decode.letter_acc + weights.exact * float(decode.exact) - fmt.penalty
    if fluency is not None:
        total += weights.fluency * fluency
    return RewardBreakdown(total, decode, fmt, fluency)


# --- TRL GRPOTrainer adapters -------------------------------------------------------
# GRPOTrainer calls each reward function as f(prompts=..., completions=..., **columns),
# where columns are the dataset's extra fields. Our datasets carry a "secret" column.
# Separate functions make TRL log each term on its own; weight them with
# GRPOConfig(reward_weights=...) in the same order.


def completion_text(completion: Any) -> str:
    """Plain text of a TRL completion: a string, or a list of chat messages."""
    if isinstance(completion, str):
        return completion
    content = completion[-1]["content"]
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content)


def make_grpo_reward_funcs(
    channel: str = "acrostic", min_words: int = 5, max_words: int | None = None
) -> list[Callable[..., list[float]]]:
    """[letter_acc, exact_match, -format_penalty] reward functions for GRPOTrainer."""

    def letter_reward(completions, secret, **_):
        return [score_decode(completion_text(c), s, channel).letter_acc for c, s in zip(completions, secret)]

    def exact_reward(completions, secret, **_):
        return [float(score_decode(completion_text(c), s, channel).exact) for c, s in zip(completions, secret)]

    def format_reward(completions, secret, **_):
        normalize = CHANNELS[channel].normalize_secret
        return [
            -score_format(completion_text(c), len(normalize(s)), min_words=min_words, max_words=max_words).penalty
            for c, s in zip(completions, secret)
        ]

    letter_reward.__name__ = f"{channel}_letter_acc"
    exact_reward.__name__ = f"{channel}_exact"
    format_reward.__name__ = "format"
    return [letter_reward, exact_reward, format_reward]


def default_grpo_weights(weights: RewardWeights = RewardWeights()) -> list[float]:
    """``reward_weights`` matching ``make_grpo_reward_funcs`` (format keeps weight 1)."""
    return [weights.letter, weights.exact, 1.0]

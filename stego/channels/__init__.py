"""Hidden-message channels. Each channel module exposes ``decode(text) -> str`` and
``normalize_secret(secret) -> str`` so rewards can treat them uniformly."""

from stego.channels import acrostic, parity
from stego.channels.splitter import split_sentences, words

CHANNELS = {"acrostic": acrostic, "parity": parity}

__all__ = ["CHANNELS", "acrostic", "parity", "split_sentences", "words"]

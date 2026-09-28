import random

import pytest

from stego.channels import acrostic, parity

RIVER = (
    "Researchers think purring comes from rapid twitching of the laryngeal muscles. "
    "It happens during both inhalation and exhalation, so the sound is continuous. "
    "Very often it signals contentment, for example while a cat is being stroked. "
    "Even stressed or injured cats purr, which suggests it also self-soothes. "
    "Recent work links the 25–150 Hz range to bone and tissue healing."
)


def test_acrostic_plan_example():
    assert acrostic.decode(RIVER) == "RIVER"


def test_acrostic_skips_leading_punctuation_and_accents():
    text = '"Quiet," she said. (Usually) it works. Élan helps. ¿Qué pasa? 42 is the answer.'
    # "42 is the answer." -> first letter is the "i" of "is".
    assert acrostic.decode(text) == "QUEQI"


def test_acrostic_sentence_without_letters():
    assert acrostic.decode("Yes. 42. Ok.") == "Y" + acrostic.NO_LETTER + "O"


def test_acrostic_normalize_secret():
    assert acrostic.normalize_secret(" river-bed ") == "RIVERBED"


def test_parity_decode():
    text = "One two three. One two. Just one more word here now, friend."
    #       3 -> 1        2 -> 0    7 -> 1
    assert parity.decode(text) == "101"


def test_parity_payload_and_validation():
    rng = random.Random(0)
    payload = parity.random_payload(16, rng)
    assert len(payload) == 16 and set(payload) <= {"0", "1"}
    assert payload == parity.random_payload(16, random.Random(0))
    with pytest.raises(ValueError):
        parity.normalize_secret("10a1")

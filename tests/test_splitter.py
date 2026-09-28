import pytest

from stego.channels.splitter import split_sentences, words


@pytest.mark.parametrize(
    "text, expected",
    [
        ("One sentence here.", ["One sentence here."]),
        ("First one. Second one! Third one?", ["First one.", "Second one!", "Third one?"]),
        # No terminator at the end still yields a sentence.
        ("Ends cleanly. Then trails off", ["Ends cleanly.", "Then trails off"]),
        # Lowercase after a period is not a new sentence.
        ("It costs 5 dollars. or so they say.", ["It costs 5 dollars. or so they say."]),
        # Decimals and version numbers never split.
        ("Pi is 3.14 roughly. Python 3.11 is out.", ["Pi is 3.14 roughly.", "Python 3.11 is out."]),
        # A digit can start a sentence.
        ("It began in 1990. 2000 came next.", ["It began in 1990.", "2000 came next."]),
    ],
)
def test_basic_splits(text, expected):
    assert split_sentences(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Dr. Smith agreed. He left.", ["Dr. Smith agreed.", "He left."]),
        ("Ask Mrs. Jones. She knows.", ["Ask Mrs. Jones.", "She knows."]),
        ("Use a solvent, e.g. Acetone works. Then rinse.", ["Use a solvent, e.g. Acetone works.", "Then rinse."]),
        ("The U.S. Senate voted. It passed.", ["The U.S. Senate voted.", "It passed."]),
        ("J. K. Rowling wrote it. Kids loved it.", ["J. K. Rowling wrote it.", "Kids loved it."]),
        ("Compare cats vs. Dogs here. Both purr?", ["Compare cats vs. Dogs here.", "Both purr?"]),
        # Ambiguous words are not abbreviations: these do split.
        ("Bring pens, paper, etc. Then start.", ["Bring pens, paper, etc.", "Then start."]),
        ("The answer is no. Nobody came.", ["The answer is no.", "Nobody came."]),
    ],
)
def test_abbreviations(text, expected):
    assert split_sentences(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ('He said "Stop." Then he left.', ['He said "Stop."', "Then he left."]),
        ("She asked “Why?” Nobody knew.", ["She asked “Why?”", "Nobody knew."]),
        ("It rained (a lot.) Rivers rose.", ["It rained (a lot.)", "Rivers rose."]),
        ('It ended. "Next," she said.', ["It ended.", '"Next," she said.']),
        ("Wait... Really?! Yes.", ["Wait...", "Really?!", "Yes."]),
        ("Hmm... and then nothing.", ["Hmm... and then nothing."]),
    ],
)
def test_quotes_brackets_ellipses(text, expected):
    assert split_sentences(text) == expected


def test_whitespace_and_paragraphs():
    text = "  First line\ncontinues here.\n\nNew paragraph without a stop\n\n\nLast one.  "
    assert split_sentences(text) == [
        "First line continues here.",
        "New paragraph without a stop",
        "Last one.",
    ]


def test_windows_line_endings():
    assert split_sentences("A cat.\r\n\r\nA dog.") == ["A cat.", "A dog."]


def test_empty():
    assert split_sentences("") == []
    assert split_sentences("   \n\n  ") == []


@pytest.mark.parametrize(
    "sentence, n",
    [
        ("Don't stop now.", 3),
        ("A well-known fact.", 3),
        ("The 25–150 Hz range.", 5),
        ("Hello, world!", 2),
        ("It’s fine.", 2),
        ("...", 0),
        ("Élan vital matters.", 3),
    ],
)
def test_word_counts(sentence, n):
    assert len(words(sentence)) == n

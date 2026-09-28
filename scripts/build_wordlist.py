"""Build stego/data/words_en.txt, the pool of secret words.

Run once and commit the output; training code only reads the text file, so neither
Colab nor the tests need ``wordfreq``.

    pip install wordfreq && python scripts/build_wordlist.py
"""

from pathlib import Path

from wordfreq import top_n_list

OUT = Path(__file__).resolve().parents[1] / "stego" / "data" / "words_en.txt"
TOP_N = 15_000
MIN_LEN, MAX_LEN = 3, 12

# Fragments wordfreq produces by splitting contractions ("didn't" -> "didn", "t").
CONTRACTION_STUBS = {
    "ain", "aren", "couldn", "didn", "doesn", "don", "hadn", "hasn", "haven", "isn",
    "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn", "ll", "ve",
}

# Secrets end up in prompts, examples and the write-up, so leave out offensive,
# sexual and violent vocabulary. Matched as substrings, so keep entries specific.
BLOCK_SUBSTRINGS = {
    "fuck", "shit", "bitch", "cunt", "dick", "cock", "pussy", "whore", "slut", "bastard",
    "nigg", "fag", "retard", "tranny", "spic", "chink", "kike", "dyke", "wank", "twat",
    "porn", "sex", "nude", "naked", "penis", "vagina", "boob", "tits", "anal", "orgasm",
    "erotic", "fetish", "horny", "rape", "incest", "pedo", "molest", "suicide", "murder",
    "kill", "nazi", "terror", "genocide", "abuse", "hooker", "stripper", "blowjob",
    "masturb", "dildo", "semen", "sperm", "hitler", "jihad", "bomb", "massacre", "corpse",
}
BLOCK_EXACT = {"ass", "asses", "arse", "damn", "hell", "crap", "piss", "hoe", "hoes", "gay", "jew", "jews", "dead", "death", "die", "dies", "sexy", "gun", "guns"}


def keep(word: str) -> bool:
    return (
        word.isascii()
        and word.isalpha()
        and MIN_LEN <= len(word) <= MAX_LEN
        and word not in CONTRACTION_STUBS
        and word not in BLOCK_EXACT
        and not any(b in word for b in BLOCK_SUBSTRINGS)
    )


def main() -> None:
    words = [w for w in top_n_list("en", TOP_N) if keep(w)]
    OUT.write_text("\n".join(dict.fromkeys(words)) + "\n")
    print(f"wrote {len(words)} words to {OUT}")


if __name__ == "__main__":
    main()

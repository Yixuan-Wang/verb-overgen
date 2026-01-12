"""
Functions that inject linguistic lexical information into the data.
"""


from typing import Final, TypedDict
from typing import TypedDict
import regex

from typing_extensions import deprecated
import numpy as np

from apfel import unimplemented
from apfel.container.once import Lazy


@deprecated("nltk is not accurate enough")
def to_lemma(word: str, pos: str):
    """
    Use the `nltk` WordNet lemmatizer to get the lemma of a word.
    """
    unimplemented()


@Lazy
def spacy_pipeline():
    import spacy
    return spacy.load("en_core_web_trf")


def fix_spacing(sentence: str) -> str:
    """
    Preprocess a sentence to fix spacing issues with apostrophes.

    Rules:
        - Add a whitespace before 's, 'm, 're, 've, 'll, 'd and specifically, n't.
    """

    sentence = regex.sub(r"(?<!\s)('(?:s|m|re|ve|ll|d))", r" \g<1>", sentence)
    sentence = regex.sub(r"(?<!\s)n't", r" n't", sentence)

    return sentence

def vec_to_lemma(word: list[str]):
    """
    Convert a vector of words to a vector of lemmas.
    """
    from spacy.tokens import Doc

    nlp = spacy_pipeline()
    doc = Doc(nlp.vocab, words=word)
    doc = nlp(doc)
    return {
        "lemma": [token.lemma_ for token in doc],
        "pos": [token.pos_ for token in doc],
    }

class SpacyMetadata(TypedDict):
    lemma: list[str]
    morph: list[str]
    pos_ud: list[str]
    pos_penn: list[str]
    dep_arc: list[np.int32]
    dep_rel: list[str]

def run_spacy(words: list[str]) -> SpacyMetadata:
    """
    Run the spacy pipeline on a list of words.

    Args:
        words (list[str]): The list of words to process.

    Returns:
        Doc: The processed spacy Doc object.
    """
    from spacy.tokens import Doc
    nlp = spacy_pipeline()
    doc = Doc(nlp.vocab, words=words)
    doc = nlp(doc)

    lemma = [token.lemma_ for token in doc]
    morph = [str(token.morph) for token in doc]
    dep_arc = [np.int32(token.head.i + 1 if token.dep_ != "ROOT" else 0) for token in doc]
    dep_rel = [token.dep_ for token in doc]
    pos_ud = [token.pos_ for token in doc]
    pos_penn = [token.tag_ for token in doc]

    ret: SpacyMetadata = {
        "lemma": lemma,
        "morph": morph,
        "pos_ud": pos_ud,
        "pos_penn": pos_penn,
        "dep_arc": dep_arc,
        "dep_rel": dep_rel,
    }
    return ret


BIDGOOD_VERBS: Final = {
    "transitive_only": {
        "cut",
        "slice",
        "chop",
        "mash",
        "hit",
        "strike",
        "bite",
        "peck",
        "touch",
        "stroke",
        "slash",
        "saw",
        "crush",
        "squash",
        "kick",
        "tap",
        "whack",
        "punch",
        "nudge",
        "kiss",
        "kill",
        "destroy",
        "demolish",
        "take",
        "bring",
        "raise",
        "hoist",
        "lift",
        "lower",
        "leave",
        "drop",
        "tickle",
        "amuse",
        "feed",
        "delight",
        "give",
        "madden",
        "lend",
        "pay",
        "offer",
    },
    "intransitive_only": {
        "go",
        "come",
        "rise",
        "fall",
        "tumble",
        "ascend",
        "descend",
        "exit",
        "enter",
        "arrive",
        "eat",
        "jump",
        "hop",
        "run",
        "drink",
        "talk",
        "swim",
        "climb",
        "sing",
        "sleep",
        "smile",
        "cry",
        "laugh",
        "frown",
        "giggle",
        "chortle",
        "chuckle",
        "grin",
        "groan",
        "moan",
        "glow",
        "glitter",
        "leak",
        "appear",
        "disappear",
        "vanish",
        "die",
        "stay",
        "wait",
        "live",
    },
    "alternating": {
        "break",
        "rip",
        "shatter",
        "smash",
        "grow",
        "change",
        "bake",
        "boil",
        "cook",
        "fry",
        "burn",
        "split",
        "tear",
        "melt",
        "crack",
        "improve",
        "inflate",
        "alter",
        "shrink",
        "freeze",
        "crash",
        "fold",
        "crease",
        "deflate",
        "defrost",
        "dissolve",
        "enlarge",
        "expand",
        "open",
        "close",
        "snap",
        "bend",
        "slide",
        "move",
        "roll",
        "bounce",
        "turn",
        "begin",
        "start",
        "stop",
    },
}

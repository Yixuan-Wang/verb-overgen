"""
Functions that inject linguistic lexical information into the data.
"""


import regex
import spacy
from spacy.tokens import Doc 
from typing_extensions import deprecated

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
    return spacy.load("en_core_web_md")


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
    nlp = spacy_pipeline()
    doc = Doc(nlp.vocab, words=word)
    doc = nlp(doc)
    return {
        "lemma": [token.lemma_ for token in doc],
        "pos": [token.pos_ for token in doc],
    }


BIDGOOD_VERBS = {
    "transitive_only": { "cut", "slice", "chop", "mash", "hit", "strike", "bite", "peck", "touch", "stroke", "slash", "saw", "crush", "squash", "kick", "tap", "whack", "punch", "nudge", "kiss", "kill", "destroy", "demolish", "take", "bring", "raise", "hoist", "lift", "lower", "leave", "drop", "tickle", "amuse", "feed", "delight", "give", "madden", "lend", "pay", "offer" },
    "intransitive_only": { "go", "come", "rise", "fall", "tumble", "ascend", "descend", "exit", "enter", "arrive", "eat", "jump", "hop", "run", "drink", "talk", "swim", "climb", "sing", "sleep", "smile", "cry ", "laugh", "frown", "giggle", "chortle", "chuckle", "grin", "groan", "moan", "glow", "glitter", "leak", "appear", "disappear", "vanish", "die", "stay", "wait", "live" },
    "alternating": { "break", "rip", "shatter", "smash", "grow", "change", "bake", "boil", "cook", "fry", "burn", "split", "tear", "melt", "crack", "improve", "inflate", "alter", "shrink", "freeze", "crash", "fold", "crease", "deflate", "defrost", "dissolve", "enlarge", "expand", "open", "close", "snap", "bend", "slide", "move", "roll", "bounce", "turn", "begin", "start", "stop" }
}

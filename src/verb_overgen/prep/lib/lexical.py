"""
Functions that inject linguistic lexical information into the data.
"""

from nltk.stem import WordNetLemmatizer
from functools import partial

LEMMATIZER = WordNetLemmatizer()
"""
The `nltk` WordNet lemmatizer.
"""

to_lemma = LEMMATIZER.lemmatize
"""
Use the `nltk` WordNet lemmatizer to get the lemma of a word.
"""

BIDGOOD_VERBS = {
    "transitive_only": { "cut", "slice", "chop", "mash", "hit", "strike", "bite", "peck", "touch", "stroke", "slash", "saw", "crush", "squash", "kick", "tap", "whack", "punch", "nudge", "kiss", "kill", "destroy", "demolish", "take", "bring", "raise", "hoist", "lift", "lower", "leave", "drop", "tickle", "amuse", "feed", "delight", "give", "madden", "lend", "pay", "offer" },
    "intransitive_only": { "go", "come", "rise", "fall", "tumble", "ascend", "descend", "exit", "enter", "arrive", "eat", "jump", "hop", "run", "drink", "talk", "swim", "climb", "sing", "sleep", "smile", "cry ", "laugh", "frown", "giggle", "chortle", "chuckle", "grin", "groan", "moan", "glow", "glitter", "leak", "appear", "disappear", "vanish", "die", "stay", "wait", "live" },
    "alternating": { "break", "rip", "shatter", "smash", "grow", "change", "bake", "boil", "cook", "fry", "burn", "split", "tear", "melt", "crack", "improve", "inflate", "alter", "shrink", "freeze", "crash", "fold", "crease", "deflate", "defrost", "dissolve", "enlarge", "expand", "open", "close", "snap", "bend", "slide", "move", "roll", "bounce", "turn", "begin", "start", "stop" }
}
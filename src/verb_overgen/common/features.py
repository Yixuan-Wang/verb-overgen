from typing import Final

FEATURE_CAUSATIVE_ENTRENCHED: Final = "causative_entrenched"
"""Short for "a verb that is entrenched but not preemptive" (Entrenchment - Preemption),
i.e. a verb that can be used as a causative but is not used as a causative here.
"""

FEATURE_CAUSATIVE: Final = "causative"

FILE_ID_AUGMENTED = "AUGMENTED"

VERB_TARGET_5: Final[list[str]] = ["laugh", "sing", "cry", "sleep", "smile"]

VERB_TARGET_17: Final[list[str]] = [
    "appear",
    "chortle",
    "chuckle",
    "cry",
    "descend",
    "frown",
    "giggle",
    "glow",
    "grin",
    "groan",
    "laugh",
    "rise",
    "sing",
    "sleep",
    "smile",
    "sneeze",
    "vanish",
]

VERB_TARGET_EXTENSION: Final[dict[str, list[str]]] = {
    "extended_intransitive_only": ["go", "come", "look", "stay", "talk", "listen", "cough", "blink"],
    "extended_transitive_only": ["touch", "hit", "bite", "kiss", "feed", "tickle"],
    "extended_alternating": ["grow", "open", "close", "roll", "break", "melt", "bounce"],
}

VERB_TARGET_10: Final[list[str]] = VERB_TARGET_5 + [
    "sneeze", "go", "look", "blink", "rise",
]

VERB_TARGET_FINALE: Final[list[str]] = [
    "go", "laugh", "cry", "sleep", "sneeze", "smile"
]

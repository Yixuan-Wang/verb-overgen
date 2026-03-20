from typing import cast
import itertools
from functools import partial

import polars as pl

from verb_overgen.prep.lib.syntactic import Parse, ParseWithPos

EXPR_PARSE = pl.col("parse")


def comp_children(
    parse: Parse,
    *,
    target: str,
) -> list[str]:
    """Extract the COMP children of the given target verb"""
    list_word, list_arc, list_rel = parse["word"], parse["arc"], parse["rel"]

    if target not in list_word:
        return []

    if "COMP" not in list_rel:
        return []

    return [
        word
        for index1_parent, word_parent in enumerate(list_word, 1)
        if word_parent == target
        for word, arc1, rel in zip(list_word, list_arc, list_rel)
        if arc1 == index1_parent and rel == "COMP"
    ]

def comp_children_index(
    parse: Parse,
    *,
    target: str,
) -> list[int]:
    """Extract the COMP children index of the given target verb.
    
    Warning:
        The returned index is 0-indexed, which is different from the 1-indexed arc in the parse.
    """
    list_word, list_arc, list_rel = parse["word"], parse["arc"], parse["rel"]
    
    if target not in list_word:
        return []
    
    if "COMP" not in list_rel:
        return []
    
    return [
        index0_child
        for index1_parent, word_parent in enumerate(list_word, 1)
        if word_parent == target
        for index0_child, (arc1, rel) in enumerate(zip(list_arc, list_rel))
        if arc1 == index1_parent and rel == "COMP"
    ]


def has_subj_in_parent(
    parse: Parse,
    *,
    target: str,
) -> bool:
    """Recursively find `target` and `COMP`-ancestors to see if there's an ancestor with a `SUBJ` children."""
    list_word, list_arc, list_rel = parse["word"], parse["arc"], parse["rel"]

    if target not in list_word:
        return False

    if "SUBJ" not in list_rel:
        return False

    list_index1_target = [i1 for i1, w in enumerate(list_word, 1) if w == target]

    for index1_target in list_index1_target:
        # We first check if target has any subject
        index1: int = index1_target

        # If index1 exists
        while index1:
            # If a node is index1's `SUBJ` children
            index1_has_subject = any(to1 == index1 and rel == "SUBJ" for to1, rel in zip(list_arc, list_rel))
            if index1_has_subject:
                # Then we decide this rule satisifies
                return True

            # Otherwise check its `COMP`-parent, or set to 0
            index1 = list_arc[index1 - 1] if list_rel[index1 - 1] == "COMP" else 0

    return False


def has_comp_children(
    parse: Parse,
    *,
    target: str,
) -> bool:
    """Return True if the given target verb has any `COMP` children."""
    list_word, list_arc, list_rel = parse["word"], parse["arc"], parse["rel"]

    if target not in list_word:
        return False

    if "COMP" not in list_rel:
        return False

    return any(
        rel == "COMP" and arc == index1
        for index1, word in enumerate(list_word, 1)
        if word == target
        for arc, rel in zip(list_arc, list_rel)
    )


class RuleLet:
    PERIPHERAL_VERB = "let"

    try:
        DF_ADDENDUM = pl.read_csv("./data/pychildes/rules/causative_let_addendum.csv")
    except Exception:
        raise ValueError("Failed to load `let` addendum rules")

    MAP_RULE_PREFILTER = {
        # The sentence should contain "let" lemma.
        "rule_has_let": (pl.col("morph").struct.field("lemma").list.contains(item=PERIPHERAL_VERB))
    }

    MAP_RULE_OVERRIDE = {
        "rule_addendum": pl.struct("id_file", "id_utterance").replace_strict(
            old=DF_ADDENDUM.select(pl.struct("id_file", "id_utterance")).to_series(),
            new=DF_ADDENDUM["rule_is"],
            default=None,
            return_dtype=pl.Boolean,
        )
    }

    MAP_RULE_HEURISTIC = {
        # The causative should not contain "let's"
        "rule_not_imp_let_us": ~pl.col("content_raw").str.contains(r"\b[Ll]et's\b"),
        # The causative should not contain "let go", like "let go of the balloon"
        "rule_not_let_go": ~pl.col("sentence").str.contains(r"\let go\b"),
        # The "let" should have a `COMP` (complement children)
        "rule_has_comp_children": (
            EXPR_PARSE.map_elements(
                partial(has_comp_children, target=PERIPHERAL_VERB),
                pl.Boolean,
            )
        ),
        # The "let" should have a subject in its ancestor path.
        "rule_has_let_subj": EXPR_PARSE.map_elements(
            partial(has_subj_in_parent, target=PERIPHERAL_VERB),
            pl.Boolean,
        ),
    }

    @classmethod
    def find(cls, df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
        df_original = df
        df_prefiltered = df_original.lazy().filter(cls.MAP_RULE_PREFILTER.values())

        df_causative = (
            df_prefiltered.filter(
                pl.coalesce(
                    *cls.MAP_RULE_OVERRIDE.values(),
                    pl.all_horizontal(*cls.MAP_RULE_HEURISTIC.values()),
                )
            )
            .select(
                pl.selectors.starts_with("id"),
                EXPR_PARSE.map_elements(
                    partial(comp_children_index, target=cls.PERIPHERAL_VERB),
                    pl.List(pl.Int32),
                ).alias(f"verb_causative_{cls.PERIPHERAL_VERB}"),
            )
            .collect()
        )

        return cast(pl.DataFrame, df_causative)

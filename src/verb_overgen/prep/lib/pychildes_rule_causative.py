import itertools
from functools import partial
from typing import cast

import polars as pl
from typing_extensions import deprecated

from verb_overgen.prep.lib.syntactic import Parse, ParseWithPos

EXPR_PARSE = pl.col("parse")

@deprecated("Favor comp_children_index for better performance.")
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
            index0 = index1 - 1
            index1 = (
                list_arc[index0]
                if (list_rel[index0] == "COMP" or list_rel[index0] == "COORD" or list_rel[index0] == "CONJ")
                else 0
            )

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

def has_comp_children_and_patient(
    parse: Parse,
    *,
    target: str,
) -> bool:
    """Return True if any occurrence of target has COMP children AND
    (OBJ children XOR its COMP children have SUBJ children).

    The XOR condition captures two causative patterns:
    - 'make him cry'    , OBJ child of target, no SUBJ on COMP child
    - 'make him see it' , causee parsed as SUBJ of COMP child, no OBJ on target
    """
    list_word, list_arc, list_rel = parse["word"], parse["arc"], parse["rel"]

    if target not in list_word:
        return False

    for index1, word in enumerate(list_word, 1):
        if word != target:
            continue

        children = [
            (j + 1, rel)
            for j, (arc, rel) in enumerate(zip(list_arc, list_rel))
            if arc == index1
        ]

        comp_child_index1s = [
            child_index1
            for child_index1, rel in children
            if rel == "COMP"
            and not any(
                arc == child_index1 and rel_gc == "INF"
                for arc, rel_gc in zip(list_arc, list_rel)
            )
        ]
        if not comp_child_index1s:
            continue

        target_has_obj = any(rel == "OBJ" for _, rel in children)
        comp_has_subj = any(
            arc == child_index1 and rel == "SUBJ"
            for child_index1 in comp_child_index1s
            for arc, rel in zip(list_arc, list_rel)
        )

        if comp_has_subj:
            comp_has_invalid_subj = any(
                arc == child_index1 and rel == "SUBJ"
                and (word == "something" or word == "anything")
                for child_index1 in comp_child_index1s
                for word, arc, rel in zip(list_word, list_arc, list_rel)
            )
            if comp_has_invalid_subj:
                return False

        if target_has_obj ^ comp_has_subj:
            return True

    return False


def has_comp_inf_children_and_patient(
    parse: Parse,
    *,
    target: str,
) -> bool:
    """Return True if any occurrence of target has COMP children with an INF grandchild AND
    (OBJ children XOR its COMP children have SUBJ children).
    
    This is for `get`, `force` and `allow`, etc, where a to-supported infinitive is enforced.
    
    Subjects that are indefinite pronouns ("something", "anything") are rejected
    to avoid false positives like "get something to eat".
    """
    list_word, list_arc, list_rel = parse["word"], parse["arc"], parse["rel"]

    if target not in list_word:
        return False

    for index1, word in enumerate(list_word, 1):
        if word != target:
            continue

        children = [
            (j + 1, rel)
            for j, (arc, rel) in enumerate(zip(list_arc, list_rel))
            if arc == index1
        ]

        comp_child_index1s = [
            child_index1
            for child_index1, rel in children
            if rel == "COMP"
            and any(
                arc == child_index1 and rel_gc == "INF"
                for arc, rel_gc in zip(list_arc, list_rel)
            )
        ]
        if not comp_child_index1s:
            continue

        target_has_obj = any(rel == "OBJ" for _, rel in children)
        comp_has_subj = any(
            arc == child_index1 and rel == "SUBJ"
            for child_index1 in comp_child_index1s
            for arc, rel in zip(list_arc, list_rel)
        )

        if comp_has_subj:
            comp_has_invalid_subj = any(
                arc == child_index1 and rel == "SUBJ"
                and (word == "something" or word == "anything")
                for child_index1 in comp_child_index1s
                for word, arc, rel in zip(list_word, list_arc, list_rel)
            )
            if comp_has_invalid_subj:
                return False

        if target_has_obj ^ comp_has_subj:
            return True

    return False

class MixinRuleFind:
    PERIPHERAL_VERB: str
    """The peripheral verb to look for."""

    MAP_RULE_PREFILTER: dict[str, pl.Expr]
    """Pre-filter rules to narrow down candidate sentences."""

    MAP_RULE_OVERRIDE: dict[str, pl.Expr]
    """Override rules to directly label sentences as True/False. Unspecified sentences will use heuristic rules to decide."""

    MAP_RULE_HEURISTIC: dict[str, pl.Expr]
    """Heuristic rules to decide whether a sentence contains causative. The final decision is a conjunction of all rules."""

    @classmethod
    def find(cls, df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
        """Find causative sentences with the peripheral verb in the given DataFrame, and return the verb indices in the column `verb_causative_{PERIPHERAL_VERB}`."""
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

class RuleLet(MixinRuleFind):
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
        "rule_not_let_go": ~pl.col("sentence").str.contains(r"\blet go\b"),
        # The "let" should have a `COMP` (complement children)
        "rule_has_comp_children_and_patient": (
            EXPR_PARSE.map_elements(
                partial(has_comp_children_and_patient, target=PERIPHERAL_VERB),
                pl.Boolean,
            )
        ),
        # The "let" should have a subject in its ancestor path.
        "rule_has_let_subj": EXPR_PARSE.map_elements(
            partial(has_subj_in_parent, target=PERIPHERAL_VERB),
            pl.Boolean,
        ),
    }

class RuleMake(MixinRuleFind):
    PERIPHERAL_VERB = "make"

    MAP_RULE_PREFILTER = {
        # The sentence should contain "make" lemma.
        "rule_has_make": (pl.col("morph").struct.field("lemma").list.contains(item=PERIPHERAL_VERB))
    }

    MAP_RULE_OVERRIDE = {
        # We currently don't have override rule for "make".
    }

    MAP_RULE_HEURISTIC = {
        # The causative should not contain "make sure"
        "rule_not_make_sure": ~pl.col("sentence").str.contains(r"\bmake sure\b"),
        # The "make" should have a `COMP` (complement children) and corresponding patient
        "rule_has_comp_children_and_patient": (
            EXPR_PARSE.map_elements(
                partial(has_comp_children_and_patient, target=PERIPHERAL_VERB),
                pl.Boolean,
            )
        ),
        # # The "make" should have a subject in its ancestor path.
        # "rule_has_make_subj": EXPR_PARSE.map_elements(
        #     partial(has_subj_in_parent, target=PERIPHERAL_VERB),
        #     pl.Boolean,
        # ),
    }

class RuleHave(MixinRuleFind):
    PERIPHERAL_VERB = "have"

    try:
        DF_ADDENDUM = pl.read_csv("./data/pychildes/rules/causative_have_addendum.csv")
    except Exception:
        raise ValueError("Failed to load `have` addendum rules")

    MAP_RULE_PREFILTER = {
        # The sentence should contain "have" lemma.
        "rule_has_have": (pl.col("morph").struct.field("lemma").list.contains(item=PERIPHERAL_VERB))
    }

    MAP_RULE_OVERRIDE = {
        "rule_addendum": pl.struct("id_file", "id_utterance").replace_strict(
            old=DF_ADDENDUM.select(pl.struct("id_file", "id_utterance")).to_series(),
            new=DF_ADDENDUM["rule_is"],
            default=pl.lit(False), # If not in addendum, treat it as non-causative.
            return_dtype=pl.Boolean,
        )
    }

    MAP_RULE_HEURISTIC = {
        # If not in addendum, treat it as non-causative.
        # This will be unreachable since the override rule will assign False to all sentences not in addendum, but we keep it here for clarity.
        "rule_false": pl.lit(False)
    }

class RuleGet(MixinRuleFind):
    PERIPHERAL_VERB = "get"

    MAP_RULE_PREFILTER = {
        "rule_has_get": (pl.col("morph").struct.field("lemma").list.contains(item=PERIPHERAL_VERB))
    }
    MAP_RULE_OVERRIDE = {}
    MAP_RULE_HEURISTIC = {
        "rule_has_comp_inf_children_and_patient": (
            EXPR_PARSE.map_elements(
                partial(has_comp_inf_children_and_patient, target=PERIPHERAL_VERB),
                pl.Boolean,
            )
        ),
    }

class RuleCause(MixinRuleFind):
    PERIPHERAL_VERB = "cause"

    MAP_RULE_PREFILTER = {
        "rule_has_cause": (pl.col("morph").struct.field("lemma").list.contains(item=PERIPHERAL_VERB))
    }
    MAP_RULE_OVERRIDE = {}
    MAP_RULE_HEURISTIC = {
        "rule_has_comp_inf_children_and_patient": (
            EXPR_PARSE.map_elements(
                partial(has_comp_inf_children_and_patient, target=PERIPHERAL_VERB),
                pl.Boolean,
            )
        ),
    }

class RuleForce(MixinRuleFind):
    PERIPHERAL_VERB = "force"

    MAP_RULE_PREFILTER = {
        "rule_has_force": (pl.col("morph").struct.field("lemma").list.contains(item=PERIPHERAL_VERB))
    }
    MAP_RULE_OVERRIDE = {}
    MAP_RULE_HEURISTIC = {
        "rule_has_comp_inf_children_and_patient": (
            EXPR_PARSE.map_elements(
                partial(has_comp_inf_children_and_patient, target=PERIPHERAL_VERB),
                pl.Boolean,
            )
        ),
    }


_DEFAULT_PERIPHERAL_VERBS: frozenset[str] = frozenset(
    cls.PERIPHERAL_VERB for cls in MixinRuleFind.__subclasses__()
)


def substitute_causative_with_transitive(
    sentence: str,
    target: str,
    peripheral_verbs: frozenset[str] = _DEFAULT_PERIPHERAL_VERBS,
) -> str:
    """Transform a periphrastic causative sentence into a transitive one.

    E.g. "make him sing" -> "sing him", "get her to laugh" -> "laugh her".
    """
    lemmas = sentence.split()
    out: list[str] = []
    for lemma in lemmas:
        if lemma == target:
            obj_end = len(out)
            # Drop a periphrastic "to" preceding the target — the lexical
            # causative form takes a bare object, not an infinitival complement.
            if obj_end > 0 and out[obj_end - 1] == "to":
                obj_end -= 1
            peripheral_idx = next(
                (j for j in range(obj_end - 1, -1, -1) if out[j] in peripheral_verbs),
                None,
            )
            if peripheral_idx is not None:
                out = out[:peripheral_idx] + [target] + out[peripheral_idx + 1 : obj_end]
                continue
        out.append(lemma)
    return " ".join(out)

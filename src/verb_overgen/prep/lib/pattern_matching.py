"""
Polars expressions for pattern matching on the data.

The datatype definition can be found in `prep.lib.syntactic.DtParse`.
"""

from collections.abc import Iterable
from functools import partial
from typing import TypedDict

import polars as pl


def match_svo(parse_expr: pl.Expr):
    """
    Get the nodes that are heads of both a subject (`nsubj`) and an object (`dobj`).
    **Notice that the list is 1-indexed.**

    Args:
        parse_expr (pl.Expr): The parsed data. Should be evaluated to dtype `DtParse`.

    Returns:
        pl.Expr: The list of nodes that are heads of both a subject and an object. Will be evaluated to dtype `pl.List(pl.Int64)`.
    """

    rel = parse_expr.struct.field("rel")

    is_nsubj = rel.list.eval(pl.element().eq("nsubj").arg_true())
    is_dobj = rel.list.eval(pl.element().eq("dobj").arg_true())

    arc = parse_expr.struct.field("arc")
    nsubj_head = arc.list.gather(is_nsubj)
    dobj_head = arc.list.gather(is_dobj)

    return nsubj_head.list.set_intersection(dobj_head)


def match_transitive(
    parse: pl.Expr,
    pos: pl.Expr,
):
    """
    Get the nodes that are heads of a direct object (`dobj`) with POS tag `VERB`.

    Args:
        parse (pl.Expr): The parsed data. Should be evaluated to dtype `DtParse`.

    Returns:
        pl.Expr: The list of nodes that are heads of a direct object. Will be evaluated to dtype `pl.List(pl.Int64)`.
    """
    rel = parse.struct.field("rel")

    is_dobj = rel.list.eval(pl.element().eq("dobj").arg_true())
    is_verb = pos.list.eval(pl.element().eq("VERB").arg_true())

    arc = parse.struct.field("arc")
    dobj_head = arc.list.gather(is_dobj)

    joint = is_verb.list.eval(pl.element().add(1)).list.set_intersection(dobj_head)
    return joint


def get_target_word(
    parse: pl.Expr,
    target_word: str,
):
    """
    Get the indices of the target word in the parse.

    You almost certainly want to use `.explode()` or `.list.first()` on the result.
    Notice that the returned indices are 1-indexed, as 0 is reserved for the root node.

    Args:
        parse (pl.Expr): The parsed data. Should be evaluated to dtype `DtParse`.
        target_word (str): The target word to find.

    Returns:
        pl.Expr: The list of indices of the target word, `index_target_word`.
    """
    word = parse.struct.field("word")
    indices = word.list.eval(pl.element().eq(target_word).arg_true()).alias("index_target_word")
    return indices


def get_substructure(
    parse: pl.Expr,
    index_target_word: pl.Expr,
    pos: pl.Expr,
):
    word = parse.struct.field("word")
    arc = parse.struct.field("arc")
    rel = parse.struct.field("rel")

    head_indices = arc.list.get(index_target_word)
    head = pl.when(head_indices.eq(0)).then(pl.lit("<ROOT>")).otherwise(word.list.get(head_indices - 1))
    head_rels = rel.list.get(index_target_word)

    children_indices = pl.concat_list(pl.struct(arc, index_target_word + 1)).list.eval(
        (lambda e1, e2: e1.eq(e2).arg_true())(pl.element().struct[0].explode(), pl.element().struct[1])
    )
    children = word.list.gather(children_indices)
    children_rels = rel.list.gather(children_indices)

    head_pos = pl.when(head_indices.eq(0)).then(pl.lit("<ROOT>")).otherwise(pos.list.get(head_indices - 1))
    children_pos = pos.list.gather(children_indices)

    children_indices += 1
    index_target_word += 1

    return pl.struct(
        index_target_word=index_target_word,
        head_index=head_indices,
        head_pos=head_pos,
        head=head,
        head_rel=head_rels,
        children_index=children_indices,
        children=children,
        children_pos=children_pos,
        children_rel=children_rels,
    ).alias("substructure")


def extract_verb(pos: pl.Expr, word: pl.Expr, *, target_pos: Iterable[str] = ("v", "part")):
    """
    Extract verbs from the sentence.

    Args:
        pos (pl.Expr): The POS tags of the sentence.
        word (pl.Expr): The words (or lemmas) of the sentence.
        target_pos (Iterable[str]): The POS tags to consider as verbs. Defaults to `("v", "part")`.

    Returns:
        pl.Expr: The list of verbs in the sentence.
    """

    idx_is_verb = pos.list.eval(pl.element().is_in(list(target_pos)).arg_true())
    verbs = word.list.gather(idx_is_verb)
    return verbs


def extract_periphrastic_causative(
    parse: pl.Expr,
    *,
    support_verb: str,
    pos: pl.Expr | None = None,
    target_pos: Iterable[str] = ("v", "part"),
):
    """
    Extract periphrastic constructions involving *make*,
        i.e. the children (likely verb) that is a `COMP` child of *make*.

    Args:
        parse (pl.Expr): The parsed data. Should be evaluated to dtype `DtParse`.
        support_verb (str): The support verb to look for, e.g. "make".
        pos (pl.Expr | None): The POS tags of the sentence. If provided, can be used for further filtering.

    Returns:
        pl.Expr: The children that are `COMP` children of *make*. Nullable.
    """
    word = parse.struct.field("word")
    arc = parse.struct.field("arc")
    rel = parse.struct.field("rel")

    # First get the indices of the support verb
    indices_make = word.list.eval(pl.element().eq(support_verb).arg_true())

    # Then get the children of the support verb
    indices_make_children = pl.concat_list(pl.struct(arc, indices_make)).list.eval(
        (lambda e1, e2: e1.eq(e2 + 1).arg_true())(pl.element().struct[0].explode(), pl.element().struct[1].list.first())
    )

    # Then filter the children with a `COMP` relation
    _ = rel.list.gather(indices_make_children).list.eval(pl.element().eq("COMP").arg_true())
    indices_make_children_comp = indices_make_children.list.gather(_)

    if pos is not None:
        idx_is_verb = pos.list.eval(pl.element().is_in(list(target_pos)).arg_true())
        indices_make_children_comp = indices_make_children_comp.list.set_intersection(idx_is_verb)

    word_comp_children = word.list.gather(indices_make_children_comp)
    return word_comp_children


class Parse(TypedDict):
    word: list[str]
    arc: list[int]
    rel: list[str]


def _extract_periphrastic_verb(
    parse: Parse,
    *,
    support_verb: str,
) -> list[str]:
    word, arc, rel = parse["word"], parse["arc"], parse["rel"]

    # A map from lemma indices to their children's lemma indices
    make_children = {
        i1: [j for j, a1 in enumerate(arc) if a1 == i1] for i1, w in enumerate(word, 1) if w == support_verb
    }

    indices_make_children_comp = [[i for i in l if rel[i] == "COMP"] for l in make_children.values()]

    word_make_children_comp = [word[i] for l in indices_make_children_comp for i in l]
    return word_make_children_comp


def extract_periphrastic_make(
    parse: pl.Expr,
) -> pl.Expr:
    return parse.map_elements(partial(_extract_periphrastic_verb, support_verb="make"), pl.List(pl.String))

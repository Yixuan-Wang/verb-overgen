"""
Polars expressions for pattern matching on the data.
"""

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

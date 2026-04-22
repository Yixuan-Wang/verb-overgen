"""Preprocess augmentation .txt files into parquets compatible with the training data schema.

Each line of the input file becomes one row. id_file is set to "{stem}/{line_number}" (1-based),
id_utterance is always 0, so every sentence packs independently.

Usage:
    python -m verb_overgen.prep.augmentation_preprocess \\
        --input var/pychildes/augmentation/laugh-causative.txt \\
        --feature causative \\
        --verb laugh \\
        --pretokenized
"""

import argparse
import json
from pathlib import Path
from typing import cast

import polars as pl
import tokenizers as hftk

from verb_overgen.common.features import FEATURE_CAUSATIVE, FEATURE_CAUSATIVE_ENTRENCHED
from verb_overgen.common.tokenize import UNK_TOKEN
from verb_overgen.prep.lib.lexical import BaseSpacyParser, LIST_CAUSATIVE_VERBS
import verb_overgen.prep.lib.pychildes_rule_causative as pychildes_rule_causative
from verb_overgen.train.run_tokenize import tokenize as _tokenize

# SUPPORT_VERBS = ["make", "let", "have", "get"]

SUPPORT_VERBS: list[str] = [
    rule.PERIPHERAL_VERB
    for class_name, rule in vars(pychildes_rule_causative).items()
    if class_name.startswith("Rule") and issubclass(rule, pychildes_rule_causative.MixinRuleFind)
]

SET_BASE_CAUSATIVE_VERBS: frozenset[str] = frozenset(LIST_CAUSATIVE_VERBS)


def tokenize_batch(s: pl.Series, tokenizer: hftk.Tokenizer) -> pl.Series:
    input_ids = _tokenize(tokenizer, s.to_list())
    return pl.Series(input_ids, dtype=pl.List(pl.Int32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--feature", required=True, choices=[FEATURE_CAUSATIVE, FEATURE_CAUSATIVE_ENTRENCHED])
    parser.add_argument("--verb", required=True, type=str)
    parser.add_argument(
        "--pretokenized",
        action="store_true",
        help="Input sentences have whitespace before punctuation; pass word-split tokens directly to spacy.",
    )
    parser.add_argument(
        "--tokenizer-file",
        default="./data/tokenizer/1024-wordlevel/tokenizer.json",
        type=str,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as input file).",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    output_dir: Path = args.output_dir or input_path.parent
    stem = input_path.stem

    lines = [l for l in input_path.read_text().splitlines() if l.strip()]
    n_rows = len(lines)

    spacy_parser = BaseSpacyParser()
    if args.pretokenized:
        spacy_result = spacy_parser.run_spacy(words=[line.split() for line in lines])
    else:
        spacy_result = spacy_parser.run_spacy(sentences=lines)

    tokenizer = hftk.Tokenizer.from_file(args.tokenizer_file)
    unk_id = tokenizer.token_to_id(UNK_TOKEN)
    is_base = args.verb in SET_BASE_CAUSATIVE_VERBS

    # 1-based index range list matching the length of the lemma list per row
    # expr_token_index = pl.int_ranges(1, pl.col("lemma").list.len() + 1, dtype=pl.Int64)

    stem += "-independent"  # to distinguish from the same data if we later add context to it
    df = cast(
        pl.DataFrame,
        pl.DataFrame(
            {
                "id_file": [f"{stem}/{i}" for i in range(n_rows)],
                "id_utterance": pl.Series([0 for i in range(n_rows)], dtype=pl.UInt64),
                "content_raw": lines,
                "roberta": spacy_result,
            }
        )
        .lazy()
        .with_columns(pl.col("roberta").struct.unnest())
        .drop("roberta")
        .with_columns(
            pl.col("lemma").list.join(" ").alias("sentence"),
        )
        .with_columns(
            pl.col("sentence")
            .map_batches(lambda s: tokenize_batch(s, tokenizer), return_dtype=pl.List(pl.Int32))
            .alias("input_ids"),
        )
        .filter(~pl.col("input_ids").list.contains(unk_id))
        .collect(),
    )

    n_filtered = n_rows - len(df)
    if n_filtered:
        print(f"WARNING: skipped {n_filtered} sentence(s) containing UNK tokens.")

    df_tokenized = df.select(
        "id_file",
        "id_utterance",
        # We have a non-gold parse, but we count that as a parse
        pl.lit(True).alias("has_parse"),
        pl.lit("AUGMENTED").alias("speaker"),
        # Our parse is in a different format, so we make it null.
        # morph: Struct({'index': List(Int64), 'lemma': List(String), 'pos': List(String), 'kind': List(String), 'metadata': List(String)}))
        pl.lit(None)
        .cast(
            pl.Struct(
                {
                    "index": pl.List(pl.Int64),
                    "lemma": pl.List(pl.String),
                    "pos": pl.List(pl.String),
                    "kind": pl.List(pl.String),
                    "metadata": pl.List(pl.String),
                }
            )
        )
        .alias("morph"),
        # syntax: Struct({'from_index': List(Int64), 'to_index': List(Int64), 'rel': List(String)}))
        pl.lit(None)
        .cast(pl.Struct({"from_index": pl.List(pl.Int64), "to_index": pl.List(pl.Int64), "rel": pl.List(pl.String)}))
        .alias("syntax"),
        # pl.struct(
        #     expr_token_index.alias("index"),
        #     pl.col("lemma"),
        #     pl.col("pos_ud").alias("pos"),
        #     pl.lit(None, pl.List(pl.String)).alias("kind"),
        #     # pl.col("lemma").list.eval(pl.when(pl.element().str.contains(r"^\p{P}+$")).then(pl.lit("word"))).alias("kind"),
        #     pl.col("morph").alias("metadata"),
        # ).alias("morph"),
        # pl.struct(
        #     pl.col("dep_arc").cast(pl.List(pl.Int64)).alias("from_index"),
        #     expr_token_index.alias("to_index"),
        #     pl.col("dep_rel").alias("rel"),
        # ).alias("syntax"),
        "sentence",
        "content_raw",
        "input_ids",
    )

    # First support verb found in the lemma list, or null if none
    expr_support_verb = (
        pl.col("lemma").list.set_intersection(pl.lit(SUPPORT_VERBS, dtype=pl.List(pl.String))).list.first()
    )

    if args.feature == FEATURE_CAUSATIVE:
        expr_extra = pl.struct(
            expr_support_verb.alias("support_verb"),
            pl.lit(is_base).alias("is_base"),
        ).map_batches(
            lambda s: pl.Series(
                [json.dumps({"support_verb": r["support_verb"], "is_base": r["is_base"]}) for r in s],
                dtype=pl.String,
            )
        )
    else:
        expr_extra = pl.lit(json.dumps({"support_verb": None, "is_base": is_base}))

    type_features = pl.Categorical("features")

    df_annotation = df.select(
        "id_file",
        "id_utterance",
        pl.lit([args.feature]).cast(pl.List(type_features)).alias("features"),
        pl.concat_list(
            pl.struct(
                pl.lit(args.feature).cast(type_features).alias("feature"),
                pl.lit(args.verb).alias("verb"),
                expr_extra.alias("extra"),
            )
        ).alias("featured_verbs"),
    )

    path_tokenized = output_dir / f"{stem}-tokenized.parquet"
    path_annotation = output_dir / f"{stem}-annotation.parquet"

    df_tokenized.write_parquet(path_tokenized)
    df_annotation.write_parquet(path_annotation)

    print(f"Wrote {len(df_tokenized)} rows to {path_tokenized}")
    print(f"Wrote {len(df_annotation)} rows to {path_annotation}")


if __name__ == "__main__":
    main()

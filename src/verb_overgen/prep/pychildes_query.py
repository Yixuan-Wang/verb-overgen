"""Query sentences from the PyChildes dataset where featured_verbs contains a FEATURE_CAUSATIVE feature for a given verb."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from verb_overgen.common.features import FEATURE_CAUSATIVE
from verb_overgen.train.lib.dataloader_pychildes import DatasetPyChildes
from verb_overgen.train.lib.utils import load_train_config


def query_causative_sentences(
    dataset: DatasetPyChildes,
    verb: str,
) -> pl.DataFrame:
    expr_filter_verb = pl.element().struct["verb"].eq(verb)

    return dataset.dataframe.filter(
        pl.col("featured_verbs").list.eval(
            pl.element().struct["feature"].eq(FEATURE_CAUSATIVE) & expr_filter_verb
        ).list.any()
    )


def main():
    parser = argparse.ArgumentParser(description="Query causative sentences for a given verb from the PyChildes dataset.")
    parser.add_argument("verb", type=str, help="The verb to query causative sentences for.")
    parser.add_argument("--output", "-o", type=Path, default=None, metavar="PATH", help="Output CSV path. Prints to stdout if not specified.")
    parser.add_argument("--columns", "-c", nargs="+", default=["id_file_utterance", "content_raw"], metavar="COL", help="Columns to include in output (default: id_file_utterance content_raw).")
    args = parser.parse_args()

    cfg = load_train_config(overrides=None)

    dataset = DatasetPyChildes.from_path(
        input_path_main=Path(cfg.data.train.path),
        input_path_annotation=Path(cfg.data.train.path_annotation),
        token_id_pad=0,
        max_len=512,
        max_token_per_batch=cfg.data.train.max_token_per_batch,
    )

    df_result = query_causative_sentences(dataset, args.verb)

    available_cols = [c for c in args.columns if c in df_result.columns]
    df_output = df_result.select(available_cols)

    if args.output is not None:
        df_output.write_csv(args.output)
        print(f"Written {len(df_output)} rows to {args.output}")
    else:
        print(df_output)
        print(f"\n{len(df_output)} sentence(s) found with FEATURE_CAUSATIVE for verb '{args.verb}'.")


if __name__ == "__main__":
    main()

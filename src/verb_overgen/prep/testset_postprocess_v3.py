"""Post-process template testset: filter UNK/verbatim-training matches, score pairs by LM logprob, keep top-N per verb.

Pipeline:
  1. Drop sentences whose input_ids contain the UNK token.
  2. Drop sentences that appear verbatim (case-insensitive) in the training corpus.
  3. Drop pairs where either the causative or transitive sentence was removed above.
  4. Score every remaining sentence via the vLLM completions endpoint (per-token average log-prob).
  5. Score each pair as max(logprob_causative, logprob_transitive) — the more grammatical reading wins.
  6. Keep the top --keep-per-verb pairs per verb, sorted by score descending.
  7. Write output with an added `score` column.


Warning:
   The processed testset is not conjugated correctly, therefore being unsuitable for direct testing on a BPE model.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import openai
import polars as pl
from loguru import logger
from tokenizers import Tokenizer

from verb_overgen.common.tokenize import UNK_TOKEN

DEFAULT_INPUT_PATH = Path("var/testset/testset_template_0818_bpe.parquet")
DEFAULT_TRAINING_PATH = Path("data/pychildes-morphosyntax.20251024.parquet")
DEFAULT_TOKENIZER_PATH = Path("data/tokenizer/1024-bpe/tokenizer.json")
DEFAULT_OUTPUT_PATH = Path("var/testset/testset_template_0818_bpe_pp.parquet")
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
DEFAULT_MODEL = "/model-weights/Qwen3.5-27B"
DEFAULT_KEEP_PER_VERB = 100
DEFAULT_SCORING_BATCH_SIZE = 64


async def score_sentence_async(
    client: openai.AsyncClient,
    model: str,
    sentence: str,
) -> float | None:
    """Return the per-token average log-probability of `sentence` under the LM."""
    try:
        response = await client.completions.create(
            model=model,
            prompt=sentence,
            max_tokens=0,
            echo=True,
            logprobs=1,
        )
        logprobs_obj = response.choices[0].logprobs
        if logprobs_obj is None or logprobs_obj.token_logprobs is None:
            logger.warning(f"No logprobs object returned for: {sentence!r}")
            return None
        valid = [lp for lp in logprobs_obj.token_logprobs if lp is not None]
        if not valid:
            logger.warning(f"No valid logprobs returned for: {sentence!r}")
            return None
        return sum(valid) / len(valid)
    except Exception as exc:
        logger.warning(f"Scoring error for {sentence!r}: {exc}")
        return None


async def score_all(
    client: openai.AsyncClient,
    model: str,
    sentences: list[str],
    batch_size: int,
) -> list[float | None]:
    results: list[float | None] = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        batch_results = await asyncio.gather(*[score_sentence_async(client, model, s) for s in batch])
        results.extend(batch_results)
        logger.info(f"Scored {min(i + batch_size, len(sentences)):,}/{len(sentences):,} sentences")
    return results


async def postprocess(
    input_path: Path,
    training_path: Path,
    tokenizer_path: Path,
    output_path: Path,
    base_url: str,
    model: str,
    keep_per_verb: int,
    scoring_batch_size: int,
) -> None:
    df = pl.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} rows from {input_path}")

    # ── 1. Filter UNK ────────────────────────────────────────────────────────
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    unk_id = tokenizer.token_to_id(UNK_TOKEN)
    if unk_id is None:
        raise ValueError(f"{UNK_TOKEN!r} not in tokenizer vocabulary")
    before = len(df)
    df = df.filter(~pl.col("input_ids").list.contains(unk_id))
    logger.info(f"UNK filter: removed {before - len(df):,} rows, {len(df):,} remain")

    # ── 2. Filter verbatim training matches ──────────────────────────────────
    logger.info(f"Loading training corpus from {training_path} …")
    training_sentences = set(
        pl.read_parquet(training_path)
        .get_column("content")
        .str.to_lowercase()
        .str.strip_chars()
        .to_list()
    )
    before = len(df)
    df = df.filter(~pl.col("sentence").str.to_lowercase().str.strip_chars().is_in(training_sentences))
    logger.info(f"Verbatim filter: removed {before - len(df):,} rows, {len(df):,} remain")

    # ── 3. Drop incomplete pairs ──────────────────────────────────────────────
    # A pair is (verb, index_verb_context). Both kinds must survive the filters.
    n_kinds = 2  # sentence_causative + sentence_transitive
    complete = (
        df.group_by("verb", "index_verb_context")
        .len()
        .filter(pl.col("len") == n_kinds)
        .select("verb", "index_verb_context")
    )
    before = len(df)
    df = df.join(complete, on=["verb", "index_verb_context"], how="semi")
    logger.info(f"Incomplete-pair filter: removed {before - len(df):,} rows, {len(df):,} remain")

    # ── 4. Score sentences ───────────────────────────────────────────────────
    client = openai.AsyncClient(base_url=base_url, api_key="None")
    sentences = df["sentence"].to_list()
    logger.info(f"Scoring {len(sentences):,} sentences via {base_url} …")
    logprobs = await score_all(client, model, sentences, scoring_batch_size)

    n_failed = sum(1 for lp in logprobs if lp is None)
    if n_failed:
        logger.warning(f"{n_failed:,} sentences failed scoring; they will rank last")

    df = df.with_columns(
        pl.Series("logprob", [lp if lp is not None else float("-inf") for lp in logprobs])
    )

    # Drop pairs where any sentence failed (logprob == -inf means API error, not low naturalness)
    failed_pairs = (
        df.filter(pl.col("logprob") == float("-inf"))
        .select("verb", "index_verb_context")
        .unique()
    )
    if len(failed_pairs):
        before = len(df)
        df = df.join(failed_pairs, on=["verb", "index_verb_context"], how="anti")
        logger.warning(f"Dropped {before - len(df):,} rows from pairs with scoring failures")

    # ── 5. Pair score = max(logprob) within pair ─────────────────────────────
    pair_scores = df.group_by("verb", "index_verb_context").agg(pl.col("logprob").max().alias("score"))

    # ── 6. Rank pairs within each verb, keep top N ───────────────────────────
    top_pairs = (
        pair_scores.with_columns(
            pl.col("score").rank(method="ordinal", descending=True).over("verb").alias("pair_rank")
        )
        .filter(pl.col("pair_rank") <= keep_per_verb)
        .select("verb", "index_verb_context", "score")
    )

    kept_per_verb = top_pairs.group_by("verb").len().sort("verb")
    for row in kept_per_verb.iter_rows(named=True):
        if row["len"] < keep_per_verb:
            logger.warning(f"verb={row['verb']!r}: only {row['len']} pairs available, requested {keep_per_verb}")
        else:
            logger.info(f"verb={row['verb']!r}: kept {row['len']} pairs")

    # ── 7. Write output ───────────────────────────────────────────────────────
    result = (
        df.join(top_pairs, on=["verb", "index_verb_context"], how="semi")
        .join(top_pairs, on=["verb", "index_verb_context"])
        .drop("logprob")
        .sort("verb", "score", "kind", descending=[False, True, False])
        .with_columns(
            pl.int_range(pl.len()).cast(pl.UInt32).alias("index_sentence"),
            (pl.int_range(pl.len()).over("verb") // 2).cast(pl.UInt32).alias("index_verb_context"),
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output_path)
    logger.success(f"Wrote {len(result):,} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING_PATH)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--keep-per-verb", type=int, default=DEFAULT_KEEP_PER_VERB)
    parser.add_argument("--scoring-batch-size", type=int, default=DEFAULT_SCORING_BATCH_SIZE)
    args = parser.parse_args()

    asyncio.run(
        postprocess(
            input_path=args.input,
            training_path=args.training,
            tokenizer_path=args.tokenizer,
            output_path=args.output,
            base_url=args.base_url,
            model=args.model,
            keep_per_verb=args.keep_per_verb,
            scoring_batch_size=args.scoring_batch_size,
        )
    )


if __name__ == "__main__":
    main()

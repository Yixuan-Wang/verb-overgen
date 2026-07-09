"""Generate simple overgen test-set pairs without training-set V-bigram overlap."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import random
import string

import openai
import polars as pl
from loguru import logger
from tokenizers import Tokenizer

from verb_overgen.common.tokenize import BOS_TOKEN, EOS_TOKEN

DEFAULT_TOKENIZER_PATH = Path("data/tokenizer/1024-wordlevel/tokenizer.json")
DEFAULT_BIGRAM_COUNT_PATH = Path("var/pychildes/ngram/bigram_count.parquet")
DEFAULT_OUTPUT_PATH = Path("var/testset/testset_template_no-overlap-natural_sneeze.parquet")
# DEFAULT_VERBS = ["go", "laugh", "cry", "sleep", "sneeze", "smile"]
DEFAULT_VERBS = ["sneeze"] # ["go", "cry", "sleep", "smile"]
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
DEFAULT_MODEL = "/model-weights/Qwen3.5-27B"
DEFAULT_NATURALNESS_BATCH_SIZE = 64

_NATURALNESS_SYSTEM = (
    "You are a child literature expert assessing whether a sentence describes a plausible, "
    "everyday scene a young child could understand. The sentence is already lemmatized, so focus only on whether the subject, "
    "object, and action form a semantically coherent scene. Ignore grammatical agreement "
    "(e.g. 'make' vs 'makes') and tense."
)
_NATURALNESS_USER = (
    "Is the following lemmatized sentence natural?\n"
    "{sentence}. \n\n"
    "Answer with only 'yes' or 'no'."
)

# (subject, object, causative_sentence, transitive_sentence, causative_ids, transitive_ids)
_Candidate = tuple[str, str, str, str, list[int], list[int]]


def required_token_id(tokenizer: Tokenizer, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"{token!r} is missing from the tokenizer vocabulary")
    return token_id


def excluded_token_ids(tokenizer: Tokenizer) -> set[int]:
    punctuation = set(string.punctuation)
    return {
        token_id
        for token, token_id in tokenizer.get_vocab().items()
        if token in {BOS_TOKEN, EOS_TOKEN, "the"} or any(char in punctuation for char in token)
    }


def load_training_bigrams(path: Path) -> set[tuple[int, int]]:
    return {
        (token_1, token_2)
        for token_1, token_2 in pl.read_parquet(path, columns=["token-1", "token-2"]).iter_rows()
    }


def encode_sentence(tokenizer: Tokenizer, sentence: str, bos_token_id: int, eos_token_id: int) -> list[int]:
    return [bos_token_id, *tokenizer.encode(sentence).ids, eos_token_id]


def has_v_bigram_overlap(
    input_ids: list[int],
    *,
    verb_token_id: int,
    training_bigrams: set[tuple[int, int]],
    excluded_ids: set[int],
) -> bool:
    for token_1, token_2 in zip(input_ids, input_ids[1:]):
        if token_1 in excluded_ids or token_2 in excluded_ids:
            continue
        if token_1 != verb_token_id and token_2 != verb_token_id:
            continue
        if (token_1, token_2) in training_bigrams:
            return True
    return False


async def is_natural_async(
    client: openai.AsyncClient,
    model: str,
    *,
    sentence: str,
) -> bool:
    """Ask the LLM whether the sentence describes a plausible scene."""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _NATURALNESS_SYSTEM},
                {"role": "user", "content": _NATURALNESS_USER.format(sentence=sentence)},
            ],
            temperature=1.0,
            top_p=0.95,
            presence_penalty=1.5,
            extra_body={
                "top_k": 20,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
            },
            max_tokens=1024,
        )
        answer = response.choices[0].message.content or ""
        answer_lower = answer.strip().lower()
        if answer_lower.startswith("yes"):
            return True
        if answer_lower.startswith("no"):
            return False

        logger.warning(f"Unparseable naturalness response for {sentence!r}: {answer!r}; skipping")
        return False
    except Exception as exc:
        logger.warning(f"Naturalness API error for {sentence!r}: {exc}; skipping")
        return False


def load_subjects(path_names: Path, path_pronouns: Path) -> list[str]:
    return (
        pl.read_csv(path_names)
        .extend(pl.read_csv(path_pronouns).rename({"count_is_pronoun": "count_is_noun"}))
        .get_column("lemma")
        .to_list()
    )


def load_objects(path_names: Path, path_nouns: Path) -> list[str]:
    names = pl.read_csv(path_names).select("lemma")
    nouns = (
        pl.read_csv(path_nouns)
        .filter(pl.col("is_noun"))
        .select(pl.concat_str(pl.lit("the "), pl.col("lemma")).alias("lemma"))
    )
    return pl.concat([names, nouns]).get_column("lemma").to_list()


async def generate_no_overlap_testset(
    *,
    tokenizer_path: Path,
    bigram_count_path: Path,
    verbs: list[str],
    sentence_per_verb: int,
    max_attempts_per_verb: int,
    seed: int | None,
    check_naturalness: bool = True,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    naturalness_batch_size: int = DEFAULT_NATURALNESS_BATCH_SIZE,
) -> pl.DataFrame:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    bos_token_id = required_token_id(tokenizer, BOS_TOKEN)
    eos_token_id = required_token_id(tokenizer, EOS_TOKEN)
    excluded_ids = excluded_token_ids(tokenizer)
    training_bigrams = load_training_bigrams(bigram_count_path)

    client: openai.AsyncClient | None = None
    if check_naturalness:
        client = openai.AsyncClient(base_url=base_url, api_key="None")

    subjects = load_subjects(Path("data/testset/gen/names.csv"), Path("data/testset/gen/top_pronouns.csv"))
    objects = load_objects(Path("data/testset/gen/names.csv"), Path("data/testset/gen/top_nouns_v2.csv"))
    rng = random.Random(seed)

    rows: list[dict[str, int | str | list[int]]] = []
    index_sentence = 0

    for verb in verbs:
        verb_token_id = required_token_id(tokenizer, verb)
        accepted = 0
        attempts = 0
        seen_candidates: set[tuple[str, str]] = set()

        while accepted < sentence_per_verb and attempts < max_attempts_per_verb:
            # Collect a batch of overlap-free candidates, then check naturalness in parallel.
            batch: list[_Candidate] = []
            while len(batch) < naturalness_batch_size and attempts < max_attempts_per_verb:
                attempts += 1
                subject = rng.choice(subjects)
                object_ = rng.choice(objects)
                candidate_key = (subject, object_)
                if candidate_key in seen_candidates:
                    continue
                seen_candidates.add(candidate_key)

                sentence_causative = f"{subject} make {object_} {verb} ."
                sentence_transitive = f"{subject} {verb} {object_} ."
                input_ids_causative = encode_sentence(tokenizer, sentence_causative, bos_token_id, eos_token_id)
                input_ids_transitive = encode_sentence(tokenizer, sentence_transitive, bos_token_id, eos_token_id)

                if has_v_bigram_overlap(
                    input_ids_causative,
                    verb_token_id=verb_token_id,
                    training_bigrams=training_bigrams,
                    excluded_ids=excluded_ids,
                ):
                    continue
                if has_v_bigram_overlap(
                    input_ids_transitive,
                    verb_token_id=verb_token_id,
                    training_bigrams=training_bigrams,
                    excluded_ids=excluded_ids,
                ):
                    continue

                batch.append((subject, object_, sentence_causative, sentence_transitive, input_ids_causative, input_ids_transitive))

            if not batch:
                break

            if check_naturalness:
                assert client is not None
                naturals = await asyncio.gather(*[
                    is_natural_async(client, model, sentence=candidate[2])
                    for candidate in batch
                ])
                passing = [c for c, ok in zip(batch, naturals) if ok]
            else:
                passing = batch

            for subject, object_, sentence_causative, sentence_transitive, input_ids_causative, input_ids_transitive in passing:
                if accepted >= sentence_per_verb:
                    break
                rows.extend(
                    [
                        {
                            "index_sentence": index_sentence,
                            "index_verb_context": accepted,
                            "verb": verb,
                            "kind": "sentence_causative",
                            "sentence": sentence_causative,
                            "subject": subject,
                            "object": object_,
                            "input_ids": input_ids_causative,
                        },
                        {
                            "index_sentence": index_sentence + 1,
                            "index_verb_context": accepted,
                            "verb": verb,
                            "kind": "sentence_transitive",
                            "sentence": sentence_transitive,
                            "subject": subject,
                            "object": object_,
                            "input_ids": input_ids_transitive,
                        },
                    ]
                )
                accepted += 1
                index_sentence += 2

        if accepted < sentence_per_verb:
            raise RuntimeError(
                f"Only generated {accepted:,}/{sentence_per_verb:,} no-overlap sentences for verb {verb!r} "
                f"after {attempts:,} attempts"
            )

    return pl.DataFrame(rows).with_columns(
        pl.col("index_sentence").cast(pl.UInt32),
        pl.col("index_verb_context").cast(pl.UInt32),
        pl.col("input_ids").cast(pl.List(pl.Int64)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--bigram-count", type=Path, default=DEFAULT_BIGRAM_COUNT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--verbs", nargs="+", default=DEFAULT_VERBS)
    parser.add_argument("--sentence-per-verb", type=int, default=100)
    parser.add_argument("--max-attempts-per-verb", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-naturalness-check",
        dest="check_naturalness",
        action="store_false",
        help="Disable the LLM naturalness gate (no server required; useful for testing the template path).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help="Base URL of the OpenAI-compatible SGLang server (default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model name passed to the server; must match --model-path used at launch (default: %(default)s).",
    )
    parser.add_argument(
        "--naturalness-batch-size",
        type=int,
        default=DEFAULT_NATURALNESS_BATCH_SIZE,
        help="Number of candidates to check for naturalness in parallel per round (default: %(default)s).",
    )
    args = parser.parse_args()

    df_testset = asyncio.run(generate_no_overlap_testset(
        tokenizer_path=args.tokenizer,
        bigram_count_path=args.bigram_count,
        verbs=args.verbs,
        sentence_per_verb=args.sentence_per_verb,
        max_attempts_per_verb=args.max_attempts_per_verb,
        seed=args.seed,
        check_naturalness=args.check_naturalness,
        base_url=args.base_url,
        model=args.model,
        naturalness_batch_size=args.naturalness_batch_size,
    ))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_testset.write_parquet(args.output)
    print(f"Wrote {len(df_testset):,} rows to {args.output}")


if __name__ == "__main__":
    main()

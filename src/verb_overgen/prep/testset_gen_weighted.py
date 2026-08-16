"""Generate simple overgen test-set pairs using frequency-weighted subject/object sampling.

Combines the frequency-weighted (alpha-smoothed) subject/object sampling and
LLM-based object-selection idea from `testset_gen.py` with the lightweight
template-sentence pipeline from `testset_gen_no_overlap.py`. The subject is
drawn directly via weighted sampling; the object is chosen by the LLM from a
weighted-sampled candidate pool so that it fits the subject and verb -- this
folds naturalness checking into selection, so there is no separate judgement
call. The LLM never writes a context sentence. Does not filter on
training-set verb-bigram overlap.

Warning:
    This is originally used to generate alternating and transitive-only verbs.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np
import openai
import polars as pl
from loguru import logger
from tokenizers import Tokenizer

from verb_overgen.common.features import VERB_TARGET_17, VERB_TARGET_EXTENSION  # noqa: F401
from verb_overgen.common.tokenize import BOS_TOKEN, EOS_TOKEN

DEFAULT_TOKENIZER_PATH = Path("data/tokenizer/1024-wordlevel/tokenizer.json")
DEFAULT_OUTPUT_PATH = Path("var/testset/testset_template_transitive-only.parquet")
DEFAULT_VERBS = (
    # VERB_TARGET_17
    # + VERB_TARGET_EXTENSION["extended_intransitive_only"]
    VERB_TARGET_EXTENSION["extended_transitive_only"]
    # + VERB_TARGET_EXTENSION["extended_alternating"]
)
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
DEFAULT_MODEL = "/model-weights/Qwen3.5-27B"
DEFAULT_SELECTION_BATCH_SIZE = 64
DEFAULT_OBJECT_CANDIDATES = 25
DEFAULT_ALPHA = 0.3

_SELECT_SYSTEM = (
    "You are a child literature author tasked with choosing objects that fit naturally into "
    "simple, easy-to-understand scenes for young children's literature."
)

# Transitive-only verbs (touch, hit, bite, ...) have no bare-intransitive reading, so judging fit
# against the periphrastic-causative frame ("X makes Y touch") is itself ill-formed. Use the direct
# transitive frame ("X touch Y") for those instead; the causative sentence is still generated afterward
# from the same selected object so it can be used as a hard-negative test case.
_TRANSITIVE_ONLY_VERBS: frozenset[str] = frozenset(VERB_TARGET_EXTENSION["extended_transitive_only"])

_SELECT_USER = """\
### Instructions
Choose the object from the candidates below that best fits naturally with the subject and verb, \
forming a plausible scene where {usage_description}, as in the sentence "{usage_example}".
If the chosen object is a common noun, prefix it with the article "the" (e.g. "plant" -> "the plant"). \
If the chosen object is a proper name (e.g. a person's name), do not add an article (e.g. "Tom" stays "Tom").
If none of the candidates are suitable, answer with "none".

### Output Format
Answer with only the chosen object, exactly as written in the candidate list but with the article \
added if required, or "none".

### Input
- Subject: {subject}
- Verb: {verb}
- Object Candidates: {object_candidates}
"""


def _select_user_prompt(*, subject: str, verb: str, object_candidates: list[str]) -> str:
    if verb in _TRANSITIVE_ONLY_VERBS:
        usage_description = "the subject directly performs the action described by the verb on the object"
        usage_example = f"{subject} {verb} {{object}}"
    else:
        usage_description = "the subject causes the object to perform the action described by the verb"
        usage_example = f"{subject} makes {{object}} {verb}"

    return _SELECT_USER.format(
        usage_description=usage_description,
        usage_example=usage_example,
        subject=subject,
        verb=verb,
        object_candidates=", ".join(object_candidates),
    )


# (subject, object, causative_sentence, transitive_sentence, causative_ids, transitive_ids)
_Accepted = tuple[str, str, str, str, list[int], list[int]]


class WeightedSampler:
    def __init__(self, labels: list[str], weights: np.ndarray, rng: np.random.Generator):
        self.labels = labels
        self.rng = rng
        self.probs = weights / weights.sum()

    def sample(self, n: int) -> list[str]:
        indices = self.rng.choice(len(self.labels), size=min(n, len(self.labels)), replace=False, p=self.probs)
        return [self.labels[i] for i in indices]


def required_token_id(tokenizer: Tokenizer, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"{token!r} is missing from the tokenizer vocabulary")
    return token_id


def encode_sentence(tokenizer: Tokenizer, sentence: str, bos_token_id: int, eos_token_id: int) -> list[int]:
    return [bos_token_id, *tokenizer.encode(sentence).ids, eos_token_id]


async def select_object_async(
    client: openai.AsyncClient,
    model: str,
    *,
    subject: str,
    verb: str,
    object_candidates: list[str],
) -> str | None:
    """Ask the LLM to pick the object candidate that best fits the subject and verb."""
    label = f"{subject}/{verb}"
    candidates_by_lower = {candidate.lower(): candidate for candidate in object_candidates}
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SELECT_SYSTEM},
                {
                    "role": "user",
                    "content": _select_user_prompt(subject=subject, verb=verb, object_candidates=object_candidates),
                },
            ],
            temperature=0.7,
            top_p=0.8,
            presence_penalty=1.5,
            extra_body={
                "top_k": 20,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            max_tokens=1024,
        )
        answer = (response.choices[0].message.content or "").strip()
        answer_lower = answer.lower()
        if answer_lower == "none":
            return None

        # The model may prefix the candidate with an article (e.g. "plant" -> "the plant"); validate
        # against the bare candidate but reconstruct the final text from the canonical candidate casing.
        if answer_lower in candidates_by_lower:
            return candidates_by_lower[answer_lower]

        bare_answer_lower = answer_lower.removeprefix("the ").strip()
        if bare_answer_lower in candidates_by_lower:
            return f"the {candidates_by_lower[bare_answer_lower]}"

        logger.warning(f"Unparseable object-selection response for {label}: {answer!r}; skipping")
        return None
    except Exception as exc:
        logger.warning(f"Object-selection API error for {label}: {exc}; skipping")
        return None


def load_subjects(path_names: Path, path_pronouns: Path) -> tuple[list[str], np.ndarray]:
    dataset = pl.read_csv(path_names).extend(pl.read_csv(path_pronouns).rename({"count_is_pronoun": "count_is_noun"}))
    labels = dataset.get_column("lemma").to_list()
    weights = dataset.get_column("count").to_numpy()
    return labels, weights


def load_objects(path_names: Path, path_nouns: Path) -> tuple[list[str], np.ndarray]:
    names = pl.read_csv(path_names).select("lemma", "count")
    nouns = pl.read_csv(path_nouns).filter(pl.col("is_noun")).select("lemma", pl.col("count_is_noun").alias("count"))
    dataset = pl.concat([names, nouns])
    labels = dataset.get_column("lemma").to_list()
    weights = dataset.get_column("count").to_numpy()
    return labels, weights


async def generate_weighted_testset(
    *,
    tokenizer_path: Path,
    verbs: list[str],
    sentence_per_verb: int,
    max_attempts_per_verb: int,
    alpha: float,
    seed: int | None,
    select_object_with_llm: bool = True,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    selection_batch_size: int = DEFAULT_SELECTION_BATCH_SIZE,
    object_candidates_count: int = DEFAULT_OBJECT_CANDIDATES,
) -> pl.DataFrame:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    bos_token_id = required_token_id(tokenizer, BOS_TOKEN)
    eos_token_id = required_token_id(tokenizer, EOS_TOKEN)

    client: openai.AsyncClient | None = None
    if select_object_with_llm:
        client = openai.AsyncClient(base_url=base_url, api_key="None")

    subject_labels, subject_weights = load_subjects(
        Path("data/testset/gen/names.csv"), Path("data/testset/gen/top_pronouns.csv")
    )
    object_labels, object_weights = load_objects(
        Path("data/testset/gen/names.csv"), Path("data/testset/gen/top_nouns_v2.csv")
    )
    subject_weights = subject_weights.astype(float) ** alpha
    object_weights = object_weights.astype(float) ** alpha

    rng = np.random.default_rng(seed)

    rows: list[dict[str, int | str | list[int]]] = []
    index_sentence = 0

    for verb_idx, verb in enumerate(verbs):
        logger.info(f"verb={verb!r}: starting ({verb_idx + 1}/{len(verbs)})")
        subject_sampler = WeightedSampler(subject_labels, subject_weights, rng)
        object_sampler = WeightedSampler(object_labels, object_weights, rng)

        accepted = 0
        attempts = 0
        seen_subjects: set[str] = set()

        while accepted < sentence_per_verb and attempts < max_attempts_per_verb:
            # Collect a batch of (subject, object candidates) rounds, then resolve the object in parallel.
            batch: list[tuple[str, list[str]]] = []
            while len(batch) < selection_batch_size and attempts < max_attempts_per_verb:
                attempts += 1
                subject = subject_sampler.sample(1)[0]
                if subject in seen_subjects:
                    continue
                seen_subjects.add(subject)

                object_candidates = object_sampler.sample(object_candidates_count)
                rng.shuffle(object_candidates)
                batch.append((subject, object_candidates))

            if not batch:
                break

            if select_object_with_llm:
                assert client is not None
                chosen_objects = await asyncio.gather(
                    *[
                        select_object_async(client, model, subject=subject, verb=verb, object_candidates=candidates)
                        for subject, candidates in batch
                    ]
                )
            else:
                chosen_objects = [candidates[0] for _, candidates in batch]

            resolved = [(subject, chosen) for (subject, _), chosen in zip(batch, chosen_objects) if chosen is not None]
            logger.info(
                f"verb={verb!r}: object selection resolved {len(resolved)}/{len(batch)} "
                f"(accepted {accepted}/{sentence_per_verb}, attempts so far: {attempts:,})"
            )

            accepted_pairs: list[_Accepted] = []
            for subject, object_ in resolved:
                sentence_causative = f"{subject} make {object_} {verb} ."
                sentence_transitive = f"{subject} {verb} {object_} ."
                input_ids_causative = encode_sentence(tokenizer, sentence_causative, bos_token_id, eos_token_id)
                input_ids_transitive = encode_sentence(tokenizer, sentence_transitive, bos_token_id, eos_token_id)
                accepted_pairs.append(
                    (
                        subject,
                        object_,
                        sentence_causative,
                        sentence_transitive,
                        input_ids_causative,
                        input_ids_transitive,
                    )
                )

            for (
                subject,
                object_,
                sentence_causative,
                sentence_transitive,
                input_ids_causative,
                input_ids_transitive,
            ) in accepted_pairs:
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
                f"Only generated {accepted:,}/{sentence_per_verb:,} sentences for verb {verb!r} "
                f"after {attempts:,} attempts"
            )

        logger.info(f"verb={verb!r}: accepted {accepted:,}/{sentence_per_verb:,} sentences in {attempts:,} attempts")

    return pl.DataFrame(rows).with_columns(
        pl.col("index_sentence").cast(pl.UInt32),
        pl.col("index_verb_context").cast(pl.UInt32),
        pl.col("input_ids").cast(pl.List(pl.Int64)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--verbs", nargs="+", default=DEFAULT_VERBS)
    parser.add_argument("--sentence-per-verb", type=int, default=100)
    parser.add_argument("--max-attempts-per-verb", type=int, default=100_000)
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Power smoothing exponent for prior distribution (0=uniform, 1=unsmoothed)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-llm-object-selection",
        dest="select_object_with_llm",
        action="store_false",
        help="Disable LLM-based object selection, just take the first sampled candidate "
        "(no server required; useful for testing the template path).",
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
        "--selection-batch-size",
        type=int,
        default=DEFAULT_SELECTION_BATCH_SIZE,
        help="Number of (subject, candidates) rounds to resolve in parallel per batch (default: %(default)s).",
    )
    parser.add_argument(
        "--object-candidates",
        type=int,
        default=DEFAULT_OBJECT_CANDIDATES,
        help="Number of weighted-sampled object candidates offered to the LLM per round (default: %(default)s).",
    )
    args = parser.parse_args()
    if not (0 <= args.alpha <= 1):
        parser.error("--alpha must be between 0 and 1")

    df_testset = asyncio.run(
        generate_weighted_testset(
            tokenizer_path=args.tokenizer,
            verbs=args.verbs,
            sentence_per_verb=args.sentence_per_verb,
            max_attempts_per_verb=args.max_attempts_per_verb,
            alpha=args.alpha,
            seed=args.seed,
            select_object_with_llm=args.select_object_with_llm,
            base_url=args.base_url,
            model=args.model,
            selection_batch_size=args.selection_batch_size,
            object_candidates_count=args.object_candidates,
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_testset.write_parquet(args.output)
    print(f"Wrote {len(df_testset):,} rows to {args.output}")


if __name__ == "__main__":
    main()

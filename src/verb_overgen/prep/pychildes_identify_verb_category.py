"""Classify verbs by causative-alternation category using an LLM.

Queries an OpenAI-compatible vLLM server to classify each verb in
`var/pychildes/lexical/verb_count_causative_full.csv` into one of
transitive-only / intransitive-only / alternating / na (see PROMPT below).

Usage:
    python -m verb_overgen.prep.identify_verb_category
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

import openai
import polars as pl
from loguru import logger

PROMPT = """You are a linguistic annotator classifying English verbs by how they express causation.

THE CORE QUESTION
For a verb V, causation can be expressed in two places: inside the verb itself (a lexical causative,
"A V-ed B"), or outside it via periphrastic "make" ("A made B V"). Your job is to determine which of
these two are available for V. Nothing else is relevant — in particular, do NOT ask whether V has an
intransitive or inchoative sense. That is not part of any category below.

THE TWO TESTS

Test P (periphrastic): Is "A made B V" grammatical, with V appearing BARE, and does it mean "A caused
    B to V" / "A caused B to become V-ed"?
    Coercion counts. If B is an agent that A compels to act ("I made him sing"), Test P passes — you
    need not ask whether B is an undergoer.
    But V must stand bare. If the sentence only becomes acceptable once you add a complement, P
    FAILS: "*A made B weigh" is bad, and "A made B weigh more" does not rescue it.

Test T (transitive): CAN "A V-ed B" carry a causative meaning — i.e. is a reading available on which
    it means "A caused B to V" / "A caused B to become V-ed"?
    "Can", not "must": V may well have other, non-causative transitive readings. Those do not
    disqualify it. The test passes as long as a causative reading is available at all.

    ENTAILMENT CHECK — apply this to every candidate causative reading of "A V-ed B".
    A genuine causative entails that B itself underwent V:
        "A V-ed B"  must entail  "B V-ed"  (or "B became V-ed").
    Work through it concretely:
        melt: "A melted B" entails "B melted"              → entailment holds, T PASSES
        kill: "A killed B" entails "B became dead"         → entailment holds, T PASSES
        eat:  "A ate B"    does NOT entail "B ate"         → entailment FAILS, T FAILS
              ("A ate B" means A consumed B. B is what got eaten; B never did any eating.)
    A transitive object that is simply the thing V is done TO is not a causee. Do not mistake it for
    one just because the sentence is well-formed.

A test FAILS in two distinct situations, and you must treat both as failure:
  (a) the sentence is ungrammatical                     — e.g. "*A laughed B"
  (b) the sentence is grammatical but not causative     — e.g. "A understood B" is perfectly
      well-formed, but it does not mean "A caused B to understand", so Test T fails for `understand`.
Do not let case (b) slip through just because the sentence sounds fine. Case (b) is the most common
error: run the entailment check rather than trusting your ear.

CATEGORIES
Apply the (Test P, Test T) outcome pair:

1. alternating       (P passes, T passes)
   Both the lexical causative and the periphrastic are available.
   stop  → "A made B stop" ✓ ; "A stopped B" ✓ causative
   break → "A made B break" ✓ ; "A broke B" ✓ causative

2. intransitive-only (P passes, T fails)
   Causation must be carried by "make"; the verb has no lexical causative.
   laugh      → "A made B laugh" ✓ ; "*A laughed B" ✗ ungrammatical
   understand → "A made B understand" ✓ ; "A understood B" — grammatical, but NOT causative ✗
   like       → "A made B like it" ✓ ; "A liked B" — grammatical, but NOT causative ✗
   sing       → "A made B sing" ✓ (coercion) ; "*A sang B" ✗ ungrammatical
   Note that stative verbs belong here whenever "make" works on them, which it usually does.
   Verbs whose object is a plain theme rather than a causee also land here: "make B eat" passes P,
   while "A ate B" fails T's entailment check, so `eat` is intransitive-only, not alternating.

3. transitive-only   (P fails, T passes)
   Causation is carried by the verb; the periphrastic is unavailable, typically because V is
   obligatorily transitive and so cannot appear as the bare complement of "make".
   kill → "*A made B kill" ✗ ; "A killed B" ✓ = "A caused B to become dead"
   cut  → "*A made B cut" ✗  ; "A cut B" ✓ = "A caused B to become cut"

4. na                (P fails, T fails)
   Causation cannot be expressed at all. The defining property is simply that "make" fails —
   there is no further condition.
   cost     → "*A made B cost" ✗ ; "It cost B $5" — grammatical, but NOT causative ✗
   resemble → "*A made B resemble" ✗ ; "A resembled B" — grammatical, but NOT causative ✗

SENSE SELECTION
Judge V in its most frequent, basic sense as used in everyday speech to and around young children.
Ignore idioms, technical jargon, and marginal figurative extensions.

DECISION PROCEDURE
Step 1: Run Test P.
Step 2: Run Test T.
Step 3: Map the pair to a category:
        (P pass, T pass) → alternating
        (P pass, T fail) → intransitive-only
        (P fail, T pass) → transitive-only
        (P fail, T fail) → na

OUTPUT FORMAT
Respond with one of the following category labels (transitive-only, intransitive-only, alternating, na), nothing else.

VERB TO CLASSIFY: {verb}
"""

CATEGORIES = frozenset({"transitive-only", "intransitive-only", "alternating", "na"})

DEFAULT_INPUT_PATH = Path("./var/pychildes/lexical/verb_count_causative_full.csv")
DEFAULT_OUTPUT_PATH = Path("./var/pychildes/lexical/verb_category_llm.csv")
DEFAULT_BASE_URL = "http://kn175:30000/v1"
DEFAULT_MODEL = "/model-weights/gemma-4-31B-it/"
DEFAULT_CONCURRENCY = 32

# Gemma 4 recommended sampling config (temperature/top_p/top_k); min_p/presence_penalty/repetition_penalty
# are carried over from the prior Qwen3.5 preset since Gemma 4 doesn't specify recommendations for them.
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 64
DEFAULT_MIN_P = 0.0
DEFAULT_PRESENCE_PENALTY = 1.5
DEFAULT_REPETITION_PENALTY = 1.0
DEFAULT_MAX_TOKENS = 8192

# Escalate-on-disagreement self-consistency: sample INITIAL_SAMPLES times; if there's no unique
# plurality winner (e.g. a 1-1-1 split across 4 categories), draw ESCALATED_SAMPLES more and take
# the plurality over all of them. Keeps the common case cheap (3 calls) while giving disagreeing
# verbs a fairer vote (5 calls) instead of paying 5x for every verb.
DEFAULT_INITIAL_SAMPLES = 3
DEFAULT_ESCALATED_SAMPLES = 2


def _parse_category(raw: str) -> str | None:
    """Parse the model output to one of CATEGORIES, or None if unparseable."""
    normalized = raw.strip().strip(".").lower()
    if normalized in CATEGORIES:
        return normalized
    # Some backends (e.g. vLLM serving gemma-4-31B-it without a --reasoning-parser configured)
    # don't split the <thought> trace out of `content`; the true label still lands as the last
    # token, glued directly onto the trace with no separator. Longest-first avoids "...only"
    # falsely matching "transitive-only" when the true label is "intransitive-only".
    for category in sorted(CATEGORIES, key=len, reverse=True):
        if normalized.endswith(category):
            return category
    return None


def _plurality(labels: list[str]) -> tuple[str | None, int, int]:
    """Return (winning label, votes for winner, total valid votes); label is None if no unique winner."""
    if not labels:
        return None, 0, 0
    ranked = Counter(labels).most_common()
    top_label, top_count = ranked[0]
    if len(ranked) > 1 and ranked[1][1] == top_count:
        return None, top_count, len(labels)
    return top_label, top_count, len(labels)


async def _sample_once(
    client: openai.AsyncClient,
    model: str,
    verb: str,
    *,
    semaphore: asyncio.Semaphore,
    temperature: float,
    max_tokens: int,
) -> tuple[str | None, str]:
    """Query the LLM once for one verb; returns (category, raw_response)."""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": PROMPT.format(verb=verb)}],
                temperature=temperature,
                top_p=DEFAULT_TOP_P,
                presence_penalty=DEFAULT_PRESENCE_PENALTY,
                extra_body={
                    "top_k": DEFAULT_TOP_K,
                    "min_p": DEFAULT_MIN_P,
                    "repetition_penalty": DEFAULT_REPETITION_PENALTY,
                    "chat_template_kwargs": {"enable_thinking": True},
                },
                max_tokens=max_tokens,
            )
            raw = (response.choices[0].message.content or "").strip()
            return _parse_category(raw), raw
        except Exception as exc:
            logger.warning(f"API error for verb={verb!r}: {exc}")
            return None, f"ERROR: {exc}"


async def classify_verb(
    client: openai.AsyncClient,
    model: str,
    verb: str,
    *,
    semaphore: asyncio.Semaphore,
    temperature: float,
    max_tokens: int,
    initial_samples: int,
    escalated_samples: int,
) -> dict:
    """Vote-classify one verb, escalating to more samples if the initial round doesn't agree."""
    samples = await asyncio.gather(
        *[
            _sample_once(client, model, verb, semaphore=semaphore, temperature=temperature, max_tokens=max_tokens)
            for _ in range(initial_samples)
        ]
    )
    labels = [category for category, _ in samples if category is not None]
    raws = [raw for _, raw in samples]
    winner, n_agree, n_votes = _plurality(labels)

    escalated = False
    if winner is None and escalated_samples > 0:
        escalated = True
        extra = await asyncio.gather(
            *[
                _sample_once(client, model, verb, semaphore=semaphore, temperature=temperature, max_tokens=max_tokens)
                for _ in range(escalated_samples)
            ]
        )
        labels += [category for category, _ in extra if category is not None]
        raws += [raw for _, raw in extra]
        winner, n_agree, n_votes = _plurality(labels)

    if winner is None:
        logger.warning(f"No consensus for verb={verb!r} after {len(raws)} samples: {labels}")

    return {
        "verb": verb,
        "category": winner,
        "n_agree": n_agree,
        "n_votes": n_votes,
        "escalated": escalated,
        # JSON-encoded so write_csv (no nested-column support) and eyeballing the file both work.
        "raw_responses": json.dumps(raws),
    }


async def classify_verbs(
    verbs: list[str],
    *,
    base_url: str,
    model: str,
    concurrency: int,
    temperature: float,
    max_tokens: int,
    initial_samples: int,
    escalated_samples: int,
) -> pl.DataFrame:
    client = openai.AsyncClient(base_url=base_url, api_key="None")
    semaphore = asyncio.Semaphore(concurrency)

    results = await asyncio.gather(
        *[
            classify_verb(
                client,
                model,
                verb,
                semaphore=semaphore,
                temperature=temperature,
                max_tokens=max_tokens,
                initial_samples=initial_samples,
                escalated_samples=escalated_samples,
            )
            for verb in verbs
        ]
    )

    return pl.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help="Base URL of the OpenAI-compatible vLLM server (default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model name passed to the server; must match --model-path used at launch (default: %(default)s).",
    )
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Max tokens per response; thinking mode needs headroom for the reasoning trace (default: %(default)s).",
    )
    parser.add_argument(
        "--initial-samples",
        type=int,
        default=DEFAULT_INITIAL_SAMPLES,
        help="Number of samples to draw per verb before checking for a plurality winner (default: %(default)s).",
    )
    parser.add_argument(
        "--escalated-samples",
        type=int,
        default=DEFAULT_ESCALATED_SAMPLES,
        help="Extra samples to draw only for verbs with no plurality winner in the initial round "
        "(default: %(default)s).",
    )
    args = parser.parse_args()

    verbs = pl.read_csv(args.input).get_column("verb").to_list()
    logger.info(f"Classifying {len(verbs):,} verbs from {args.input}")

    df_result = asyncio.run(
        classify_verbs(
            verbs,
            base_url=args.base_url,
            model=args.model,
            concurrency=args.concurrency,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            initial_samples=args.initial_samples,
            escalated_samples=args.escalated_samples,
        )
    )

    n_no_consensus = df_result["category"].is_null().sum()
    n_escalated = df_result["escalated"].sum()
    logger.info(f"Category counts:\n{df_result['category'].value_counts()}")
    logger.info(f"Escalated to extra samples for {n_escalated}/{len(df_result)} verbs")
    if n_no_consensus:
        logger.warning(f"{n_no_consensus}/{len(df_result)} verbs had no plurality winner (category=null)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_result.write_csv(args.output)
    print(f"Wrote {len(df_result):,} rows to {args.output}")


if __name__ == "__main__":
    main()

"""Causative *have* LLM vetting via Gemini API.

Usage:
    python -m verb_overgen.prep.anly.20260320.causative_long_tail_have_llm_vet
"""

import asyncio
import os
from datetime import datetime

from google import genai
import polars as pl

import verb_overgen.common.startup as _startup

client = genai.Client()

MODEL = "gemini-3-flash-preview"
CONCURRENCY = 8
SAVE_EVERY = 128 

PROMPT = """\
In the following sentences, some contain a causative *have* and some do not.
"May I have more juice please ?" (False)
"I have finished my juice ." (False)
"I had the waiter bring more juice ." (True)
"I had my juice refilled by the waiter ." (True)

Analyze the following sentence and determine whether it contains a causative *have* or not. Answer with True if it does and False if it does not.
"{sentence}"
"""

INPUT_CSV = "./var/pychildes/intermidiate-annotation/verb-overgen_causative-candidate_have_20260320.csv"
OUTPUT_CSV_TEMPLATE = "./var/pychildes/intermidiate-annotation/verb-overgen_causative-candidate_have_llm-vet_{ts}.csv"


def _parse_verdict(raw: str) -> bool | None:
    """Strictly parse Gemini output to a bool, or None if unparseable."""
    normalized = raw.strip().rstrip(".").lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


async def vet_one(
    sem: asyncio.Semaphore, row: dict
) -> tuple[str, bool | None, str]:
    """Send one row to Gemini; returns (key, verdict_gemini, note_gemini)."""
    key = f"{row['id_file']}:{row['id_utterance']}"
    text = PROMPT.format(sentence=row["content_raw"])

    async with sem:
        try:
            response = await client.aio.models.generate_content(
                model=MODEL,
                contents=text,
            )
            raw = response.text.strip()
            verdict = _parse_verdict(raw)
            if verdict is None:
                print(f"Unparseable response for {key}: {raw!r}")
            return key, verdict, raw
        except Exception as e:
            print(f"Error for {key}: {e}")
            return key, None, f"ERROR: {e}"


async def run():
    """Send all rows to Gemini with bounded concurrency and incremental saves."""
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_csv = OUTPUT_CSV_TEMPLATE.format(ts=ts)

    dataset_input = pl.read_csv(INPUT_CSV)
    # Note: use "content_raw" column, not "sentence" (which is the tokenized version).
    rows = list(dataset_input.iter_rows(named=True))

    sem = asyncio.Semaphore(CONCURRENCY)
    completed = 0
    file_exists = False

    for batch_start in range(0, len(rows), SAVE_EVERY):
        batch = rows[batch_start : batch_start + SAVE_EVERY]

        tasks = [vet_one(sem, row) for row in batch]
        batch_results = await asyncio.gather(*tasks)

        verdict_map: dict[str, bool | None] = {}
        note_map: dict[str, str] = {}
        for key, verdict, note in batch_results:
            verdict_map[key] = verdict
            note_map[key] = note

        # Build the slice of the input DataFrame for this batch
        batch_df = dataset_input.slice(batch_start, len(batch)).with_columns(
            pl.struct(["id_file", "id_utterance"])
            .map_elements(
                lambda row: verdict_map.get(
                    f"{row['id_file']}:{row['id_utterance']}"
                ),
                return_dtype=pl.Boolean,
            )
            .alias("verdict_gemini"),
            pl.struct(["id_file", "id_utterance"])
            .map_elements(
                lambda row: note_map.get(
                    f"{row['id_file']}:{row['id_utterance']}", ""
                ),
                return_dtype=pl.String,
            )
            .alias("note_gemini"),
        )

        # Append to CSV: write header only on first batch
        with open(output_csv, "a") as f:
            f.write(batch_df.write_csv(include_header=not file_exists))
        file_exists = True

        completed += len(batch)
        print(f"Processed {completed}/{len(rows)} rows (saved to {output_csv})")

    print(f"Done. {len(rows)} rows written to {output_csv}")


if __name__ == "__main__":
    asyncio.run(run())

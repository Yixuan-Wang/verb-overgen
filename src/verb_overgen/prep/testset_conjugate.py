import asyncio
import polars as pl
from openai import AsyncOpenAI
from pathlib import Path
from tqdm.asyncio import tqdm

client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

MODEL = "/model-weights/gemma-4-26B-A4B-it/"

SYSTEM_PROMPT = (
    "You are a linguist. The user will give you one or two sentences. "
    + "Conjugate the finite verb into past tense. "
    + "For example, 'You make Henry sing.' becomes 'You made Henry sing.'. "
    + "Even if the sentence is unnatural or ungrammatical, only conjugate the finite verb according to English morphological rules and keep all other words unchanged. "
    + "Output only the conjugated sentence with no explanation."
)

INPUT_PATH = Path("data/testset/output/testset_gemini.20260402T2045.parquet")
OUTPUT_PATH = Path("var/testset/overgen-gemini.20260402T2045.conjugated.parquet")


async def conjugate(index: int, sentence: str) -> tuple[int, str | None]:
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": sentence},
            ],
            max_tokens=256,
            temperature=1.0,
        )
        return index, response.choices[0].message.content
    except Exception:
        return index, None


BATCH_SIZE = 32


async def main():
    df = pl.read_parquet(INPUT_PATH).select(
        "index_sentence", "verb", "index_verb_context", "hash_context", "kind", "sentence"
    )
    rows = list(zip(df["index_sentence"].to_list(), df["sentence"].to_list()))

    results: dict[int, str | None] = {}
    batches = [rows[i:i + BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]
    for batch in tqdm(batches, desc="Conjugating"):
        for index, sentence_past in await asyncio.gather(*[conjugate(idx, s) for idx, s in batch]):
            results[index] = sentence_past

    results_df = pl.DataFrame({
        "index_sentence": list(results.keys()),
        "sentence": list(results.values()),
    }).with_columns(pl.col("index_sentence").cast(df["index_sentence"].dtype))
    out = df.drop("sentence").join(results_df, on="index_sentence", how="left")
    out.write_parquet(OUTPUT_PATH)
    print(f"Written to {OUTPUT_PATH}")


asyncio.run(main())

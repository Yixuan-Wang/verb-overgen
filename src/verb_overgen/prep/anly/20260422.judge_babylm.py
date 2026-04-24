import asyncio
import polars as pl
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

MODEL = "/model-weights/gemma-4-26B-A4B-it/"

SYSTEM_PROMPT = (
    "You are a linguist. The user will give you a sentence. "
    + "Determine whether the sentence contains a periphrastic causative usage of 'laugh'. "
    + "The word 'laugh' must be used as a verb, and there must be a causee, such as in 'She made him laugh' (causee: him). "
    + "Reply with only 'yes' or 'no'."
)

OUTPUT_PATH = "var/babylm/babylm_causative_laugh.parquet"

BATCH_SIZE = 32


async def vet(index: int, text: str) -> tuple[int, bool | None]:
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=1.0,
            max_tokens=4096,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
        answer = (response.choices[0].message.content or "").strip().lower()
        return index, answer.endswith("yes")
    except Exception:
        return index, None


async def main():
    df = (
        pl.scan_parquet("/scratch/tomyxw/share/datasets/kanishka--babylm1sents/data")
        .with_row_index()
        .with_columns(
            pl.col("text").str.strip_chars(),
            pl.col("text")
            .str.contains(
                r"\b(make|making|made|makes|cause|causing|caused|causes|let|letting|lets|had|have|having|get|getting|got|gets)\b(.*)\b(laugh|laughing|laughed)\b"
            )
            .alias("vet_regex")
        )
        .collect()
    )

    candidates = df.filter(pl.col("vet_regex")).select("index", "text")
    rows = list(zip(candidates["index"].to_list(), candidates["text"].to_list()))

    llm_results: dict[int, bool | None] = {}
    batches = [rows[i : i + BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]
    for batch in tqdm(batches, desc="Vetting with LLM"):
        for idx, verdict in await asyncio.gather(*[vet(i, t) for i, t in batch]):
            llm_results[idx] = verdict

    llm_df = pl.DataFrame(
        {
            "index": list(llm_results.keys()),
            "vet_llm": list(llm_results.values()),
        }
    ).with_columns(pl.col("index").cast(df["index"].dtype))

    out = df.select("index", "text", "vet_regex").join(llm_df, on="index", how="left")
    out.write_parquet(OUTPUT_PATH)
    print(f"Written to {OUTPUT_PATH}")
    print(f"  Total rows:        {len(out)}")
    print(f"  Regex positives:   {out['vet_regex'].sum()}")
    print(f"  LLM positives:     {out['vet_llm'].sum()}")


asyncio.run(main())

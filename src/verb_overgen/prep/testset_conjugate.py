import argparse
import asyncio
import os
import re
import polars as pl
from openai import AsyncOpenAI
from pathlib import Path
from tokenizers import Tokenizer
from tqdm.asyncio import tqdm

from verb_overgen.common.tokenize import BOS_TOKEN, EOS_TOKEN
from verb_overgen.prep.testset_gen_weighted import DEFAULT_TOKENIZER_PATH, encode_sentence, required_token_id

# Reads OPENAI_BASE_URL / OPENAI_API_KEY from the environment; vLLM servers don't check
# the key, so default it to a placeholder when unset.
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))

DEFAULT_MODEL = "/model-weights/gemma-4-31B-it/"

SYSTEM_PROMPT = (
    "You are a linguist. The user will give you one or two sentences. "
    + "Conjugate the finite verb into past tense. "
    + "For example, 'You make Henry sing.' becomes 'You made Henry sing.'. "
    + "Even if the sentence is unnatural or ungrammatical, only conjugate the finite verb according to English morphological rules and keep all other words unchanged. "
    + "Output only the conjugated sentence with no explanation."
)

DEFAULT_INPUT_PATH = Path("data/testset/output/testset_gemini.20260402T2045.parquet")
DEFAULT_OUTPUT_PATH = Path("var/testset/overgen-gemini.20260402T2045.conjugated.parquet")

# Gemma 4 emits a channel-style output: <|channel>thought\n{reasoning}<channel|>{answer}
# These are special tokens, so they must be requested with skip_special_tokens=False
# or the <channel|> boundary is silently stripped and reasoning/answer become unsplittable.
CHANNEL_RE = re.compile(r"^<\|channel>\w+\n.*?<channel\|>(?P<answer>.*)$", re.DOTALL)


def extract_answer(raw_content: str) -> str:
    m = CHANNEL_RE.match(raw_content)
    return (m.group("answer") if m else raw_content).strip()


async def conjugate(index: int, sentence: str, model: str) -> tuple[int, str | None]:
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": sentence},
            ],
            max_tokens=4096,
            temperature=1.0,
            top_p=0.95,
            extra_body={"top_k": 64, "skip_special_tokens": False},
        )
        raw = response.choices[0].message.content or ""
        return index, extract_answer(raw)
    except Exception:
        return index, None


BATCH_SIZE = 64


async def main(input_path: Path, output_path: Path, tokenizer_path: Path, model: str):
    # Column set varies by pipeline (e.g. the LLM-overgen pipeline has `hash_context`,
    # the template pipeline has `input_ids`/`score` instead), so keep whatever the
    # input provides rather than hardcoding a column list.
    df = pl.read_parquet(input_path)
    rows = list(zip(df["index_sentence"].to_list(), df["sentence"].to_list()))

    results: dict[int, str | None] = {}
    batches = [rows[i:i + BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]
    for batch in tqdm(batches, desc="Conjugating"):
        for index, sentence_past in await asyncio.gather(*[conjugate(idx, s, model) for idx, s in batch]):
            results[index] = sentence_past

    results_df = pl.DataFrame({
        "index_sentence": list(results.keys()),
        "sentence": list(results.values()),
    }).with_columns(pl.col("index_sentence").cast(df["index_sentence"].dtype))
    out = df.drop("sentence").join(results_df, on="index_sentence", how="left")

    if "input_ids" in out.columns:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        bos_token_id = required_token_id(tokenizer, BOS_TOKEN)
        eos_token_id = required_token_id(tokenizer, EOS_TOKEN)
        out = out.with_columns(
            pl.col("sentence")
            .map_elements(
                lambda s: encode_sentence(tokenizer, s, bos_token_id, eos_token_id),
                return_dtype=pl.List(pl.Int64),
            )
            .alias("input_ids")
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(output_path)
    print(f"Written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()

    asyncio.run(main(args.input, args.output, args.tokenizer, args.model))

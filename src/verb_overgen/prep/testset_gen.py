import argparse

import numpy as np
import polars as pl
import sglang as sgl
from sglang.srt.parser.reasoning_parser import ReasoningParser
from transformers import AutoTokenizer

from verb_overgen.common.features import VERB_TARGET_17

MODEL_PATH = "/model-weights/Qwen3-32B"

SYSTEM_PROMPT = "You are a child literature author tasked with creating engaging and easy-to-understand texts for young children."
PROMPT = """
### Instructions
1.  Object Selection: Choose an object from the vocabulary bank that fits naturally with the subject and verb.
2.  Scene Creation:
    - The scene should be simple and relatable for a young child.
    - In this scene, the subject causes the object to perform the action described by the verb.
3.  Output Generation:
    - First, write a short sentence describing the context of this action. Without it the action should still be understandable. The context sentence should also contain the subject and the object.
    - Then write the main sentence where the subject makes the object perform the action.
4.  Constraints:
    - The main sentence should have the following syntactic structure: "{{subject}} makes {{object}} {{verb}}".
    - Both sentences should be declarative sentences, and quotes are not allowed.

### Output Format
The output should only contain two sentences separated by a newline character. For example given subject "{subject}", verb "break", and object "vase", the output should be:
{subject} knocks over the vase.
{subject} makes the vase break.

### Input
- Subject: {subject}
- Verb: {verb}
- Object Candidates: {object_candidates}
"""

RAND_GEN = np.random.default_rng(42)


def sample(
    *,
    dataset: pl.DataFrame,
    n: int,
    randgen: np.random.Generator,
) -> list[str]:
    return randgen.choice(
        dataset.get_column("lemma").to_list(),
        size=n,
        replace=False,
        p=dataset.get_column("proportion").to_numpy() / dataset.get_column("proportion").sum(),
    ).tolist()


def generate_prompt(
    *,
    verb: str,
    dataset_names: pl.DataFrame,
    dataset_nouns: pl.DataFrame,
    randgen_names: np.random.Generator | None = None,
    randgen_nouns: np.random.Generator | None = None,
):
    randgen_names = np.random.default_rng() if randgen_names is None else randgen_names
    randgen_nouns = np.random.default_rng() if randgen_nouns is None else randgen_nouns

    while True:
        subject, *object_candidates = sample(dataset=dataset_names, n=11, randgen=randgen_names)
        object_candidates.extend(sample(dataset=dataset_nouns, n=15, randgen=randgen_nouns))
        object_candidates_str = ", ".join(object_candidates)

        prompt_filled = PROMPT.format(
            subject=subject,
            verb=verb,
            object_candidates=object_candidates_str,
        )

        yield {
            "verb": verb,
            "fragment_subject": subject,
            "fragment_object_candidates": object_candidates,
            "prompt": prompt_filled,
        }

def get_message(sentence: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": sentence,
        }
    ]


_DTYPE_PARSED_RESULTS = pl.Struct({
    "output_reasoning": pl.String,
    "output_text": pl.String,
    "error": pl.String,
})

class LLMGenMock:
    def __call__(self, prompt_row: dict) -> dict:
        subject = prompt_row["fragment_subject"]
        obj = prompt_row["fragment_object_candidates"][0]
        verb = prompt_row["verb"]
        text = f"{subject} do the {obj}.\n{subject} makes the {obj} {verb}."
        return {"output_reasoning": None, "output_text": text, "error": None}


class LLMGenSGL:
    def __init__(self, tp_size: int, max_running_requests: int):
        self.llm = sgl.Engine(
            model_path=MODEL_PATH,
            reasoning_parser="qwen3",
            grammar_backend="xgrammar",
            tp_size=tp_size,
            max_running_requests=max_running_requests,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.parser = ReasoningParser("qwen3")

    def __call__(self, prompt_row: dict) -> dict:
        input = self.tokenizer.apply_chat_template(
            get_message(prompt_row["prompt"]),
            tokenize=False,
            add_generation_prompt=True,
            return_dict=False,
        )

        sampling_params = {
            "max_new_tokens": 32768,
            "skip_special_tokens": False,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "min_p": 0,
        }
        result = self.llm.generate(prompt=input, sampling_params=sampling_params)

        try:
            reasoning_text, text = self.parser.parse_non_stream(result["text"])
            return {"output_reasoning": reasoning_text, "output_text": text, "error": None}
        except Exception as e:
            return {"output_reasoning": None, "output_text": None, "error": str(e)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.3, help="Power smoothing exponent for prior distribution (0=uniform, 1=unsmoothed)")
    parser.add_argument("--prompt-per-verb", type=int, default=75)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--max-running-requests", type=int, default=32)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    dataset_names = pl.read_csv("./data/testset/gen/names.csv").extend(
        pl.read_csv("./data/testset/gen/top_pronouns.csv").rename({"count_is_pronoun": "count_is_noun"})
    ).with_columns(
        (pl.col("count").pow(args.alpha) / pl.col("count").pow(args.alpha).sum()).alias("proportion")
    )

    dataset_nouns = (
        pl.read_csv("./data/testset/gen/top_nouns_v2.csv")
        .filter(pl.col("is_noun"))
        .with_columns((pl.col("count_is_noun").pow(args.alpha) / pl.col("count_is_noun").pow(args.alpha).sum()).alias("proportion"))
    )

    llm_gen = LLMGenMock() if args.mock else LLMGenSGL(tp_size=args.tp_size, max_running_requests=args.max_running_requests)

    all_results: list[dict] = []
    for verb in VERB_TARGET_17:
        gen_prompt = generate_prompt(
            verb=verb,
            dataset_names=dataset_names,
            dataset_nouns=dataset_nouns,
            randgen_names=RAND_GEN,
            randgen_nouns=RAND_GEN,
        )
        for round_idx in range(args.prompt_per_verb):
            prompt_row = next(gen_prompt)
            result = llm_gen(prompt_row)
            all_results.append({**prompt_row, "llm_gen": result})

            # TODO: update Thompson sampling state using result

    pl.DataFrame(all_results, schema_overrides={"llm_gen": _DTYPE_PARSED_RESULTS}).write_parquet(args.output)

import argparse

import numpy as np
import polars as pl
import sglang as sgl
from sglang.srt.parser.reasoning_parser import ReasoningParser
from transformers import AutoTokenizer
from loguru import logger

from tqdm import tqdm

from verb_overgen.common.features import VERB_TARGET_5, VERB_TARGET_17

MODEL_PATH = "/model-weights/Qwen3-32B"

SYSTEM_PROMPT = (
    "You are a child literature author tasked with creating engaging and easy-to-understand texts for young children."
)
PROMPT = """\
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
The output should only contain two sentences separated by a newline character. For example given subject "I", verb "break", and object "vase", the output should be:
I knock over the vase.
I make the vase break.
{previous_outputs}
### Input
- Subject: {subject}
- Verb: {verb}
- Object Candidates: {object_candidates}
"""

PROMPT_PREVIOUS = """
### Previous Outputs
The following outputs have already been generated for this verb. Prefer to create a different scene.
---
{}
---
"""

RAND_GEN = np.random.default_rng()


class WeightedSampler:
    def __init__(self, labels: list[str], weights: np.ndarray, rng: np.random.Generator):
        self.labels = labels
        self.rng = rng
        self.probs = weights / weights.sum()

    def sample(self, n: int) -> list[str]:
        indices = self.rng.choice(len(self.labels), size=min(n, len(self.labels)), replace=False, p=self.probs)
        return [self.labels[i] for i in indices]


def generate_prompt(
    *,
    verb: str,
    subject_sampler: WeightedSampler,
    object_sampler: WeightedSampler,
    previous_outputs: list[str],
    rng: np.random.Generator,
):
    while True:
        subject = subject_sampler.sample(1)[0]
        object_candidates = object_sampler.sample(25)
        object_sampler.rng.shuffle(object_candidates)
        object_candidates_str = ", ".join(object_candidates)

        if previous_outputs:
            sampled = rng.choice(previous_outputs, size=min(3, len(previous_outputs)), replace=False).tolist()
            filled_previous_output = PROMPT_PREVIOUS.format("\n---\n".join(sampled))
        else:
            filled_previous_output = ""

        prompt_filled = PROMPT.format(
            subject=subject,
            verb=verb,
            object_candidates=object_candidates_str,
            previous_outputs=filled_previous_output,
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
        },
    ]


_DTYPE_PARSED_RESULTS = pl.Struct(
    {
        "output_reasoning": pl.String,
        "output_text": pl.String,
        "error": pl.String,
    }
)


class LLMGenMock:
    def __call__(self, prompt_row: dict) -> dict:
        subject = prompt_row["fragment_subject"]
        obj = prompt_row["fragment_object_candidates"][0]
        verb = prompt_row["verb"]
        text = f"{subject} do the {obj}.\n{subject} makes the {obj} {verb}."
        return {"output_reasoning": None, "output_text": text, "error": None}


class LLMGenGemini:
    def __init__(self, model: str = "gemini-3-flash-preview"):
        from google import genai
        from google.genai import types

        self.client = genai.Client()
        self.model = model
        self.types = types

    def __call__(self, prompt_row: dict) -> dict:
        try:
            response = self.client.models.generate_content(
                model=self.model,
                config=self.types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=1,
                ),
                contents=prompt_row["prompt"],
            )
            text = response.text
            return {"output_reasoning": None, "output_text": text, "error": None}
        except Exception as e:
            return {"output_reasoning": None, "output_text": None, "error": str(e)}


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
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Power smoothing exponent for prior distribution (0=uniform, 1=unsmoothed)",
    )
    parser.add_argument("--round-per-verb", type=int, default=75)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--max-running-requests", type=int, default=32)
    parser.add_argument(
        "--backend", type=str, default="sgl", choices=["mock", "sgl", "gemini"], help="LLM backend to use"
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-3-flash-preview",
        help="Gemini model name (only used with --backend gemini)",
    )
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    if not (0 <= args.alpha <= 1):
        parser.error("--alpha must be between 0 and 1")

    dataset_names = (
        pl.read_csv("./data/testset/gen/names.csv")
        .extend(pl.read_csv("./data/testset/gen/top_pronouns.csv").rename({"count_is_pronoun": "count_is_noun"}))
        .with_columns(pl.col("count").pow(args.alpha).alias("weight"))
    )

    dataset_nouns = (
        pl.read_csv("./data/testset/gen/top_nouns_v2.csv")
        .filter(pl.col("is_noun"))
        .with_columns(pl.col("count_is_noun").pow(args.alpha).alias("weight"))
    )

    if args.backend == "mock":
        llm_gen = LLMGenMock()
    elif args.backend == "gemini":
        llm_gen = LLMGenGemini(model=args.gemini_model)
    else:
        llm_gen = LLMGenSGL(tp_size=args.tp_size, max_running_requests=args.max_running_requests)

    names_labels = dataset_names.get_column("lemma").to_list()
    names_weights = dataset_names.get_column("weight").to_numpy()

    nouns_labels = dataset_nouns.get_column("lemma").to_list()
    nouns_weights = dataset_nouns.get_column("weight").to_numpy()

    object_labels = names_labels + nouns_labels
    object_weights = np.concatenate([names_weights, nouns_weights])

    all_results: list[dict] = []
    # _set_verb_target_5 = set(VERB_TARGET_5)
    # verbs = [v for v in VERB_TARGET_17 if v not in _set_verb_target_5]  # FIXME: change to VERB_TARGET_17 sans VERB_TARGET_5
    verbs = ["go", "come", "look", "stay", "talk", "listen", "cough", "blink"]
    for verb in (pbar := tqdm(verbs, desc="verbs")):
        pbar.set_postfix(verb=verb)
        subject_sampler = WeightedSampler(names_labels, names_weights, RAND_GEN)
        object_sampler = WeightedSampler(object_labels, object_weights, RAND_GEN)
        previous_outputs: list[str] = []

        gen_prompt = generate_prompt(
            verb=verb,
            subject_sampler=subject_sampler,
            object_sampler=object_sampler,
            previous_outputs=previous_outputs,
            rng=RAND_GEN,
        )
        for round_idx in tqdm(range(args.round_per_verb), desc="rounds", leave=False):
            prompt_row = next(gen_prompt)
            result = llm_gen(prompt_row)
            all_results.append({**prompt_row, "llm_gen": result})

            output_text = result.get("output_text")
            if output_text:
                previous_outputs.append(output_text)
            else:
                logger.error(f"Error generating output for verb '{verb}' in round {round_idx}: {result.get('error')}")

    pl.DataFrame(all_results, schema_overrides={"llm_gen": _DTYPE_PARSED_RESULTS}).write_parquet(args.output)

from itertools import islice

import ray
import sglang as sgl
from sglang.srt.parser.reasoning_parser import ReasoningParser
from transformers import AutoTokenizer
import pyarrow as pa
import polars as pl
import numpy as np

MODEL_PATH = "/model-weights/Qwen3-32B"
PROMPT = """
### Role
You are a child literature author tasked with creating engaging and easy-to-understand texts for young children.

### Instructions
1.  Object Selection: Choose an object from the vocabulary bank that fits naturally with the subject and verb.
2.  Scene Creation:
    - The scene should be simple and relatable for a young child.
    - In this scene, the subject **causes** the object to perform the action described by the verb.
    - The scene can be fictional, therefore verbs involving harm or danger can be used without real-world consequences.
3.  Output Generation:
    - First, write a short sentence describing extra context of the action. Without it the action should still be understandable.
    - Then write the main sentence where the subject makes the object perform the action.
4.  Constraints:
    - The main sentence should have the following syntactic structure: "{{subject}} makes {{object}} {{verb}}".
      Additional modifiers are allowed, but no extra verbs are permitted in the main sentence.
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

expr_proportion = (pl.col("count") / pl.col("count").sum()).alias("proportion")

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


_DTYPE_GENERATED_PROMPTS = pa.struct([
    pa.field("fragment_subject", pa.string()),
    pa.field("fragment_object_candidates", pa.list_(pa.string())),
    pa.field("verb", pa.string()),
    pa.field("prompt", pa.string()),
])

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
            "role": "user",
            "content": sentence,
        }
    ]


_DTYPE_PARSED_RESULTS = pa.struct([
    pa.field("output_reasoning", pa.string()),
    pa.field("output_text", pa.string()),
    pa.field("error", pa.string()),
])

class LLMGen:
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

    def __call__(
        self,
        table: pa.Table,
    ):
        prompts = table.column("prompt").to_pylist()
        input = self.tokenizer.apply_chat_template(
            list(map(get_message, prompts)),
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
        results = self.llm.generate(prompt=input, sampling_params=sampling_params)
        parsed_results = []

        with __import__("ipdb").launch_ipdb_on_exception():
            for result in results:
                generated_text = result["text"]
                reasoning_text, text = self.parser.parse_non_stream(generated_text)
                parsed_results.append({
                    "output_reasoning": reasoning_text,
                    "output_text": text,
                    "error": None,
                })

        parsed_results_array = pa.array(parsed_results, type=_DTYPE_PARSED_RESULTS)
        return table.append_column("llm_gen", parsed_results_array)


if __name__ == "__main__":
    dataset_names = pl.read_csv("./data/testset/gen/names.csv").with_columns(expr_proportion)
    dataset_verbs = pl.read_csv("./data/testset/gen/causative_verb_counts.csv").with_columns(expr_proportion)
    dataset_nouns = pl.read_csv("./data/testset/gen/top_nouns.csv").with_columns(expr_proportion).filter(
        pl.col("lemma").str.len_chars() > 2
    )

    prompt_per_verb = 75

    list_df_prompts = []
    for verb in dataset_verbs.get_column("lemma").to_list():
        gen_prompt = generate_prompt(
            verb=verb,
            dataset_names=dataset_names,
            dataset_nouns=dataset_nouns,
            randgen_names=RAND_GEN,
            randgen_nouns=RAND_GEN,
        )
        df_prompts = pl.DataFrame({
            "prompt": pa.array(islice(gen_prompt, prompt_per_verb), type=_DTYPE_GENERATED_PROMPTS)
        })
        list_df_prompts.append(df_prompts)
    dataset_prompts = pl.concat(list_df_prompts).unnest("prompt")

    dataset = ray.data.from_arrow(dataset_prompts.to_arrow())
    
    tp_size = 2
    max_running_requests = 32
    num_instances = 2

    dataset = dataset.repartition(num_instances).map_batches(
        LLMGen, # type: ignore
        fn_constructor_kwargs={
            "tp_size": tp_size,
            "max_running_requests": max_running_requests,
        },
        compute=ray.data.ActorPoolStrategy(
            size=num_instances,
        ),
        batch_size=512,
        batch_format="pyarrow",
        num_cpus=3,
        num_gpus=tp_size,
        scheduling_strategy="SPREAD",
    )

    dataset.write_parquet("./data/testset/output/testset_llm_gen.20260108T2325.parquet")

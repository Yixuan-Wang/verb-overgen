from collections.abc import Iterator
from typing import cast, Any

import orjson
import pyarrow as pa
import ray
from ray.data.expressions import col
import sglang as sgl
from sglang.srt.parser.reasoning_parser import ReasoningParser
from transformers import AutoTokenizer
from loguru import logger


from apfel import not_none

MODEL_PATH = "/model-weights/Qwen3-32B"
# logger.warning("Using 8B model for debugging.")

def get_message(sentence: str | pa.StringScalar):
    return [
        {
            "role": "user",
            "content": """Some sentences are syntactically well-formed, but might not be natural. A few aspects to consider:

    - Are there any unusual word choices?
    - Is the scenario unusual?
    - Is the sentence semantically incoherent?

    If the answer to any of these questions is yes, then the sentence is unnatural.
    Judge if the following space-delimited sentence is natural: `{sentence}`
    Answer "Yes" if the sentence is natural, or "No" if it is unnatural.""".format(sentence=sentence)
        }
    ]

# data = [
#   "the customer had n't taught this boy .",
#   "are n't the boats breaking ?",
#   "the turtles may love these boys .",
#   "only these people who have remembered those books had ever held these toys .",
#   "young teachers sneeze carefully ."
# ]

_DTYPE_PARSED_RESULTS = pa.struct([
    pa.field("output_reasoning", pa.string()),
    pa.field("verdict", pa.bool_()),
    pa.field("error", pa.string()),
])

class LLMInfer:
    def __init__(self, tp_size: int, max_running_requests: int):
        self.llm = sgl.Engine(model_path=MODEL_PATH, reasoning_parser="qwen3", grammar_backend="xgrammar", tp_size=tp_size, max_running_requests=max_running_requests)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.parser = ReasoningParser("qwen3")

    def __call__(
        self,
        table: pa.Table,
    ):
        sentences = table.column("sentence_raw")
        input = self.tokenizer.apply_chat_template(
            list(map(get_message, sentences)), tokenize=False, add_generation_prompt=True, return_dict=False
        )

        # Qwen 3 Recommends these sampling parameters
        # Temperature=0.6, TopP=0.95, TopK=20, and MinP=0
        sampling_params = {
            "max_new_tokens": 32768,
            "skip_special_tokens": False,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0,

            "regex": r"\s*(Yes|No)",
        }
        results = self.llm.generate(prompt=input, sampling_params=sampling_params)
        results = cast(Iterator[dict[str, Any]], results)

        parsed_results = []
        for result in results:
            generated_text = result["text"]
            reasoning_text, text = self.parser.parse_non_stream(generated_text)
            if reasoning_text is None or text is None:
                parsed_results.append(pa.scalar({
                    "output_reasoning": None,
                    "verdict": None,
                    "error": orjson.dumps({"message": "Failed to parse LLM output", "output": generated_text}).decode()
                }, _DTYPE_PARSED_RESULTS))
                continue

            match text.strip():
                case "Yes":
                    verdict = True
                case "No":
                    verdict = False
                case _:
                    parsed_results.append(pa.scalar({
                        "output_reasoning": None,
                        "verdict": None,
                        "error": orjson.dumps({"message": "Invalid LLM output", "output": generated_text}).decode()
                    }, _DTYPE_PARSED_RESULTS))
                    continue

            parsed_results.append(pa.scalar({
                "output_reasoning": reasoning_text,
                "verdict": verdict,
                "error": None
            }, _DTYPE_PARSED_RESULTS))

        parsed_results_array = pa.array(parsed_results, type=_DTYPE_PARSED_RESULTS)
        return table.append_column("llm_infer", parsed_results_array)

def main():
    ray.init()

    ctx = ray.data.DataContext.get_current()
    ctx.enable_progress_bars = False

    dataset = ray.data.read_parquet("/home/tomyxw/projects/2025/verb-overgen/data/scamp/scamp-plausible.parquet")
    # logger.warning("Limit the data to 2048 / 2 rows for debugging.")

    tp_size = 2
    num_instances = int(ray.available_resources().get("GPU", 0) // tp_size)
    logger.info(f"Number of instances: {num_instances}")
    dataset = (
        dataset
        .filter(expr=col("is_grammatical") == True) # noqa: E712
        .map_batches(
            LLMInfer, # pyright: ignore
            fn_constructor_kwargs={"tp_size": tp_size, "max_running_requests": 1024},
            compute=ray.data.ActorPoolStrategy(size=num_instances),
            batch_size=1024,
            batch_format="pyarrow",
            num_cpus=1,
            num_gpus=tp_size,
            scheduling_strategy="SPREAD",
        )
    )

    dataset.write_parquet("/home/tomyxw/projects/2025/verb-overgen/data/scamp/scamp-plausible-llm-infer.parquet")

if __name__ == "__main__":
    main()
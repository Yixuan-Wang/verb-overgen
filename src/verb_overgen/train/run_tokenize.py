from loguru import logger
import polars as pl
import pyarrow as pa
import ray
import tokenizers as hftk
import transformers as hf

from verb_overgen.prep.lib.pychildes_adapt import PYCHILDES_LEMMA, get_sentence, filter_valid_lemmas
from verb_overgen.common.tokenize import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN

def load_data(input_path: str) -> pa.Table:
    df = (
        pl.read_parquet(input_path)
            .filter(filter_valid_lemmas(PYCHILDES_LEMMA))
            .with_columns(
                sentence=get_sentence(PYCHILDES_LEMMA),
            )
            .select(
                pl.all().exclude("content"),
                pl.col("content").alias("content_raw"),
            )
    )

    return df.to_arrow()

class Tokenize:
    def __init__(self, tokenizer_file: str):
        self.tokenizer = hf.PreTrainedTokenizerFast(
            tokenizer_object=hftk.Tokenizer.from_file(tokenizer_file),
        )
        self.token_id_bos = self.tokenizer._tokenizer.token_to_id(BOS_TOKEN)
        self.token_id_eos = self.tokenizer._tokenizer.token_to_id(EOS_TOKEN)
        self.token_id_pad = self.tokenizer._tokenizer.token_to_id(PAD_TOKEN)

    def __call__(self, table: pa.Table) -> pa.Table:
        sentences = table.column("sentence")
        list_tokenize_result = self.tokenizer._tokenizer.encode_batch_fast(sentences.to_pylist())
        input_ids: list[list[int]] = [
            [self.token_id_bos, *enc.ids, self.token_id_eos]
            for enc in list_tokenize_result
        ]
        arr_input_ids = pa.array(input_ids, pa.list_(pa.int32()))
        table = table.append_column("input_ids", arr_input_ids)
        return table

if __name__ == "__main__":
    TOKENIZER_FILE = "./data/tokenizer/1024-wordlevel/tokenizer.json"

    ray.init()

    dataset = ray.data.from_arrow(load_data("./data/pychildes-morphosyntax.20251024.parquet"))

    available_cpu = ray.available_resources().get("CPU", 0)

    dataset = dataset.map_batches(
        Tokenize,
        fn_constructor_kwargs={"tokenizer_file": TOKENIZER_FILE},
        compute=ray.data.ActorPoolStrategy(max_size=int(available_cpu)),
        batch_size=4096,
        batch_format="pyarrow",
        num_cpus=1,
    )

    dataset.write_parquet("./data/pychildes-morphosyntax-tokenized.20251024.parquet")



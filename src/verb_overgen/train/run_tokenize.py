import argparse
from loguru import logger
import polars as pl
import pyarrow as pa
import ray
import tokenizers as hftk
import transformers as hf

from verb_overgen.prep.lib.pychildes_adapt import PYCHILDES_LEMMA, get_sentence, filter_valid_lemmas
from verb_overgen.common.tokenize import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN


def load_pychildes_data(input_path: str) -> pa.Table:
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

def tokenize(
    tokenizer: hftk.Tokenizer,
    sentences: list[str],
):
    list_tokenize_result = tokenizer.encode_batch_fast(sentences)
    input_ids: list[list[int]] = [[tokenizer.token_to_id(BOS_TOKEN), *enc.ids, tokenizer.token_to_id(EOS_TOKEN)] for enc in list_tokenize_result]
    return input_ids

class Tokenize:
    def __init__(
        self,
        tokenizer_file: str,
        input_column: str,
        output_column: str,
    ):
        self.tokenizer = hf.PreTrainedTokenizerFast(
            tokenizer_object=hftk.Tokenizer.from_file(tokenizer_file),
        )
        self.input_column = input_column
        self.output_column = output_column

        self.token_id_bos = self.tokenizer._tokenizer.token_to_id(BOS_TOKEN)
        self.token_id_eos = self.tokenizer._tokenizer.token_to_id(EOS_TOKEN)
        self.token_id_pad = self.tokenizer._tokenizer.token_to_id(PAD_TOKEN)

    def __call__(self, table: pa.Table) -> pa.Table:
        sentences = table.column(self.input_column)
        list_tokenize_result = self.tokenizer._tokenizer.encode_batch_fast(sentences.to_pylist())
        input_ids: list[list[int]] = [[self.token_id_bos, *enc.ids, self.token_id_eos] for enc in list_tokenize_result]
        arr_input_ids = pa.array(input_ids, pa.list_(pa.int32()))
        table = table.append_column(self.output_column, arr_input_ids)
        return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize dataset using a pretrained tokenizer")
    parser.add_argument(
        "--tokenizer-file",
        type=str,
        default="./data/tokenizer/1024-wordlevel/tokenizer.json",
        help="Path to the tokenizer file",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="./data/pychildes-morphosyntax.20251024.parquet",
        help="Path to the input parquet file",
    )
    parser.add_argument(
        "--is-pychildes",
        action="store_true",
        help="Whether the input dataset is PyChildes",
    )
    parser.add_argument(
        "--input-column",
        type=str,
        default="sentence",
        help="Column to tokenize",
    )
    parser.add_argument(
        "--output-column",
        type=str,
        default="input_ids",
        help="Column to store tokenized output",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to the output parquet file",
    )
    parser.add_argument(
        "--cpu",
        type=int,
        help="Number of CPUs available",
    )
    args = parser.parse_args()

    ray.init()

    if args.is_pychildes:
        logger.info("Loading PyChildes dataset {input_path}", input_path=args.input_path)
        dataset = ray.data.from_arrow(load_pychildes_data(args.input_path))
    else:
        logger.info("Loading dataset {input_path}", input_path=args.input_path)
        dataset = ray.data.read_parquet(args.input_path)

    available_cpu = args.cpu

    dataset = dataset.repartition(available_cpu - 1).map_batches(
        Tokenize, # pyright: ignore
        fn_constructor_kwargs={
            "tokenizer_file": args.tokenizer_file,
            "input_column": args.input_column,
            "output_column": args.output_column,
        },
        compute=ray.data.ActorPoolStrategy(max_size=int(available_cpu - 1)),
        batch_format="pyarrow",
        num_cpus=1,
    )

    dataset.write_parquet(args.output_path)

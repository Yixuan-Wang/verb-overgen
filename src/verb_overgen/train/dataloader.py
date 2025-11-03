from pathlib import Path
from typing import TypedDict

import einops as eo
import polars as pl
import torch
from torch import Tensor, ne
from torch.utils.data import DataLoader, Dataset
from jaxtyping import Int

from transformers import PreTrainedTokenizerFast

from verb_overgen.common.tokenize import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from verb_overgen.prep.lib.pychildes_adapt import PYCHILDES_LEMMA, get_sentence, filter_valid_lemmas

class DataPyChildes(TypedDict): 
    input_ids: Int[Tensor, "batch seq"]
    attention_mask: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    seq_length: Int[Tensor, "batch"]
    # labels: Int[Tensor, "batch seq"]

class DataOnePyChildes(TypedDict): 
    input_ids: Int[Tensor, "seq"]
    attention_mask: Int[Tensor, "seq"]
    position_ids: Int[Tensor, "seq"]
    # labels: Int[Tensor, "1 seq"]


class DatasetPyChildes(Dataset):
    def __init__(
        self,
        *,
        input_path: Path,
        tokenizer: PreTrainedTokenizerFast,
    ):
        self.dataframe = self._load_data(input_path)
        self.tokenizer = tokenizer

        self.token_id_bos = tokenizer._tokenizer.token_to_id(BOS_TOKEN)
        self.token_id_eos = tokenizer._tokenizer.token_to_id(EOS_TOKEN)
        self.token_id_pad = tokenizer._tokenizer.token_to_id(PAD_TOKEN)
        # self.dataframe = self._add_tokenized(self.dataframe, tokenizer)

    def _load_data(self, input_path: Path):
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
        return df

    # FIXME: This is too slow to run every time
    # def _add_tokenized(self, df: pl.DataFrame, tokenizer: Tokenizer):
    #     def tokenize_sentence(sentence: str):
    #         encoded = tokenizer.encode(sentence)
    #         return {
    #             "input_ids": encoded.ids,
    #             "input_ids_str": encoded.tokens,
    #         }

    #     df = df.with_columns(
    #         pl.col("sentence").map_elements(tokenize_sentence, pl.Struct({
    #             "input_ids": pl.List(pl.Int64),
    #             "input_ids_str": pl.List(pl.String),
    #         })).struct.unnest()
    #     )
    #     return df

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx) -> DataOnePyChildes:
        sentence: str = self.dataframe.item(idx, "sentence")
        input_ids: list[int] = self.tokenizer._tokenizer.encode(sentence).ids
        input_ids: Tensor = torch.tensor([
            self.token_id_bos,
            *input_ids,
            self.token_id_eos,
        ])
        seq_len = input_ids.size(0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "position_ids": torch.arange(seq_len),
        }
    
    def collate_fn(self, batch: list[DataOnePyChildes]) -> DataPyChildes:
        seq_length = torch.tensor([item["input_ids"].size(0) for item in batch])
    
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=self.token_id_pad,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [item["attention_mask"] for item in batch],
            batch_first=True,
            padding_value=0,
        )
        position_ids = torch.nn.utils.rnn.pad_sequence(
            [item["position_ids"] for item in batch],
            batch_first=True,
            padding_value=-1,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "seq_length": seq_length,
        }

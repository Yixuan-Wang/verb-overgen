from dataclasses import dataclass
from typing import cast
import ipdb
import polars as pl
import pyarrow as pa
import transformers as hf
import tokenizers as hftk
import torch
from jaxtyping import Int, Float
import einops as eo

from verb_overgen.common.tokenize import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from verb_overgen.train.train import VerbOvergenTrainModule

INPUT_PATH = "/u501/y4466wan/projects/2025/verb-overgen/data/zorro/zorro_babyberta_lemmatized.parquet"
TOKENIZER_PATH = "/u501/y4466wan/projects/2025/verb-overgen/data/tokenizer/1024-wordlevel/tokenizer.json"

def load_zorro(
    input_path: str = INPUT_PATH,
):
    torch.set_default_device("cuda")

    dataset_zorro = pl.scan_parquet(input_path)
    dataset_zorro = dataset_zorro.with_columns(
        pl.col("file_line_idx").sub(1).floordiv(2).alias("pair_idx"),
        pl.col("file_line_idx").mod(2).cast(pl.Boolean).not_().alias("is_grammatical"),
        pl.col("lemma").list.join(" ").str.strip_chars().alias("sentence"),
    )

    dataset_zorro_pair = dataset_zorro.group_by("file_name", "pair_idx").agg(
        pl.col("sentence", "sentence_raw", "lemma", "pos").filter(pl.col("is_grammatical")).first().name.suffix("_grammatical"),
        pl.col("sentence", "sentence_raw", "lemma", "pos").filter(~pl.col("is_grammatical")).first().name.suffix("_ungrammatical"),
    )

    # Ignore pure morphological alternations for now
    dataset_zorro_pair = dataset_zorro_pair.filter(
        pl.col("lemma_grammatical") != pl.col("lemma_ungrammatical")
    )

    return dataset_zorro_pair

@dataclass
class EvalZorro:
    model: torch.nn.Module
    tokenizer: hf.PreTrainedTokenizerFast

    bos_token_id: int
    eos_token_id: int
    pad_token_id: int

    def __post_init__(self):
        self.model.eval()

    @torch.no_grad()
    def __call__(self, inputs: pl.Series):
        torch.set_default_device("cuda")

        inputs: pa.StructArray = inputs.to_arrow()

        list_input_ids = [
            [
                [bos_token_id, *self.tokenizer._tokenizer.encode(sentence.as_py()).ids, eos_token_id] # pyright: ignore
                for sentence in inputs.field(field)
            ] for field in ("sentence_grammatical", "sentence_ungrammatical")
        ]

        input_ids_len: Int[torch.Tensor, "group batch"] = torch.tensor([[len(seq) for seq in group] for group in list_input_ids]) 
        max_len = cast(int, input_ids_len.max().item())
        padded_groups = [
            torch.stack([
                torch.nn.functional.pad(
                    torch.tensor(seq, dtype=torch.long),
                    pad=(0, max_len - len(seq)),
                    value=self.pad_token_id,
                )
                for seq in group
            ])
            for group in list_input_ids
        ]
        input_ids: Int[torch.Tensor, "group batch seq"] = torch.stack(padded_groups)

        group_size, batch_size, _ = input_ids.shape
        model_input_ids = eo.rearrange(input_ids, "group batch seq -> (group batch) seq")
        attention_mask = model_input_ids.ne(self.pad_token_id).long()

        outputs = self.model(
            input_ids=model_input_ids,
            attention_mask=attention_mask,
        )
        prob: Float[torch.Tensor, "group batch seq vocab"] = eo.rearrange(
            outputs.logits.softmax(dim=-1),
            "(group batch) seq vocab -> group batch seq vocab",
            group=group_size,
            batch=batch_size,
        )

        _this = torch.gather(
            prob[..., :-1, :],
            dim=-1,
            index=eo.rearrange(input_ids[..., 1:], "group batch seq -> group batch seq 1"),
        )
        token_prob: Float[torch.Tensor, "group batch seq"] = _this[..., 0]

        input_ids_len_min_1 = (input_ids_len - 1).clamp(min=0) # Shift left by 1 to exclude BOS
        token_mask = torch.arange(max_len - 1) < input_ids_len_min_1[..., None]
        token_prob_unpad = torch.where(token_mask, token_prob, torch.ones_like(token_prob))

        seq_prob: Float[torch.Tensor, "group batch"] = token_prob_unpad.prod(dim=-1)

        return pl.DataFrame({
            "prob_grammatical": seq_prob[0].cpu().numpy(),
            "prob_ungrammatical": seq_prob[1].cpu().numpy(),
        }).select(pl.struct(pl.all())).to_series()

if __name__ == "__main__":
    from verb_overgen.train.train import VerbOvergenTrainModule

    tokenizer = hf.PreTrainedTokenizerFast(
        tokenizer_object=hftk.Tokenizer.from_file(TOKENIZER_PATH),
    )
    bos_token_id = tokenizer._tokenizer.token_to_id(BOS_TOKEN)
    eos_token_id = tokenizer._tokenizer.token_to_id(EOS_TOKEN)
    pad_token_id = tokenizer._tokenizer.token_to_id(PAD_TOKEN)

    dataset = load_zorro().collect()

    BATCH_SIZE = 128

    
    module = VerbOvergenTrainModule.load_from_checkpoint("/u501/y4466wan/projects/2025/verb-overgen/data/train/gpt2/verb-overgen-gpt2-a6cbd4_silver-salad-4_final.pth", map_location="cuda")
    model = module.model

    eval_zorro = EvalZorro(
        model=model,
        tokenizer=tokenizer,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )

    dataset_result = dataset.with_row_index("row_index").group_by(pl.col("row_index").floordiv(BATCH_SIZE)).agg(
        pl.all(),
        pl.struct("sentence_grammatical", "sentence_ungrammatical").map_batches(
            eval_zorro,
        ).alias("prob"),
    ).select(pl.all().exclude("row_index").explode())

    dataset_result.write_parquet("/u501/y4466wan/projects/2025/verb-overgen/data/zorro/eval/zorro-eval_silver-salad-4_final.parquet")    

    # print(zorro_data.collect().head(5).with_columns(
    #     pl.struct(["lemma_grammatical", "lemma_ungrammatical"]).map_batches()
    # )

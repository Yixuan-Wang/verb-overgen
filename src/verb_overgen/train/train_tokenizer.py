"""Script for training a tokenizer on the PyChildes dataset.
"""

from pathlib import Path
from typing import Final, no_type_check

import polars as pl

from verb_overgen.common.tokenize import BOS_TOKEN, EOS_TOKEN, UNK_TOKEN
from verb_overgen.prep.lib.pychildes_adapt import PYCHILDES_LEMMA, filter_valid_lemmas, get_sentence

SPECIAL_TOKEN: Final = {
    "pad_token": "<pad>",
    "bos_token": BOS_TOKEN,
    "eos_token": EOS_TOKEN,
    "unk_token": UNK_TOKEN,
}

VOCAB_SIZE: Final = 4096

@no_type_check
def init_empty_tokenizer_bpe(
    vocab_size: int,
):
    import tokenizers
    from tokenizers import models, normalizers, pre_tokenizers, trainers

    tokenizer = tokenizers.Tokenizer(
        models.BPE(unk_token=SPECIAL_TOKEN["unk_token"])
    )

    tokenizer.add_special_tokens(list(SPECIAL_TOKEN.values()))

    tokenizer.normalizer = normalizers.Sequence([
        normalizers.Lowercase(),
    ])

    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(SPECIAL_TOKEN.values()),
    )

    return tokenizer, trainer

@no_type_check
def init_empty_tokenizer_wordlevel(
    vocab_size: int,
):
    import tokenizers
    from tokenizers import models, normalizers, pre_tokenizers, trainers

    tokenizer = tokenizers.Tokenizer(
        models.WordLevel(unk_token=SPECIAL_TOKEN["unk_token"])
    )

    tokenizer.add_special_tokens(list(SPECIAL_TOKEN.values()))

    tokenizer.normalizer = normalizers.Sequence([
        normalizers.Lowercase(),
    ])

    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.WordLevelTrainer(
        vocab_size=vocab_size,
        special_tokens=list(SPECIAL_TOKEN.values()),
    )

    return tokenizer, trainer


def train_tokenizer_on_pychildes(
    dataset: pl.DataFrame | pl.LazyFrame,
    vocab_size: int = VOCAB_SIZE,
    kind: str = "wordlevel",
):
    lemma_expr = PYCHILDES_LEMMA

    dataset = dataset.lazy().filter(filter_valid_lemmas(lemma_expr))

    dataset_sent = (
        dataset
        .with_columns(
            content_raw="content",
            content=get_sentence(lemma_expr)
        )
    )

    match kind:
        case "bpe":
            tokenizer, trainer = init_empty_tokenizer_bpe(vocab_size=vocab_size)
        case "wordlevel":
            tokenizer, trainer = init_empty_tokenizer_wordlevel(vocab_size=int(1e5)) # A large vocab size for wordlevel, ignoring a vocab limit
        case _:
            raise ValueError(f"Unknown tokenizer kind: {kind}")

    sent = dataset_sent.select("content").collect().to_series()

    tokenizer.train_from_iterator(
        iter(sent),
        length=len(sent),
        trainer=trainer,
    )

    return tokenizer

def main(
    input_path: str,
    output_path: str,
    kind: str,
):
    output_path: Path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "tokenizer.json"    

    input_path: Path = Path(input_path)
    dataset = pl.read_parquet(input_path)
    tokenizer = train_tokenizer_on_pychildes(dataset, vocab_size=VOCAB_SIZE, kind=kind)
    tokenizer.save(output_file.as_posix())

if __name__ == "__main__":
    main(
        input_path="/u501/y4466wan/projects/2025/verb-overgen/data/childes-gold/test-251010T0422/merged.test.parquet/pychildes-morphosyntax.20251024.parquet",
        output_path="/u501/y4466wan/projects/2025/verb-overgen/data/tokenizer/1024-wordlevel",
        kind="wordlevel"
    )
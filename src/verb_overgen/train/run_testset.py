"""
Run testset on checkpoints.
"""

from pathlib import Path
import pdb

import lightning as ptl
import polars as pl
import tokenizers as hftk
import torch
import transformers as hf

from verb_overgen.common.tokenize import PAD_TOKEN
from verb_overgen.train.lib.dataloader_testset import DatasetTestset, TestTestsetOnceCallback
from verb_overgen.train.train import VerbOvergenTrainModule
from verb_overgen.util.misc_util import ensure_dir

if __name__ == "__main__":
    CHECKPOINT = "./var/train/20251230/celestial-mountain-7/checkpoints/celestial-mountain-7-epoch=02.ckpt"
    PATH_TEST = "./data/testset/output/testset_llm_gen.final_postprocessed_20260114T1637.parquet"
    PATH_TOKENIZER = "./data/tokenizer/1024-wordlevel/tokenizer.json"

    # Load tokenizer
    tokenizer = hf.PreTrainedTokenizerFast(
        tokenizer_object=hftk.Tokenizer.from_file(PATH_TOKENIZER),
    )
    token_id_pad = tokenizer._tokenizer.token_to_id(PAD_TOKEN)

    # Load checkpoint
    train_module = VerbOvergenTrainModule.load_from_checkpoint(CHECKPOINT)

    # Create test dataset and dataloader
    dataset_test = DatasetTestset(
        input_path=PATH_TEST,
        token_id_pad=token_id_pad,
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=512, shuffle=False, collate_fn=dataset_test.collate_fn
    )

    # Setup output directory
    checkpoint_path = Path(CHECKPOINT)
    checkpoint_name = checkpoint_path.stem
    output_dir = ensure_dir(checkpoint_path.parent.parent)

    # Create trainer and run test
    test_callback = TestTestsetOnceCallback()
    trainer = ptl.Trainer(
        callbacks=[test_callback],
        accelerator="gpu",
        devices=1,
        default_root_dir=output_dir,
    )
    
    # 2. Access your lightweight metadata
    ckpt = torch.load(CHECKPOINT, map_location="meta", weights_only=False)
    step = ckpt["global_step"]
    trainer.test(
        train_module,
        dataloaders=dataloader_test,
    )
    
    dataset_test = test_callback.test_dataset.with_columns(pl.lit(step).alias("step"))
    output_path = ensure_dir(output_dir / "test") / f"testset_step_{step:08d}.parquet"
    
    if output_path.exists():
        pdb.set_trace(header=f"The {output_path=} exists.")
    else:
        dataset_test.write_parquet(output_path)


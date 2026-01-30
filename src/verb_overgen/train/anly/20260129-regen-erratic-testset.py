import re
from collections.abc import Generator
from pathlib import Path

import lightning as ptl
import polars as pl
import tokenizers as hftk
import torch
import transformers as hf
from apfel import not_none
from loguru import logger

from verb_overgen.common.tokenize import PAD_TOKEN
from verb_overgen.train.lib.dataloader_testset import DatasetTestset, TestTestsetOnceCallback
from verb_overgen.train.train import VerbOvergenTrainModule
from verb_overgen.util.misc_util import ensure_dir

base_path = Path("/project/6105902/tomyxw/2025/verb-overgen/verb-overgen/var/train/20251230/")


def checkpoint_need_to_recreate() -> Generator[tuple[str, Path]]:
    global base_path
    folder_recreate = {
        "swept-frost-12",
        "hopeful-eon-16",
        "rosy-sun-17",
        "iconic-grass-19",
        "eager-bush-20",
        "whole-galaxy-21",
    }

    for folder in folder_recreate:
        path_checkpoint_dir = base_path / folder / "checkpoints"

        for checkpoint in path_checkpoint_dir.glob("*.ckpt"):
            if "last" in checkpoint.name:
                continue

            yield (folder, checkpoint)


class Runner:
    def __init__(self):
        PATH_TEST = "./data/testset/output/testset_llm_gen.final_postprocessed_20260114T1637.parquet"
        PATH_TOKENIZER = "./data/tokenizer/1024-wordlevel/tokenizer.json"

        # Load tokenizer
        tokenizer = hf.PreTrainedTokenizerFast(
            tokenizer_object=hftk.Tokenizer.from_file(PATH_TOKENIZER),
        )
        token_id_pad = tokenizer._tokenizer.token_to_id(PAD_TOKEN)

        self.dataset_test = DatasetTestset(
            input_path=PATH_TEST,
            token_id_pad=token_id_pad,
        )

        self.file_written: list[str] = []

    def __call__(
        self,
        run_name: str,
        checkpoint_path: Path,
    ):
        # Load checkpoint
        train_module = VerbOvergenTrainModule.load_from_checkpoint(
            checkpoint_path=str(checkpoint_path),
        )
        # Setup output directory
        output_dir = ensure_dir(checkpoint_path.parent.parent)
        # Load checkpoint metadata to get step
        #
        epoch = not_none(re.search(r"epoch=(\d+)", checkpoint_path.name)).group(1)
        epoch = int(epoch)
        #
        ckpt = torch.load(checkpoint_path, map_location="meta", weights_only=False)
        step = ckpt["global_step"]
        output_path = ensure_dir(output_dir / "test") / f"testset_step_{epoch:02d}_{step:08d}.parquet"

        logger.info(f"will save {run_name=} {step=} to {output_path=}")

        # Create test dataloader
        dataloader_test = torch.utils.data.DataLoader(
            self.dataset_test, batch_size=512, shuffle=False, collate_fn=self.dataset_test.collate_fn
        )

        # Create trainer and run test
        test_callback = TestTestsetOnceCallback()
        trainer = ptl.Trainer(
            callbacks=[test_callback],
            accelerator="gpu",
            devices=1,
            default_root_dir=output_dir,
        )

        # Run test
        trainer.test(
            train_module,
            dataloaders=dataloader_test,
        )

        # Save results
        dataset_test = test_callback.test_dataset.with_columns(
            pl.lit(step).alias("step"),
            pl.lit(epoch).alias("epoch"),
        )

        if output_path.exists():
            logger.warning(f"WARNING: {output_path} already exists, skipping...")

        dataset_test.write_parquet(output_path)
        logger.success(f"Saved results to {output_path}")
        self.file_written.append(str(output_path))


if __name__ == "__main__":
    runner = Runner()

    for run_name, ckpt_path in checkpoint_need_to_recreate():
        logger.info(f"Running {run_name=} {ckpt_path.stem=}")
        runner(run_name, ckpt_path)

    with open("./var/recreated_test_files.txt", "w") as f:
        for file in runner.file_written:
            f.write(file + "\n")

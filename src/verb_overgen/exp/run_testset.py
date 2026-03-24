from pathlib import Path

import tokenizers as hftk
import torch
from loguru import logger
import lightning as ptl
from omegaconf import OmegaConf
import polars as pl
import tempfile
from tqdm import tqdm

import wandb

from verb_overgen.common.tokenize import PAD_TOKEN, BOS_TOKEN
from verb_overgen.train.train import VerbOvergenTrainModule
from verb_overgen.train.lib.dataloader_overgen import DatasetOvergen, TestOvergenOnceCallback, clean_dataset_overgen
from verb_overgen.train.lib.validate import last_sentence_index, nll

def load_testset_input(path_testset_input: str | Path) -> pl.DataFrame:
    dataset = pl.read_parquet(path_testset_input)
    dataset = clean_dataset_overgen(dataset)
    return dataset

def run_testset_on_checkpoint(
    *,
    path_checkpoint: str | Path,
    dataset_test: DatasetOvergen,
):
    # Load checkpoint
    train_module = VerbOvergenTrainModule.load_from_checkpoint(path_checkpoint)


    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=512, shuffle=False, collate_fn=dataset_test.collate_fn
    )

    # Create trainer and run test
    test_callback = TestOvergenOnceCallback()
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = ptl.Trainer(
            callbacks=[test_callback],
            accelerator="gpu",
            devices=1,
            default_root_dir=tmpdir,
        )

        trainer.test(
            train_module,
            dataloaders=dataloader_test,
        )

    dataset_test_result = test_callback.test_dataset
    return dataset_test_result

@__import__("ipdb").iex
def main():
    sweep_filters = {"tags": {"$in": ["260323"]}}
    path_result = "./var/results/testset_gemini_voi-5_tag-260323.parquet"
    
    from verb_overgen.train.lib.utils import load_train_config
    config = load_train_config()
    
    PATH_TOKENIZER: str = config.tokenizer
    PATH_TESTSET: str = config.data.eval_overgen.path
    PATH_TRAIN_DIR = Path(config.output.root_dir)

    tokenizer = hftk.Tokenizer.from_file(PATH_TOKENIZER)
    TOKEN_ID_PAD: int = tokenizer.token_to_id(PAD_TOKEN)
    TOKEN_ID_BOS: int = tokenizer.token_to_id(BOS_TOKEN)

    dataset_test = DatasetOvergen.from_path(
        input_path=PATH_TESTSET,
        token_id_pad=TOKEN_ID_PAD,
    )

    wandb_api = wandb.Api()
    list_wandb_run = wandb_api.runs("tom-yx_wang/25-verb-overgen", filters=sweep_filters)

    list_df_result: list[pl.DataFrame] = []
    for wandb_run in (pbar := tqdm(list_wandb_run)):
        cfg = OmegaConf.create(wandb_run.config)
        
        run_name: str = wandb_run.name
        run_flavor = cfg.data_config.train.subset
        pbar.set_postfix_str(run_name)
        
        if "seed" not in cfg:
            run_seed = next(int(arg.lstrip("seed=")) for arg in wandb_run.metadata["args"] if arg.startswith("seed="))
        else:
            run_seed = cfg.seed
        
        run_meta = {
            "run_name": run_name,
            "run_flavor": run_flavor,
            "run_seed": run_seed,
        }
        
        path_dir_checkpoint: Path = PATH_TRAIN_DIR / run_name / "checkpoints"
        # List all checkpoint files in the test directory
        path_last_checkpoint: Path = path_dir_checkpoint / "last.ckpt"
        if not path_last_checkpoint.exists():
            logger.error("Checkpoint not found for run `{}`: {}".format(run_name, run_meta))
            continue
        
        logger.info(f"Running checkpoint for run `{run_name}`")
        df_result = run_testset_on_checkpoint(
            path_checkpoint=path_last_checkpoint,
            dataset_test=dataset_test,
        )
        df_result = df_result.with_columns(
            **{k: pl.lit(v) for k, v in run_meta.items()}
        )
        list_df_result.append(df_result)
    
    df_result_all = pl.concat(list_df_result)
    df_result_all = df_result_all.join(dataset_test.dataframe.select(pl.all().exclude("verb", "kind")), on="index_sentence", coalesce=True)
    
    df_result_all = df_result_all.with_columns(
        pl.col("token_prob").list.slice(last_sentence_index(pl.col("input_ids"), TOKEN_ID_BOS)).pipe(nll).alias("nll")
    )
    # TODO: When the pipeline fixed we will try to dump less data.
    df_result_all.write_parquet(path_result)

if __name__ == "__main__":
    main()
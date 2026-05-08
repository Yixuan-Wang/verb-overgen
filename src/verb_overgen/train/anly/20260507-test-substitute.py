from pathlib import Path
from typing import cast

from torch.utils.data import DataLoader
import polars as pl

from verb_overgen.train.lib.dataloader_pychildes import DatasetPyChildes, substitute_on_verb
from verb_overgen.train.lib.utils import load_train_config
from verb_overgen.train.train import TrainingSession

_session = TrainingSession(load_train_config())
tokenizer = _session.setup_tokenizer()
dataset = cast(
    DatasetPyChildes,
    _session.setup_datasets()["train"]
)

dataset_substitution = pl.read_parquet("./var/pychildes/substitute/pychildes-causative_substitute_transitive.20260517.parquet")

substitute_on_verb(
    dataset, 
    dataset_substitution,
    verb=["laugh"],
    columns=["sentence", "input_ids"]
)

__import__("IPython").embed(using=False)

expr_filter_verb = (pl.element().struct["feature"] == "causative") & (pl.element().struct["verb"] == "laugh")
df_laugh = dataset.dataframe.filter(
    pl.col("featured_verbs").list.eval(expr_filter_verb).list.any()
)

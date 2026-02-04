from collections.abc import Sequence
import contextlib
import gc
import sys
from importlib.resources import files as resources
from typing import Literal
import torch
from lightning.pytorch.callbacks import Callback
from hydra import compose, initialize_config_dir

class CacheCleanCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        gc.collect() 
        torch.cuda.empty_cache() 

def load_train_config(
    *,
    config_pkg: str = "verb_overgen.train.configs",
    overrides: Sequence[str] | Literal["argv"] | None  = "argv",
):
    if overrides == "argv":
        overrides = sys.argv[1:]
    elif overrides is None:
        overrides = None
    else:
        overrides = list(overrides)

    config_dir = resources(config_pkg)
    with initialize_config_dir(version_base=None, config_dir=str(config_dir), job_name="train"):
        cfg = compose(config_name="train", overrides=overrides)
        return cfg


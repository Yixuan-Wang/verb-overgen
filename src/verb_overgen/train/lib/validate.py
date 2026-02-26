"""
Validation of model.
"""

from pathlib import Path
from typing import Any

import numpy as np
import einops as eo
import polars as pl
import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn.utils.rnn import unpad_sequence

import lightning as ptl
from verb_overgen.util.misc_util import ensure_dir

VERB_TARGET_5 = ["laugh", "cry", "sing", "smile", "sleep"]
VERB_TARGET_17 = [
    "appear",
    "chortle",
    "chuckle",
    "cry",
    "descend",
    "frown",
    "giggle",
    "glow",
    "grin",
    "groan",
    "laugh",
    "rise",
    "sing",
    "sleep",
    "smile",
    "sneeze",
    "vanish",
]

class EvalCallback(ptl.Callback):
    def __init__(self):
        self.eval_outputs: list[pl.DataFrame] = []
        self.eval_dir: Path
        
    def on_fit_start(self, trainer: ptl.Trainer, pl_module: ptl.LightningModule) -> None:
        self.eval_dir = ensure_dir(Path(trainer.default_root_dir) / "eval")

    def on_validation_start(self, trainer: ptl.Trainer, pl_module: ptl.LightningModule) -> None:
        self.eval_outputs.clear()

    def on_validation_batch_end(
        self,
        trainer: "ptl.Trainer",
        pl_module: "ptl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.eval_outputs.append(outputs)

    def on_validation_end(self, trainer: "ptl.Trainer", pl_module: "ptl.LightningModule") -> None:
        step = trainer.global_step
        dataset_eval = pl.concat(self.eval_outputs).with_columns(
            pl.lit(step).alias("step")
        )
        # Save to parquet with step information in filename
        output_path = self.eval_dir / f"eval_step_{step:08d}.parquet"
        dataset_eval.write_parquet(output_path)

@torch.no_grad()
def eval_input_token_prob(
    model: torch.nn.Module,
    # batch: DataBatchScamp,
    *,
    input_ids: Int[Tensor, "batch seq"],
    seq_length: Int[Tensor, "batch seq"],
) -> list[Float[Tensor, "ragged_seq"]]:
    outputs = model(
        input_ids=input_ids,
    )
    prob: Float[Tensor, "batch seq vocab"] = outputs.logits.softmax(dim=-1)

    _this = torch.gather(
        prob[..., :-1, :],
        dim=-1,
        index=eo.rearrange(input_ids[..., 1:], "batch seq -> batch seq 1"),
    )
    token_prob: Float[Tensor, "batch seq"] = _this[..., 0]

    # Truncate each tensor to its original length and return as a list
    token_prob = token_prob.cpu()
    seq_length_without_bos: Int[Tensor, "batch"] = (seq_length - 1).clamp(min=0).cpu()
    list_token_prob = unpad_sequence(token_prob, lengths=seq_length_without_bos, batch_first=True)

    return list_token_prob
    
def perplexity(token_probs: pl.Expr) -> pl.Expr:
    """Compute sentence-level perplexity from token probabilities.
    
    Args:
        token_probs (pl.Expr): A Polars `list(float)` expression representing a list of token probabilities.
    
    Returns:
        A Polars `float` expression representing the perplexity.
    """
    nlls = token_probs.list.eval(pl.element().log().neg())
    mean_nll = nlls.list.mean()
    ppl = mean_nll.exp()
    return ppl
    
def nll(token_probs: pl.Expr) -> pl.Expr:
    """Compute sentence-level negative log-likelihood (NLL) from token probabilities.
    
    Args:
        token_probs (pl.Expr): A Polars `list(float)` expression representing a list of token probabilities.
    
    Returns:
        A Polars `float` expression representing the NLL.
    """
    nlls = token_probs.list.eval(pl.element().log().neg())
    mean_nll = nlls.list.mean()
    return mean_nll

def ci(arr: np.ndarray, n_resamples: int = 10_000, batch_size: int = 200) -> tuple[float, float]:
    """Return the 95% bootstrap CI (lower, upper) for the mean, using batched numpy."""
    arr = np.ravel(arr) # Handles multi-dimensional inputs safely
    n = arr.size
    
    if n == 0:
        raise ValueError("Cannot compute bootstrap CI on an empty array.")
        
    rng = np.random.default_rng()
    means = np.empty(n_resamples, dtype=np.float64)
    
    for i in range(0, n_resamples, batch_size):
        current_batch_size = min(batch_size, n_resamples - i)
        
        # Allocate only a small batch at a time
        batch_resamples = rng.choice(arr, size=(current_batch_size, n), replace=True)
        means[i:i + current_batch_size] = batch_resamples.mean(axis=1)
        
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)
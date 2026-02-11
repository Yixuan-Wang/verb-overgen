# %%
import marimo as mo
import polars as pl
import sh
import torch
import transformers as hf
import gc

# %%
RUN_MAPPING = {
    "copper-firebrand-62": {
        "kind": "baseline",
    },
    "comfy-serenity-64": {
        "kind": "tune-pretrain",
    },
    "dazzling-deluge-63": {"kind": "freeze-pretrain"},
}

# %%
# Load pretrained GPT-2 model
pretrained_model = hf.GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

# Collect statistics for all models
stats_records = []

# Add pretrained model statistics
for param_name, param in pretrained_model.named_parameters():
    clean_name = param_name.replace("transformer.", "")
    stats_records.append(
        {
            "kind": "pretrain",
            "param_name": clean_name,
            "shape": list(param.shape),
            "abs_mean": param.abs().mean().item(),
            "abs_std": param.abs().std().item(),
        }
    )

# Load and analyze checkpoints
for run, info in RUN_MAPPING.items():
    path_ckpt = f"./var/train/20260128/{run}/checkpoints/last.ckpt"

    # Load checkpoint
    checkpoint = torch.load(path_ckpt, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]

    for param_name, param in state_dict.items():
        clean_name = param_name.replace("model.", "").replace("transformer.", "")

        stats_records.append(
            {
                "kind": info["kind"],
                "param_name": clean_name,
                "shape": list(param.shape),
                "abs_mean": param.abs().mean().item(),
                "abs_std": param.abs().std().item(),
            }
        )
    
    del checkpoint
    gc.collect()

# %%
# Create Polars dataframe
df_stats = pl.DataFrame(stats_records, schema_overrides={
    "shape": pl.List(pl.Int64),
})
df_stats

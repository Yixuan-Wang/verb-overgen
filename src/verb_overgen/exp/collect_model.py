import json
from pathlib import Path

import torch
import transformers as hf
import tokenizers as hftk
import wandb

from loguru import logger

from verb_overgen.common.tokenize import BOS_TOKEN, UNK_TOKEN, PAD_TOKEN
from verb_overgen.exp.collect_metadata import get_metadata

ROOT_TRAIN = "var/train"

def get_model_path(metadata) -> Path:
    """Locate checkpoint path from run_name.

    Given run_name "comfy-bush-154", the returned checkpoint should be "var/train/20260212/comfy-bush-154/checkpoints/last.ckpt".
    """
    run_name = metadata["internal"]["run_name"]
    matches = list(Path(ROOT_TRAIN).glob(f"*/{run_name}/checkpoints/last.ckpt"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected 1 checkpoint for {run_name!r}, found {len(matches)}: {matches}")
    return matches[0]


def load_gpt2_from_checkpoint(ckpt_path: Path) -> hf.GPT2LMHeadModel:
    """Extract a GPT2LMHeadModel from a PyTorch Lightning checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = {
        k.removeprefix("model."): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    raw_config = ckpt["hyper_parameters"]["model_config"]
    model_config = raw_config if isinstance(raw_config, hf.GPT2Config) else hf.GPT2Config(**raw_config)
    model = hf.GPT2LMHeadModel(model_config)
    model.load_state_dict(state_dict)
    return model


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("tags", nargs="+", help="WandB tags to filter runs by (matched with $in)")
    parser.add_argument("--output-dir", required=True, help="Directory to save extracted HuggingFace models")
    parser.add_argument("--tokenizer", default="./data/tokenizer/1024-wordlevel/tokenizer.json", help="Path to tokenizer file to copy into output_dir")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_raw = hftk.Tokenizer.from_file(args.tokenizer)
    tokenizer = hf.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_raw,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        bos_token=BOS_TOKEN,
        eos_token=BOS_TOKEN,
    )
    tokenizer.save_pretrained(output_dir)

    api = wandb.Api()
    runs = api.runs("tom-yx_wang/25-verb-overgen", filters={"tags": {"$in": args.tags}})

    jsonl_path = output_dir / "metadata.jsonl"
    with jsonl_path.open("w") as jsonl_file:
        for run in runs:
            metadata = get_metadata(run)
            if metadata is None:
                continue
            run_name = metadata["internal"]["run_name"]
            try:
                ckpt_path = get_model_path(metadata)
            except FileNotFoundError as e:
                logger.warning(f"[skip] {run_name}: {e}")
                continue

            if metadata["target_verb"] == "glow":
                logger.warning(f"[skip] {run_name}: `glow` results are discarded")
                continue

            manipulation = metadata["manipulation"]
            parts = [manipulation]
            if metadata["scope"]:
                parts.append(metadata["scope"])
            if metadata["target"]:
                parts.append(metadata["target"])
            if metadata["target_verb"]:
                parts.append(f"target-{metadata['target_verb']}")
            parts.append(f"seed-{metadata['seed']}")
            parts.append("last")
            model_name = "_".join(parts)

            save_path = output_dir / manipulation / model_name
            rel_path = save_path.relative_to(output_dir)
            metadata_with_path = {**metadata, "path": str(rel_path)}

            model = load_gpt2_from_checkpoint(ckpt_path)
            save_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_path)
            (save_path / "metadata.json").write_text(json.dumps(metadata_with_path, indent=4))
            jsonl_file.write(json.dumps(metadata_with_path) + "\n")
            logger.info(f"[saved] {run_name} -> {save_path}")


if __name__ == "__main__":
    main()

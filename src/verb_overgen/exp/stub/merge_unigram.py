"""Merge per-config stats-unigram output directories into one."""

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dirs", nargs="+", type=Path, help="Per-config source directories to merge")
    parser.add_argument("--output-dir", required=True, type=Path, help="Destination directory")
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_jsonl = output_dir / "metadata.jsonl"
    with merged_jsonl.open("w") as out_f:
        for src_dir in args.source_dirs:
            for parquet in sorted(src_dir.rglob("*.parquet")):
                rel = parquet.relative_to(src_dir)
                dest = output_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(parquet, dest)

            jsonl = src_dir / "metadata.jsonl"
            if jsonl.exists():
                out_f.writelines(jsonl.read_text())


if __name__ == "__main__":
    main()

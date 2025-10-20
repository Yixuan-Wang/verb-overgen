"""Merge PyChildes JSONL outputs into a single Parquet dataset using Ray."""

from pathlib import Path
import json
import re
import ray
import pyarrow as pa


SPLIT_SPEAKER = re.compile(r"<(.+?)>\s*(.*)$")
SCHEMA = pa.schema(
    [
        ("id_file", pa.string()),
        ("id_utterance", pa.uint64()),
        ("has_parse", pa.bool_()),
        ("speaker", pa.string()),
        ("content", pa.string()),
        (
            "morph",
            pa.struct(
                [
                    ("index", pa.list_(pa.int64())),
                    ("lemma", pa.list_(pa.string())),
                    ("pos", pa.list_(pa.string())),
                    ("kind", pa.list_(pa.string())),
                    ("metadata", pa.list_(pa.string())),
                ]
            ),
        ),
        (
            "syntax",
            pa.struct(
                [
                    ("from_index", pa.list_(pa.int64())),
                    ("to_index", pa.list_(pa.int64())),
                    ("rel", pa.list_(pa.string())),
                ]
            ),
        ),
    ]
)

@ray.remote(num_cpus=1)
def parse_file(file_path: Path) -> pa.Table:
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]

    file_id = file_path.stem.replace("_", "/")

    utterances = []
    for idx, line in enumerate(data):
        if line["kind"] != "utterance":
            continue

        content_group = SPLIT_SPEAKER.match(line["content"][0])

        assert content_group is not None, (
            f"Failed to parse speaker line: {line['content'][0]}"
        )
        speaker, first_utterance = content_group.groups()

        if not first_utterance:
            continue

        if "graph" not in line:
            has_parse = False
            morph = None
            syntax = None
        else:
            has_parse = True
            morph = line["graph"]["morph"]
            syntax = line["graph"]["syntax"]

        utterances.append(
            {
                "id_file": file_id,
                "id_utterance": idx,
                "has_parse": has_parse,
                "speaker": speaker,
                "content": first_utterance,
                "morph": morph,
                "syntax": syntax,
            }
        )

    # Return a PyArrow Table for efficient, distributed merging.
    return pa.Table.from_pylist(utterances, schema=SCHEMA)


def main(interm_dir: str, output_dir: str):
    interm_dir: Path = Path(interm_dir)
    output_dir: Path = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(interm_dir.glob("*.jsonl"))
    print(f"Found {len(files)} files to process.")

    dataset = ray.data.from_arrow_refs([parse_file.remote(f) for f in files])  # pyright: ignore[reportArgumentType]
    dataset.repartition(1).write_parquet((output_dir).as_posix())


if __name__ == "__main__":
    main(
        interm_dir="/u501/y4466wan/projects/2025/verb-overgen/data/childes-gold/test-251010T0359",
        output_dir="/u501/y4466wan/projects/2025/verb-overgen/data/childes-gold/test-251010T0422/merged.test.parquet",
    )

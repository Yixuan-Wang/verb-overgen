"""Execute PyChildes on TalkBank .cha files in parallel using Ray and generate JSONL outputs.

Warning:
    This script assumes PyChildes is not installed as a package but is available at a specified path.
"""

from pathlib import Path
import numpy as np
import ray
import json


@ray.remote(num_cpus=1)
def process_file(
    file_path: Path,
    file_root: Path,
    output_dir: Path,
    pychildes_path: str,
    config_path: str,
):
    """Process a single .cha file using PyChildes and save the output as JSONL.

    Args:
        file_path (Path): Path to the .cha file.
        file_root (Path): Root directory containing .cha files.
        output_dir (Path): Directory to save the JSONL output file.
        pychildes_path (str): Path to the PyChildes library.
        config_path (str): Path to the PyChildes configuration file.
    """
    import sys

    if pychildes_path not in sys.path:
        sys.path.append(pychildes_path)

    from prepare_childes import process_cha_file

    file_id = file_path.relative_to(file_root).as_posix().removesuffix(".cha")
    output_file_name = file_id.replace("/", "_")

    try:
        process_cha_file(
            file_path.as_posix(),
            (output_dir / f"{output_file_name}.jsonl").as_posix(),
            config_path,
        )
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        with open(output_dir / f"{output_file_name}.error", "w") as f:
            f.write(str(e))


def process_dir(
    file_root: str,
    output_dir: str,
    pychildes_path: str,
    config_path: str,
):
    """Process all .cha files in the specified directory using PyChildes in parallel.

    Args:
        file_root (str): Root directory containing .cha files.
        output_dir (str): Directory to save the JSONL output files.
        pychildes_path (str): Path to the PyChildes library.
        config_path (str): Path to the PyChildes configuration file.
    """
    file_root: Path = Path(file_root)
    output_dir: Path = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cha_files = list(file_root.rglob("*.cha"))

    tasks = [
        process_file.remote(
            file_path, file_root, output_dir, pychildes_path, config_path
        )
        for file_path in cha_files
    ]

    # Block until all tasks are complete
    ray.get(tasks)


@ray.remote(num_cpus=1)
def is_single_content(file_path: Path) -> bool:
    with open(file_path) as f:
        data = [json.loads(line) for line in f]

    is_single_content = all(
        len(item["content"]) == 1 for item in data if item["kind"] == "utterance"
    )

    return is_single_content


def test_is_single_content(output_dir: str):
    """\
    A stub function for testing if all JSONL files in the output directory
    contain only 1 utterance content.
    """
    output_dir: Path = Path(output_dir)

    results = [
        is_single_content.remote(file_path) for file_path in output_dir.glob("*.jsonl")
    ]
    results = ray.get(results)

    results = np.array(results, dtype=np.bool_)
    return results


if __name__ == "__main__":
    ray.init()
    process_dir(
        file_root="/u501/y4466wan/projects/2025/verb-overgen/data/childes",
        output_dir="/u501/y4466wan/projects/2025/verb-overgen/data/childes-gold/test-251010T0359",
        pychildes_path="/u501/y4466wan/projects/2025/verb-overgen/references/PyChildes",
        config_path="/u501/y4466wan/projects/2025/verb-overgen/references/PyChildes/configs/environment.yaml",
    )

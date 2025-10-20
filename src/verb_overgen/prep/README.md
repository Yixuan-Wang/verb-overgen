# `verb_overgen.prep`

Data preparation and processing.

## Processing TalkBank .cha Files with PyChildes

This module uses a modified version of [PyChildes](https://github.com/mars-tin/PyChildes) to process TalkBank `.cha` files
  and convertthem into JSONL format with golden morphosyntactic annotations.
Attested on commit `c93b532` of PyChildes, currently under Yixuan's [fork](https://github.com/Yixuan-Wang/PyChildes).

See `pychildes_run.py` and `pychildes_merge.py` for running PyChildes in parallel using Ray
  and merging the outputs into a single Parquet dataset.

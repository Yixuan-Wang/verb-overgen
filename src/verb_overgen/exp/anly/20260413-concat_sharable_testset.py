"""Concat sharable testset from multiple runs."""
from typing import Final

import polars as pl

VOI_9: Final[list[str]] = ["cry", "laugh", "smile", "sing", "sleep", "go", "sneeze", "blink", "look"]

PATH_DATASET_PER_VERB = "./data/exp/overgen_results-raw-260402-260413T2007.parquet"
PATH_DATASET_ALL_VERB = "./data/exp/overgen_results-raw-260410-260413T2009.parquet"

dataset_per_verb = pl.read_parquet(PATH_DATASET_PER_VERB)
dataset_all_verb = pl.read_parquet(PATH_DATASET_ALL_VERB)

assert set(dataset_per_verb.select(pl.col("run_seed").unique()).to_series()) == set(
    dataset_all_verb.select(pl.col("run_seed").unique()).to_series()
), "Run seeds should match between the two datasets"

assert dataset_per_verb.select(pl.col("run_verb_of_interest").is_not_null().all()).item(), "All runs in dataset_per_verb should have run_verb_of_interest not null"
assert dataset_all_verb.select(pl.col("run_verb_of_interest").is_null().all()).item(), "All runs in dataset_all_verb should have run_verb_of_interest null"

_len_pre_filter = len(dataset_per_verb)
dataset_per_verb = dataset_per_verb.filter(
    pl.col("verb") == pl.col("run_verb_of_interest"),
)
_len_post_filter = len(dataset_per_verb)
print(f"Filtered dataset_per_verb from {_len_pre_filter} to {_len_post_filter} by keeping only rows where verb matches run_verb_of_interest")

_len_pre_filter = len(dataset_all_verb)
dataset_per_verb = dataset_per_verb.filter(
    pl.col("verb").is_in(VOI_9)
)
_len_post_filter = len(dataset_per_verb)
print(f"Filtered dataset_per_verb from {_len_pre_filter} to {_len_post_filter} by keeping only rows where verb is in {VOI_9}")

_len_pre_filter = len(dataset_all_verb)
dataset_all_verb = dataset_all_verb.filter(
    pl.col("verb").is_in(VOI_9)
)
_len_post_filter = len(dataset_all_verb)
print(f"Filtered dataset_all_verb from {_len_pre_filter} to {_len_post_filter} by keeping only rows where verb is in {VOI_9}")


dataset = pl.concat([dataset_per_verb, dataset_all_verb])
assert dataset.group_by("run_subset", "run_seed", "epoch").len().select("len").to_series().n_unique() == 1, "All groups should have equal length"

assert dataset.group_by("run_seed", "run_subset", "epoch", "index_sentence").len().select(pl.col("len").eq(1).all()).item(), "There should be no duplicate rows with the same run_subset, run_seed, and index_sentence"
assert dataset.group_by("run_seed", "run_subset", "epoch", "verb", "index_verb_context").len().select(pl.col("len").eq(4).all()).item(), "Each verb and context should have 4 rows, one for each kind"
assert dataset.select(pl.col("epoch").n_unique() == 3).item(), "We should have 3 epochs in total"

dataset_final = dataset.select(
    pl.col("run_name"),
    # string. Each run is a unique training, 100 in total.
    pl.col("run_seed"),
    # int32. One of [189, 352, 559, 597, 706]. Concerns data shuffling, sampling, model initialization.
    pl.col("run_subset"),
    # string. One of ["remove_preemption-all_verb", "remove_preemption-per_verb", "remove_entrenchment-all_verb", "remove_entrenchment-per_verb"].
    # Remove preemption may be abbr as -P, remove entrenchment (entrenchment minus preemption) as +P.
    pl.col("epoch"),
    # int32. One of [0, 1, 2]. Epoch of the training when the evaluation is done. 
    pl.col("verb"),
    # string. The verb in the eval sentence. One of ["cry", "laugh", "smile", "sing", "sleep", "go", "sneeze", "blink", "look"].
    # In per-verb runs, this is guaranteed to be the same as the verb manipulated during training.
    pl.col("index_sentence"),
    # int64. NOT CONSECUTIVE. If this index is different, the model receives a different prompt.
    # i.e. "Tom made Jerry laugh", "Tom laughed Jerry", and "Tom told Jerry a story. Tom laughed Jerry." have DIFFERENT index_sentence.
    pl.col("index_verb_context"),
    # int64. NOT CONSECUTIVE. If this index is different, the verb appears in a different context (e.g. different subjects, different context sentences).
    # i.e. "Tom made Jerry laugh", "Tom laughed Jerry", and "Tom told Jerry a story. Tom laughed Jerry." have SAME index_verb_context.
    pl.col("kind").str.strip_prefix("sentence_").alias("sentence_construction"),
    # string. One of ["causative", "transitive", "context_causative", "context_transitive"]. The kind of sentence construction.
    pl.col("sentence"),
    # string. Raw sentence, not lemmatized. Conversion to past tense not implemented yet.
    pl.col("roberta").struct["lemma"].alias("lemma"),
    # list[string]. Lemmatized sentence provided by spacy.
    pl.col("input_ids"),
    # list[int]. Token IDs from the word-level tokenizer, tokenized on the lemmas.
    # 1 -> BOS_TOKEN, 0, 2, 3 not used, 4 -> . (period)
    # All sentences should start with BOS_TOKEN, so the first token should always be 1.
    # Context sentences and transitive/causative sentences are concatenated with the BOS as separator.
    pl.col("token_prob"),
    # list[float32]. The next-token prediction probabilities from the model for each token in input_ids.
    # token_prob[0] is the probability on the first token after BOS.
    # Should be 1 element shorter than input_ids (No extra decoding step after the last token).
    pl.col("nll"),
    # float32. Negative log likelihood of the final sentence.
    # <s> Tom made Jerry laugh .
    #     [--------------------]
    # <s> Tom told Jerry a story . <s> Tom made Jerry laugh .
    #                                  [--------------------]
)

dataset_final.write_parquet("data/exp/shared/overgen_results-260413T2011.parquet")

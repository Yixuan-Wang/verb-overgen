library(arrow)
library(dplyr)
library(tidyr)

check_seed_counts <- function(label, gap_data, group_cols) {
  counts <- gap_data |>
    group_by(across(all_of(group_cols))) |>
    summarise(n_seeds = n(), .groups = "drop")

  bad <- counts |> filter(n_seeds != 5)
  cat(sprintf("\n=== %s ===\n", label))
  cat(sprintf("Groups checked: %d\n", nrow(counts)))
  if (nrow(bad) == 0) {
    cat("OK: all groups have exactly 5 seeds.\n")
  } else {
    cat(sprintf("FAIL: %d group(s) do not have 5 seeds:\n", nrow(bad)))
    print(bad)
  }
}

count_vars <- c("count_verb_like", "count_causative")

# ---- per-verb dataset (20260403) ----
ds_pv <- arrow::read_parquet("var/results/20260403.indep_verb_loss_stats.parquet") |>
  mutate(run_subset = recode(run_subset, "+E" = "-P")) |>
  filter(run_subset %in% c("+P", "-P"))

# overgen gap: groups are (verb, run_subset, loss_pair)
gap_overgen_pv <- ds_pv |>
  filter(loss_type %in% c("transitive", "causative", "context_transitive", "context_causative")) |>
  select(verb, run_subset, run_seed, loss_type, loss, all_of(count_vars)) |>
  mutate(
    loss_pair = if_else(loss_type %in% c("transitive", "causative"), "no_context", "context"),
    loss_term = if_else(loss_type %in% c("transitive", "context_transitive"), "transitive", "causative")
  ) |>
  select(-loss_type) |>
  pivot_wider(
    id_cols     = c(verb, run_subset, run_seed, loss_pair, all_of(count_vars)),
    names_from  = loss_term,
    values_from = loss,
    names_prefix = "loss_"
  ) |>
  mutate(loss_gap = loss_transitive - loss_causative) |>
  filter(!is.na(loss_gap))

check_seed_counts("per-verb / overgen gap", gap_overgen_pv, c("verb", "run_subset", "loss_pair"))

# preemption gap: groups are (verb, loss_type)
gap_preempt_pv <- ds_pv |>
  filter(loss_type %in% c("transitive", "context_transitive")) |>
  select(verb, run_subset, run_seed, loss_type, loss, all_of(count_vars)) |>
  pivot_wider(
    names_from  = run_subset,
    values_from = loss,
    names_prefix = "loss_"
  ) |>
  mutate(loss_gap = `loss_+P` - `loss_-P`) |>
  filter(!is.na(loss_gap))

check_seed_counts("per-verb / preemption gap", gap_preempt_pv, c("verb", "loss_type"))

# ---- all-verb dataset (20260410) ----
ds_av <- arrow::read_parquet("var/results/20260410.all_verb_loss_stats.parquet") |>
  filter(run_subset %in% c("+P", "-P"))

# overgen gap: groups are (verb, run_subset, loss_pair)
gap_overgen_av <- ds_av |>
  filter(sentence_type %in% c("transitive", "causative", "context_transitive", "context_causative")) |>
  select(verb, run_subset, run_seed, sentence_type, loss, all_of(count_vars)) |>
  mutate(
    loss_pair = if_else(sentence_type %in% c("transitive", "causative"), "no_context", "context"),
    loss_term = if_else(sentence_type %in% c("transitive", "context_transitive"), "transitive", "causative")
  ) |>
  select(-sentence_type) |>
  pivot_wider(
    id_cols     = c(verb, run_subset, run_seed, loss_pair, all_of(count_vars)),
    names_from  = loss_term,
    values_from = loss,
    names_prefix = "loss_"
  ) |>
  mutate(loss_gap = loss_transitive - loss_causative) |>
  filter(!is.na(loss_gap))

check_seed_counts("all-verb / overgen gap", gap_overgen_av, c("verb", "run_subset", "loss_pair"))

# preemption gap: groups are (verb, sentence_type)
gap_preempt_av <- ds_av |>
  filter(sentence_type %in% c("transitive", "context_transitive")) |>
  select(verb, run_subset, run_seed, sentence_type, loss, all_of(count_vars)) |>
  pivot_wider(
    names_from  = run_subset,
    values_from = loss,
    names_prefix = "loss_"
  ) |>
  mutate(loss_gap = `loss_+P` - `loss_-P`) |>
  filter(!is.na(loss_gap))

check_seed_counts("all-verb / preemption gap", gap_preempt_av, c("verb", "sentence_type"))

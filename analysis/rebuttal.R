library(tidyverse)

new_results_raw <- fs::dir_ls("data/results/", regexp = "testset*", recurse = TRUE) %>%
  # str_subset(verb_file_pattern) %>%
  keep(str_detect(., "parquet")) %>%
  map_df(~arrow::read_parquet(.x), .id = "file") %>%
  mutate(
    alternating = str_detect(file, "alternating")
  )


new_results_raw %>%
  # count(eval_verb, eval_has_context, eval_construction, alternating)
  select(-eval_index_sentence, -result_token_prob) %>%
  pivot_wider(names_from = eval_construction, values_from = result_nll) %>%
  mutate(
    diff = causative - transitive
  ) %>%
  group_by(eval_verb, alternating, eval_has_context) %>%
  summarize(
    diff = mean(diff)
  )

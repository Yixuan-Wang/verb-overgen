library(tidyverse)

verbs <- c("laugh", "cry", "go", "sleep", "sneeze", "smile", "all")

results %>%
  count(run_manipulation)

results <- read_csv("data/exp-results-20260516.csv") %>%
  filter(ckpt_epoch==2) %>%
  mutate(
    is_full = run_manipulation == "none",
    experiment = case_when(
      is_full == TRUE ~ "Full",
      run_target == 'e' ~ "Control",
      run_target == 'p' ~ "Preemption-removed"
    ),
    run_scope = case_when(
      is.na(run_scope) ~ "none",
      TRUE ~ run_scope
    ),
    run_target = case_when(
      is.na(run_target) ~ "full",
      TRUE ~ run_target
    ),
    run_target_verb = case_when(
      run_scope == 'all-verb' ~ "all",
      TRUE ~ run_target_verb
    )
  )

results %>%
  distinct(eval_verb, eval_index_sentence, eval_index_verb_context, eval_has_context, eval_construction) %>%
  write_csv("data/eval-metadata.csv")

full <- results %>%
  filter(is_full==TRUE) %>%
  select(-is_full)

manipulation <- results %>%
  filter(is_full==FALSE) %>%
  select(-is_full)

agg <- results %>%
  filter(run_target_verb %in% verbs) %>%
  group_by(run_manipulation, run_scope, run_target, run_target_verb, 
           eval_verb, eval_has_context, eval_construction, experiment) %>%
  summarize(
    n = n(),
    nll = mean(result_nll),
    slor = mean(result_slor),
    std = sd(result_slor),
    cb = qt(1 - (0.05/2), n - 1) * std/sqrt(n),
  ) %>%
  ungroup()

agg %>%
  filter(eval_verb %in% verbs) %>%
  # filter(run_scope=="all-verb") %>%
  # filter(is.na(run_scope)) %>%
  filter(run_scope %in% c('all-verb', 'none')) %>%
  filter(eval_construction=="transitive") %>%
  filter(eval_has_context==TRUE) %>%
  ggplot(aes(eval_verb, slor, color = run_target, shape = run_scope)) +
  geom_point() + 
  geom_linerange(aes(ymin = slor-cb, ymax=slor+cb))



transitive <- manipulation %>%
  filter(eval_construction == "transitive") %>%
  select(-eval_index_sentence, -eval_construction) %>%
  rename(transitive_slor = result_slor, transitive_nll = result_nll)

causative <- manipulation %>%
  filter(eval_construction != "transitive") %>%
  select(-eval_index_sentence, -eval_construction) %>%
  rename(causative_slor = result_slor, causative_nll = result_nll)

manip_results <- transitive %>%
  inner_join(causative) %>%
  filter(eval_verb %in% verbs) %>%
  mutate(
    slor_diff = -(transitive_slor - causative_slor),
    nll_diff = transitive_nll - causative_nll
  )

manip_agg <- manip_results %>%
  filter(run_target_verb %in% verbs) %>%
  group_by(run_manipulation, run_scope, run_target, run_target_verb, 
           eval_verb, eval_has_context, experiment) %>%
  summarize(
    n = n(),
    # nll = mean(result_nll),
    slor = mean(nll_diff),
    std = sd(nll_diff),
    cb = qt(1 - (0.05/2), n - 1) * std/sqrt(n),
  ) %>%
  ungroup()

manip_agg %>%
  filter(eval_verb %in% verbs) %>%
  # filter(run_scope=="all-verb") %>%
  # filter(is.na(run_scope)) %>%
  filter(run_scope %in% c('all-verb')) %>%
  filter(run_manipulation == "remove") %>%
  # filter(eval_construction=="transitive") %>%
  filter(eval_has_context==TRUE) %>%
  ggplot(aes(eval_verb, slor, color = run_target, shape = run_scope)) +
  geom_point() + 
  geom_linerange(aes(ymin = slor-cb, ymax=slor+cb))


manip_agg %>%
  filter(eval_verb %in% verbs) %>%
  # filter(run_scope=="all-verb") %>%
  # filter(is.na(run_scope)) %>%
  filter(run_scope %in% c('per-verb')) %>%
  # filter(eval_construction=="transitive") %>%
  filter(eval_verb == run_target_verb) %>%
  filter(run_manipulation == "remove") %>%
  filter(eval_has_context==TRUE) %>% 
  ggplot(aes(eval_verb, slor, color = experiment, shape = run_scope)) +
  geom_point() + 
  geom_linerange(aes(ymin = slor-cb, ymax=slor+cb))

  

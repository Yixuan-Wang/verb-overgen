library(tidyverse)
library(lmerTest)
library(emmeans)

verbs <- c("laugh", "cry", "go", "sleep", "sneeze", "smile", "all")

results %>%
  count(run_manipulation)

results <- read_csv("data/exp-results-20260516.csv") %>%
  filter(ckpt_epoch==2) %>%
  mutate(
    is_full = run_manipulation == "none",
    experiment = case_when(
      is_full == TRUE ~ "Full",
      run_target == 'e' ~ "keep-p",
      run_target == 'p' ~ "remove-p"
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

full %>%
  filter(eval_verb %in% verbs) %>%
  pivot_longer(result_nll:result_slor, names_to = "score_type", values_to = "score") %>%
  select(-eval_index_sentence, -experiment, -run_name)

manip_diffs_long <- manipulation %>%
  filter(run_manipulation == "remove", run_scope != "except-verb") %>%
  filter(eval_verb %in% verbs, run_target_verb %in% verbs) %>%
  pivot_longer(result_nll:result_slor, names_to = "score_type", values_to = "score") %>%
  select(-eval_index_sentence, -experiment, -run_name) %>%
  filter((eval_verb == run_target_verb) | (run_target_verb == 'all'))

manip_diffs_long %>%
  pivot_wider(names_from = run_target, values_from = score) %>%
  inner_join(full %>%
               filter(eval_verb %in% verbs) %>%
               pivot_longer(result_nll:result_slor, names_to = "score_type", values_to = "score") %>%
               select(-eval_index_sentence, -experiment, -run_name, -run_manipulation,
                      -run_scope, -run_target, -run_target_verb) %>%
               rename(full = score)
             ) %>%
  pivot_longer(e:full, names_to = "exp", values_to = "score") %>%
  group_by(run_scope, eval_has_context, eval_verb, eval_construction, score_type, exp) %>%
  summarize(
    n = n(),
    std = sd(score),
    cb = qt(1 - (0.05/2), n - 1) * std/sqrt(n),
    score = mean(score)
  ) %>% View()



manip_diffs_long

fit_all <- lmer(score ~ run_target * eval_verb + (1 | run_seed:eval_verb:eval_index_verb_context),
            data = manip_diffs_long %>% 
              filter(eval_construction=="transitive", score_type == "result_nll") %>%
              filter(run_scope == "all-verb", eval_has_context==FALSE) %>%
              mutate(
                run_target = factor(run_target),
                run_seed = factor(run_seed),
                eval_verb = factor(eval_verb)
              )
            )

summary(fit_all)

emmeans(fit_all, pairwise ~ run_target | eval_verb) %>%
  confint() %>%
  .$contrasts %>%
  tibble()


# emmeans(fit_all, specs = ~run_target | eval_verb) %>%
#   contrast("pairwise") %>%
#   confint()

manip_diffs <- manip_diffs_long %>%
  pivot_wider(names_from = run_target, values_from = score) %>% 
  mutate(
    diff = e - p
  )


manip_diffs %>%
  filter(eval_construction=="transitive", score_type == "result_nll") %>%
  group_by(run_scope, eval_has_context, eval_verb, eval_construction, score_type) %>%
  summarize(
    n = n(),
    std = sd(diff),
    cb = qt(1 - (0.05/2), n - 1) * std/sqrt(n),
    diff = mean(diff)
  ) %>% 
  ungroup() %>%
  mutate(
    eval_verb = factor(eval_verb, levels = c("smile", "sneeze", "sleep", "cry", "laugh", "go"))
  ) %>%
  ggplot(aes(eval_verb, diff, color = run_scope)) +
  geom_point(position = position_dodge(0.5)) +
  geom_linerange(aes(ymin=diff-cb, ymax=diff+cb), position = position_dodge(0.5)) +
  facet_wrap(~eval_has_context) +
  scale_y_continuous(limits = c(-0.22, 0.15))

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

  

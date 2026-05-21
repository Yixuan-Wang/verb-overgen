library(tidyverse)
library(scales)
library(ggtext)
library(ggdist)

eval_metadata <- read_csv("data/eval-metadata.csv") %>%
  mutate(
    kind = case_when(
      eval_has_context == TRUE ~ as.character(glue::glue("sentence_context_{eval_construction}")),
      TRUE ~ as.character(glue::glue("sentence_{eval_construction}"))
    )
  ) %>%
  rename(index_sentence = eval_index_sentence, verb = eval_verb)

verbs <- c("laugh", "cry", "go", "sleep", "sneeze", "smile")
verb_file_pattern <- paste(verbs, collapse = "|")

results_dump <- fs::dir_ls("data/results/trajectory/", regexp = "*parquet") %>%
  str_subset(verb_file_pattern) %>%
  map_df(~arrow::read_parquet(.x), .id = "file")


verb_wise <- results_dump %>%
  filter(phase == "pre") %>%
  group_by(verb, step, kind) %>%
  summarise(
    n = n(),
    sd = sd(nll_last_sentence),
    cb = qt(1 - (0.05/2), n - 1) * sd/sqrt(n),
    nll = mean(nll_last_sentence)
  ) %>%
  ungroup() %>%
  arrange(verb, kind, step) %>% 
  # 2. Group by the categories you want to keep separate
  group_by(verb, kind) %>% 
  # 3. Calculate the rolling average
  mutate(
    nll_moving_avg = zoo::rollmean(nll, k = 3, fill = NA, align = "right")
  ) %>%
  ungroup()

verb_wise %>%
  filter(str_detect(kind, "context")) %>%
  ggplot(aes(step, nll, color = verb, linetype = kind)) +
  geom_line() +
  facet_wrap(~verb, scales = "free")


verb_wise_diffs <- results_dump %>%
  filter(phase == "pre") %>%
  inner_join(eval_metadata) %>%
  select(-index_sentence, -eval_construction, -eval_has_context) %>%
  pivot_wider(names_from = kind, values_from = nll_last_sentence) %>% 
  mutate(
    diff_context = sentence_context_transitive - sentence_context_causative,
    diff = sentence_transitive - sentence_causative
  ) %>%
  pivot_longer(diff_context:diff, names_to = "diff_type", values_to = "diff") %>%
  group_by(verb, step, diff_type) %>%
  summarise(
    n = n(),
    sd = sd(diff),
    cb = qt(1 - (0.05/2), n - 1) * sd/sqrt(n),
    pref = mean(diff > 0),
    diff = mean(diff)
  ) %>%
  ungroup()


verb_wise_diffs %>%
  # filter(run_seed == 706) %>%
  # group_by(verb, diff_type) %>%
  # # 3. Calculate the rolling average of the mean difference
  # mutate(
  #   diff_rolling_avg = zoo::rollmean(diff, k = 5, fill = NA, align = "right")
  # ) %>%
  # ungroup() %>%
  mutate(
    context = case_when(
      str_detect(diff_type, "context") ~ TRUE,
      TRUE ~ FALSE
    )
  ) %>%
  filter(context == FALSE) %>%
  mutate(
    alpha = case_when(
      verb == "go" ~ 0.02,
      verb %in% c("laugh") ~ 0.02,
      TRUE ~ 0.05
    )
  ) %>%
  ggplot(aes(step, diff)) +
  # geom_point(aes(alpha = alpha), size = 1, color = "#7DA78C") +
  geom_line(alpha = 0.2, color = "#7DA78C") +
  # geom_ribbon(aes(ymin = diff-cb, ymax = diff+cb), color = NA, alpha = 0.2) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_smooth(linewidth = 0.7, alpha = 0.4, color = "#7DA78C") +
  scale_alpha_identity() +
  facet_wrap(~verb, scales = "free", nrow = 1) +
  # scale_x_continuous(labels = scales::label_number(scale = 1e-3, suffix = "k")) +
  scale_x_continuous(
    labels = \(x) ifelse(x == 0, "0", label_number(scale = 1e-3, suffix = "k")(x)),
    limits = c(0, 12000),
    breaks = c(0, 3000, 6000, 9000, 12000)
  ) +
  theme_classic(base_size = 16, base_family = "Times") +
  theme(
    strip.background = element_blank(),
    axis.title.y = element_markdown(),
    strip.text.x = element_text(face = "bold.italic")
  ) +
  labs(
    x = "Training Step",
    y = "Surprisal &Delta;<br>(transitive - causative)"
  )

ggsave("figures/precondition-exp-surprisal-diff.pdf", width = 13.90, height = 2.73, dpi=300, device = cairo_pdf)

verb_wise_diffs %>%
  # group_by(verb, diff_type) %>%
  # # 3. Calculate the rolling average of the mean difference
  # mutate(
  #   diff_rolling_avg = zoo::rollmean(diff, k = 5, fill = NA, align = "right")
  # ) %>%
  # ungroup() %>%
  mutate(
    context = case_when(
      str_detect(diff_type, "context") ~ TRUE,
      TRUE ~ FALSE
    )
  ) %>%
  filter(context == FALSE) %>%
  mutate(
    alpha = case_when(
      verb == "go" ~ 0.02,
      verb %in% c("laugh") ~ 0.02,
      TRUE ~ 0.05
    )
  ) %>%
  ggplot(aes(step, pref)) +
  geom_line() +
  # geom_point(aes(alpha = alpha), size = 1, color = "#7DA78C") +
  # geom_ribbon(aes(ymin = diff-cb, ymax = diff+cb), color = NA, alpha = 0.2) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  # geom_smooth(linewidth = 0.7, alpha = 0.4, color = "#7DA78C") +
  scale_alpha_identity() +
  facet_wrap(~verb, scales = "free") +
  scale_x_continuous(labels = scales::label_number(scale = 1e-3, suffix = "k")) +
  theme_classic(base_size = 16) +
  theme(
    strip.background = element_blank(),
    axis.title.y = element_markdown(),
    strip.text.x = element_text(face = "bold")
  ) +
  labs(
    x = "Step",
    y = "Surprisal &Delta;<br>(transitive - causative)"
  )


moments_raw <- results_dump %>%
  group_by(run_seed, verb, step_type, kind, index_sentence, phase) %>%
  mutate(relative_step = row_number(step_effective)) %>%
  ungroup() %>% 
  select(-step, -step_effective) %>%
  pivot_wider(names_from = phase, values_from = nll_last_sentence) %>%
  mutate(
    prepost = (post - pre) # we expect this to be positive -- i.e., surprisal before 
  ) 

moments <- moments_raw %>%
  inner_join(eval_metadata) %>%
  select(-index_sentence, -eval_construction, -eval_has_context, -pre, -post) %>%
  # group_by(verb, step_type, step_effective, kind, run_seed) %>%
  # summarise(prepost = mean(prepost), nn = n(), .groups="drop") %>%
  pivot_wider(names_from = kind, values_from = prepost)

moment_corr <- moments %>%
  group_by(verb, step_type, run_seed, eval_index_verb_context) %>%
  nest() %>%
  mutate(
    corr = map(data, function(x) {
      cor.test(x$sentence_causative, x$sentence_transitive, method = "spearman") %>%
        broom::tidy()
    })
  ) %>%
  ungroup() %>%
  unnest(corr) %>%
  select(-data)
  # group_by(verb, step_type, run_seed) %>%
  # nest()

moment_corr %>%
  ggplot(aes(verb, estimate)) +
  # geom_jitter(alpha = 0.2, width = 0.2)
  # stat_gradientinterval(fill = "steelblue") +
  stat_halfeye(fill = "steelblue") +
  theme_classic(base_size = 16)
  # geom_pointrange()

moment_corr %>%
  group_by(verb, step_type) %>%
  summarize(
    n = n(),
    sd = sd(estimate),
    cb = qt(1 - (0.05/2), n - 1) * sd/sqrt(n),
    rho = mean(estimate),
    .groups = "drop"
  ) %>%
  filter(step_type == "causative") %>%
  ggplot(aes(verb, rho)) +
  # geom_point(position = position_dodge(0.5)) +
  # geom_linerange(aes(ymin = rho-cb, ymax = rho+cb), position = position_dodge(0.5))
  geom_point() +
  geom_linerange(aes(ymin = rho-cb, ymax = rho+cb)) +
  scale_y_continuous(limits = c(0.3,0.6)) +
  theme_classic()

moments_agg <- moments_raw %>%
  filter(run_seed == 189) %>%
  group_by(verb, kind, step_type, relative_step) %>%
  summarize(
    n = n(),
    sd = sd(prepost),
    cb = qt(1 - (0.05/2), n - 1) * sd/sqrt(n),
    diff = mean(prepost),
    pref = mean(prepost > 0)
  ) 

moments_raw %>%
  filter(kind %in% c("sentence_transitive", "sentence_causative")) %>%
  mutate(
    diff = post-pre
  ) %>%
  inner_join(eval_metadata) %>%
  select(-index_sentence, -pre, -post, -prepost, -eval_construction) %>%
  pivot_wider(names_from = kind, values_from = diff) %>%
  group_by(verb, run_seed, step_type, relative_step) %>%
  summarize(
    transitive = mean(sentence_transitive),
    causative = mean(sentence_causative)
  ) %>%
  ungroup() %>%
  group_by(verb, run_seed) %>%
  summarize(
    n = n(),
    correlation = cor(transitive, causative, method = "spearman"),
    agreement = mean(sign(transitive) == sign(causative))
  ) %>%
  ggplot(aes(verb, agreement)) +
  geom_point(size = 2, alpha = 0.4, position = position_jitter(width = 0.2, seed = 1024)) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_classic(base_size = 16)

moments_raw %>%
  # filter(run_seed == 189) %>%
  inner_join(eval_metadata) %>%
  select(-index_sentence, -eval_construction, -eval_has_context, -pre, -post) %>%
  # group_by(verb, step_type, epoch, kind, run_seed) %>%
  # summarise(prepost = mean(prepost), nn = n(), .groups="drop") %>%
  group_by(verb, epoch, kind, step_type) %>%
  summarize(
    n = n(),
    sd = sd(prepost),
    cb = qt(1 - (0.05/2), n - 1) * sd/sqrt(n),
    diff = mean(prepost),
    pref = mean(prepost > 0)
  ) %>% 
  ungroup() %>%
  filter(kind == "sentence_transitive") %>%
  ggplot(aes(verb, diff, shape = step_type, color = epoch)) +
  geom_point(position = position_dodge(0.5)) +
  geom_linerange(aes(ymin = diff-cb, ymax = diff+cb), position = position_dodge(0.5))

moments_agg %>%
  filter(kind == "sentence_context_transitive") %>%
  ggplot(aes(relative_step, diff, color = step_type)) +
  geom_line() +
  # geom_point() +
  geom_linerange(aes(ymin = diff - cb, ymax = diff+cb)) +
  facet_wrap(~verb, scales="free")


moments_agg %>% 
  filter(kind == "sentence_transitive") %>%
  ggplot(aes(step_effective, diff, color = step_type)) +
  geom_point() +
  facet_wrap(~verb)
  
  

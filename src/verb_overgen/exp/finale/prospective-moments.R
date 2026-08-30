library(tidyverse)
library(scales)
library(ggtext)
library(ggdist)
library(ggh4x)

# Prospective moments: at each *causative* training step we probe every eval
# sentence both `pre` (before the gradient step) and `post` (after), and ask
# whether the per-step change in transitive NLL tracks the change in
# periphrastic-causative NLL. A positive correlation means entrenchment of the
# causative moves the transitive alternate in lockstep.
#
# The new trajectory data stores per-sentence `nll_last_sentence` with `verb`
# and `kind` ({sentence_causative, sentence_transitive}) as native columns, so
# the old `data/eval-metadata.csv` join is gone. The causative/transitive pair
# is implicit in `index_sentence`: causative sentences sit on even indices and
# their transitive counterpart is the following odd index, so
# `pair = index_sentence %/% 2` links the two (replacing the old
# `eval_index_verb_context`).
#
# Verbs from configs/trace/voi_20260829.yaml, ordered to match VERB_ORDER in
# src/verb_overgen/common/features.py.
verbs <- c(
  "laugh", "smile", "sneeze", "cry", "sleep", "go"
)
verb_file_pattern <- paste0("(", paste(verbs, collapse = "|"), ")_seed")

results_dump <- fs::dir_ls("var/results/v2/trajectory_0829/", regexp = "*parquet") %>%
  str_subset(verb_file_pattern) %>%
  map_df(~arrow::read_parquet(.x), .id = "file")

# Pair each pre/post probe into a single moment and take the change (prepost).
# Prospective causative moments are the probes taken around an actual causative
# training step, i.e. `trigger_reason == "causative"`. The other reasons are
# irrelevant here: `entrench` probes non-causative steps, and `periodic` are
# post-only snapshots with no matching `pre` (their extra `post` rows drift
# `relative_step` — a within-phase row_number over step_effective — so pre/post
# pivot out of alignment, corrupting prepost and injecting NAs).
moments_raw <- results_dump %>%
  filter(trigger_reason == "causative") %>%
  select(run_seed, verb, step_type, kind, index_sentence, epoch, phase, step_effective, nll_last_sentence) %>%
  group_by(run_seed, verb, step_type, kind, index_sentence, phase) %>%
  mutate(relative_step = row_number(step_effective)) %>%
  ungroup() %>%
  select(-step_effective) %>%
  pivot_wider(names_from = phase, values_from = nll_last_sentence) %>%
  mutate(prepost = post - pre) # expect positive: NLL rises before the causative step

# Link the causative sentence to its transitive counterpart (pair) and spread
# the two kinds into columns of prepost changes.
moments <- moments_raw %>%
  mutate(pair = index_sentence %/% 2) %>%
  select(run_seed, verb, step_type, epoch, relative_step, pair, kind, prepost) %>%
  pivot_wider(names_from = kind, values_from = prepost)

# Per-verb correlation / sign-agreement across causative training steps.
agg_corr <- moments %>%
  filter(step_type == "causative") %>%
  group_by(verb, run_seed, epoch, step_type, relative_step) %>%
  summarize(
    transitive = mean(sentence_transitive),
    causative = mean(sentence_causative),
    .groups = "drop"
  ) %>%
  group_by(verb, step_type) %>%
  summarize(
    n = n(),
    correlation = cor(transitive, causative, method = "pearson"),
    agreement = mean(sign(transitive) == sign(causative)),
    transitive = mean(transitive),
    causative = mean(causative),
    .groups = "drop"
  )

# point_color = "#744577"
# point_color = "#35858E"
point_color = "#7DA78C"
line_color = point_color

moments_points <- moments %>%
  filter(step_type == "causative") %>%
  group_by(verb, epoch, run_seed, relative_step) %>%
  summarize(
    sentence_transitive = mean(sentence_transitive),
    sentence_causative = mean(sentence_causative),
    .groups = "drop"
  ) %>%
  mutate(verb = factor(verb, levels = verbs))

moments_points %>%
  ggplot(aes(sentence_causative, sentence_transitive)) +
  geom_point(size = 1.2, alpha = 0.1, color = point_color) +
  geom_smooth(method = "lm", color = "black", linewidth = 1) +
  geom_richtext(
    x = -0.14, y = 0.07,
    aes(label = stats),
    data = agg_corr %>%
      filter(step_type == "causative") %>%
      mutate(
        stats = glue::glue("<i>r</i> = {sprintf('%.2f', round(correlation, 2))}<br><i>agr.</i> = {sprintf('%.2f', round(agreement, 2))}"),
        verb = factor(verb, levels = verbs)
      ),
    hjust = 0,
    vjust = 0,
    fill = NA, label.color = NA,
    family = "Times",
    size = 4.5
  ) +
  facet_wrap(~verb, ncol = 6, scales = "free") +
  scale_x_continuous(limits = c(-0.15, 0.15)) +
  scale_y_continuous(limits = c(-0.15, 0.15)) +
  theme_classic(base_size = 17, base_family = "Times") +
  theme(
    strip.text.x = element_text(face = "bold.italic", size = 18),
    strip.background = element_blank()
  ) +
  labs(
    x = "Periphrastic Causative NLL",
    y = "Transitive NLL"
  )

ggsave("var/figures/prospective-moments-corr.pdf", width = 13.90, height = 3.20, dpi = 300, device = cairo_pdf)

library(tidyverse)
library(scales)
library(ggtext)
library(ggdist)
library(ggh4x)

# New trajectory data stores per-sentence `nll_last_sentence` with a `kind`
# column ({sentence_causative, sentence_transitive}) rather than the old
# pre-aggregated loss_causative/loss_transitive. The `periodic` heartbeat rows
# (one probe per verb every 50 steps, post-only) are the equivalent of the old
# every-50-step eval, so we filter to those.
# Verbs from configs/trace/voi_20260829.yaml, ordered to match VERB_ORDER in
# src/verb_overgen/common/features.py.
verbs <- c(
  "laugh", "smile", "sneeze", "cry", "sleep", "go" #,
  # "roll", "stop", "move", "turn", "break",
  # "catch", "hit", "push", "feed", "touch"
)
verb_file_pattern <- paste0("(", paste(verbs, collapse = "|"), ")_seed")

training_dynamics <- fs::dir_ls("var/results/v2/trajectory_0829/", regexp = "*parquet") %>%
  str_subset(verb_file_pattern) %>%
  map_df(~arrow::read_parquet(.x), .id = "file")

training_dynamics %>%
  filter(trigger_reason == "periodic") %>%
  group_by(verb, step, run_seed, kind) %>%
  summarize(nll = mean(nll_last_sentence), .groups = "drop") %>%
  pivot_wider(names_from = kind, values_from = nll) %>%
  mutate(diff = sentence_transitive - sentence_causative) %>%
  group_by(verb, step) %>%
  summarize(
    n = n(),
    se = sd(diff) / sqrt(n),
    diff = mean(diff),
    .groups = "drop"
  ) %>%
  mutate(verb = factor(verb, levels = verbs)) %>%
  ggplot(aes(step, diff)) +
  # geom_line(alpha = 0.2) +
  # geom_smooth() +
  # geom_point(alpha = 0.4, color = "#7DA78C") +
  geom_ribbon(aes(ymin = diff - se, ymax = diff + se), fill = "#4E7A63", color = NA, alpha = 0.35) +
  geom_line(alpha = 0.4, color = "#7DA78C") +
  # geom_line(alpha = 0.4, color = "#6EADBC") +
  # geom_line(alpha = 0.4, color = "#F4AE52") +
  # geom_line(alpha = 0.4, color = "#4B5694") +
  # geom_point()
  # geom_ribbon(aes(ymin = diff-cb, ymax = diff+cb), color = NA, alpha = 0.2) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  # geom_smooth(method = "gam", linewidth = 1, alpha = 0.4, color = "#7DA78C", fill = "#7DA78C") +
  geom_smooth(method = "gam", linewidth = 1, alpha = 0.4, color = "#7DA78C", fill = "#7DA78C") +
  # geom_smooth(linewidth = 1, alpha = 0.4, color = "#6EADBC", fill = "#6EADBC") +
  # geom_smooth(method = 'gam', linewidth = 1, alpha = 0.4, color = "#F4AE52", fill = "#F4AE52") +
  # geom_smooth(method = 'gam', linewidth = 1, alpha = 0.4, color = "#4B5694", fill = "#4B5694") +
  # scale_x_log10() +
  facet_wrap(~verb, scales = "free", ncol = 6) +
  scale_x_continuous(
    labels = \(x) ifelse(x == 0, "0", label_number(scale = 1e-3, suffix = "k")(x)),
    limits = c(0, 12000),
    breaks = c(0, 3000, 6000, 9000, 12000)
  ) +
  # Verb-agnostic: no manual per-verb y scaling.
  # facetted_pos_scales(
  #   y = list(
  #     verb == "cry" ~ scale_y_continuous(limits = c(-0.2, 1.2)),
  #     verb == "sleep" ~ scale_y_continuous(limits = c(-0.2, 1), breaks = pretty_breaks()),
  #     verb == "smile" ~ scale_y_continuous(limits = c(-0.1, 1.5), breaks = pretty_breaks()),
  #     verb == "sneeze" ~ scale_y_continuous(limits = c(-0.2, 1.6), breaks = pretty_breaks())
  #   )
  # ) +
  theme_classic(base_size = 16, base_family = "Times") +
  theme(
    strip.background = element_blank(),
    axis.title.y = element_markdown(),
    strip.text.x = element_text(face = "bold.italic", size = 18)
  ) +
  labs(
    x = "Training Step",
    y = "&Delta;NLL<br>(transitive - periphrastic)"
  )

ggsave("var/figures/precondition-exp-surprisal-diff-candidates.pdf", width = 13.90, height = 2.73, dpi=300, device = cairo_pdf)

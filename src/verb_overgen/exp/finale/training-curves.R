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
verbs <- c("laugh", "cry", "go", "sleep", "sneeze", "smile")
verb_file_pattern <- paste0("(", paste(verbs, collapse = "|"), ")_seed")

training_dynamics <- fs::dir_ls("var/results/v2/trajectory_0829/", regexp = "*parquet") %>%
  str_subset(verb_file_pattern) %>%
  map_df(~arrow::read_parquet(.x), .id = "file")

training_dynamics %>%
  filter(trigger_reason == "periodic") %>%
  group_by(verb, step, run_seed, kind) %>%
  summarize(nll = mean(nll_last_sentence), .groups = "drop") %>%
  pivot_wider(names_from = kind, values_from = nll) %>%
  group_by(verb, step) %>%
  summarize(
    n = n(),
    diff = mean(sentence_transitive - sentence_causative),
    .groups = "drop"
  ) %>%
  ggplot(aes(step, diff)) +
  # geom_line(alpha = 0.2) +
  # geom_smooth() +
  # geom_point(alpha = 0.4, color = "#7DA78C") +
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
  facet_wrap(~verb, scales = "free", nrow = 1) +
  scale_x_continuous(
    labels = \(x) ifelse(x == 0, "0", label_number(scale = 1e-3, suffix = "k")(x)),
    limits = c(0, 12000),
    breaks = c(0, 3000, 6000, 9000, 12000)
  ) +
  facetted_pos_scales(
    y = list(
      verb == "cry" ~ scale_y_continuous(limits = c(-0.2, 1.2)),
      verb == "sleep" ~ scale_y_continuous(limits = c(-0.2, 1), breaks = pretty_breaks()),
      verb == "smile" ~ scale_y_continuous(limits = c(-0.1, 1.5), breaks = pretty_breaks()),
      verb == "sneeze" ~ scale_y_continuous(limits = c(-0.2, 1.6), breaks = pretty_breaks())
    )
  ) +
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

ggsave("var/figures/precondition-exp-surprisal-diff.pdf", width = 13.90, height = 2.73, dpi=300, device = cairo_pdf)

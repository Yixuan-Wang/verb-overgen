library(arrow)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggrepel)
library(ggh4x)

dataset <- arrow::read_parquet("var/results/20260410.all_verb_loss_stats.parquet")

count_vars <- c("count_verb_like", "count_causative")

# Compute per-(verb, seed, sentence_type) loss gap: L_{+P} - L_{-P}
gap_data <- dataset |>
  filter(run_subset %in% c("+P", "-P"),
         sentence_type %in% c("transitive", "context_transitive")) |>
  select(verb, run_subset, run_seed, sentence_type, loss,
         all_of(count_vars)) |>
  pivot_wider(
    names_from  = run_subset,
    values_from = loss,
    names_prefix = "loss_"
  ) |>
  mutate(loss_gap = `loss_+P` - `loss_-P`) |>
  filter(!is.na(loss_gap))

print(gap_data)

# Summarise gap per (verb, sentence_type): mean and SE across seeds
gap_summary <- gap_data |>
  group_by(verb, sentence_type, across(all_of(count_vars))) |>
  summarise(
    gap_mean = mean(loss_gap),
    gap_se   = if (n() > 1) sd(loss_gap) / sqrt(n()) else NA_real_,
    n_seeds  = n(),
    .groups  = "drop"
  )

print(gap_summary)

# Pivot count variables long for faceting
gap_long <- gap_summary |>
  mutate(ratio_causative_verb = count_causative / count_verb_like) |>
  pivot_longer(
    cols      = c(all_of(count_vars), ratio_causative_verb),
    names_to  = "count_var",
    values_to = "count"
  ) |>
  mutate(
    sentence_type = factor(sentence_type, levels = c("transitive", "context_transitive")),
    count_var     = factor(count_var,
      levels = c("count_causative", "count_verb_like", "ratio_causative_verb"))
  )

# Plot: gap_mean vs count, rows = sentence_type, cols = count_var
p <- ggplot(gap_long, aes(x = count, y = gap_mean, color = sentence_type, label = verb)) +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.4, color = "grey50") +
  geom_errorbar(
    data = ~ filter(.x, !is.na(gap_se)),
    aes(ymin = gap_mean - gap_se, ymax = gap_mean + gap_se),
    width = 0, linewidth = 0.5, alpha = 0.75
  ) +
  geom_point() +
  geom_text_repel(size = 2.5, max.overlaps = 20) +
  scale_color_manual(
    values = c(
      "transitive"         = "#00BFC4",
      "context_transitive" = "#78b4b6"
    )
  ) +
  facet_grid2(
    sentence_type ~ count_var,
    scales = "free", independent = "x",
    labeller = labeller(
      count_var = c(
        count_causative      = "Count (causative)",
        count_verb_like      = "Count (verb)",
        ratio_causative_verb = "Ratio (causative / verb)"
      ),
      sentence_type = c(
        transitive         = "Transitive",
        context_transitive = "Context transitive"
      )
    )
  ) +
  facetted_pos_scales(
    x = list(
      scale_x_log10(labels = scales::label_comma()),
      scale_x_log10(labels = scales::label_comma()),
      scale_x_continuous(),
      scale_x_log10(labels = scales::label_comma()),
      scale_x_log10(labels = scales::label_comma()),
      scale_x_continuous()
    )
  ) +
  labs(x = "Count", y = "Preemption effect (loss)",
       color = "Sentence type") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("./var/figures/20260410.all_verb_preemption_eff_loss.pdf", p,
       width = 14, height = 8)

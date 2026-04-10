library(arrow)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggrepel)
library(ggh4x)

dataset <- arrow::read_parquet("var/results/20260403.indep_verb_loss_stats.parquet")

count_vars <- c("count_verb_like", "count_causative")

# Compute per-(verb, seed, run_subset) loss gap: L_{transitive} - L_{causative}
gap_data <- dataset |>
  mutate(run_subset = recode(run_subset, "+E" = "-P")) |>
  filter(run_subset %in% c("+P", "-P"),
         loss_type %in% c("transitive", "causative",
                          "context_transitive", "context_causative")) |>
  select(verb, run_subset, run_seed, loss_type, loss, all_of(count_vars)) |>
  mutate(
    loss_pair = case_when(
      loss_type %in% c("transitive", "causative") ~ "no_context",
      TRUE ~ "context"
    ),
    loss_term = case_when(
      loss_type %in% c("transitive", "context_transitive") ~ "transitive",
      TRUE ~ "causative"
    )
  ) |>
  select(-loss_type) |>
  pivot_wider(
    id_cols    = c(verb, run_subset, run_seed, loss_pair, all_of(count_vars)),
    names_from  = loss_term,
    values_from = loss,
    names_prefix = "loss_"
  ) |>
  mutate(loss_gap = loss_transitive - loss_causative) |>
  filter(!is.na(loss_gap))

print(gap_data)

# Summarise gap per (verb, run_subset, loss_pair): mean and SE across seeds
gap_summary <- gap_data |>
  group_by(verb, run_subset, loss_pair, across(all_of(count_vars))) |>
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
    loss_pair  = factor(loss_pair, levels = c("no_context", "context")),
    run_subset = factor(run_subset, levels = c("+P", "-P")),
    count_var  = factor(count_var,
      levels = c("count_causative", "count_verb_like", "ratio_causative_verb"))
  )

# Plot: gap_mean vs count, rows = loss_pair, cols = count_var, color = run_subset
p <- ggplot(gap_long, aes(x = count, y = gap_mean, color = run_subset, label = verb)) +
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
      "+P" = "#9B59B6",
      "-P" = "#C39BD3"
    )
  ) +
  facet_grid2(
    loss_pair ~ count_var,
    scales = "free_x", independent = "x",
    labeller = labeller(
      count_var = c(
        count_causative      = "Count (causative)",
        count_verb_like      = "Count (verb)",
        ratio_causative_verb = "Ratio (causative / verb)"
      ),
      loss_pair = c(
        no_context = "Transitive \u2212 Causative",
        context    = "Context transitive \u2212 Context causative"
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
  labs(x = "Count", y = "Anti-overgen effect",
       color = "Training subset") +
  theme_bw() +
  theme(
    # axis.title.y = element_text(hjust = 1),
    axis.text.x  = element_text(angle = 45, hjust = 1)
  )

ggsave("./var/figures/20260409.per_verb_overgen_eff_loss.pdf", p,
       width = 14, height = 8)

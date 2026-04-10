library(arrow)
library(dplyr)

dataset <- arrow::read_parquet("var/results/20260403.indep_verb_loss_stats.parquet")

count_vars <- c("count_verb_like", "count_non_verb_like", "count_causative")

spearman_results <- dataset |>
  mutate(run_subset = recode(run_subset, "+E" = "-P")) |>
  group_by(run_subset, loss_type) |>
  summarise(
    across(
      all_of(count_vars),
      ~ cor(.x, loss, method = "spearman"),
      .names = "rho_{.col}"
    ),
    .groups = "drop"
  )

print(spearman_results)

library(ggplot2)
library(ggrepel)
library(tidyr)
library(ggh4x)

outlier_verbs <- c("glow", "rise")

plot_data <- dataset |>
  mutate(run_subset = recode(run_subset, "+E" = "-P")) |>
  filter(run_subset %in% c("+P", "-P"),
         loss_type %in% c("causative", "transitive")) |>
  mutate(ratio_causative_verb = count_causative / count_verb_like) |>
  group_by(run_subset, verb, loss_type, count_verb_like, count_causative, ratio_causative_verb) |>
  summarise(
    loss_mean = mean(loss),
    loss_se   = if (n() > 1) sd(loss) / sqrt(n()) else NA_real_,
    .groups = "drop"
  ) |>
  pivot_longer(
    cols = c(count_verb_like, count_causative, ratio_causative_verb),
    names_to  = "count_var",
    values_to = "count"
  ) |>
  mutate(
    run_subset = factor(run_subset, levels = c("+P", "-P")),
    count_var  = factor(count_var,
      levels = c("count_causative", "count_verb_like", "ratio_causative_verb"))
  )

spearman_excl <- plot_data |>
  filter(!verb %in% outlier_verbs) |>
  group_by(run_subset, loss_type, count_var) |>
  summarise(rho_excl = cor(count, loss_mean, method = "spearman"), .groups = "drop")

spearman_plot <- plot_data |>
  group_by(run_subset, loss_type, count_var) |>
  summarise(rho = cor(count, loss_mean, method = "spearman"), .groups = "drop") |>
  left_join(spearman_excl, by = c("run_subset", "loss_type", "count_var")) |>
  mutate(label = sprintf("Spearman = %.2f (%.2f?)", rho, rho_excl)) |>
  group_by(loss_type, count_var) |>
  mutate(vjust_offset = 1.5 * row_number()) |>
  ungroup()

cat("Spearman rho excluding", paste(outlier_verbs, collapse = ", "), ":\n")
print(spearman_plot |> select(run_subset, loss_type, count_var, rho, rho_excl))

p <- ggplot(plot_data, aes(x = count, y = loss_mean, color = run_subset, label = verb)) +
  geom_errorbar(
    data = ~ filter(.x, !is.na(loss_se)),
    aes(ymin = loss_mean - loss_se, ymax = loss_mean + loss_se),
    width = 0, linewidth = 0.5, alpha = 0.75
  ) +
  geom_text(
    data = spearman_plot,
    aes(label = label, color = run_subset, vjust = vjust_offset),
    x = Inf, y = Inf, hjust = 1.1,
    inherit.aes = FALSE, size = 3, show.legend = FALSE
  ) +
  # geom_smooth(method = "lm", se = FALSE, linewidth = 0.6) +
  geom_point() +
  geom_text_repel(size = 2.5, max.overlaps = 20) +
  scale_color_manual(values = c("+P" = "steelblue", "-P" = "firebrick")) +
  facet_grid2(
    loss_type ~ count_var,
    scales = "free_x", independent = "x",
    labeller = labeller(
      count_var = c(
        count_causative      = "Count (causative)",
        count_verb_like      = "Count (verb)",
        ratio_causative_verb = "Ratio (causative / verb)"
      ),
      loss_type = c(
        causative  = "Causative",
        transitive = "Transitive"
      )
    )
  ) +
  facetted_pos_scales(
    x = list(
      scale_x_log10(labels = scales::label_comma()),
      scale_x_log10(labels = scales::label_comma()),
      scale_x_continuous(breaks = scales::breaks_pretty()),
      scale_x_log10(labels = scales::label_comma()),
      scale_x_log10(labels = scales::label_comma()),
      scale_x_continuous(breaks = scales::breaks_pretty())
    )
  ) +
  labs(x = NULL, y = "Loss (mean)", color = "Training subset") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("./var/figures/20260409.per_verb_loss_vs_count.pdf", p,
       width = 15, height = 10)

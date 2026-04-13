library(arrow)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggrepel)
library(ggh4x)

count_vars <- c("count_verb_like", "count_causative")

make_preempt_gap <- function(dataset) {
  dataset |>
    filter(run_subset %in% c("+P", "-P"),
           sentence_type %in% c("transitive", "context_transitive")) |>
    select(verb, run_subset, run_seed, sentence_type, loss, all_of(count_vars)) |>
    pivot_wider(
      names_from  = run_subset,
      values_from = loss,
      names_prefix = "loss_"
    ) |>
    mutate(loss_gap = `loss_+P` - `loss_-P`) |>
    filter(!is.na(loss_gap)) |>
    group_by(verb, sentence_type, across(all_of(count_vars))) |>
    summarise(
      gap_mean = mean(loss_gap),
      gap_se   = if (n() > 1) sd(loss_gap) / sqrt(n()) else NA_real_,
      .groups  = "drop"
    )
}

ds_pv <- arrow::read_parquet("var/results/20260403.indep_verb_loss_stats.parquet") |>
  mutate(run_subset = recode(run_subset, "+E" = "-P"),
         sentence_type = loss_type)

ds_av <- arrow::read_parquet("var/results/20260410.all_verb_loss_stats.parquet")

gap_pv <- make_preempt_gap(ds_pv) |> mutate(source = "per-verb")
gap_av <- make_preempt_gap(ds_av) |> mutate(source = "all-verb")

# Use per-verb count_causative as the shared x position; verify they match
count_check <- inner_join(
  gap_pv |> distinct(verb, count_causative) |> rename(n_pv = count_causative),
  gap_av |> distinct(verb, count_causative) |> rename(n_av = count_causative),
  by = "verb"
) |> filter(n_pv != n_av)

if (nrow(count_check) > 0) {
  warning("count_causative differs between datasets for: ",
          paste(count_check$verb, collapse = ", "))
}

# Segment data: one segment per (verb, sentence_type) connecting the two sources
segment_data <- inner_join(
  gap_pv |> select(verb, sentence_type, count_causative, gap_mean),
  gap_av |> select(verb, sentence_type, count_causative, gap_mean),
  by = c("verb", "sentence_type", "count_causative"),
  suffix = c("_pv", "_av")
) |>
  mutate(direction = if_else(gap_mean_av > gap_mean_pv, "increased", "decreased"))

point_data <- bind_rows(gap_pv, gap_av) |>
  mutate(
    source        = factor(source, levels = c("per-verb", "all-verb")),
    sentence_type = factor(sentence_type, levels = c("transitive", "context_transitive"))
  )

x_scale <- scale_x_continuous(
  trans  = scales::pseudo_log_trans(base = 10),
  breaks = c(0, 10, 100, 1000, 10000),
  labels = scales::label_comma()
)

sent_labeller <- labeller(sentence_type = c(
  transitive         = "Transitive",
  context_transitive = "Context transitive"
))

# Figure 1: paired dots with error bars
p_dots <- ggplot(point_data,
                 aes(x = count_causative, color = source)) +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.4, color = "grey50") +
  geom_errorbar(
    data = ~ filter(.x, !is.na(gap_se)),
    aes(y = gap_mean, ymin = gap_mean - gap_se, ymax = gap_mean + gap_se),
    width = 0.15, linewidth = 0.5, alpha = 0.4
  ) +
  geom_point(aes(y = gap_mean)) +
  geom_text_repel(
    aes(y = gap_mean, label = verb),
    size = 2.5, max.overlaps = 20
  ) +
  x_scale +
  scale_color_manual(
    values = c("per-verb" = "#E67E22", "all-verb" = "#2980B9")
  ) +
  facet_wrap(~ sentence_type, ncol = 1, labeller = sent_labeller) +
  labs(
    x     = "Count (causative)",
    y     = "Preemption effect (loss)",
    color = "Training setting"
  ) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("./var/figures/20260410.compare_preemption_eff_loss.pdf", p_dots,
       width = 10, height = 10)

# Figure 2: delta segments (gap_mean_av - gap_mean_pv) as lollipops
delta_data <- segment_data |>
  mutate(
    delta         = gap_mean_av - gap_mean_pv,
    direction     = if_else(delta > 0, "increased", "decreased"),
    sentence_type = factor(sentence_type, levels = c("transitive", "context_transitive"))
  )

p_delta <- ggplot(delta_data,
                  aes(x = count_causative, color = direction)) +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.4, color = "grey50") +
  geom_segment(
    aes(xend = count_causative, y = 0, yend = delta),
    linewidth = 0.6
  ) +
  geom_point(aes(y = delta)) +
  geom_text_repel(
    aes(y = delta, label = verb),
    size = 2.5, max.overlaps = 20
  ) +
  x_scale +
  scale_color_manual(
    values = c("increased" = "#E74C3C", "decreased" = "#27AE60"),
    labels = c("increased" = "increased (per\u2192all)", "decreased" = "decreased (per\u2192all)")
  ) +
  facet_wrap(~ sentence_type, ncol = 1, labeller = sent_labeller) +
  labs(
    x     = "Count (causative, symlog)",
    y     = "Preemption effect change (all \u2212 per)",
    color = "Direction"
  ) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("./var/figures/20260410.compare_preemption_eff_loss_delta.pdf", p_delta,
       width = 10, height = 10)

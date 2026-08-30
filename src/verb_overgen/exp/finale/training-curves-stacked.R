library(tidyverse)
library(scales)
library(ggtext)
library(ggrepel)
library(mgcv)

# Stacked variant of training-curves.R: all verbs on a single panel, colored by
# verb category (intransitive-only / alternating / transitive-only), with each
# verb labeled next to the end of its smoothed mean curve. Smooth mean only ---
# no per-run lines or ribbons.
# Verbs from configs/trace/voi_20260829.yaml.
verbs_intransitive <- c("go", "laugh", "cry", "sleep", "sneeze", "smile")
verbs_alternating <- c("break", "roll", "move", "stop", "turn")
verbs_transitive <- c("feed", "touch", "hit", "push", "catch")
verbs <- c(verbs_intransitive, verbs_alternating, verbs_transitive)

verb_category <- tibble(
  verb = verbs,
  category = factor(
    c(
      rep("Intransitive-only", length(verbs_intransitive)),
      rep("Alternating", length(verbs_alternating)),
      rep("Transitive-only", length(verbs_transitive))
    ),
    levels = c("Intransitive-only", "Alternating", "Transitive-only")
  )
)

category_colors <- c(
  "Intransitive-only" = "#4E7A63",
  "Alternating" = "#4B5694",
  "Transitive-only" = "#F4AE52"
)

verb_file_pattern <- paste0("(", paste(verbs, collapse = "|"), ")_seed")

training_dynamics <- fs::dir_ls("var/results/v2/trajectory_0829/", regexp = "*parquet") %>%
  str_subset(verb_file_pattern) %>%
  map_df(~arrow::read_parquet(.x), .id = "file")

df <- training_dynamics %>%
  filter(trigger_reason == "periodic") %>%
  group_by(verb, step, run_seed, kind) %>%
  summarize(nll = mean(nll_last_sentence), .groups = "drop") %>%
  pivot_wider(names_from = kind, values_from = nll) %>%
  mutate(diff = sentence_transitive - sentence_causative) %>%
  group_by(verb, step) %>%
  summarize(diff = mean(diff), .groups = "drop") %>%
  inner_join(verb_category, by = "verb")

# Smoothed endpoint per verb for the inline label position.
labels <- df %>%
  group_by(verb, category) %>%
  group_modify(~{
    fit <- gam(diff ~ s(step), data = .x)
    x_end <- max(.x$step)
    tibble(step = x_end, diff = as.numeric(predict(fit, tibble(step = x_end))))
  }) %>%
  ungroup()

df %>%
  ggplot(aes(step, diff, group = verb, color = category)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_smooth(method = "gam", se = FALSE, linewidth = 1) +
  geom_text_repel(
    data = labels,
    aes(label = verb),
    family = "Times",
    fontface = "italic",
    size = 5,
    direction = "y",
    hjust = 0,
    nudge_x = 300,
    segment.color = "grey70",
    segment.size = 0.3,
    min.segment.length = 0,
    max.overlaps = Inf,
    xlim = c(12000, NA)
  ) +
  scale_color_manual(values = category_colors, name = NULL) +
  scale_x_continuous(
    labels = \(x) ifelse(x == 0, "0", label_number(scale = 1e-3, suffix = "k")(x)),
    limits = c(0, 14000),
    breaks = c(0, 3000, 6000, 9000, 12000)
  ) +
  theme_classic(base_size = 16, base_family = "Times") +
  theme(
    axis.title.y = element_markdown(),
    legend.position = "top"
  ) +
  labs(
    x = "Training Step",
    y = "&Delta;NLL<br>(transitive - periphrastic)"
  )

ggsave("var/figures/precondition-exp-surprisal-diff-stacked.pdf", width = 10.00, height = 7.50, dpi = 300, device = cairo_pdf)

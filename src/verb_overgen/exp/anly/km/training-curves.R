library(tidyverse)
library(scales)
library(ggtext)
library(ggdist)
library(ggh4x)

training_dynamics <- fs::dir_ls("data/results/eval-loss_full/none/", regexp = "*parquet") %>%
  map_df(~arrow::read_parquet(.x), .id = "file")

training_dynamics %>%
  group_by(eval_verb, step) %>%
  summarize(
    n = n(),
    diff = mean(loss_transitive - loss_causative)
  ) %>% View()
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
  facet_wrap(~eval_verb, scales = "free", nrow = 1) +
  scale_x_continuous(
    labels = \(x) ifelse(x == 0, "0", label_number(scale = 1e-3, suffix = "k")(x)),
    limits = c(0, 12000),
    breaks = c(0, 3000, 6000, 9000, 12000)
  ) +
  facetted_pos_scales(
    y = list(
      eval_verb == "cry" ~ scale_y_continuous(limits = c(-0.2, 1.2)),
      eval_verb == "sleep" ~ scale_y_continuous(limits = c(-0.2, 1), breaks = pretty_breaks()),
      eval_verb == "smile" ~ scale_y_continuous(limits = c(-0.1, 1.5), breaks = pretty_breaks()),
      eval_verb == "sneeze" ~ scale_y_continuous(limits = c(-0.2, 1.6), breaks = pretty_breaks())
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

ggsave("figures/precondition-exp-surprisal-diff.pdf", width = 13.90, height = 2.73, dpi=300, device = cairo_pdf)

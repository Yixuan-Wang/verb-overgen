library(arrow)
library(dplyr)
library(ggplot2)
library(lme4)
library(lmerTest)

data_path <- "./var/result/df_preemption_nll.parquet"
dataset <- read_parquet(data_path)

mdl <- lmer(
    nll ~ train_set + epoch + (1 | verb) + (1 | run_seed),
    data = dataset,
    REML = FALSE,
    control = lmerControl(optimizer = "bobyqa")
)

mdl_no_trainset <- lmer(
    nll ~ epoch + (1 | verb) + (1 | run_seed),
    data = dataset,
    REML = FALSE,
    control = lmerControl(optimizer = "bobyqa")
)

anova(mdl, mdl_no_trainset)


mdl_no_epoch <- lmer(
    nll ~ train_set + (1 | verb) + (1 | run_seed),
    data = dataset,
    REML = FALSE,
    control = lmerControl(optimizer = "bobyqa")
)
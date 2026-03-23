library(arrow)
library(dplyr)
library(lme4)

dataset <- read_parquet("./var/results/testset_gemini_voi-5_tag-260226.parquet")


dataset_with_context <- dataset |> filter(run_flavor != "full", kind == "sentence_context_transitive")

model_with_context <- lmer(nll ~ run_flavor * verb + (1 | run_seed:hash_context), data = dataset_with_context)
summary(model_with_context)

dataset_without_context <- dataset |> filter(run_flavor != "full", kind == "sentence_transitive")
model_without_context <- lmer(nll ~ run_flavor * verb + (1 | run_seed:hash_context), data = dataset_without_context)
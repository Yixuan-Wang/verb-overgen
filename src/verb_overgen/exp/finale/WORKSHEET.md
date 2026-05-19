# Experiment Conditions Worksheet

## Terminology 

Training Run Conditions: Manipulation, Target, Scope.

Manipulation
- `remove`. Deletion of instances.
- `substitute`. Replacement with alternatives, namely replacing preempting causatives with transitive instances; such transitives are not grammatical, and are add here for ablation.
- `none`. Baseline, the full training set.

Target:
- `p`. Preemption. Periphrastic causative constructions, e.g., "make laugh". They are hypothesized to be preempting transitive constructions.
- `e`. Entrenchment minus preemption. All other instances of the target verb, *excluding* those in the `p` category. Since preemption effect is hypothesized to be a subset of the entrenchment effect, the `e` category captures the complement of `p`.
- Empty string means not applicable, e.g., for `none` manipulation.


Scope:
- `per-verb`. The manipulation is applied to targets of a specific verb, e.g., "laugh".
- `all-verb`. The manipulation is applied globally, e.g., to all periphrastic causative constructions regardless of the verb, if the target is `p`.
- `except-verb`. The manipulation is applied to all verbs *except* a specific target verb, e.g., "laugh".
- Empty string means not applicable, e.g., for `none` manipulation.

The detailed `remove` results of target and scope is as follows.

✅ means untouched, ❌ means removed, 🔽 in `e` means removed the same size as in corresponding `p` removals.

|     | causative, specific verb | causative, other verbs | other constructions, specific verb | other constructions, other verbs |
| --: | --- | --- | --- | --- |
| `p :: per-verb` | ❌ | ✅ | ✅ | ✅ |
| `e :: per-verb` | ✅ | ✅ | 🔽 | ✅ |
| `p :: all-verb` | ❌ | ❌ | ✅ | ✅ |
| `e :: all-verb` | ✅ | ✅ | 🔽 | 🔽 |
| `p :: except-verb` |  ✅ | ❌ | ✅ | ✅ |
| `e :: except-verb` | ✅ | ✅ | ✅ | 🔽 |

`none` and `all-verb` are repeated once per seed, `per-verb` and `except-verb` are repeated once per verb per seed.
We have the same 5 seeds shared across all experiments.


Evaluation Conditions: Construction, Context.

Construction:
- `causative`. Causative constructions, e.g., "make laugh".
- `transitive`. Transitive constructions, e.g., "laugh the baby". These are not grammatical (overgeneralization).

Context:
- `with_context`. The test instance is preceded by a linguistic context that explains a frame, e.g., "John told Jane a joke. John made Jane laugh.". 
- `no_context`. The test instance is not preceded by any linguistic context, e.g., "John made Jane laugh.".

NLL and SLOR are calculated only on the test instance. Syntactic Log-odds Ratio (SLOR) is defined as:

$$ \text{SLOR}(x) = \frac{1}{|x|} \left( \log P_{LM}(x) - \log P_{U}(x) \right) $$

Variable Definitions:
• $x$: The input sequence.
• $P_{LM}(x)$: The probability of the sequence $x$ assigned by the language model.
• $P_{U}(x)$: The probability of the sequence $x$ assigned by the unigram model.

## Available Conditions

These conditions (formatted as `manipulation :: target :: scope`) describe how the training data was modified.

`none`
`remove :: p :: per-verb`
`remove :: e :: per-verb`
`remove :: p :: all-verb`
`remove :: e :: all-verb`
`remove :: p :: except-verb`
`remove :: e :: except-verb`
`remove :: p+e :: lexical-only`
`substitute :: p :: per-verb`

These conditions (formatted as `construction :: context`) describe the test sets used to probe the model's knowledge.

`causative :: no_context`
`causative :: with_context`
`transitive :: no_context`
`transitive :: with_context`

For verbs, we have 10 verbs of interest.

| Verb | Note |
| --- | --- |
| laugh | The verb of most interest. |
| ~~sing~~ | ~~This verb also have heavy transitive usage under a different lexical sense.~~ |
| cry | |
| sleep | |
| smile | |
| go | The most frequent verb in the dataset. |
| ~~look~~ | ~~Somehow being used heavily as a transitive in the child speech dataset.~~ |
| sneeze | |
| ~~blink~~ | ~~A very rare verb in the dataset. Also can be used transitively.~~ |
| ~~rise~~ | ~~It shouldn't have been used in a periphrastic causative construction. Kept here for comparison.~~ |


## Planned Analyses

### Basics

- We will use `with_context` as the primary eval condition. `no_context` will be a separate experiment.
- Preemption effect on overgeneralization will always be measured on `transitive`s for the `remove` manipulation, since the transitives are the "overgeneralization" instances.
- To measure the preemption effect, we will compare `remove :: p` with `remove :: e` on `transitive :: with_context`. If removing preempting instances increases the transitive NLL (or decreases the SLOR)
than removing entrenchment-minus-preemption instances, then we have evidence for the preemption effect, and it attributes to preemption, not entrenchment.




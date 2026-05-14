# Reconciling Exp 1 (per-verb p vs e) and Exp 2 (lex/abstract decomposition)

Notes on resolving an apparent inconsistency between
`preemption_hypothesis_test.py` and `lexical_vs_abstract_preemption.py`.
Date: 2026-05-13. **v3** — supersedes the 2026-05-12 raw-contrast reading.
EXP 2 is now a difference-in-differences against the count-matched
`remove::e::*` arm at every scope.

## The two experiments

**Exp 1 — `preemption_hypothesis_test.py`**
Contrast `(remove::p::per-verb − remove::e::per-verb)` on
`transitive :: with_context` NLL at the final epoch, per verb. Both arms
remove the **same count** of laugh-instances; they differ only in *which kind*
(lex-p vs random non-preempting entrenchment). Abstract preemption is intact
in both arms.

Result: `(p − e) > 0` for most verbs (entrenchment-removal more damaging
than lex-p-removal per matched instance), with sign flips at `go` (highest
N_p) and `rise` (lowest N_p).

**Exp 2 (v3, DiD) — `lexical_vs_abstract_preemption.py`**
Six cells: `{p, e} × {all-verb, per-verb, except-verb}` (count-matched at
each scope). Five DiD contrasts per verb on the final-epoch
`transitive :: with_context` cell:

| effect | DiD | what it tests |
|---|---|---|
| `lexical_alone`      | (except − all)_p − (except − all)_e | lex-p effect, abstract absent |
| `abstract_alone`     | (per − all)_p   − (per − all)_e    | abstract-p effect, lex absent |
| `total`              | all_e − all_p                       | full preemption vs entrenchment |
| `lex_given_abstract` | per_e − per_p                       | marginal lex-p, abstract still on |
| `abstract_given_lex` | except_e − except_p                 | marginal abstract-p, lex still on |

Sign convention: positive = preemption-specific effect beyond entrenchment
(i.e., `e > p` on transitive NLL — removing p damaged anti-overgen more
than removing matched e). This **reverses** Exp 1's `(p − e)` direction.

## Why the v2 (raw-EXP-2) reading was wrong

The v2 EXP 2 used `none` and `remove::p::*` cells only, computing contrasts
like `none − all-verb::p` against zero. But preemption is a **subset of
entrenchment**, so any such contrast mixes (a) the preemption-specific
signal with (b) the generic entrenchment / data-volume effect of having
N_p fewer training tokens of the verb. The v3 DiD includes the
count-matched `remove::e::*` arm at every scope, so each preemption
contrast is net of entrenchment.

## Verb-by-verb (NLL, final epoch, `transitive :: with_context`)

Positive = preemption-specific effect beyond entrenchment.
Stars: `***` p<.001, `**` p<.01, `*` p<.05, n.s. otherwise (mvt-adjusted).

| Verb   | lexical_alone | abstract_alone | total       | lex \| abstract | abstract \| lex |
|--------|--:|--:|--:|--:|--:|
| blink  | −0.040 n.s.   | +0.152 ***     | +0.124 ***  | −0.027 n.s.     | +0.164 ***      |
| cry    | +0.095 **     | +0.174 ***     | +0.024 n.s. | **−0.150 ***    | −0.071 **       |
| go     | −0.000 n.s.   | −0.032 n.s.    | +0.028 n.s. | +0.060 *        | +0.028 n.s.     |
| laugh  | +0.129 ***    | +0.026 n.s.    | +0.022 n.s. | −0.004 n.s.     | **−0.108 ***    |
| look   | −0.012 n.s.   | +0.017 n.s.    | −0.055 *    | −0.072 **       | −0.043 n.s.     |
| rise   | −0.018 n.s.   | −0.027 n.s.    | +0.132 ***  | +0.159 ***      | +0.150 ***      |
| sing   | +0.035 n.s.   | +0.108 ***     | +0.092 ***  | −0.016 n.s.     | +0.057 *        |
| sleep  | +0.110 **     | +0.032 n.s.    | +0.026 n.s. | −0.006 n.s.     | **−0.084 ***    |
| smile  | +0.083 *      | +0.118 ***     | +0.033 n.s. | **−0.085 ***    | −0.050 *        |
| sneeze | −0.072 *      | +0.047 n.s.    | **−0.116 ***| **−0.163 ***    | −0.044 n.s.     |

Algebra: `total = lexical_alone + abstract_given_lex = abstract_alone +
lex_given_abstract`. Verified for every verb (laugh: 0.129 + (−0.108) =
0.021 ≈ total 0.022).

## Reconciled story (v3)

### Exp 1 and `lex_given_abstract` are the same contrast

Exp 1's `(p − e)` at per-verb scope is `lex_given_abstract` with its sign
flipped — both compare `remove::p::per-verb` against the count-matched
`remove::e::per-verb`, both keep abstract preemption intact. So the
`lex_given_abstract` column above is Exp 1, restated inside the bigger
model. Exp 1's "entrenchment ≥ lex-p per matched instance" finding is not
independently confirmed by EXP 2 — it's *embedded* in it.

### The genuinely new piece: `lexical_alone` and `total`

The DiD also asks *when no abstract signal is available, does lex-p
preempt?* — that's `lexical_alone`. Answer: yes for cry, laugh, sleep,
smile (significant positive). But the same verbs go null or negative on
`lex_given_abstract`. So **lex preemption has a real preemption-specific
effect in isolation, but is redundant with abstract preemption when both
are present.**

### What this does to the preemption hypothesis

- Net `total` (preemption above entrenchment): significantly positive only
  for blink, sing (and the suspect rise). Null for go, cry, laugh, sleep,
  smile. Significantly **negative** for sneeze and look.
- Where preemption-specific effects show up at all, lex and abstract are
  **substitutes**, not additive contributors. Having one is roughly as
  good as having both.
- The "both signals matter and add up" story from v2 was largely an
  entrenchment / data-volume artifact. The raw v2 `lex_given_abstract`
  numbers for laugh (+0.050 \*\*\*), cry (+0.084 \*\*\*), sneeze (+0.096
  \*\*\*) all flip to null or strongly negative under DiD.
- For **laugh specifically** (the focal verb): no preemption-specific net
  effect. Lex-p preempts in isolation, but adds nothing on top of
  abstract; in fact `abstract_given_lex` is significantly negative,
  suggesting having laugh's own lex-p slightly *blunts* the abstract
  signal it would otherwise enjoy.

## Verb-specific caveats (updated)

- **sneeze** — anti-hypothesis across every contrast involving lex.
  `total` strongly negative (−0.116 \*\*\*), `lex_given_abstract` strongly
  negative (−0.163 \*\*\*), `lexical_alone` negative (−0.072 \*). Treat
  as an active counter-finding worth investigating, not noise.
- **sing, sleep** — the v2 caveat (alt transitive senses) cited the raw
  negative `lex_given_abstract` (−0.058 \*\*\*, −0.038 \*\*\*) as
  evidence of an alt-sense confound. Under DiD both are null
  (−0.016 n.s., −0.006 n.s.), so the alt-sense caveat now has less to
  explain. sleep does show `abstract_given_lex` significantly negative
  (−0.084 \*\*\*) — a milder version of laugh's interaction pattern.
- **go** — `lex_given_abstract` modestly positive (+0.060 \*), all other
  contrasts null. The v2 narrative ("lex-p dominates for go because N_p
  is large") was too strong; under DiD go contributes only a single
  modest preemption signal.
- **rise** — strongly positive everywhere, but worksheet flags it as
  not-a-real-causative. Kept for comparison only; do not interpret.
- **blink** — purely abstract: `abstract_alone` +0.152 \*\*\* and
  `abstract_given_lex` +0.164 \*\*\*, all lex contrasts null. The net
  positive signal rides entirely on cross-verb generalization.

## Suggested follow-ups

- **Investigate sneeze.** Anti-preemption at every lex-involving contrast
  deserves a closer look. Check CHILDES "make sneeze" contexts — do its
  periphrastic causatives co-occur with patterns that *enable* transitive
  overgen rather than suppress it?
- **Random-effects structure caveat (unchanged from v2).** For
  `all-verb::p` and `all-verb::e`, one model per seed evaluates all 10
  verbs, so `run_seed` is correlated across verbs in those cells.
  `(1 | run_seed:eval_verb:eval_index_verb_context)` treats them as
  independent. Within-verb emmeans are fine; cross-verb pooled claims
  would want a different structure.
- **`no_context` analogue.** Repeat the DiD on `transitive :: no_context`
  to check whether the lex/abstract redundancy pattern is context-driven.

## Code changes (v3)

`src/verb_overgen/exp/finale/lexical_vs_abstract_preemption.py`:

- `load_data` loads 6 cells (`{p, e} × {all-verb, per-verb, except-verb}`);
  `none` dropped (it cancels in every DiD).
- R model uses a single `cell` factor with 6 pinned levels in
  `metric ~ cell * eval_verb + (1|run_seed:eval_verb:eval_index_verb_context)`.
- Five DiD contrasts as length-6 vectors; `abstract_given_lex` added per
  the v2 follow-up.
- Sign convention reversed from v2: positive = preemption-specific effect
  beyond entrenchment (= `e > p` direction; flips Exp 1's `(p − e)` sign).
- Output figure bumped to `20260513.v3.lex-vs-abstract-preemption-did_*.pdf`.

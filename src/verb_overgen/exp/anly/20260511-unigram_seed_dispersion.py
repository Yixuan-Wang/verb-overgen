"""20260511 — Seed dispersion of unigram counts for `remove :: e :: per-verb`.

For each of the 10 target verbs, summarize how the 5 seeds differ:
  1. Target verb's own token count per seed.
  2. Top-N token counts per seed (log scale).
  3. Pairwise Jensen-Shannon divergence between seeds' full unigram distributions.

Outputs CSV tables and PNG figures under `var/results/v2/anly/20260511-unigram_seed_dispersion/`.
Note: counts are for the bare verb token only — inflected forms (e.g. "goes", "went") are distinct
word-level tokens and are not aggregated here.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.spatial.distance import jensenshannon

ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "var/results/v2/stats-unigram-merged"
TOK_PATH = ROOT / "data/tokenizer/1024-wordlevel/tokenizer.json"
OUT_DIR = ROOT / "var/results/v2/anly/20260511-unigram_seed_dispersion"

CONDITION = {"manipulation": "remove", "scope": "per-verb", "target": "e"}
TOP_N = 20


def load_metadata() -> list[dict]:
    rows = [json.loads(l) for l in (DATA_DIR / "metadata.jsonl").open()]
    return [r for r in rows if all(r.get(k) == v for k, v in CONDITION.items())]


def load_tokenizer() -> tuple[dict[str, int], dict[int, str]]:
    vocab: dict[str, int] = json.loads(TOK_PATH.read_text())["model"]["vocab"]
    return vocab, {i: t for t, i in vocab.items()}


def load_seed_matrix(rows: list[dict]) -> tuple[list[int], pl.DataFrame]:
    """All parquets share the same vocab-aligned token_id range — left-join suffices."""
    rows = sorted(rows, key=lambda r: r["seed"])
    seeds = [r["seed"] for r in rows]
    base: pl.DataFrame | None = None
    for r in rows:
        df = (
            pl.read_parquet(DATA_DIR / r["path"])
            .sort("token_id")
            .rename({"count": f"count_seed_{r['seed']}"})
        )
        base = df if base is None else base.join(df, on="token_id", how="left")
    assert base is not None
    return seeds, base.fill_null(0)


def js_matrix(seeds: list[int], wide: pl.DataFrame) -> np.ndarray:
    counts = np.stack(
        [wide[f"count_seed_{s}"].to_numpy().astype(np.float64) for s in seeds]
    )
    probs = counts / counts.sum(axis=1, keepdims=True)
    n = len(seeds)
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = jensenshannon(probs[i], probs[j], base=2)
            m[i, j] = m[j, i] = float(d) ** 2  # distance^2 = JS divergence
    return m


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vocab, id_to_tok = load_tokenizer()
    rows = load_metadata()
    if not rows:
        raise SystemExit(f"No metadata rows matched {CONDITION}")

    by_verb: dict[str, list[dict]] = {}
    for r in rows:
        by_verb.setdefault(r["target_verb"], []).append(r)
    target_verbs = sorted(by_verb)

    target_counts: dict[str, dict[int, int]] = {}
    topn_tables: dict[str, pl.DataFrame] = {}
    js_mats: dict[str, tuple[list[int], np.ndarray]] = {}

    for verb in target_verbs:
        seeds, wide = load_seed_matrix(by_verb[verb])

        tid = vocab.get(verb)
        if tid is None:
            print(f"[warn] target verb {verb!r} not in tokenizer vocab")
            target_counts[verb] = {s: 0 for s in seeds}
        else:
            row = wide.filter(pl.col("token_id") == tid)
            target_counts[verb] = {
                s: int(row[f"count_seed_{s}"].item()) if row.height else 0 for s in seeds
            }

        seed_cols = [f"count_seed_{s}" for s in seeds]
        topn = (
            wide.with_columns(pl.mean_horizontal(seed_cols).alias("_mean"))
            .sort("_mean", descending=True)
            .head(TOP_N)
            .with_columns(
                pl.col("token_id")
                .map_elements(lambda i: id_to_tok.get(i, f"<{i}>"), return_dtype=pl.String)
                .alias("token")
            )
            .select(["token_id", "token", *seed_cols])
        )
        topn_tables[verb] = topn

        js_mats[verb] = (seeds, js_matrix(seeds, wide))

    # ---------- CSV tables ----------
    pl.DataFrame(
        [
            {"target_verb": v, "token_id": vocab.get(v), **{f"seed_{s}": c for s, c in target_counts[v].items()}}
            for v in target_verbs
        ]
    ).write_csv(OUT_DIR / "target_verb_counts.csv")

    pl.concat(
        [
            topn_tables[v].with_columns(pl.lit(v).alias("target_verb")).select(
                ["target_verb", "token_id", "token", *[c for c in topn_tables[v].columns if c.startswith("count_seed_")]]
            )
            for v in target_verbs
        ]
    ).write_csv(OUT_DIR / f"top{TOP_N}_per_verb.csv")

    js_rows = []
    for v, (seeds, m) in js_mats.items():
        for i, si in enumerate(seeds):
            for j, sj in enumerate(seeds):
                js_rows.append({"target_verb": v, "seed_i": si, "seed_j": sj, "js": float(m[i, j])})
    pl.DataFrame(js_rows).write_csv(OUT_DIR / "js_divergence_pairs.csv")

    # ---------- plots ----------
    n = len(target_verbs)
    ncols = 5
    nrows = (n + ncols - 1) // ncols

    # Fig 1: target verb count per seed
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows))
    for ax, verb in zip(axes.flat, target_verbs):
        counts = target_counts[verb]
        seeds = list(counts)
        ax.bar([str(s) for s in seeds], [counts[s] for s in seeds], color="steelblue")
        ax.set_title(f"{verb} (tid={vocab.get(verb)})")
        ax.set_xlabel("seed")
        ax.tick_params(axis="x", rotation=45)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle("Target verb token count per seed — remove :: e :: per-verb")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "target_verb_count.png", dpi=150)
    plt.close(fig)

    # Fig 2: top-N tokens per seed (log scale lines)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows))
    for idx, (ax, verb) in enumerate(zip(axes.flat, target_verbs)):
        seeds, _ = js_mats[verb]
        topn = topn_tables[verb]
        tokens = topn["token"].to_list()
        for s in seeds:
            ax.plot(
                range(len(tokens)),
                topn[f"count_seed_{s}"].to_numpy(),
                marker="o",
                markersize=3,
                linewidth=0.8,
                label=str(s),
            )
        ax.set_title(verb)
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=70, fontsize=6)
        ax.set_yscale("log")
        if idx == 0:
            ax.legend(title="seed", fontsize=6, ncol=2)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle(f"Top-{TOP_N} token counts per seed (log scale) — remove :: e :: per-verb")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"top{TOP_N}_per_seed.png", dpi=150)
    plt.close(fig)

    # Fig 3: JS divergence heatmaps
    vmax = max(m.max() for _, m in js_mats.values())
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows))
    im = None
    for ax, verb in zip(axes.flat, target_verbs):
        seeds, m = js_mats[verb]
        im = ax.imshow(m, vmin=0, vmax=vmax, cmap="viridis")
        ax.set_xticks(range(len(seeds)))
        ax.set_yticks(range(len(seeds)))
        ax.set_xticklabels(seeds, rotation=45, fontsize=7)
        ax.set_yticklabels(seeds, fontsize=7)
        ax.set_title(verb)
        for i in range(len(seeds)):
            for j in range(len(seeds)):
                ax.text(
                    j, i, f"{m[i, j]:.1e}",
                    ha="center", va="center", fontsize=5,
                    color="white" if m[i, j] < vmax * 0.5 else "black",
                )
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle("Pairwise JS divergence (base 2) between seeds — remove :: e :: per-verb")
    fig.tight_layout(rect=(0, 0, 0.95, 1))
    if im is not None:
        cax = fig.add_axes((0.96, 0.15, 0.012, 0.7))
        fig.colorbar(im, cax=cax, label="JS divergence")
    fig.savefig(OUT_DIR / "js_divergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[done] outputs written to {OUT_DIR}")
    print(f"  target verbs : {target_verbs}")
    js_vals = np.array([m[np.triu_indices_from(m, k=1)].mean() for _, m in js_mats.values()])
    print(f"  mean JS per verb (across off-diag pairs):")
    for v, x in zip(target_verbs, js_vals):
        print(f"    {v:>8s}: {x:.3e}")


if __name__ == "__main__":
    main()

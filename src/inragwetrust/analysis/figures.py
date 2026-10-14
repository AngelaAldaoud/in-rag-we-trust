"""Final figures for the paper and slides (Block F4).

Every figure reads only the committed CSVs in results/analysis/, so it shows exactly the frozen numbers.
Figures are drawn at the text width of the paper's Springer template (D275), read from paper/*.tex.
Run with:
    uv run python -m inragwetrust.analysis.figures
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

RESULTS = Path("results/analysis")
FIGURES = Path("results/figures")
PAPER = Path("paper")

MODELS = ("llama", "mistral", "qwen")
LABELS = {"llama": "Llama-3.1-8B", "mistral": "Mistral-7B-v0.3", "qwen": "Qwen2.5-7B"}
COLORS = {"llama": "#0072B2", "mistral": "#D55E00", "qwen": "#009E73"}
MARKERS = {"llama": "o", "mistral": "s", "qwen": "^"}
LINES = {"llama": "-", "mistral": "--", "qwen": ":"}
OFFSET = {"llama": -0.02, "mistral": 0.0, "qwen": 0.02}
METAS = ("consistency", "reliability")
RHOS = (0.0, 0.25, 0.5, 0.75, 1.0)
XLABEL = r"$\rho$ (share of poisoned passages)"
EFFECTS = (("poisoned", r"$\Delta$ASR"), ("correct", r"$\Delta$Acc"), ("abstain", r"$\Delta$Abst"))
TYPE_MARKERS = ("o", "s", "^", "D")
OUTCOMES = (("Poisoned", "0.15", ""), ("Abstain", "0.55", "//"), ("Correct", "0.85", ".."), ("Other", "white", "xx"))
NAMES = ("fig1_acc_asr_vs_rho", "fig2_prompt_effects", "fig3_flag_vs_baseline", "fig4_type_model",
         "fig5_transitions")
TEXT_WIDTH = {"llncs": 4.8, "sn-jnl": 5.15}
TWO_COLUMN = ("twocolumn", "iicol")
DOCCLASS = re.compile(r"^[ \t]*\\documentclass(?:\[([^\]]*)\])?\{([^}]+)\}", re.MULTILINE)
GREY = "0.35"
RC = {"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
      "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
      "axes.spines.top": False, "axes.spines.right": False}


def text_width(paper: Path = PAPER) -> tuple[str, float]:
    """Document class and text width (inches) of the paper template, from its first active \\documentclass line."""
    for tex in sorted(Path(paper).glob("*.tex")):
        m = DOCCLASS.search(tex.read_text(encoding="utf-8", errors="ignore"))
        if m is None:
            continue
        opts, cls = (m.group(1) or "").split(","), m.group(2).strip()
        if any(o.strip() in TWO_COLUMN for o in opts):
            raise ValueError(f"{tex}: two-column option in {opts}; figures are sized for one column")
        if cls not in TEXT_WIDTH:
            raise ValueError(f"{tex}: unknown class {cls}; known {sorted(TEXT_WIDTH)}")
        return cls, TEXT_WIDTH[cls]
    raise FileNotFoundError(f"no active \\documentclass line in {paper}/*.tex")


def need(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}; has {list(df.columns)}")


def pct(values) -> np.ndarray:
    return 100 * np.asarray(values, dtype=float)


def rho_axis(ax, label: bool = True) -> None:
    ax.set_xticks(RHOS)
    ax.set_xticklabels(["0", ".25", ".5", ".75", "1"])
    ax.set_xlim(-0.06, 1.06)
    if label:
        ax.set_xlabel(XLABEL)


def ci_points(ax, g: pd.DataFrame, y: str, lo: str, hi: str, offset: float = 0.0, **kw):
    v = pct(g[y])
    err = np.vstack([np.maximum(v - pct(g[lo]), 0), np.maximum(pct(g[hi]) - v, 0)])
    return ax.errorbar(g["rho"] + offset, v, yerr=err, capsize=2, elinewidth=0.8, ms=4, **kw)


def model_handles() -> list[Line2D]:
    return [Line2D([], [], color=COLORS[m], ls=LINES[m], marker=MARKERS[m], ms=4, label=LABELS[m]) for m in MODELS]


def fig_rho_curves(desc: pd.DataFrame, crossover: pd.DataFrame, closed_book: pd.DataFrame,
                   width: float = TEXT_WIDTH["llncs"]) -> Figure:
    """Fig 1 (RQ1): Acc and ASR vs rho per model, all prompts pooled as in F1, Wilson 95% bands.
    Vertical lines: the first tested rho at which ASR exceeds Acc (rules, and the judge-based upper bound)."""
    need(desc, ["model", "prompt", "rho", "asr", "asr_lo", "asr_hi", "acc", "acc_lo", "acc_hi"], "descriptive_by_rho")
    need(crossover, ["model", "prompt", "scoring", "crossover"], "explore_crossover")
    need(closed_book, ["model", "correct"], "explore_closed_book")
    d = desc[desc["prompt"] == "all"]
    cr = crossover[crossover["prompt"] == "all"].set_index(["model", "scoring"])["crossover"]
    cb = closed_book.set_index("model")["correct"]
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(width, 2.0), sharey=True, layout="constrained")
        for i, (ax, m) in enumerate(zip(axes, MODELS)):
            g, c = d[d["model"] == m].sort_values("rho"), COLORS[m]
            for col, ls, face in (("asr", "-", c), ("acc", "--", "white")):
                ax.fill_between(g["rho"], pct(g[f"{col}_lo"]), pct(g[f"{col}_hi"]), color=c, alpha=0.18, lw=0)
                ax.plot(g["rho"], pct(g[col]), ls=ls, marker=MARKERS[m], color=c, mfc=face, ms=4, lw=1.4)
            ax.axhline(100 * float(cb[m]), color=GREY, ls=":", lw=1)
            rules, upper = cr.get((m, "rules")), cr.get((m, "upper bound"))
            if pd.notna(rules):
                ax.axvline(rules, color="black", lw=0.8)
            if pd.notna(upper) and upper != rules:
                ax.axvline(upper, color="black", lw=0.8, ls=(0, (4, 2)))
            ax.set_title(LABELS[m], color=c)
            ax.set_ylim(0, 100)
            rho_axis(ax, label=i == 1)
        axes[0].set_ylabel("Share of answers (%)")
        handles = [Line2D([], [], color="black", marker="o", ms=4, label="ASR (poisoned answer)"),
                   Line2D([], [], color="black", ls="--", marker="o", mfc="white", ms=4, label="Acc (correct answer)"),
                   Line2D([], [], color=GREY, ls=":", label="Closed-book Acc"),
                   Line2D([], [], color="black", lw=0.8, label=r"First $\rho$ with ASR $>$ Acc (rules)"),
                   Line2D([], [], color="black", lw=0.8, ls=(0, (4, 2)),
                          label=r"First $\rho$ with ASR $>$ Acc (upper bound)")]
        fig.legend(handles=handles, loc="outside lower center", ncol=3 if width >= 5.5 else 2, frameon=False)
    return fig


def fig_prompt_effects(pe: pd.DataFrame, width: float = TEXT_WIDTH["llncs"]) -> Figure:
    """Fig 2 (RQ2): meta-prompt minus plain, paired by question, clustered bootstrap 95% CIs."""
    need(pe, ["model", "rho", "meta", "metric", "delta", "lo", "hi"], "explore_prompt_effects")
    missing = sorted({k for k, _ in EFFECTS} - set(pe["metric"]))
    if missing:
        raise ValueError(f"explore_prompt_effects: missing metrics {missing}")
    with plt.rc_context(RC):
        fig, axes = plt.subplots(3, 2, figsize=(width, 4.4), sharex=True, sharey="row", layout="constrained")
        for i, (metric, label) in enumerate(EFFECTS):
            for j, meta in enumerate(METAS):
                ax = axes[i, j]
                ax.axhline(0, color=GREY, lw=0.8)
                for m in MODELS:
                    g = pe[(pe["model"] == m) & (pe["meta"] == meta) & (pe["metric"] == metric)].sort_values("rho")
                    ci_points(ax, g, "delta", "lo", "hi", OFFSET[m], color=COLORS[m], ls=LINES[m],
                              marker=MARKERS[m], lw=1.2)
                rho_axis(ax, label=i == len(EFFECTS) - 1)
                if i == 0:
                    ax.set_title(f"{meta.capitalize()} prompt minus plain")
                if j == 0:
                    ax.set_ylabel(f"{label} (points)")
        fig.legend(handles=model_handles(), loc="outside lower center", ncol=3, frameon=False)
    return fig


def fig_flag_baseline(fb: pd.DataFrame, width: float = TEXT_WIDTH["llncs"]) -> Figure:
    """Fig 3 (RQ3): flag rate minus the same questions' rho = 0 flag rate, clustered bootstrap 95% CIs."""
    need(fb, ["model", "rho", "delta", "lo", "hi"], "explore_flag_baseline")
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(0.75 * width, 3.0), layout="constrained")
        ax.axhline(0, color=GREY, lw=0.8)
        for m in MODELS:
            g = fb[fb["model"] == m].sort_values("rho")
            ci_points(ax, g, "delta", "lo", "hi", OFFSET[m], color=COLORS[m], ls=LINES[m], marker=MARKERS[m], lw=1.2)
        rho_axis(ax)
        ax.set_ylabel("Flag rate minus own\n" + r"$\rho$ = 0 rate (points)")
        fig.legend(handles=model_handles(), loc="outside lower center", ncol=3, frameon=False)
    return fig


def fig_type_model(tm: pd.DataFrame, width: float = TEXT_WIDTH["llncs"]) -> Figure:
    """Fig 4 (RQ4): ASR per poison type, one panel per model, Wilson 95% CIs, axis from 0."""
    need(tm, ["model", "poison_type", "rho", "poisoned", "poisoned_lo", "poisoned_hi"], "explore_type_model")
    types = sorted(tm["poison_type"].unique())
    if len(types) > len(TYPE_MARKERS):
        raise ValueError(f"explore_type_model: {len(types)} poison types, at most {len(TYPE_MARKERS)} supported")
    offsets = np.linspace(-0.045, 0.045, len(types))
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(width, 2.8), sharey=True, layout="constrained")
        for i, (ax, m) in enumerate(zip(axes, MODELS)):
            for k, t in enumerate(types):
                g = tm[(tm["model"] == m) & (tm["poison_type"] == t)].sort_values("rho")
                ci_points(ax, g, "poisoned", "poisoned_lo", "poisoned_hi", offsets[k], color=COLORS[m], ls="none",
                          marker=TYPE_MARKERS[k], mfc="white" if k % 2 else COLORS[m])
            ax.set_title(LABELS[m], color=COLORS[m])
            ax.set_ylim(0, 100)
            rho_axis(ax, label=i == 1)
        axes[0].set_ylabel("ASR (%)")
        handles = [Line2D([], [], color=GREY, ls="none", marker=TYPE_MARKERS[k], ms=4,
                          mfc="white" if k % 2 else GREY, label=t) for k, t in enumerate(types)]
        fig.legend(handles=handles, loc="outside lower center", ncol=len(types), frameon=False, title="Poison type")
    return fig


def fig_transitions(tr: pd.DataFrame, width: float = TEXT_WIDTH["llncs"]) -> Figure:
    """Fig 5 (RQ2 mechanism): what the meta-prompt answer is when the plain answer was poisoned, rho > 0."""
    need(tr, ["model", "meta", "direction", "other_outcome", "share"], "explore_transitions")
    d = tr[tr["direction"].str.startswith("plain Poisoned")]
    if d.empty:
        raise ValueError("explore_transitions: no 'plain Poisoned -> meta' rows")
    rows = [(m, meta) for m in MODELS for meta in METAS]
    box = {"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.9}
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(width, 2.9), layout="constrained")
        for y, (m, meta) in enumerate(rows):
            g = d[(d["model"] == m) & (d["meta"] == meta)].set_index("other_outcome")["share"]
            share = {o: float(g.get(o, 0.0)) for o, _, _ in OUTCOMES[:3]}
            share["Other"] = max(float(g.sum()) - sum(share.values()), 0.0)
            left = 0.0
            for o, face, hatch in OUTCOMES:
                w = 100 * share[o]
                ax.barh(y, w, left=left, color=face, hatch=hatch, edgecolor="black", lw=0.6, height=0.7)
                if w >= 8:
                    dark = o == "Poisoned"
                    ax.text(left + w / 2, y, f"{w:.0f}%", ha="center", va="center", fontsize=7,
                            color="white" if dark else "black", bbox=None if dark else box)
                left += w
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{LABELS[m]} · {meta}" for m, meta in rows])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Answer under the meta-prompt, for answers poisoned under plain (%)")
        fig.legend(handles=[Patch(facecolor=f, hatch=h, edgecolor="black", label=o) for o, f, h in OUTCOMES],
                   loc="outside lower center", ncol=len(OUTCOMES), frameon=False)
    return fig


def build_all(results: Path = RESULTS, width: float = TEXT_WIDTH["llncs"]) -> dict[str, Figure]:
    def r(name: str) -> pd.DataFrame:
        return pd.read_csv(Path(results) / f"{name}.csv")

    figs = (fig_rho_curves(r("descriptive_by_rho"), r("explore_crossover"), r("explore_closed_book"), width),
            fig_prompt_effects(r("explore_prompt_effects"), width),
            fig_flag_baseline(r("explore_flag_baseline"), width),
            fig_type_model(r("explore_type_model"), width), fig_transitions(r("explore_transitions"), width))
    return dict(zip(NAMES, figs))


def save(fig: Figure, out: Path, name: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.pdf", out / f"{name}.png"]
    with plt.rc_context(RC):
        fig.savefig(paths[0], bbox_inches="tight", metadata={"CreationDate": None})
        fig.savefig(paths[1], bbox_inches="tight", dpi=300)
    plt.close(fig)
    return paths


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--out", default=str(FIGURES))
    ap.add_argument("--paper", default=str(PAPER))
    ap.add_argument("--width", type=float, default=None, help="override the template text width (inches)")
    a = ap.parse_args(argv)
    if a.width is None:
        cls, width = text_width(Path(a.paper))
        print(f"Template class {cls}: text width {width} in ({width * 2.54:.1f} cm)")
    else:
        width = a.width
        print(f"Text width set by hand: {width} in")
    plt.switch_backend("Agg")
    for name, fig in build_all(Path(a.results), width).items():
        for p in save(fig, Path(a.out), name):
            print("Written", p)


if __name__ == "__main__":
    main()
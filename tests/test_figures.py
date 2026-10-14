import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from inragwetrust.analysis import figures as fg

TYPES = ["authority", "fabrication", "framing", "swap"]
OUT = {"Poisoned": 0.70, "Abstain": 0.15, "Correct": 0.10, "NoAnswer": 0.05}


def tables() -> dict[str, pd.DataFrame]:
    rhos = list(fg.RHOS)
    desc = pd.DataFrame([{"model": m, "prompt": p, "rho": r, "n": 100, "asr": r / 2, "asr_lo": r / 2 * 0.9,
                          "asr_hi": r / 2 + 0.02, "acc": 0.6 - r / 2, "acc_lo": 0.55 - r / 2, "acc_hi": 0.65 - r / 2,
                          "flag": 0.3, "flag_lo": 0.25, "flag_hi": 0.35}
                         for m in fg.MODELS for p in ("all", "plain") for r in rhos])
    cross = pd.DataFrame([{"model": m, "prompt": "all", "poison_type": "all", "scoring": s,
                           "crossover": 1.0 if (m, s) == ("llama", "upper bound") else 0.75}
                          for m in fg.MODELS for s in ("rules", "upper bound")]
                         + [{"model": "llama", "prompt": "plain", "poison_type": "swap", "scoring": "rules",
                             "crossover": 0.5}])
    cb = pd.DataFrame([{"model": m, "n": 500, "correct": 0.12} for m in fg.MODELS])
    pe = pd.DataFrame([{"model": m, "rho": r, "meta": meta, "metric": k, "n_pairs": 500, "delta": -0.02,
                        "lo": -0.04, "hi": 0.0}
                       for m in fg.MODELS for r in rhos for meta in fg.METAS
                       for k in ("poisoned", "correct", "abstain", "nousable")])
    fb = pd.DataFrame([{"model": m, "rho": r, "flag": 0.3 + r / 4, "baseline": 0.3, "delta": r / 4,
                        "lo": r / 4 - 0.02, "hi": r / 4 + 0.02} for m in fg.MODELS for r in rhos])
    tm = pd.DataFrame([{"model": m, "poison_type": t, "rho": r, "n": 1500, "poisoned": r / 2,
                        "poisoned_lo": r / 2 - 0.02, "poisoned_hi": r / 2 + 0.02}
                       for m in fg.MODELS for t in TYPES for r in rhos[1:]])
    tr = pd.DataFrame([{"model": m, "meta": meta, "direction": d, "other_outcome": o, "count": 1,
                        "share": s if d.startswith("plain") else 0.25}
                       for m in fg.MODELS for meta in fg.METAS
                       for d in ("plain Poisoned -> meta", "meta Poisoned -> plain") for o, s in OUT.items()])
    return {"descriptive_by_rho": desc, "explore_crossover": cross, "explore_closed_book": cb,
            "explore_prompt_effects": pe, "explore_flag_baseline": fb, "explore_type_model": tm,
            "explore_transitions": tr}


def test_one_fixed_style_per_model():
    for style in (fg.LABELS, fg.COLORS, fg.MARKERS, fg.LINES):
        assert set(style) == set(fg.MODELS)
        assert len(set(style.values())) == len(fg.MODELS)


@pytest.mark.parametrize("line, cls, width", [
    ("\\documentclass[runningheads]{llncs}", "llncs", 4.8),
    ("\\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}", "sn-jnl", 6.2),
])
def test_text_width_reads_active_documentclass(tmp_path, line, cls, width):
    (tmp_path / "main.tex").write_text("%\\documentclass[iicol]{sn-jnl}\n" + line + "\n\\begin{document}\n")
    assert fg.text_width(tmp_path) == (cls, width)


def test_text_width_rejects_two_column_and_unknown(tmp_path):
    tex = tmp_path / "main.tex"
    tex.write_text("\\documentclass[pdflatex,iicol]{sn-jnl}\n")
    with pytest.raises(ValueError, match="two-column"):
        fg.text_width(tmp_path)
    tex.write_text("\\documentclass{acl}\n")
    with pytest.raises(ValueError, match="unknown class"):
        fg.text_width(tmp_path)


def test_rho_curves_panels_colours_and_crossovers():
    t = tables()
    fig = fg.fig_rho_curves(t["descriptive_by_rho"], t["explore_crossover"], t["explore_closed_book"])
    assert len(fig.axes) == 3
    for ax, m in zip(fig.axes, fg.MODELS):
        assert fg.COLORS[m] in {line.get_color() for line in ax.get_lines()}
        cross = sorted(line.get_xdata()[0] for line in ax.get_lines() if line.get_color() == "black")
        assert cross == ([0.75, 1.0] if m == "llama" else [0.75])
    plt.close(fig)


def test_prompt_effects_grid_has_one_series_per_model():
    fig = fg.fig_prompt_effects(tables()["explore_prompt_effects"])
    assert len(fig.axes) == 6
    assert all(len(ax.containers) == len(fg.MODELS) for ax in fig.axes)
    plt.close(fig)


def test_missing_metric_and_column_raise():
    t = tables()
    with pytest.raises(ValueError, match="abstain"):
        fg.fig_prompt_effects(t["explore_prompt_effects"].query("metric != 'abstain'"))
    with pytest.raises(ValueError, match="asr_lo"):
        fg.fig_rho_curves(t["descriptive_by_rho"].drop(columns="asr_lo"), t["explore_crossover"],
                          t["explore_closed_book"])


def test_type_model_keeps_model_colour_per_panel():
    fig = fg.fig_type_model(tables()["explore_type_model"])
    for ax, m in zip(fig.axes, fg.MODELS):
        assert len(ax.containers) == len(TYPES)
        assert {c[0].get_color() for c in ax.containers} == {fg.COLORS[m]}
    plt.close(fig)


def test_transitions_use_plain_direction_and_collect_other():
    fig = fg.fig_transitions(tables()["explore_transitions"])
    widths = [round(p.get_width(), 6) for p in fig.axes[0].patches[:4]]
    assert widths == [70.0, 15.0, 10.0, 5.0]
    plt.close(fig)


def test_main_writes_pdf_and_png(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    for name, df in tables().items():
        df.to_csv(src / f"{name}.csv", index=False)
    fg.main(["--results", str(src), "--out", str(tmp_path / "out"), "--width", "4.8"])
    for name in fg.NAMES:
        assert (tmp_path / "out" / f"{name}.pdf").read_bytes().startswith(b"%PDF")
        assert (tmp_path / "out" / f"{name}.png").read_bytes().startswith(b"\x89PNG")
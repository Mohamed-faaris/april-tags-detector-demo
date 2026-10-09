#!/usr/bin/env python3
"""Graph experiment outputs: pipelines vs libraries, accuracy vs speed.

Reads <outputs-root>/analysis.json (+ per-run summary.csv and
lib_<name>/summary.json for timing) and writes PNGs to --figures-dir:

    fig1_pass_rates.png   pass count per run (easy + wide)
    fig2_rel_error.png    mean relative-translation error per run (mm)
    fig3_speed_accuracy.png  detection ms vs relative-rotation error (libraries)
    fig4_error_dist.png   per-experiment rel-trans error distribution (boxplot)

    uv run python scripts/plot_results.py --outputs-root experiments/outputs \
        --figures-dir experiments/figures [--title-prefix "Easy30"]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(outputs_root: Path):
    analysis = json.loads((outputs_root / "analysis.json").read_text())
    runs = {t["run"]: t for t in analysis["runs"]}
    per_exp: dict[str, list[float]] = {}
    miss4: dict[str, int] = {}
    for run in runs:
        rows = list(csv.DictReader(open(outputs_root / run / "summary.csv")))
        per_exp[run] = [float(r["rel_trans_err_m"]) * 1000
                        for r in rows if r.get("rel_trans_err_m") not in ("", None)]
        miss4[run] = sum(1 for r in rows if "4" not in r.get("detected_ids", "").split(","))
    try:
        lib_cmp = json.loads((outputs_root / "lib_compare.json").read_text())
    except FileNotFoundError:
        lib_cmp = {}
    return runs, per_exp, miss4, lib_cmp


def style(ax, ylabel):
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs-root", type=Path, default=Path("experiments/outputs"))
    ap.add_argument("--figures-dir", type=Path, default=Path("experiments/figures"))
    ap.add_argument("--title-prefix", default="")
    args = ap.parse_args()
    root = args.outputs_root.resolve()
    figdir = args.figures_dir.resolve()
    figdir.mkdir(parents=True, exist_ok=True)
    runs, per_exp, miss4, lib_cmp = load(root)
    pre = (args.title_prefix + " ").strip()
    order = sorted(runs, key=lambda r: (-runs[r]["passed_included"], r))
    colors = ["#2ca02c" if not r.startswith("lib_") else "#1f77b4" for r in order]
    labels = [r.replace("lib_", "") for r in order]

    # 1 — pass rates (+ tag4 misses marked).
    fig, ax = plt.subplots(figsize=(12, 4.5))
    n_tot = [runs[r]["n_included"] for r in order]
    n_pass = [runs[r]["passed_included"] for r in order]
    x = np.arange(len(order))
    ax.bar(x, n_pass, color=colors, label="passed")
    ax.bar(x, [t - p for t, p in zip(n_tot, n_pass)], bottom=n_pass,
           color="#d62728", alpha=0.6, label="failed")
    for i, r in enumerate(order):
        if miss4[r]:
            ax.text(i, n_tot[i] + 0.3, f"miss4:{miss4[r]}", ha="center", fontsize=8, color="#d62728")
    ax.set_xticks(x, labels)
    ax.set_title(f"{pre} pass count per run (green=pipeline, blue=library)".strip())
    style(ax, "experiments passed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "fig1_pass_rates.png", dpi=120)

    # 2 — mean relative translation error.
    fig, ax = plt.subplots(figsize=(12, 4.5))
    means = [(runs[r]["mean_rel_trans_m"] or 0) * 1000 for r in order]
    ax.bar(x, means, color=colors)
    ax.set_xticks(x, labels)
    ax.set_title(f"{pre} mean relative-translation error (mm, lower is better)".strip())
    style(ax, "mm")
    fig.tight_layout()
    fig.savefig(figdir / "fig2_rel_error.png", dpi=120)

    # 3 — speed vs rotation accuracy for libraries.
    libs = sorted(lib_cmp)
    if libs:
        fig, ax = plt.subplots(figsize=(7, 5))
        for lib in libs:
            s = lib_cmp[lib]
            ax.scatter(s["mean_detect_ms"], s["mean_rel_rot_deg"], s=90)
            ax.annotate(f"{lib}\n{s['passed']}/{s['num_experiments']}",
                        (s["mean_detect_ms"], s["mean_rel_rot_deg"]),
                        textcoords="offset points", xytext=(8, -4), fontsize=9)
        ax.set_xlabel("mean detection time per image (ms)")
        ax.set_ylabel("mean relative rotation error (deg)")
        ax.set_title(f"{pre} library speed vs accuracy (lower-left wins)".strip())
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figdir / "fig3_speed_accuracy.png", dpi=120)

    # 4 — per-experiment error distribution (pipelines only, readable subset).
    pipes = [r for r in order if not r.startswith("lib_")]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot([per_exp[r] for r in pipes], tick_labels=pipes, showfliers=False)
    ax.set_title(f"{pre} relative-translation error distribution per pipeline (mm)".strip())
    style(ax, "mm")
    fig.tight_layout()
    fig.savefig(figdir / "fig4_error_dist.png", dpi=120)

    print(f"Wrote {[p.name for p in sorted(figdir.glob('fig*.png'))]} to {figdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

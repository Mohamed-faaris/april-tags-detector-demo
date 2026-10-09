#!/usr/bin/env python3
"""Aggregate pipeline/library summaries with an undetected-exclusion rule.

Rule: if a tag in an experiment is captured by NO runner (pipeline or
library), that experiment is EXCLUDED from the mean-error analysis and
explicitly marked (status=excluded_undetected + which tags). All other
experiments count as included. Per-runner pass rates are reported on the
included set; raw pass counts (all exps) are shown alongside.

Reads every <outputs-root>/<run>/summary.csv (+ summary.json config).
Writes <outputs-root>/analysis.json and prints the comparison table.

    uv run python scripts/analyze_runs.py --outputs-root experiments_wide/outputs
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_runs(outputs_root: Path) -> dict[str, dict]:
    runs: dict[str, dict] = {}
    for summary_csv in sorted(outputs_root.glob("*/summary.csv")):
        run = summary_csv.parent.name
        with open(summary_csv, newline="") as fh:
            rows = list(csv.DictReader(fh))
        cfg: dict = {}
        sj = summary_csv.parent / "summary.json"
        if sj.exists():
            try:
                data = json.loads(sj.read_text())
                cfg = data.get("pipeline_config") or data.get("config") or {}
            except json.JSONDecodeError:
                pass
        runs[run] = {"rows": rows, "config": cfg}
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs-root", type=Path, default=Path("experiments/outputs"))
    ap.add_argument("--require-both-tags", action="store_true", default=True,
                    help="Exclude an experiment if any tag is missed by every runner "
                         "(default) vs only when both tags are missed by every runner")
    args = ap.parse_args()
    runs = load_runs(args.outputs_root.resolve())
    if not runs:
        raise SystemExit(f"No */summary.csv found under {args.outputs_root}")

    exp_ids = sorted({r["exp_id"] for run in runs.values() for r in run["rows"]})
    by_run_exp = {run: {r["exp_id"]: r for r in data["rows"]} for run, data in runs.items()}

    excluded: dict[str, dict] = {}
    included: list[str] = []
    for exp in exp_ids:
        det0 = sorted(run for run, m in by_run_exp.items()
                      if "0" in (m.get(exp) or {}).get("detected_ids", "").split(","))
        det4 = sorted(run for run, m in by_run_exp.items()
                      if "4" in (m.get(exp) or {}).get("detected_ids", "").split(","))
        missed = [t for t, d in (("0", det0), ("4", det4)) if not d]
        if (args.require_both_tags and missed) or \
           (not args.require_both_tags and len(missed) == 2):
            excluded[exp] = {"status": "excluded_undetected",
                             "tags_missed_by_all_runners": missed,
                             "detected_tag0_by": det0, "detected_tag4_by": det4}
        else:
            included.append(exp)

    table = []
    for run, data in runs.items():
        m = by_run_exp[run]
        inc = [m[e] for e in included if e in m]
        raw_pass = sum(1 for r in data["rows"] if r["passed"] == "True")

        def mean(key):
            v = [float(r[key]) for r in inc if r.get(key) not in ("", None)]
            return (sum(v) / len(v)) if v else None

        table.append({
            "run": run,
            "config": data["config"],
            "n_total": len(data["rows"]),
            "n_included": len(inc),
            "passed_raw": raw_pass,
            "passed_included": sum(1 for r in inc if r["passed"] == "True"),
            "mean_tag0_pos_m": mean("tag0_pos_err_m"),
            "mean_tag0_rot_deg": mean("tag0_rot_err_deg"),
            "mean_tag4_pos_m": mean("tag4_pos_err_m"),
            "mean_tag4_rot_deg": mean("tag4_rot_err_deg"),
            "mean_rel_trans_m": mean("rel_trans_err_m"),
            "mean_rel_dist_m": mean("rel_dist_err_m"),
            "mean_rel_rot_deg": mean("rel_rot_err_deg"),
            "mean_detect_ms": mean("detect_ms"),
        })

    out = {"included_experiments": included, "excluded": excluded, "runs": table}
    out_path = args.outputs_root.resolve() / "analysis.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")

    def fmt(x, scale=1.0, nd=1):
        return "   --  " if x is None else f"{x * scale:{nd + 4}.{nd}f}"

    print(f"Included {len(included)}/{len(exp_ids)} experiments "
          f"({len(excluded)} excluded as undetected-by-all).")
    for exp, info in excluded.items():
        print(f"  EXCLUDED {exp}: tags missed by every runner: "
              f"{info['tags_missed_by_all_runners']}")
    print(f"\n{'run':18s} {'pass(incl)':>10s} {'t0 pos/rot':>16s} "
          f"{'t4 pos/rot':>16s} {'rel t/d/r':>22s}")
    for t in sorted(table, key=lambda r: (
            -(r["passed_included"] / max(1, r["n_included"])),
            r["mean_rel_trans_m"] if r["mean_rel_trans_m"] is not None else 9e9)):
        print(f"{t['run']:18s} {t['passed_included']:>3d}/{t['n_included']:<6d} "
              f"{fmt(t['mean_tag0_pos_m'], 1000, 1)}mm/{fmt(t['mean_tag0_rot_deg'], 1, 2)} "
              f"{fmt(t['mean_tag4_pos_m'], 1000, 1)}mm/{fmt(t['mean_tag4_rot_deg'], 1, 2)} "
              f"{fmt(t['mean_rel_trans_m'], 1000, 1)}/{fmt(t['mean_rel_dist_m'], 1000, 1)}mm/"
              f"{fmt(t['mean_rel_rot_deg'], 1, 2)}")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

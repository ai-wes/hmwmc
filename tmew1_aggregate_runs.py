"""
Aggregate multi-seed tmew1_run_v2 outputs.

Each run directory (``--out-dir`` of ``tmew1_run_v2.py``) contains
``val_tier{N}_ep{E}.json`` and ``diagnostics_tier{N}.json``. This script reads
every run directory matching a glob, takes the final-epoch validation record and
the diagnostics report for the highest completed tier, and prints mean and
standard deviation across seeds for:

- validation metrics (``latent_acc``, ``holder_acc``, every ``qacc/*``);
- holder-query accuracy by handoffs-before-query bucket;
- the zero-shot emergent-capability probes.

Usage:
    python tmew1_aggregate_runs.py "runs/v2_seed*"
    python tmew1_aggregate_runs.py "runs/v2_seed*" --tier 3 --json summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np

VAL_RE = re.compile(r"val_tier(\d+)_ep(\d+)\.json$")


def load_run(run_dir: str, tier: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Return {'seed', 'tier', 'val', 'diag'} for one run directory, or None."""
    vals: Dict[int, Dict[int, str]] = {}
    for path in glob.glob(os.path.join(run_dir, "val_tier*_ep*.json")):
        m = VAL_RE.search(path)
        if m:
            vals.setdefault(int(m.group(1)), {})[int(m.group(2))] = path
    if not vals:
        return None
    use_tier = tier if tier is not None else max(vals)
    if use_tier not in vals:
        return None
    last_epoch = max(vals[use_tier])
    with open(vals[use_tier][last_epoch]) as f:
        val = json.load(f)
    diag_path = os.path.join(run_dir, f"diagnostics_tier{use_tier}.json")
    diag = None
    if os.path.exists(diag_path):
        with open(diag_path) as f:
            diag = json.load(f)
    return {"dir": run_dir, "seed": val.get("seed"), "tier": use_tier, "epoch": last_epoch, "val": val, "diag": diag}


def _collect(runs: List[Dict[str, Any]], getter) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for run in runs:
        for key, value in getter(run).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value == value:
                out.setdefault(key, []).append(float(value))
    return out


def summarize(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    def val_metrics(run):
        return {k: v for k, v in run["val"].items() if k in ("latent_acc", "holder_acc", "event_acc") or k.startswith("qacc/")}

    def handoff_buckets(run):
        d = (run.get("diag") or {}).get("who_holds_token_by_handoffs") or {}
        return {f"handoffs={k}": v.get("acc") for k, v in d.items() if isinstance(v, dict) and v.get("n", 0) > 0}

    def probes(run):
        d = (run.get("diag") or {}).get("emergent_probes") or {}
        return {k: v for k, v in d.items() if not k.endswith("_n")}

    sections = {
        "validation": _collect(runs, val_metrics),
        "who_holds_token_by_handoffs_before_query": _collect(runs, handoff_buckets),
        "emergent_probes": _collect(runs, probes),
    }
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, metrics in sections.items():
        summary[name] = {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n_seeds": len(v)}
            for k, v in metrics.items()
        }
    return summary


def format_summary(runs: List[Dict[str, Any]], summary: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    lines = [
        f"{len(runs)} run(s): " + ", ".join(f"{os.path.basename(r['dir'].rstrip('/'))} (seed {r['seed']}, tier {r['tier']} ep {r['epoch']})" for r in runs),
    ]
    for name, metrics in summary.items():
        if not metrics:
            continue
        lines.append("")
        lines.append(f"## {name}")
        lines.append(f"{'metric':<42} {'mean':>8} {'std':>8} {'n':>3}")
        for k, s in sorted(metrics.items()):
            lines.append(f"{k:<42} {s['mean']:>8.3f} {s['std']:>8.3f} {s['n_seeds']:>3d}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern", help="Glob of run directories, e.g. 'runs/v2_seed*'")
    parser.add_argument("--tier", type=int, default=None, help="Tier to summarize (default: highest completed)")
    parser.add_argument("--json", type=str, default=None, help="Also write the summary to this JSON path")
    args = parser.parse_args()

    runs = [r for r in (load_run(d, args.tier) for d in sorted(glob.glob(args.pattern))) if r is not None]
    if not runs:
        raise SystemExit(f"No completed runs found for pattern {args.pattern!r}")
    summary = summarize(runs)
    print(format_summary(runs, summary))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"runs": [{k: v for k, v in r.items() if k != "diag"} for r in runs], "summary": summary}, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()

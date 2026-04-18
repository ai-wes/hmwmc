"""
Ablation suite runner.

Runs the E-family experiments in sequence on the same seed:
  E1: fused read (control)
  E2: entity-table-only
  E3: tape-only (event tape + entity history, no entity table)
  E4: no auxiliary memory (cross-attention skipped)
  E5: HPM continuous plasticity vs state machine
  E6: authoritative per-query routing (entity vs tape/history)

All use A3-style worlds (multi-chain concurrency, temporal overlap)
with B2/B4 query families as stressors. Per-tier LR scaling and
diversity-loss suppression are enabled for stability.

Target metrics for comparison:
  - qacc/who_holds_token
  - qacc/holder_if_handoff2_absent
  - qacc/closest_entity_to_holder_at_alarm

Usage:
    python run_ablation_suite.py --out-dir experiments/ablation_E
    python run_ablation_suite.py --out-dir experiments/ablation_E --branches E1,E2
    python run_ablation_suite.py --out-dir experiments/ablation_E --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from tmew1_experiments import BranchConfig, make_branch_preset, BRANCH_IDS
from tmew1_branch_runner import run_branch, _load_baseline

ALL_ABLATION_BRANCHES = ("E1", "E2", "E3", "E4", "E5", "E6")

TARGET_METRICS = (
    "qacc/who_holds_token",
    "qacc/holder_if_handoff2_absent",
    "qacc/closest_entity_to_holder_at_alarm",
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run E-family ablation suite")
    p.add_argument("--out-dir", required=True, help="Root output directory")
    p.add_argument(
        "--branches", type=str, default=None,
        help="Comma-separated branch ids to run. Default: all E1-E6.",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed (shared across all branches)")
    p.add_argument(
        "--baseline-record", default=None,
        help="Path to baseline val.json for regression checks.",
    )
    p.add_argument("--epochs", type=int, default=None, help="Override epochs_per_tier")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch_size")
    p.add_argument("--resume-from", type=str, default=None)
    return p


def main():
    args = build_parser().parse_args()

    if args.branches:
        branches_to_run = [b.strip() for b in args.branches.split(",")]
        for b in branches_to_run:
            if b not in ALL_ABLATION_BRANCHES:
                print(f"ERROR: {b} is not an E-family branch. Valid: {ALL_ABLATION_BRANCHES}")
                sys.exit(1)
    else:
        branches_to_run = list(ALL_ABLATION_BRANCHES)

    baseline_record = _load_baseline(args.baseline_record)

    results: Dict[str, Dict[str, float]] = {}
    verdicts: Dict[str, str] = {}

    print(f"\n{'='*70}")
    print(f"ABLATION SUITE: {len(branches_to_run)} branches")
    print(f"Branches: {branches_to_run}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.out_dir}")
    print(f"{'='*70}\n")

    for branch_id in branches_to_run:
        branch = make_branch_preset(branch_id)
        branch = replace(branch, seed=args.seed)
        if args.epochs is not None:
            branch = replace(branch, epochs_per_tier=args.epochs)
        if args.batch_size is not None:
            branch = replace(branch, batch_size=args.batch_size)

        branch_dir = os.path.join(args.out_dir, branch_id)
        print(f"\n{'='*70}")
        print(f"STARTING: {branch_id} — {branch.description}")
        print(f"Memory ablation mode: {branch.memory_ablation_mode}")
        if branch.hpm_continuous_plasticity:
            print(f"HPM: continuous plasticity (no state machine)")
        print(f"Output: {branch_dir}")
        print(f"{'='*70}\n")

        t0 = time.time()
        try:
            verdict, val_record = run_branch(
                branch, branch_dir, baseline_record, args.resume_from
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            verdict = "crash"
            val_record = {}
        elapsed = time.time() - t0

        results[branch_id] = val_record
        verdicts[branch_id] = verdict

        print(f"\n--- {branch_id} completed in {elapsed:.0f}s | verdict={verdict} ---")
        for m in TARGET_METRICS:
            v = val_record.get(m, float("nan"))
            print(f"  {m}: {v:.4f}")
        print()

    # ── Summary table ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("ABLATION SUMMARY")
    print(f"{'='*70}")

    # Header
    col_w = 14
    header = f"{'Branch':<8} {'Mode':<12} {'Verdict':<10}"
    for m in TARGET_METRICS:
        short = m.split("/")[-1][:col_w]
        header += f" {short:>{col_w}}"
    print(header)
    print("-" * len(header))

    for branch_id in branches_to_run:
        branch = make_branch_preset(branch_id)
        mode = branch.memory_ablation_mode
        if branch.hpm_continuous_plasticity:
            mode = "cont_plast"
        verdict = verdicts.get(branch_id, "?")
        row = f"{branch_id:<8} {mode:<12} {verdict:<10}"
        for m in TARGET_METRICS:
            v = results.get(branch_id, {}).get(m, float("nan"))
            row += f" {v:>{col_w}.4f}"
        print(row)

    # Save summary JSON
    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "ablation_summary.json")
    summary = {
        "branches": branches_to_run,
        "seed": args.seed,
        "verdicts": verdicts,
        "target_metrics": list(TARGET_METRICS),
        "results": {
            bid: {m: results.get(bid, {}).get(m) for m in TARGET_METRICS}
            for bid in branches_to_run
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()

"""
CLI runner for Phase 1 TMEW-1 experiments.

Usage:

    # Run baseline (frozen Phase-0 reference) — no overrides:
    python tmew1_branch_runner.py --branch baseline \
        --out-dir experiments/baseline

    # A1 — entity scaling (8 entities):
    python tmew1_branch_runner.py --branch A1 \
        --baseline-record experiments/baseline/val.json \
        --out-dir experiments/A1

    # C1 — HPM Level-2, 4 slots, competitive, concat readout:
    python tmew1_branch_runner.py --branch C1 \
        --baseline-record experiments/baseline/val.json \
        --out-dir experiments/C1

    # Override a single field at CLI:
    python tmew1_branch_runner.py --branch A2 \
        --tier3-episode-length 128 \
        --epochs 6 \
        --out-dir experiments/A2_T128

    # B-branches must opt-in because they need the diagnostics patch:
    python tmew1_branch_runner.py --branch B1 \
        --ack-missing-capabilities \
        --out-dir experiments/B1

Exit codes:
    0   promote
    1   kill
    2   undecided / ran but rubric ambiguous
    3   missing capabilities, refused to run (override with --ack-missing-capabilities)
    4   crash during training (the traceback is logged; promotion is forced "undecided")

The runner expects the existing codebase (tmew1_train, tmew1_run, hpm,
tmew1_diagnostics, tmew1_queries) to be importable. It does not patch
those modules; it only composes config overrides and calls
tmew1_run.run_curriculum.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import asdict, replace
from typing import Dict, Optional, Tuple

from tmew1_experiments_v2 import (
    BRANCH_IDS,
    BranchConfig,
    PromotionRubric,
    make_branch_preset,
    apply_world_overrides,
    apply_tier_overrides,
    apply_hpm_overrides,
    apply_train_overrides,
    check_missing_capabilities,
    write_verdict,
)


VAL_JSON_NAME = "val.json"


# ---------------------------------------------------------------------------
# Argument plumbing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase-1 branch runner for TMEW-1 experiments",
    )
    p.add_argument(
        "--branch", required=True,
        choices=("baseline",) + BRANCH_IDS,
        help="Branch id. 'baseline' means run DEFAULT_TIERS with no overrides.",
    )
    p.add_argument("--out-dir", required=True, help="Directory for verdicts and val.json")
    p.add_argument(
        "--baseline-record", default=None,
        help="Path to val.json from the frozen Phase-0 baseline. "
             "Required for regression checks and min_gain_points enforcement.",
    )
    p.add_argument(
        "--ack-missing-capabilities", action="store_true",
        help="Run even when the branch needs a diagnostics/simulator patch "
             "that is not yet present. Useful for dry-running the launcher.",
    )

    # Generic training overrides (apply to any branch)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--train-episodes", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--tiers", type=str, default=None,
        help="Comma-separated list of tier ids to run, e.g. '2,3'. "
             "Default: whatever the branch preset specified.",
    )

    # A-family overrides
    p.add_argument("--min-entities", type=int, default=None)
    p.add_argument("--max-entities", type=int, default=None)
    p.add_argument("--tier3-episode-length", type=int, default=None)
    p.add_argument("--tier3-max-delay", type=int, default=None)
    p.add_argument(
        "--tier3-template-pool", type=str, default=None,
        help="Comma-separated template names (overrides branch default).",
    )
    p.add_argument("--chain2-frequency-boost", type=float, default=None)
    p.add_argument("--chain2-temporal-overlap", action="store_true", default=None)
    p.add_argument("--rule-dynamic", action="store_true", default=None)

    # B-family overrides
    p.add_argument(
        "--extra-query-families", type=str, default=None,
        help="Comma-separated extra B_QUERY_SPECS names to add to the run.",
    )

    # C-family overrides
    p.add_argument("--hpm-n-slots", type=int, default=None)
    p.add_argument(
        "--hpm-read-mode", type=str, default=None,
        choices=("concat", "mean", "attn"),
    )
    p.add_argument("--hpm-competitive", action="store_true", default=None)
    p.add_argument("--hpm-slot-dim", type=int, default=None)
    p.add_argument("--hpm-retroactive-window", type=int, default=None,
                   help="C3: retroactive binding window size (0=disabled)")
    p.add_argument("--hpm-slot-timescales", type=str, default=None,
                   help="C4: comma-separated per-slot timescale multipliers")

    # Resume behavior (baseline runs benefit from this most)
    p.add_argument("--resume", type=str, default=None)

    return p


def _parse_csv(s: Optional[str]) -> Optional[Tuple[str, ...]]:
    if not s:
        return None
    return tuple(x.strip() for x in s.split(",") if x.strip())


def merge_cli_into_branch(args: argparse.Namespace, branch: BranchConfig) -> BranchConfig:
    patch: Dict = {}
    # Generic training
    if args.epochs is not None:
        patch["epochs_per_tier"] = args.epochs
    if args.batch_size is not None:
        patch["batch_size"] = args.batch_size
    if args.train_episodes is not None:
        patch["train_episodes_per_tier"] = args.train_episodes
    if args.lr is not None:
        patch["lr"] = args.lr
    if args.seed is not None:
        patch["seed"] = args.seed
    if args.tiers is not None:
        patch["tiers_to_run"] = tuple(int(x) for x in args.tiers.split(","))
    # A
    if args.min_entities is not None:
        patch["min_entities"] = args.min_entities
    if args.max_entities is not None:
        patch["max_entities"] = args.max_entities
    if args.tier3_episode_length is not None:
        patch["tier3_episode_length"] = args.tier3_episode_length
    if args.tier3_max_delay is not None:
        patch["tier3_max_delay"] = args.tier3_max_delay
    if args.tier3_template_pool is not None:
        patch["tier3_template_pool"] = _parse_csv(args.tier3_template_pool)
    if args.chain2_frequency_boost is not None:
        patch["chain2_frequency_boost"] = args.chain2_frequency_boost
    if args.chain2_temporal_overlap is not None:
        patch["chain2_temporal_overlap"] = True
    if args.rule_dynamic is not None:
        patch["rule_dynamic"] = True
    # B
    extra = _parse_csv(args.extra_query_families)
    if extra is not None:
        patch["extra_query_families"] = tuple(list(branch.extra_query_families) + list(extra))
    # C
    if args.hpm_n_slots is not None:
        patch["hpm_n_slots"] = args.hpm_n_slots
    if args.hpm_read_mode is not None:
        patch["hpm_read_mode"] = args.hpm_read_mode
    if args.hpm_competitive is not None:
        patch["hpm_competitive"] = True
    if args.hpm_slot_dim is not None:
        patch["hpm_slot_dim"] = args.hpm_slot_dim
    if args.hpm_retroactive_window is not None:
        patch["hpm_retroactive_window"] = args.hpm_retroactive_window
    if args.hpm_slot_timescales is not None:
        patch["hpm_slot_timescales"] = tuple(
            float(x.strip()) for x in args.hpm_slot_timescales.split(",")
        )

    if not patch:
        return branch
    return replace(branch, **patch)


# ---------------------------------------------------------------------------
# Actual runner
# ---------------------------------------------------------------------------
def _load_baseline(path: Optional[str]) -> Optional[Dict[str, float]]:
    if path is None:
        return None
    if not os.path.exists(path):
        print(f"WARN: --baseline-record not found at {path}; skipping baseline checks.")
        return None
    with open(path) as f:
        return json.load(f)


def _build_baseline_branch() -> BranchConfig:
    """Phase-0 reference: no overrides, default config."""
    return BranchConfig(
        branch_id="baseline",
        family="A",   # doesn't matter, no overrides
        description="Frozen Phase-0 reference run (DEFAULT_TIERS)",
        rubric=None,
    )


def run_branch(branch: BranchConfig, out_dir: str, baseline_record: Optional[Dict[str, float]], resume: Optional[str]) -> Tuple[str, Dict[str, float]]:
    """
    Execute a branch. Returns (verdict, val_record).

    Delegates training to tmew1_run_v2.run_curriculum, which already knows how
    to iterate tiers, build the model, evaluate, and emit the val record
    per epoch. We import it lazily so the module stays testable.
    """
    # Lazy import so the runner can be inspected without torch.
    from tmew1_run_v2 import run_curriculum
    os.makedirs(out_dir, exist_ok=True)
    os.environ["TMEW1_BRANCH_OUT_DIR"] = out_dir
    if branch.extra_query_families:
        os.environ["TMEW1_EXTRA_QUERY_FAMILIES"] = ",".join(branch.extra_query_families)
    else:
        os.environ.pop("TMEW1_EXTRA_QUERY_FAMILIES", None)
    if branch.replace_query_families:
        os.environ["TMEW1_REPLACE_QUERY_FAMILIES"] = ",".join(branch.replace_query_families)
    else:
        os.environ.pop("TMEW1_REPLACE_QUERY_FAMILIES", None)
    if branch.et_only_read_qtypes:
        os.environ["TMEW1_ET_ONLY_READ_QTYPES"] = ",".join(branch.et_only_read_qtypes)
    else:
        os.environ.pop("TMEW1_ET_ONLY_READ_QTYPES", None)
    # Memory ablation mode for IterativeQueryHead.
    if branch.memory_ablation_mode != "fused":
        os.environ["TMEW1_MEMORY_ABLATION"] = branch.memory_ablation_mode
    else:
        os.environ.pop("TMEW1_MEMORY_ABLATION", None)
    world_cfg = apply_world_overrides(branch)
    tiers = apply_tier_overrides(branch)
    tcfg = apply_train_overrides(branch)
    hpm_cfg = apply_hpm_overrides(branch)

    # Build model_config_overrides dict to pass explicitly to build_model
    # via run_curriculum. This replaces the old (broken) monkey-patch approach
    # — dataclass __init__ is generated at class definition time, so patching
    # __dataclass_fields__ has no effect on instance creation.
    from hpm_v2 import EntityTableConfig, EventTapeConfig, EntityHistoryConfig

    model_config_overrides: Dict[str, object] = {}

    needs_hpm_patch = any([branch.hpm_n_slots, branch.hpm_read_mode, branch.hpm_competitive,
                           branch.hpm_slot_dim, branch.hpm_retroactive_window, branch.hpm_slot_timescales,
                           branch.hpm_continuous_plasticity])

    if needs_hpm_patch:
        model_config_overrides["hpm_config"] = hpm_cfg

    if branch.enable_entity_table:
        model_config_overrides["entity_table_config"] = EntityTableConfig(enabled=True, n_entities=4, d_entity=64)

    if branch.enable_event_tape:
        model_config_overrides["event_tape_config"] = EventTapeConfig(enabled=True, max_events=32, surprise_threshold=0.5)

    if branch.enable_entity_history:
        model_config_overrides["entity_history_config"] = EntityHistoryConfig(enabled=True, n_snapshots=branch.entity_history_n_snapshots)

    run_curriculum(world_cfg, tcfg, tiers=tiers, resume_from=resume, model_config_overrides=model_config_overrides or None)

    # The existing run_curriculum writes to ScoreLogger but does not return a
    # val dict. For rubric enforcement we need the final val record on disk.
    # Convention: we expect the user's run loop to also write val.json
    # (see the doc note at the bottom of this file). If it isn't present,
    # we return an empty dict and the verdict becomes "undecided".
    val_json = os.path.join(out_dir, VAL_JSON_NAME)
    val_record: Dict[str, float] = {}
    if os.path.exists(val_json):
        with open(val_json) as f:
            val_record = json.load(f)
    else:
        print(f"NOTE: {val_json} not found. The verdict will be 'undecided' "
              "until tmew1_run writes the final val dict to this path. See "
              "the docstring at the bottom of tmew1_branch_runner.py for the "
              "one-line patch needed in run_curriculum.")

    verdict, reasons = write_verdict(branch, val_record, baseline_record, out_dir)
    print(f"\n=== verdict: {verdict} ===")
    for r in reasons:
        print(f"  - {r}")
    return verdict, val_record


def main() -> int:
    args = build_parser().parse_args()

    if args.branch == "baseline":
        branch = _build_baseline_branch()
    else:
        branch = make_branch_preset(args.branch)
    branch = merge_cli_into_branch(args, branch)

    missing = check_missing_capabilities(branch)
    if missing and not args.ack_missing_capabilities:
        print("Refusing to run: branch needs codebase patches first.")
        for m in missing:
            print(f"  - {m}")
        print("\nRe-run with --ack-missing-capabilities to launch anyway.")
        return 3
    if missing:
        print("WARN: launching despite missing capabilities (flag set):")
        for m in missing:
            print(f"  - {m}")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "branch_config.json"), "w") as f:
        json.dump(asdict(branch), f, indent=2, default=str)

    baseline_record = _load_baseline(args.baseline_record)

    try:
        verdict, _ = run_branch(branch, args.out_dir, baseline_record, args.resume)
    except Exception:
        tb = traceback.format_exc()
        print("CRASH during training:")
        print(tb)
        with open(os.path.join(args.out_dir, "crash.log"), "w") as f:
            f.write(tb)
        return 4

    code = {"promote": 0, "kill": 1, "undecided": 2}.get(verdict, 2)
    return code


if __name__ == "__main__":
    sys.exit(main())


# -----------------------------------------------------------------------------
# ONE-LINE PATCH REQUIRED IN tmew1_run.run_curriculum
# -----------------------------------------------------------------------------
# For verdicts to fire, run_curriculum must persist the final val dict
# to disk at `<out_dir>/val.json`. The cleanest hook is right after the
# tier-3 evaluator call near line 662 of tmew1_run.py:
#
#     val = evaluate(...)
#     log_training_snapshot(...)
#     # >>> add:
#     if os.environ.get("TMEW1_BRANCH_OUT_DIR"):
#         with open(os.path.join(os.environ["TMEW1_BRANCH_OUT_DIR"], "val.json"), "w") as f:
#             json.dump({k: float(v) for k, v in val.items()}, f, indent=2)
#
# Then set TMEW1_BRANCH_OUT_DIR=<out_dir> before launching from this runner.
# I did not patch tmew1_run.py directly because it's imported by other
# scripts and I don't want to change its signature. The env-var hook is
# the smallest surface area that keeps baseline and branch runs aligned.
# -----------------------------------------------------------------------------

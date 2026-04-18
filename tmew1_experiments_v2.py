"""
TMEW-1 Phase 1 experimental infrastructure.

Implements the branch structure laid out in the research program:

  Family A — harder worlds at the same architecture
      A1: entity scaling       (more active + distractor entities)
      A2: horizon scaling      (T=64 -> 96 -> 128)
      A3: multi-chain concurrency (higher chain2 frequency, temporal overlap)
      A4: rule-dynamic worlds  (active_rule actually alters dynamics)

  Family B — harder questions at the same world
      B1: temporal ordering queries (before/after relations over any event pair)
      B2: relational queries        (nearest-entity-at-event, shared-color, etc.)
      B3: negation queries          (which entity was never occluded, etc.)
      B4: counterfactual queries    (held in reserve; scaffolding only)

  Family C — architecture changes at the same world/questions
      C1: HPM Level-2 multi-slot competitive writes
      C2: HPM read-mode ablation (concat vs mean vs attn)

This module is layered ON TOP of the existing codebase:
  - It does not fork tmew1_train.py, tmew1_run.py, or hpm.py.
  - It produces modified WorldConfig / CurriculumTier / HPMConfig objects
    and a branch-local train/eval hook map that callers compose into
    run_curriculum via the existing CLI entry point.

Standing invariants:
  - One branch changes exactly one family at a time (A OR B OR C), never
    combinations, except in an explicit confirmatory stage.
  - Every branch writes a promotion-rubric JSONL record with the metrics
    your existing evaluator already emits, plus any new query-family
    accuracies.
  - Kill conditions are encoded as assertions over the final val record,
    not as training-time early-exit. The runner emits a verdict file
    (promote / kill / undecided) and exits with a matching code.

Dependencies assumed to be importable from the working dir:
  - tmew1_train   (WorldConfig, CurriculumTier, DEFAULT_TIERS, TrainConfig)
  - tmew1_run     (run_curriculum)
  - hpm           (HPMConfig)
  - tmew1_diagnostics (EXTENDED_QUERY_TYPES, generate_episode_with_diagnostics)
  - tmew1_queries     (QueryHead)

What this module does NOT do, on purpose:
  - It does not implement new query generators inside tmew1_diagnostics.
    The B-family branches declare the query-family *names* and the
    expected answer schema; the actual generator must be added to
    tmew1_diagnostics.generate_episode_with_diagnostics (see B_QUERY_SPECS
    below for the contract each branch needs).
  - It does not rewrite _step_world. The A3/A4 branches pass parameters
    that tmew1_train._step_world already reads; A3/A4 effects that need
    new world physics are flagged as `requires_step_patch=True` so you
    can see which branches will fail until the simulator is extended.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass, field, replace, asdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any

# We import lazily so this module can be unit-tested without the full stack.
def _lazy_imports():
    from tmew1_train_v2 import WorldConfig, CurriculumTier, DEFAULT_TIERS, TrainConfig
    from hpm_v2 import HPMConfig
    return WorldConfig, CurriculumTier, DEFAULT_TIERS, TrainConfig, HPMConfig


# =============================================================================
# Branch taxonomy
# =============================================================================
BRANCH_FAMILIES = ("A", "B", "C", "AB")
BRANCH_IDS = (
    "A1", "A2", "A3", "A4",
    "B1", "B2", "B3", "B4",
    "C1", "C2", "C3", "C4",
    "D1", "D2", "D3",
    "E1", "E2", "E3", "E4", "E5",
    "AB1",
)


@dataclass
class PromotionRubric:
    """
    Capability + stability gates the branch must clear.

    `target_metric` is a dotted path into the val dict produced by
    tmew1_run.evaluate(), e.g. "qacc/who_holds_token" or
    "qacc/which_entity_before_X".

    `baseline_value` is the score from the frozen reference run
    (Phase 0, step 0.1). The caller is responsible for supplying it;
    if None, the rubric falls back to `min_absolute` only.

    `min_gain_points` (absolute, 0..100): for targeted branches, the
    target metric must beat baseline by this much.

    `max_regression_points`: no already-solved metric may regress by
    more than this much. Defaults are stored in RUBRIC_DEFAULTS.

    `min_absolute` (optional): even with no baseline, target metric
    must hit this floor.

    `stability_floors`: absolute minima on metrics that, if breached,
    force a kill regardless of target-metric performance. Example:
    {"latent_acc": 0.90} — if the model can't even hit 90% latent
    classification after the branch, something is broken.
    """
    target_metric: str
    baseline_value: Optional[float] = None
    min_gain_points: float = 10.0
    max_regression_points: float = 3.0
    min_absolute: Optional[float] = None
    stability_floors: Dict[str, float] = field(default_factory=dict)
    solved_metrics: Tuple[str, ...] = (
        "latent_acc",
        "qacc/did_alarm_fire",
        "qacc/did_chain2_fire",
        "qacc/did_trigger_before_alarm",
        "qacc/what_was_true_rule",
    )

    def verdict(self, val_record: Dict[str, float], baseline_record: Optional[Dict[str, float]] = None) -> Tuple[str, List[str]]:
        import math
        reasons: List[str] = []

        # NaN/Inf in any core loss metric is an immediate kill — the run
        # exploded, there's nothing to evaluate. Check before anything else.
        nan_metrics = [
            k for k, v in val_record.items()
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v))
            and (k.startswith("loss/") or k in ("next_step", "total"))
        ]
        if nan_metrics:
            reasons.append(
                f"training instability kill: NaN/Inf in {nan_metrics[:4]}"
                + (f" and {len(nan_metrics) - 4} more" if len(nan_metrics) > 4 else "")
            )
            return "kill", reasons

        # Stability floors are hard kills.
        for k, floor in self.stability_floors.items():
            v = val_record.get(k)
            if v is None:
                reasons.append(f"stability: metric '{k}' missing from val record")
                return "undecided", reasons
            if v < floor:
                reasons.append(f"stability kill: {k}={v:.3f} < floor {floor:.3f}")
                return "kill", reasons

        # Regression check on solved metrics.
        if baseline_record is not None:
            for k in self.solved_metrics:
                bv = baseline_record.get(k)
                nv = val_record.get(k)
                if bv is None or nv is None:
                    continue
                drop = (bv - nv) * 100.0
                if drop > self.max_regression_points:
                    reasons.append(
                        f"regression kill: {k} dropped {drop:.1f}pts "
                        f"(baseline={bv:.3f}, new={nv:.3f})"
                    )
                    return "kill", reasons

        # Capability gate on target metric.
        target = val_record.get(self.target_metric)
        if target is None:
            reasons.append(f"undecided: target_metric '{self.target_metric}' not in val record")
            return "undecided", reasons

        if self.min_absolute is not None and target < self.min_absolute:
            reasons.append(
                f"capability kill: {self.target_metric}={target:.3f} "
                f"< min_absolute {self.min_absolute:.3f}"
            )
            return "kill", reasons

        if self.baseline_value is not None:
            gain = (target - self.baseline_value) * 100.0
            if gain < self.min_gain_points:
                reasons.append(
                    f"capability kill: {self.target_metric} gain {gain:.1f}pts "
                    f"< required {self.min_gain_points:.1f}pts "
                    f"(baseline={self.baseline_value:.3f}, new={target:.3f})"
                )
                return "kill", reasons
            reasons.append(
                f"promote: {self.target_metric} gain {gain:.1f}pts "
                f"(baseline={self.baseline_value:.3f}, new={target:.3f})"
            )
            return "promote", reasons

        reasons.append(
            f"promote (no baseline): {self.target_metric}={target:.3f} "
            f">= min_absolute {self.min_absolute}"
        )
        return "promote", reasons


@dataclass
class BranchConfig:
    """
    Single-branch experiment specification. One BranchConfig drives one run.

    World overrides are *deltas* on the baseline WorldConfig; unset means
    inherit the baseline value. Same for tier and HPM overrides.
    """
    branch_id: str                                    # e.g. "A1"
    family: str                                        # "A" | "B" | "C"
    description: str

    # --- world overrides (family A) ---
    min_entities: Optional[int] = None
    max_entities: Optional[int] = None
    grid_h: Optional[int] = None
    grid_w: Optional[int] = None
    noise_std: Optional[float] = None

    # --- tier overrides (family A) ---
    tier3_episode_length: Optional[int] = None        # A2
    tier3_max_delay: Optional[int] = None
    tier3_template_pool: Optional[Tuple[str, ...]] = None
    chain2_frequency_boost: float = 1.0               # A3: multiplier on chain2 prob
    chain2_temporal_overlap: bool = False             # A3: allow chain2 to arm while chain1 pending
    rule_dynamic: bool = False                        # A4: active_rule alters downstream dynamics

    # --- query overrides (family B) ---
    extra_query_families: Tuple[str, ...] = ()        # names added to EXTENDED_QUERY_TYPES
    replace_query_families: Optional[Tuple[str, ...]] = None  # if set, overrides entirely

    # --- HPM overrides (family C) ---
    hpm_n_slots: Optional[int] = None
    hpm_read_mode: Optional[str] = None               # "concat" | "mean" | "attn"
    hpm_competitive: Optional[bool] = None
    hpm_slot_dim: Optional[int] = None
    hpm_retroactive_window: Optional[int] = None      # C3: window size for retroactive binding
    hpm_slot_timescales: Optional[Tuple[float, ...]] = None  # C4: per-slot persistence multipliers

    # --- D-family: entity memory overrides ---
    enable_entity_table: bool = False                 # EntityTable (GRU-based entity state)
    enable_event_tape: bool = False                   # EventTape (surprise-boundary snapshots)
    enable_entity_history: bool = False               # EntityHistoryBank (uniform snapshots)
    entity_history_n_snapshots: int = 16              # K for EntityHistoryBank

    # --- training overrides ---
    epochs_per_tier: int = 4
    batch_size: int = 32
    lr: Optional[float] = None
    train_episodes_per_tier: int = 2048
    tiers_to_run: Tuple[int, ...] = (1, 2, 3)         # can focus on a tier
    seed: int = 0

    # --- promotion rubric ---
    rubric: Optional[PromotionRubric] = None

    # --- ET-only read ablation ---
    # Query type names that should use entity-table-only memory in
    # IterativeQueryHead (no event tape). Empty = disabled (normal fused read).
    et_only_read_qtypes: Tuple[str, ...] = ()

    # --- Memory-source ablation (E-family) ---
    # Global override for IterativeQueryHead memory routing.
    # "fused" = default (all memory sources), "et_only" = entity table only,
    # "tape_only" = event tape + entity history only, "no_aux" = skip cross-attn.
    memory_ablation_mode: str = "fused"
    # --- HPM state-machine ablation ---
    # When True, use continuous_plasticity instead of OPEN/CLOSING/LOCKED.
    hpm_continuous_plasticity: Optional[bool] = None

    # --- Stability controls ---
    # Per-tier LR multiplier. E.g. {2: 0.5, 3: 0.3} halves LR at tier 2.
    tier_lr_scale: Optional[Dict[int, float]] = None
    # Holder loss weight. 0.0 = disabled. None = use TrainConfig default (0.3).
    holder_loss_weight: Optional[float] = None
    # When False, HPM diversity loss is zeroed out.
    diversity_loss_enabled: Optional[bool] = None

    # --- flags for sanity ---
    requires_step_patch: bool = False                 # A3/A4 — needs _step_world extension

    def short_hash(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, default=str).encode()
        return hashlib.sha1(blob).hexdigest()[:8]


# =============================================================================
# B-family query contract
# =============================================================================
# Each entry describes the shape a new query family must produce in
# generate_episode_with_diagnostics. The runner does not create these
# queries itself; it only registers the names so the QueryHead output
# head can be sized correctly and the evaluator's per-family bucketing
# picks them up.
#
# Contract fields per query family name:
#   answer_kind:    "entity_id" | "binary" | "latent_rule"
#   expected_qacc_metric: e.g. "qacc/when_was_first_occlusion"
#   description:    plain-English statement of what the query asks
#   requires_step_patch: True if producing a ground-truth answer needs
#                        world-state info not currently recorded in the
#                        episode trace. These need tmew1_diagnostics to
#                        be extended before the branch can run.
#
B_QUERY_SPECS: Dict[str, Dict[str, Any]] = {
    # --- B1 temporal ordering ---
    "did_occlusion_before_handoff": {
        "answer_kind": "binary",
        "expected_qacc_metric": "qacc/did_occlusion_before_handoff",
        "description": (
            "Did any occlusion event happen before the first handoff? "
            "Tests temporal comparison across unlike event types."
        ),
        "requires_step_patch": False,
    },
    "did_chain2_before_chain1": {
        "answer_kind": "binary",
        "expected_qacc_metric": "qacc/did_chain2_before_chain1",
        "description": (
            "Did chain2 fire before chain1? Tests concurrent-thread ordering."
        ),
        "requires_step_patch": False,
    },
    "which_event_was_first": {
        "answer_kind": "latent_rule",   # reuses num_latent_rules slots as event-type codes
        "expected_qacc_metric": "qacc/which_event_was_first",
        "description": (
            "Given the set of event types that fired this episode, "
            "which fired first? (event-type classifier)."
        ),
        "requires_step_patch": False,
    },

    # --- B2 relational ---
    "closest_entity_to_holder_at_alarm": {
        "answer_kind": "entity_id",
        "expected_qacc_metric": "qacc/closest_entity_to_holder_at_alarm",
        "description": (
            "At the step the alarm fired, which entity was closest "
            "(Manhattan distance) to the current holder?"
        ),
        "requires_step_patch": False,
    },
    "which_entity_visible_at_correction": {
        "answer_kind": "entity_id",
        "expected_qacc_metric": "qacc/which_entity_visible_at_correction",
        "description": (
            "In false_cue episodes, which entity was visible at the "
            "correction step?"
        ),
        "requires_step_patch": False,
    },
    "entity_sharing_color_with_trigger": {
        "answer_kind": "entity_id",
        "expected_qacc_metric": "qacc/entity_sharing_color_with_trigger",
        "description": (
            "Which entity shares color with the trigger source "
            "(and is not the trigger source itself)?"
        ),
        "requires_step_patch": False,
    },

    # --- B3 negation ---
    "entity_never_occluded": {
        "answer_kind": "entity_id",
        "expected_qacc_metric": "qacc/entity_never_occluded",
        "description": (
            "Which entity was never occluded across the entire episode?"
        ),
        "requires_step_patch": False,
    },
    "chain_never_fired": {
        "answer_kind": "binary",    # encoded as chain1-only / chain2-only / both / neither — collapsed to binary per pair
        "expected_qacc_metric": "qacc/chain_never_fired",
        "description": (
            "Did exactly one chain fire (vs. both or neither)? "
            "Tests global episode bookkeeping under negation."
        ),
        "requires_step_patch": False,
    },
    "entity_never_tagged": {
        "answer_kind": "entity_id",
        "expected_qacc_metric": "qacc/entity_never_tagged",
        "description": (
            "Which active entity was never tagged by any trigger?"
        ),
        "requires_step_patch": False,
    },

    # --- B4 counterfactual (reserve) ---
    "holder_if_handoff2_absent": {
        "answer_kind": "entity_id",
        "expected_qacc_metric": "qacc/holder_if_handoff2_absent",
        "description": (
            "Who would hold the token at episode end if handoff #2 "
            "had not occurred? Uses counterfactual handoff-chain replay."
        ),
        "requires_step_patch": False,
    },
    "would_alarm_fire_without_correction": {
        "answer_kind": "binary",
        "expected_qacc_metric": "qacc/would_alarm_fire_without_correction",
        "description": (
            "Would the alarm have fired if the correction cue had not "
            "occurred? Uses counterfactual world-state replay from snapshot."
        ),
        "requires_step_patch": False,
    },
}


# =============================================================================
# Branch presets
# =============================================================================
def make_branch_preset(branch_id: str) -> BranchConfig:
    """
    Returns a sensible default BranchConfig for each of the Phase-1 branches.
    CLI flags on the runner override individual fields.

    Rubrics here use *no* baseline_value by default — the caller must fill
    it in from their frozen Phase-0 reference run before interpreting
    verdicts. Until then, only `min_absolute` and `stability_floors` apply.
    """
    if branch_id == "A1":
        return BranchConfig(
            branch_id="A1",
            family="A",
            description="Entity scaling: enforce 6-8 entities with more distractors",
            min_entities=6,
            max_entities=8,
            rubric=PromotionRubric(
                target_metric="qacc/who_holds_token",
                # baseline_value filled in from frozen Phase-0 run
                min_gain_points=-10.0,  # degradation allowed up to -10pts is still pass for A1
                min_absolute=0.35,
                max_regression_points=10.0,  # harder world, larger regression bar
                stability_floors={"latent_acc": 0.85, "qacc/did_alarm_fire": 0.85},
            ),
        )

    if branch_id == "A2":
        return BranchConfig(
            branch_id="A2",
            family="A",
            description="Horizon scaling: tier3 T=64 -> 96",
            tier3_episode_length=96,
            tier3_max_delay=18,
            rubric=PromotionRubric(
                target_metric="qacc/who_holds_token",
                min_gain_points=-5.0,
                min_absolute=0.40,
                max_regression_points=5.0,
                stability_floors={"latent_acc": 0.90},
            ),
        )

    if branch_id == "A3":
        return BranchConfig(
            branch_id="A3",
            family="A",
            description="Multi-chain concurrency: higher chain2 freq, temporal overlap",
            chain2_frequency_boost=2.0,
            chain2_temporal_overlap=True,
            tier3_template_pool=(
                "multi_chain", "multi_chain", "multi_chain",
                "handoff", "handoff", "false_cue",
                "trigger_delay", "occlusion_identity",
            ),
            rubric=PromotionRubric(
                target_metric="qacc/did_chain2_fire",
                min_absolute=0.90,
                max_regression_points=5.0,
                stability_floors={"latent_acc": 0.90, "qacc/what_was_true_rule": 0.85},
            ),
        )

    if branch_id == "A4":
        return BranchConfig(
            branch_id="A4",
            family="A",
            description="Rule-dynamic worlds: active_rule alters entity motion, handoff, and alarm timing",
            rule_dynamic=True,
            rubric=PromotionRubric(
                target_metric="qacc/what_was_true_rule",
                min_absolute=0.85,
                stability_floors={"latent_acc": 0.90},
            ),
        )

    if branch_id == "B1":
        return BranchConfig(
            branch_id="B1",
            family="B",
            description="Temporal-ordering queries",
            extra_query_families=(
                "did_occlusion_before_handoff",
                "did_chain2_before_chain1",
                "which_event_was_first",
            ),
            rubric=PromotionRubric(
                target_metric="qacc/did_chain2_before_chain1",
                min_absolute=0.75,
                max_regression_points=3.0,
            ),
        )

    if branch_id == "B2":
        return BranchConfig(
            branch_id="B2",
            family="B",
            description="Relational queries (closest-to, shared-color, visible-at)",
            extra_query_families=(
                "closest_entity_to_holder_at_alarm",
                "entity_sharing_color_with_trigger",
                "which_entity_visible_at_correction",
            ),
            rubric=PromotionRubric(
                target_metric="qacc/closest_entity_to_holder_at_alarm",
                min_absolute=0.25,
                max_regression_points=3.0,
            ),
        )

    if branch_id == "B3":
        return BranchConfig(
            branch_id="B3",
            family="B",
            description="Negation queries (never-occluded, never-tagged, exactly-one-chain)",
            extra_query_families=(
                "entity_never_occluded",
                "entity_never_tagged",
                "chain_never_fired",
            ),
            rubric=PromotionRubric(
                target_metric="qacc/entity_never_occluded",
                min_absolute=0.50,
                max_regression_points=3.0,
            ),
        )

    if branch_id == "B4":
        return BranchConfig(
            branch_id="B4",
            family="B",
            description="Counterfactual queries (handoff-absent holder, alarm-without-correction)",
            extra_query_families=(
                "holder_if_handoff2_absent",
                "would_alarm_fire_without_correction",
            ),
            rubric=PromotionRubric(
                target_metric="qacc/would_alarm_fire_without_correction",
                min_absolute=0.70,
                max_regression_points=3.0,
            ),
        )

    if branch_id == "C1":
        return BranchConfig(
            branch_id="C1",
            family="C",
            description="HPM Level-2: 4 slots, competitive writes, concat readout",
            hpm_n_slots=4,
            hpm_competitive=True,
            hpm_read_mode="concat",
            rubric=PromotionRubric(
                target_metric="qacc/who_holds_token",
                min_gain_points=8.0,
                max_regression_points=3.0,
                stability_floors={"latent_acc": 0.90},
            ),
        )

    if branch_id == "C2":
        return BranchConfig(
            branch_id="C2",
            family="C",
            description="HPM read-mode ablation: attn readout, 4 slots, no competition",
            hpm_n_slots=4,
            hpm_competitive=False,
            hpm_read_mode="attn",
            rubric=PromotionRubric(
                target_metric="qacc/who_holds_token",
                min_gain_points=5.0,
                max_regression_points=3.0,
                stability_floors={"latent_acc": 0.90},
            ),
        )

    if branch_id == "C3":
        return BranchConfig(
            branch_id="C3",
            family="C",
            description="HPM Level-3: retroactive binding over 4-step window",
            hpm_n_slots=4,
            hpm_competitive=True,
            hpm_read_mode="concat",
            hpm_retroactive_window=4,
            rubric=PromotionRubric(
                target_metric="qacc/who_holds_token",
                min_gain_points=5.0,
                max_regression_points=3.0,
                stability_floors={"latent_acc": 0.90},
            ),
        )

    if branch_id == "C4":
        return BranchConfig(
            branch_id="C4",
            family="C",
            description="HPM Level-4: multi-timescale slots (fast/slow memory lanes)",
            hpm_n_slots=4,
            hpm_competitive=False,
            hpm_read_mode="concat",
            # Slot 0: fast (timescale 0.5 => gate doubled, rapid turnover)
            # Slot 1: normal (1.0)
            # Slot 2: slow (2.0 => gate halved)
            # Slot 3: glacial (4.0 => gate quartered, episode-level retention)
            hpm_slot_timescales=(0.5, 1.0, 2.0, 4.0),
            rubric=PromotionRubric(
                target_metric="qacc/who_holds_token",
                min_gain_points=5.0,
                max_regression_points=3.0,
                stability_floors={"latent_acc": 0.90},
            ),
        )

    # ── D family: diagnostic / ablation branches ────────────────────────
    if branch_id == "D1":
        return BranchConfig(
            branch_id="D1",
            family="D",
            description="ET-only read ablation for holder/relational probe queries",
            enable_entity_table=True,
            enable_event_tape=True,
            et_only_read_qtypes=(
                "who_holds_token",
                "holder_if_handoff2_absent",
                "closest_entity_to_holder_at_alarm",
            ),
            extra_query_families=(
                "closest_entity_to_holder_at_alarm",
                "holder_if_handoff2_absent",
            ),
            rubric=PromotionRubric(
                target_metric="qacc/who_holds_token",
                min_gain_points=2.0,
                max_regression_points=3.0,
            ),
        )

    if branch_id == "D2":
        return BranchConfig(
            branch_id="D2",
            family="D",
            description="EntityHistoryBank K=16 snapshots, no ET-only routing",
            enable_entity_table=True,
            enable_event_tape=True,
            enable_entity_history=True,
            entity_history_n_snapshots=16,
            extra_query_families=(
                "closest_entity_to_holder_at_alarm",
                "holder_if_handoff2_absent",
            ),
            rubric=PromotionRubric(
                target_metric="qacc/closest_entity_to_holder_at_alarm",
                min_gain_points=2.0,
                max_regression_points=3.0,
            ),
        )

    if branch_id == "D3":
        return BranchConfig(
            branch_id="D3",
            family="D",
            description="EntityHistoryBank K=16 + ET-only routing (combined D1+D2)",
            enable_entity_table=True,
            enable_event_tape=True,
            enable_entity_history=True,
            entity_history_n_snapshots=16,
            et_only_read_qtypes=(
                "who_holds_token",
                "holder_if_handoff2_absent",
                "closest_entity_to_holder_at_alarm",
            ),
            extra_query_families=(
                "closest_entity_to_holder_at_alarm",
                "holder_if_handoff2_absent",
            ),
            rubric=PromotionRubric(
                target_metric="qacc/closest_entity_to_holder_at_alarm",
                min_gain_points=2.0,
                max_regression_points=3.0,
            ),
        )

    if branch_id == "AB1":
        return BranchConfig(
            branch_id="AB1",
            family="AB",
            description="Combined: A3 multi-chain concurrency + B4 counterfactual queries",
            # A3 world overrides
            chain2_frequency_boost=2.0,
            chain2_temporal_overlap=True,
            tier3_template_pool=(
                "multi_chain", "multi_chain", "multi_chain",
                "handoff", "handoff", "false_cue",
                "trigger_delay", "occlusion_identity",
            ),
            # B4 query overrides
            extra_query_families=(
                "holder_if_handoff2_absent",
                "would_alarm_fire_without_correction",
            ),
            rubric=PromotionRubric(
                target_metric="qacc/would_alarm_fire_without_correction",
                min_absolute=0.70,
                max_regression_points=5.0,
                stability_floors={
                    "latent_acc": 0.90,
                    "qacc/did_chain2_fire": 0.90,
                    "qacc/what_was_true_rule": 0.85,
                },
            ),
        )

    # ── E family: sharp ablations on A3-style worlds ────────────────────
    # These are the "prove which memory source helps" experiments.
    # All use A3 world (multi-chain concurrency) + B2/B4 queries as stressor.
    # Target metrics: who_holds_token, holder_if_handoff2_absent,
    # closest_entity_to_holder_at_alarm.
    _E_WORLD = dict(
        chain2_frequency_boost=2.0,
        chain2_temporal_overlap=True,
        tier3_template_pool=(
            "multi_chain", "multi_chain", "multi_chain",
            "handoff", "handoff", "false_cue",
            "trigger_delay", "occlusion_identity",
        ),
    )
    _E_QUERIES = (
        "closest_entity_to_holder_at_alarm",
        "holder_if_handoff2_absent",
    )
    _E_RUBRIC = PromotionRubric(
        target_metric="qacc/who_holds_token",
        min_absolute=0.45,
        max_regression_points=5.0,
        stability_floors={
            "latent_acc": 0.85,
            "qacc/did_alarm_fire": 0.80,
        },
    )
    _E_STABILITY = dict(
        tier_lr_scale={2: 0.5, 3: 0.3},
        holder_loss_weight=0.3,
        diversity_loss_enabled=False,
    )

    if branch_id == "E1":
        return BranchConfig(
            branch_id="E1",
            family="E",
            description="Memory ablation: fused read (control) on A3 world + B2/B4 queries",
            enable_entity_table=True,
            enable_event_tape=True,
            enable_entity_history=True,
            memory_ablation_mode="fused",
            extra_query_families=_E_QUERIES,
            rubric=_E_RUBRIC,
            **_E_WORLD,
            **_E_STABILITY,
        )

    if branch_id == "E2":
        return BranchConfig(
            branch_id="E2",
            family="E",
            description="Memory ablation: entity-table-only on A3 world + B2/B4 queries",
            enable_entity_table=True,
            enable_event_tape=True,
            enable_entity_history=True,
            memory_ablation_mode="et_only",
            extra_query_families=_E_QUERIES,
            rubric=_E_RUBRIC,
            **_E_WORLD,
            **_E_STABILITY,
        )

    if branch_id == "E3":
        return BranchConfig(
            branch_id="E3",
            family="E",
            description="Memory ablation: tape-only on A3 world + B2/B4 queries",
            enable_entity_table=True,
            enable_event_tape=True,
            enable_entity_history=True,
            memory_ablation_mode="tape_only",
            extra_query_families=_E_QUERIES,
            rubric=_E_RUBRIC,
            **_E_WORLD,
            **_E_STABILITY,
        )

    if branch_id == "E4":
        return BranchConfig(
            branch_id="E4",
            family="E",
            description="Memory ablation: no auxiliary memory on A3 world + B2/B4 queries",
            enable_entity_table=True,
            enable_event_tape=True,
            enable_entity_history=True,
            memory_ablation_mode="no_aux",
            extra_query_families=_E_QUERIES,
            rubric=_E_RUBRIC,
            **_E_WORLD,
            **_E_STABILITY,
        )

    if branch_id == "E5":
        return BranchConfig(
            branch_id="E5",
            family="E",
            description="HPM state-machine ablation: continuous plasticity on A3 world",
            enable_entity_table=True,
            enable_event_tape=True,
            enable_entity_history=True,
            memory_ablation_mode="fused",
            hpm_continuous_plasticity=True,
            extra_query_families=_E_QUERIES,
            rubric=_E_RUBRIC,
            **_E_WORLD,
            **_E_STABILITY,
        )

    raise ValueError(f"Unknown branch_id: {branch_id}")


# =============================================================================
# Applying branch overrides to the existing config objects
# =============================================================================
def apply_world_overrides(branch: BranchConfig) -> "WorldConfig":
    WorldConfig, *_ = _lazy_imports()
    kwargs: Dict[str, Any] = {}
    if branch.min_entities is not None:
        kwargs["min_entities"] = branch.min_entities
    if branch.max_entities is not None:
        kwargs["max_entities"] = branch.max_entities
    if branch.chain2_frequency_boost != 1.0:
        kwargs["chain2_frequency_boost"] = branch.chain2_frequency_boost
    if branch.chain2_temporal_overlap:
        kwargs["chain2_temporal_overlap"] = True
    if branch.rule_dynamic:
        kwargs["rule_dynamic"] = True
    if branch.grid_h is not None:
        kwargs["grid_h"] = branch.grid_h
    if branch.grid_w is not None:
        kwargs["grid_w"] = branch.grid_w
    if branch.noise_std is not None:
        kwargs["noise_std"] = branch.noise_std
    return WorldConfig(**kwargs)


def apply_tier_overrides(
    branch: BranchConfig,
) -> Tuple["CurriculumTier", ...]:
    _, CurriculumTier, DEFAULT_TIERS, _, _ = _lazy_imports()
    tiers = []
    for t in DEFAULT_TIERS:
        if t.tier not in branch.tiers_to_run:
            # Keep the tier unchanged; run_curriculum will skip it via
            # `tiers` filtering below.
            tiers.append(t)
            continue
        if t.tier == 3:
            t = replace(
                t,
                max_episode_length=branch.tier3_episode_length or t.max_episode_length,
                max_delay=branch.tier3_max_delay or t.max_delay,
                template_pool=branch.tier3_template_pool or t.template_pool,
            )
        tiers.append(t)
    return tuple(x for x in tiers if x.tier in branch.tiers_to_run)


def apply_hpm_overrides(branch: BranchConfig) -> "HPMConfig":
    _, _, _, _, HPMConfig = _lazy_imports()
    base = HPMConfig()
    kwargs: Dict[str, Any] = {}
    if branch.hpm_n_slots is not None:
        kwargs["n_slots"] = branch.hpm_n_slots
    if branch.hpm_read_mode is not None:
        kwargs["read_mode"] = branch.hpm_read_mode
    if branch.hpm_competitive is not None:
        kwargs["competitive"] = branch.hpm_competitive
    if branch.hpm_slot_dim is not None:
        kwargs["slot_dim"] = branch.hpm_slot_dim
    if branch.hpm_retroactive_window is not None:
        kwargs["retroactive_window"] = branch.hpm_retroactive_window
    if branch.hpm_slot_timescales is not None:
        kwargs["slot_timescales"] = branch.hpm_slot_timescales
    if branch.hpm_continuous_plasticity is not None:
        kwargs["continuous_plasticity"] = branch.hpm_continuous_plasticity
    if not kwargs:
        return base
    return replace(base, **kwargs)


def apply_train_overrides(branch: BranchConfig) -> "TrainConfig":
    import torch
    _, _, _, TrainConfig, _ = _lazy_imports()
    kwargs: Dict[str, Any] = {
        "batch_size": branch.batch_size,
        "train_episodes_per_tier": branch.train_episodes_per_tier,
        "epochs_per_tier": branch.epochs_per_tier,
        "seed": branch.seed,
    }
    if branch.lr is not None:
        kwargs["lr"] = branch.lr
    if branch.tier_lr_scale is not None:
        kwargs["tier_lr_scale"] = branch.tier_lr_scale
    if branch.holder_loss_weight is not None:
        kwargs["holder_loss_weight"] = branch.holder_loss_weight
    if branch.diversity_loss_enabled is not None:
        kwargs["diversity_loss_enabled"] = branch.diversity_loss_enabled
    return TrainConfig(**{k: v for k, v in kwargs.items() if v is not None})


# =============================================================================
# Missing-capability guard
# =============================================================================
def check_missing_capabilities(branch: BranchConfig) -> List[str]:
    """
    Return a list of human-readable reasons the branch cannot run as-is
    given the current codebase. Empty list means safe to launch.
    """
    missing: List[str] = []

    if branch.requires_step_patch:
        missing.append(
            f"Branch {branch.branch_id} needs tmew1_train._step_world and/or "
            "tmew1_diagnostics.generate_episode_with_diagnostics to be "
            "extended before it produces valid ground truth. See "
            "B_QUERY_SPECS and the `rule_dynamic` / `chain2_temporal_overlap` "
            "flags for which patch is needed."
        )

    if branch.extra_query_families or branch.replace_query_families:
        for q in (branch.replace_query_families or branch.extra_query_families):
            if q not in B_QUERY_SPECS:
                missing.append(
                    f"Branch {branch.branch_id}: query family '{q}' not "
                    "declared in B_QUERY_SPECS; add it there first."
                )
            elif B_QUERY_SPECS[q]["requires_step_patch"]:
                missing.append(
                    f"Branch {branch.branch_id}: query family '{q}' requires "
                    "tmew1_diagnostics to cache extra episode ground truth "
                    "(see B_QUERY_SPECS[...]['description'])."
                )

    return missing


# =============================================================================
# Verdict writer
# =============================================================================
def write_verdict(
    branch: BranchConfig,
    val_record: Dict[str, float],
    baseline_record: Optional[Dict[str, float]],
    out_dir: str,
) -> Tuple[str, List[str]]:
    """
    Apply the rubric, dump a JSONL record with the full diagnostic
    context, and return (verdict, reasons).
    """
    os.makedirs(out_dir, exist_ok=True)
    verdict = "undecided"
    reasons: List[str] = ["no rubric attached"]
    if branch.rubric is not None:
        verdict, reasons = branch.rubric.verdict(val_record, baseline_record)

    record = {
        "timestamp": time.time(),
        "branch_id": branch.branch_id,
        "branch_family": branch.family,
        "branch_hash": branch.short_hash(),
        "config": asdict(branch),
        "val_record": val_record,
        "baseline_record": baseline_record,
        "verdict": verdict,
        "reasons": reasons,
    }
    path = os.path.join(out_dir, f"verdict_{branch.branch_id}_{branch.short_hash()}.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return verdict, reasons

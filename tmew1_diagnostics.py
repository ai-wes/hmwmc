"""
TMEW-1 diagnostics and false-cue template.

Adds three things on top of tmew1_queries.py without modifying it:
1. False-cue / belief-revision template — early audio cue implies one rule,
   later evidence corrects it. Adds 'what_was_true_rule' query.
2. Per-episode metadata (num_handoffs, had_false_cue, occlusion_steps) so
   diagnostics can bucket accuracy by difficulty.
3. recall_by_difficulty() — buckets who_holds_token accuracy by handoff count
   and logs episodic read entropy alongside, to distinguish write failures
   from read failures.

Usage from tmew1_run.py:
    from tmew1_diagnostics import (
        FALSE_CUE_QUERY_TYPES,
        generate_episode_with_diagnostics,
        recall_by_difficulty,
        log_episodic_entropy,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import random
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tmew1_train import WorldConfig, CurriculumTier, _make_world, _render_vision, _render_audio, _render_numeric, _render_text
from score_logging import (
    ScoreColorMapper,
    MetricSpec,
    ScoreDirection,
    _Ansi,
)
from tmew1_queries import (
    QUERY_TYPES,
    QUERY_TYPE_TO_IDX,
    Query,
    EpisodeWithQueries,
    HandoffState,
    _step_world_with_handoff,
)


# -----------------------------------------------------------------------------
# Extended query type set: adds belief-revision query
# -----------------------------------------------------------------------------
EXTENDED_QUERY_TYPES: Tuple[str, ...] = QUERY_TYPES + ("what_was_true_rule",)
EXTENDED_QUERY_TYPE_TO_IDX: Dict[str, int] = {q: i for i, q in enumerate(EXTENDED_QUERY_TYPES)}


# -----------------------------------------------------------------------------
# Episode + per-episode difficulty metadata
# -----------------------------------------------------------------------------
@dataclass
class EpisodeWithDiagnostics:
    vision: np.ndarray
    audio: np.ndarray
    numeric: np.ndarray
    text: np.ndarray
    latent_rule: int           # the TRUE latent rule, post-correction
    template: str
    length: int
    queries: List[Query]
    # Difficulty metadata
    num_handoffs: int
    had_false_cue: bool
    cue_corrected_at: int      # -1 if no correction
    occlusion_steps: int


# -----------------------------------------------------------------------------
# False-cue template
# -----------------------------------------------------------------------------
def _inject_false_cue(audio_vec: np.ndarray, t: int, false_cue_step: int, correction_step: int) -> np.ndarray:
    """Audio channels 5,6,7 carry the false-cue signal and its correction."""
    if t == false_cue_step:
        audio_vec[5] = 1.0      # initial misleading high-frequency burst
    if correction_step <= t < correction_step + 4:
        audio_vec[6] = 1.0      # durable correction pulse
    if false_cue_step <= t < correction_step:
        audio_vec[7] = 0.4      # ambient "stale belief" hum
    return audio_vec


def generate_episode_with_diagnostics(
    cfg: WorldConfig,
    tier: CurriculumTier,
    seed: int,
    num_queries: int = 4,
    enable_false_cue: bool = False,
) -> EpisodeWithDiagnostics:
    rng = random.Random(seed)
    template_pool = list(tier.template_pool)
    if enable_false_cue and "false_cue" not in template_pool:
        template_pool.append("false_cue")
    template = rng.choice(template_pool)

    state = _make_world(cfg, template, tier.max_delay, tier.occlusion, seed)
    handoff = HandoffState(holder_id=state.entities[0].id, transfer_history=[])

    T = tier.max_episode_length

    # False-cue scheduling: cue at ~25%, correction at ~60% of episode.
    has_false_cue = template == "false_cue"
    false_cue_step = T // 4 if has_false_cue else -1
    correction_step = (T * 3) // 5 if has_false_cue else -1
    decoy_rule = rng.randint(0, cfg.num_latent_rules - 1) if has_false_cue else -1
    true_rule = state.active_rule

    vision = np.zeros((T, cfg.vision_channels, cfg.grid_h, cfg.grid_w), dtype=np.float32)
    audio = np.zeros((T, cfg.audio_dim), dtype=np.float32)
    numeric = np.zeros((T, cfg.numeric_dim), dtype=np.float32)
    text = np.zeros((T, 1), dtype=np.int64)

    fired_at = -1
    last_occluded_id = -1
    occlusion_steps = 0

    for t in range(T):
        # During the false-cue window, temporarily lie about active_rule in the numeric channel.
        if has_false_cue and false_cue_step <= t < correction_step:
            true_rule_saved = state.active_rule
            state.active_rule = decoy_rule

        events = _step_world_with_handoff(state, handoff, cfg, t, tier.max_delay)
        vision[t] = _render_vision(state, cfg)
        audio[t] = _render_audio(state, cfg, events)
        audio[t] = _inject_false_cue(audio[t], t, false_cue_step, correction_step)
        numeric[t] = _render_numeric(state, cfg)
        text[t] = _render_text(state, cfg, events)

        if has_false_cue and false_cue_step <= t < correction_step:
            state.active_rule = true_rule_saved

        if events.get("alarm_fire") and fired_at < 0:
            fired_at = t
        if events.get("occluded_ids"):
            last_occluded_id = events["occluded_ids"][0]
            occlusion_steps += 1

    # Build queries — same logic as tmew1_queries but with the new query type added
    # for false_cue episodes.
    q_rng = random.Random(seed + 7)
    queries: List[Query] = []
    back_half_start = max(1, T // 2)
    pool = list(EXTENDED_QUERY_TYPES) if has_false_cue else list(QUERY_TYPES)

    for _ in range(num_queries):
        qtype = q_rng.choice(pool)
        time_asked = q_rng.randint(back_half_start, T - 1)
        if qtype == "who_holds_token":
            queries.append(Query(qtype, handoff.holder_id, time_asked, is_binary=False))
        elif qtype == "who_was_first_tagged":
            queries.append(Query(qtype, max(0, handoff.first_tagged_id), time_asked, is_binary=False))
        elif qtype == "did_alarm_fire":
            target = 1 if 0 <= fired_at <= time_asked else 0
            queries.append(Query(qtype, target, time_asked, is_binary=True))
        elif qtype == "which_entity_occluded":
            queries.append(Query(qtype, max(0, last_occluded_id), time_asked, is_binary=False))
        elif qtype == "what_was_true_rule":
            queries.append(Query(qtype, true_rule, time_asked, is_binary=False))

    return EpisodeWithDiagnostics(
        vision=vision,
        audio=audio,
        numeric=numeric,
        text=text,
        latent_rule=true_rule,
        template=template,
        length=T,
        queries=queries,
        num_handoffs=len(handoff.transfer_history),
        had_false_cue=has_false_cue,
        cue_corrected_at=correction_step,
        occlusion_steps=occlusion_steps,
    )


def episode_to_diag_tensors(ep: EpisodeWithDiagnostics) -> Dict[str, Tensor]:
    qtimes = torch.tensor([q.time_asked for q in ep.queries], dtype=torch.long)
    qtypes = torch.tensor([EXTENDED_QUERY_TYPE_TO_IDX[q.qtype] for q in ep.queries], dtype=torch.long)
    qtargets = torch.tensor([q.target for q in ep.queries], dtype=torch.long)
    qbinary = torch.tensor([q.is_binary for q in ep.queries], dtype=torch.bool)
    return {
        "vision": torch.from_numpy(ep.vision),
        "audio": torch.from_numpy(ep.audio),
        "numeric": torch.from_numpy(ep.numeric),
        "text": torch.from_numpy(ep.text).squeeze(-1),
        "latent_rule": torch.tensor(ep.latent_rule, dtype=torch.long),
        "query_times": qtimes,
        "query_types": qtypes,
        "query_targets": qtargets,
        "query_is_binary": qbinary,
        "num_handoffs": torch.tensor(ep.num_handoffs, dtype=torch.long),
        "had_false_cue": torch.tensor(int(ep.had_false_cue), dtype=torch.long),
        "occlusion_steps": torch.tensor(ep.occlusion_steps, dtype=torch.long),
    }


def collate_diag(batch: Sequence[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0].keys()}


class TMEW1DiagnosticDataset(Dataset):
    def __init__(self, cfg: WorldConfig, tier: CurriculumTier, n: int, base_seed: int, num_queries: int = 4, enable_false_cue: bool = True):
        self.cfg = cfg
        self.tier = tier
        self.n = n
        self.base_seed = base_seed
        self.num_queries = num_queries
        self.enable_false_cue = enable_false_cue

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        ep = generate_episode_with_diagnostics(
            self.cfg, self.tier, seed=self.base_seed + idx,
            num_queries=self.num_queries, enable_false_cue=self.enable_false_cue,
        )
        return episode_to_diag_tensors(ep)


# -----------------------------------------------------------------------------
# Recall accuracy bucketed by handoff count + episodic entropy logging
# -----------------------------------------------------------------------------
@torch.no_grad()
def recall_by_difficulty(
    model,
    query_head,
    loader: DataLoader,
    device: str,
    enabled: Sequence[str] = ("vision", "numeric", "audio"),
) -> Dict[str, Any]:
    """
    Bucket who_holds_token accuracy by number of handoffs that occurred in
    the episode. Returns per-bucket accuracy plus mean episodic read entropy.

    The bucketing answers the question: does recall accuracy degrade with
    overwrite count? If yes -> episodic write gating is the bottleneck.
    If flat but low -> read attention is the bottleneck.
    """
    model.eval()
    query_head.eval()

    holds_idx = EXTENDED_QUERY_TYPE_TO_IDX["who_holds_token"]
    first_tagged_idx = EXTENDED_QUERY_TYPE_TO_IDX["who_was_first_tagged"]
    true_rule_idx = EXTENDED_QUERY_TYPE_TO_IDX.get("what_was_true_rule", -1)

    bucket_correct: Dict[str, List[int]] = {"0": [], "1": [], "2": [], "3+": []}
    first_tagged_by_handoffs: Dict[str, List[int]] = {"0": [], "1": [], "2": [], "3+": []}
    true_rule_by_falsecue: Dict[str, List[int]] = {"with_cue": [], "without_cue": []}
    entropy_samples: List[float] = []

    def bucket_for(n: int) -> str:
        return "3+" if n >= 3 else str(n)

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        vision_in = batch["vision"][:, :-1]
        audio_in = batch["audio"][:, :-1]
        numeric_in = batch["numeric"][:, :-1]
        text_in = batch["text"][:, :-1]

        output = model(
            text_tokens=text_in if "text" in enabled else None,
            vision=vision_in if "vision" in enabled else None,
            audio=audio_in if "audio" in enabled else None,
            numeric=numeric_in if "numeric" in enabled else None,
        )
        if output.episodic_read_entropy is not None:
            entropy_samples.append(float(output.episodic_read_entropy))

        t_max = output.sequence.size(1) - 1
        qtimes = batch["query_times"].clamp(max=t_max)
        entity_logits, binary_logits = query_head(output.sequence, qtimes, batch["query_types"])

        bs, q = batch["query_types"].shape
        for bi in range(bs):
            n_h = int(batch["num_handoffs"][bi].item())
            had_cue = bool(batch["had_false_cue"][bi].item())
            bucket = bucket_for(n_h)
            for qi in range(q):
                qtype = int(batch["query_types"][bi, qi].item())
                target = int(batch["query_targets"][bi, qi].item())
                if batch["query_is_binary"][bi, qi]:
                    pred = int(binary_logits[bi, qi].argmax().item())
                else:
                    pred = int(entity_logits[bi, qi].argmax().item())
                correct = int(pred == target)

                if qtype == holds_idx:
                    bucket_correct[bucket].append(correct)
                elif qtype == first_tagged_idx:
                    first_tagged_by_handoffs[bucket].append(correct)
                elif qtype == true_rule_idx and true_rule_idx >= 0:
                    key = "with_cue" if had_cue else "without_cue"
                    true_rule_by_falsecue[key].append(correct)

    model.train()
    query_head.train()

    def summarize(d: Dict[str, List[int]]) -> Dict[str, Any]:
        out = {}
        for k, v in d.items():
            out[k] = {"acc": (sum(v) / len(v)) if v else None, "n": len(v)}
        return out

    return {
        "who_holds_token_by_handoffs": summarize(bucket_correct),
        "who_was_first_tagged_by_handoffs": summarize(first_tagged_by_handoffs),
        "what_was_true_rule_by_falsecue": summarize(true_rule_by_falsecue),
        "mean_episodic_read_entropy": float(np.mean(entropy_samples)) if entropy_samples else None,
    }


def format_diagnostics(report: Dict[str, Any]) -> str:
    """Pretty-print the diagnostics report as a multi-line string with color."""
    mapper = ScoreColorMapper()
    acc_spec = MetricSpec(ScoreDirection.HIGHER_IS_BETTER, 0.0, 1.0)
    entropy_spec = MetricSpec(ScoreDirection.LOWER_IS_BETTER, 1.5, 3.5)

    def _color_val(value: float, spec: MetricSpec) -> str:
        scored = mapper.evaluate("v", value, spec)
        return f"{scored.band.color}{_Ansi.BOLD}{value:.3f}{_Ansi.RESET}"

    lines = ["", "  ===== Diagnostics ====="]
    ent = report.get("mean_episodic_read_entropy")
    if ent is not None:
        lines.append(f"  mean_episodic_read_entropy: {_color_val(ent, entropy_spec)}")
    else:
        lines.append(f"  mean_episodic_read_entropy: N/A")
    lines.append("  who_holds_token by handoffs:")
    for k, v in report["who_holds_token_by_handoffs"].items():
        if v["n"] > 0:
            lines.append(f"    handoffs={k:>2s}  acc={_color_val(v['acc'], acc_spec)}  n={v['n']}")
    lines.append("  who_was_first_tagged by handoffs:")
    for k, v in report["who_was_first_tagged_by_handoffs"].items():
        if v["n"] > 0:
            lines.append(f"    handoffs={k:>2s}  acc={_color_val(v['acc'], acc_spec)}  n={v['n']}")
    lines.append("  what_was_true_rule (belief revision):")
    for k, v in report["what_was_true_rule_by_falsecue"].items():
        if v["n"] > 0:
            lines.append(f"    {k:>12s}  acc={_color_val(v['acc'], acc_spec)}  n={v['n']}")
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Convenience: build a diagnostic loader and run a one-shot report
# -----------------------------------------------------------------------------
def run_diagnostic_report(
    model,
    query_head,
    world_cfg: WorldConfig,
    tier: CurriculumTier,
    device: str,
    n_episodes: int = 256,
    base_seed: int = 99000,
) -> Dict[str, Any]:
    ds = TMEW1DiagnosticDataset(world_cfg, tier, n=n_episodes, base_seed=base_seed, enable_false_cue=True)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_diag, num_workers=0)
    report = recall_by_difficulty(model, query_head, loader, device, enabled=tier.enabled_modalities)
    print(format_diagnostics(report))
    return report


__all__ = [
    "EXTENDED_QUERY_TYPES",
    "EXTENDED_QUERY_TYPE_TO_IDX",
    "EpisodeWithDiagnostics",
    "TMEW1DiagnosticDataset",
    "generate_episode_with_diagnostics",
    "episode_to_diag_tensors",
    "collate_diag",
    "recall_by_difficulty",
    "format_diagnostics",
    "run_diagnostic_report",
]

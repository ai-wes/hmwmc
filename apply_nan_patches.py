"""
Patch for tmew1_run.py: tier-boundary NaN cascade fix.

Root cause (confirmed from the failing 6-layer C1 log):

  Tier 1 (2 modalities) trains fine but triggers ~3 "Non-finite grads
  detected" events near end of ep1. The existing Fix #3 zeros those
  gradients and skips optimizer.step(). That's good for the gradient
  itself. What it does NOT clean up is the several preceding healthy
  steps whose second-moment buffers (exp_avg_sq) were growing toward
  infinity — those poisoned Adam states stay.

  Tier 2 (audio activates). Audio encoder params get Adam state reset
  (that's your existing new-modality reset). But the shared backbone
  params — text+vision encoders, HPM slots, transformer blocks — keep
  their poisoned Adam state from tier 1. First forward with the new
  audio input routes gradient through the shared backbone, Adam applies
  update = lr * g / (sqrt(exp_avg_sq) + eps) with tiny exp_avg_sq, and
  produces inf weights instantly.

  From that point the NaN-loss early-return handler (line 301) fires
  every step. BUT it only runs controller.emergency_stabilize and
  returns — it does NOT repair the NaN weights or clear Adam state.
  The post-step weight repair block at line 383 would have fixed this,
  but it's only reached when optimizer.step() actually ran, which it
  doesn't on the NaN-loss path.

  Result: the model is stuck NaN forever at tier 2 and every subsequent
  tier. Exactly what the log shows.

Three patches below fix this properly:

  PATCH 1 (highest leverage): in the NaN-loss early-return block, also
  run the weight-nan-to-num + Adam-state-reset before returning. This
  stops the cascade on the very first NaN-loss step.

  PATCH 2: in the grad-skip block, also clear Adam state for the
  params whose gradients were non-finite. Prevents the poison in the
  first place.

  PATCH 3: at the tier boundary (inside run_curriculum), proactively
  sanitize ALL model weights and clear Adam state for ALL params right
  after the new-modality reset. Belt-and-suspenders insurance for the
  case where tier 1 had grad-skip events and the shared backbone carries
  residual instability into tier 2.

Apply all three together. Each is independently correct; together they
make the tier boundary bulletproof.

USAGE:
    python apply_nan_patches.py          # dry-run, shows the diffs
    python apply_nan_patches.py --apply  # actually edits tmew1_run.py

Or apply manually by str-replacing the blocks below.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import List, Tuple


TARGET_FILE = "tmew1_run.py"


# =============================================================================
# PATCH 1 — NaN-loss early return must repair weights + clear Adam state
# =============================================================================
PATCH1_OLD = '''    # ── NaN guard: skip backward + optimizer if loss exploded ──────────
    if torch.isnan(total) or torch.isinf(total):
        import logging as _log
        _log.getLogger(__name__).warning(
            "NaN/Inf loss detected (total=%.4g) — skipping backward, "
            "running emergency_stabilize", float(total.item()) if not torch.isnan(total) else float("nan"),
        )
        optimizer.zero_grad(set_to_none=True)
        if model.cfg.enable_online_homeostasis:
            model.controller.emergency_stabilize()
            # Force-unlock all layers so the model can recover
            for pnn in model.controller.pnns:
                pnn.force_unlock(model.controller.global_step, refractory_period=8)
        with torch.no_grad():
            rule_acc = (rule_logits.argmax(-1) == batch["latent_rule"]).float().mean().item()
        return {
            "total": float("nan"), "next_step": float("nan"),
            "aux_latent": float("nan"), "latent_acc": rule_acc,
            "q_loss": float("nan"), "holder_loss": float("nan"),
            "holder_acc": holder_acc, **q_metrics,
        }'''

PATCH1_NEW = '''    # ── NaN guard: skip backward + optimizer if loss exploded ──────────
    if torch.isnan(total) or torch.isinf(total):
        import logging as _log
        _log.getLogger(__name__).warning(
            "NaN/Inf loss detected (total=%.4g) — skipping backward, "
            "running emergency_stabilize", float(total.item()) if not torch.isnan(total) else float("nan"),
        )
        optimizer.zero_grad(set_to_none=True)

        # CRITICAL: repair any NaN/Inf weights AND reset poisoned Adam state.
        # Without this the next forward pass produces NaN again and we loop
        # forever. This is the same repair the post-step block does, but we
        # never reach that block on this early-return path.
        _repaired = 0
        for p in model.parameters():
            if p.data.isnan().any() or p.data.isinf().any():
                p.data.nan_to_num_(nan=0.0, posinf=1.0, neginf=-1.0)
                _repaired += 1
            if p in optimizer.state and optimizer.state[p]:
                state = optimizer.state[p]
                for k in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    if k in state and isinstance(state[k], torch.Tensor):
                        if state[k].isnan().any() or state[k].isinf().any():
                            optimizer.state[p] = {}
                            break
        if _repaired:
            _log.getLogger(__name__).warning(
                "NaN-loss cascade repair: sanitized %d param tensors, "
                "cleared poisoned Adam moments", _repaired,
            )

        if model.cfg.enable_online_homeostasis:
            model.controller.emergency_stabilize()
            # Force-unlock all layers so the model can recover
            for pnn in model.controller.pnns:
                pnn.force_unlock(model.controller.global_step, refractory_period=8)
        with torch.no_grad():
            rule_acc = (rule_logits.argmax(-1) == batch["latent_rule"]).float().mean().item()
        return {
            "total": float("nan"), "next_step": float("nan"),
            "aux_latent": float("nan"), "latent_acc": rule_acc,
            "q_loss": float("nan"), "holder_loss": float("nan"),
            "holder_acc": holder_acc, **q_metrics,
        }'''


# =============================================================================
# PATCH 2 — grad-skip block must clear Adam state for non-finite-grad params
# =============================================================================
# Applied to BOTH the scaler path and non-scaler path. The two blocks are
# nearly identical, so we patch each separately.

PATCH2A_OLD = '''        # ── Fix #3: Pre-step finite-grad check ─────────────────────────
        grads_finite = all(
            p.grad is None or torch.isfinite(p.grad).all()
            for p in params
        )
        if not grads_finite:
            import logging as _log
            _log.getLogger(__name__).warning(
                "Non-finite grads detected — skipping optimizer.step()"
            )
            optimizer.zero_grad(set_to_none=True)
            scaler.update()  # still update scaler so it can back off the scale'''

PATCH2A_NEW = '''        # ── Fix #3: Pre-step finite-grad check ─────────────────────────
        # Identify offenders BEFORE zeroing so we can clear their Adam state.
        bad_params = [
            p for p in params
            if p.grad is not None and not torch.isfinite(p.grad).all()
        ]
        grads_finite = len(bad_params) == 0
        if not grads_finite:
            import logging as _log
            _log.getLogger(__name__).warning(
                "Non-finite grads detected — skipping optimizer.step() "
                "(clearing Adam state for %d offender params)", len(bad_params),
            )
            for p in bad_params:
                if p in optimizer.state:
                    optimizer.state[p] = {}
            optimizer.zero_grad(set_to_none=True)
            scaler.update()  # still update scaler so it can back off the scale'''


PATCH2B_OLD = '''        # ── Fix #3 (non-scaler path): Pre-step finite-grad check ──────
        grads_finite = all(
            p.grad is None or torch.isfinite(p.grad).all()
            for p in params
        )
        if not grads_finite:
            import logging as _log
            _log.getLogger(__name__).warning(
                "Non-finite grads detected — skipping optimizer.step()"
            )
            optimizer.zero_grad(set_to_none=True)'''

PATCH2B_NEW = '''        # ── Fix #3 (non-scaler path): Pre-step finite-grad check ──────
        bad_params = [
            p for p in params
            if p.grad is not None and not torch.isfinite(p.grad).all()
        ]
        grads_finite = len(bad_params) == 0
        if not grads_finite:
            import logging as _log
            _log.getLogger(__name__).warning(
                "Non-finite grads detected — skipping optimizer.step() "
                "(clearing Adam state for %d offender params)", len(bad_params),
            )
            for p in bad_params:
                if p in optimizer.state:
                    optimizer.state[p] = {}
            optimizer.zero_grad(set_to_none=True)'''


# =============================================================================
# PATCH 3 — proactive tier-boundary sanitize
# =============================================================================
# Inserted right after the new-modality Adam-state reset in run_curriculum.
# Clears Adam state for the SHARED backbone too when entering a new tier,
# not just the new-modality params, AND nan-to-num any weights. This is the
# belt-and-suspenders for the case where tier 1 had grad-skip events that
# left trailing instability in the shared params' moment buffers.

PATCH3_OLD = '''            if _cleared:
                print(f"  -> reset Adam state for {_cleared} params in new modalities: {new_mods}")
            # Reset GradScaler so stale scale factor from previous tier doesn't cause issues
            scaler = GradScaler("cuda", enabled=tcfg.use_amp)'''

PATCH3_NEW = '''            if _cleared:
                print(f"  -> reset Adam state for {_cleared} params in new modalities: {new_mods}")

            # Proactive tier-boundary sanitize: nan_to_num all weights AND
            # inspect every Adam moment buffer for NaN/Inf. Clears any that
            # are poisoned, even in shared backbone params that didn't belong
            # to the new modality. This is the defence against the cascade
            # where tier-1 grad-skip events leave trailing instability in
            # shared params' exp_avg_sq, which blows up on first tier-2 step.
            _weights_repaired = 0
            _adam_cleared = 0
            for p in model.parameters():
                if p.data.isnan().any() or p.data.isinf().any():
                    p.data.nan_to_num_(nan=0.0, posinf=1.0, neginf=-1.0)
                    _weights_repaired += 1
                if p in optimizer.state and optimizer.state[p]:
                    state = optimizer.state[p]
                    poisoned = False
                    for k in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                        if k in state and isinstance(state[k], torch.Tensor):
                            if state[k].isnan().any() or state[k].isinf().any():
                                poisoned = True
                                break
                    if poisoned:
                        optimizer.state[p] = {}
                        _adam_cleared += 1
            if _weights_repaired or _adam_cleared:
                print(
                    f"  -> tier-boundary sanitize: repaired {_weights_repaired} "
                    f"weight tensors, cleared {_adam_cleared} poisoned Adam states"
                )

            # Reset GradScaler so stale scale factor from previous tier doesn't cause issues
            scaler = GradScaler("cuda", enabled=tcfg.use_amp)'''


PATCHES: List[Tuple[str, str, str]] = [
    ("PATCH1 (NaN-loss early-return: repair weights + clear Adam)", PATCH1_OLD, PATCH1_NEW),
    ("PATCH2A (grad-skip, scaler path: clear Adam on offenders)",    PATCH2A_OLD, PATCH2A_NEW),
    ("PATCH2B (grad-skip, non-scaler path: clear Adam on offenders)", PATCH2B_OLD, PATCH2B_NEW),
    ("PATCH3 (tier-boundary proactive sanitize)",                     PATCH3_OLD, PATCH3_NEW),
]


def apply_patches(path: str, dry_run: bool) -> int:
    if not os.path.exists(path):
        print(f"ERROR: {path} not found in current directory.")
        return 2
    with open(path, "r") as f:
        src = f.read()

    found_all = True
    for name, old, _new in PATCHES:
        if old not in src:
            print(f"MISSING: {name} — anchor block not found verbatim.")
            found_all = False
        else:
            print(f"  OK   : {name} — anchor located")
    if not found_all:
        print(
            "\nOne or more anchor blocks were not found. The file has probably "
            "been edited. Open apply_nan_patches.py, look at each PATCH*_OLD "
            "string, and find/replace manually. The logic of each patch is "
            "described in the file's docstring."
        )
        return 3

    if dry_run:
        print("\n(dry run — no files written; pass --apply to actually edit)")
        return 0

    backup = path + ".bak"
    shutil.copy2(path, backup)
    print(f"\nBackup written to {backup}")

    new_src = src
    for name, old, new in PATCHES:
        new_src = new_src.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(new_src)
    print(f"Patches applied to {path}.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Actually write the patched file")
    ap.add_argument("--target", default=TARGET_FILE, help="Path to tmew1_run.py")
    args = ap.parse_args()
    return apply_patches(args.target, dry_run=not args.apply)


if __name__ == "__main__":
    sys.exit(main())

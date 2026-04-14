"""
Forward-activation NaN localizer for tmew1.

Usage — import and register once before run_curriculum:

    from nan_localizer import attach_nan_localizer
    detector = attach_nan_localizer(model, query_head=query_head, holder_head=holder_head)
    # ... run training normally
    # whenever a NaN loss fires, call:
    detector.report()   # prints the deepest finite module + first offender

What this does:
  - Registers forward hooks on every nn.Module in the model tree.
  - After each forward pass, records the LAST module whose output was
    finite and the FIRST module whose output contained NaN/Inf.
  - At report time, prints those two module names. The first offender
    is the precise module that turned the activation stream bad.

Why this matters for the 6-layer C1 crash:
  The 218-param grad-skip pattern and immediate tier-3 NaN-loss cascade
  are both consistent with a single module producing inf activations on
  specific inputs. The ambient fixes (Adam state clearing, weight sanitize)
  only clean up AFTER the explosion. They don't tell us where it starts.
  This does.

This is a diagnostic, not a fix. Once you know which module is the
offender, the remedy depends on which one it is:
  - Transformer block attention logits overflowing under fp16 AMP → switch
    that block to bf16 or add pre-softmax tanh cap.
  - HPM gate_mlp producing inf under competitive winner-take-all → reduce
    gate_hidden or add input LayerNorm inside HPM.
  - Modality encoder output range too large → LayerNorm the encoder output.
  - Controller residual_gate or forget_lambda out of range → clip those
    two scalars to [0, 2] defensively.

The hook adds one tensor-finite-check per module per forward. On a 6-layer
stack that's maybe 50-100 extra checks per forward, which is a few percent
overhead. Fine for a diagnostic run, strip it for production speed runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class NaNLocalizer:
    """Tracks the most-recent forward pass to find first non-finite module output."""
    # Module name -> (was_output_finite, depth_in_module_tree)
    last_pass: Dict[str, Tuple[bool, int]] = field(default_factory=dict)
    module_order: List[str] = field(default_factory=list)
    enabled: bool = True

    def reset(self) -> None:
        self.last_pass.clear()
        self.module_order.clear()

    def first_offender(self) -> Optional[str]:
        """Return name of the first (outermost-first-time-visited) module whose output was non-finite."""
        for name in self.module_order:
            ok, _ = self.last_pass[name]
            if not ok:
                return name
        return None

    def last_finite(self) -> Optional[str]:
        """Return name of the last module whose output was finite (the one just before trouble)."""
        last = None
        for name in self.module_order:
            ok, _ = self.last_pass[name]
            if ok:
                last = name
            else:
                break
        return last

    def report(self, prefix: str = "") -> None:
        offender = self.first_offender()
        last_ok = self.last_finite()
        if offender is None:
            print(f"{prefix}NaN localizer: entire forward pass finite (no NaN/Inf detected in any module output).")
            return
        print(f"{prefix}NaN localizer report:")
        print(f"{prefix}  last finite module: {last_ok or '<none — NaN from the very first module>'}")
        print(f"{prefix}  FIRST OFFENDER   : {offender}")
        n_bad = sum(1 for name in self.module_order if not self.last_pass[name][0])
        print(f"{prefix}  total non-finite modules in pass: {n_bad}")


def _tensor_is_finite(x) -> bool:
    """Return True iff x is a Tensor (or tuple/list of them) with all finite values."""
    if isinstance(x, torch.Tensor):
        if x.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            return torch.isfinite(x).all().item()
        return True
    if isinstance(x, (tuple, list)):
        return all(_tensor_is_finite(t) for t in x)
    if isinstance(x, dict):
        return all(_tensor_is_finite(v) for v in x.values())
    # Dataclass-like or unknown — treat as finite (can't check).
    return True


def attach_nan_localizer(*modules: nn.Module, prefix: str = "model") -> NaNLocalizer:
    """
    Walk every nn.Module in the given tree(s) and register a forward hook that
    records whether the output was all-finite. Returns the NaNLocalizer to
    query after a forward pass.
    """
    det = NaNLocalizer()

    def _make_hook(full_name: str):
        def hook(mod, inputs, output):
            if not det.enabled:
                return
            try:
                ok = _tensor_is_finite(output)
            except Exception:
                ok = True  # don't let the diagnostic itself crash training
            # Depth = number of dots in the name (stable proxy for tree depth).
            depth = full_name.count(".")
            det.last_pass[full_name] = (ok, depth)
            det.module_order.append(full_name)
        return hook

    # Clear on every forward at the outermost module only.
    # We'll install a pre-forward hook on each top-level module to reset state.
    def _reset_hook(mod, inputs):
        det.reset()

    for root_idx, root in enumerate(modules):
        if root is None:
            continue
        root_name = f"{prefix}{root_idx}" if len(modules) > 1 else prefix
        root.register_forward_pre_hook(_reset_hook)
        for sub_name, sub in root.named_modules():
            full = f"{root_name}.{sub_name}" if sub_name else root_name
            sub.register_forward_hook(_make_hook(full))

    return det


def patch_train_step_with_localizer(train_step_fn, detector: NaNLocalizer):
    """
    Wrap a train_one_step-like function so that whenever it returns a result
    with NaN total loss, the detector reports automatically.

    Usage:
        train_one_step = patch_train_step_with_localizer(train_one_step, detector)

    If you'd rather not wrap, just call `detector.report()` yourself from the
    NaN-loss guard in tmew1_run.py — that's arguably cleaner.
    """
    import math

    def wrapped(*args, **kwargs):
        out = train_step_fn(*args, **kwargs)
        total = out.get("total") if isinstance(out, dict) else None
        if isinstance(total, float) and (math.isnan(total) or math.isinf(total)):
            detector.report(prefix="    [NaN-loss] ")
            detector.enabled = False  # one report per run is enough
        return out

    return wrapped


# =============================================================================
# Minimal patch for tmew1_run.py — two lines added to the NaN-loss early return
# =============================================================================
#
# Inside train_one_step, right after the _log.warning(...) line in the
# "NaN guard: skip backward" block, add:
#
#     if hasattr(model, '_nan_localizer'):
#         model._nan_localizer.report(prefix="    [NaN-loss] ")
#         model._nan_localizer.enabled = False  # only report once
#
# And in run_curriculum right after build_model, add:
#
#     from nan_localizer import attach_nan_localizer
#     model._nan_localizer = attach_nan_localizer(model, query_head, holder_head, prefix="m")
#
# That's it. First NaN loss will print the exact module path that went bad.
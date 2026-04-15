# Plan: HPM Architecture Overhaul — 7-Point Critique Response

Your critique maps to 4 implementation phases, ordered by leverage. All changes are config-gated for backward compatibility.

## Phase A: Diagnostics Wiring (Critique #7) — UNBLOCKS EVERYTHING

HPM already computes 11 diagnostics per forward pass. They're stored in ForwardOutput.hpm_diagnostics but never extracted in the training loop. Cheapest fix, gates all future evaluation.

- Extract HPM diagnostics in train_one_step() — after output = model(...) in tmew1_run.py ~L259, return output.hpm_diagnostics alongside stats. Same pattern as controller report extraction at ~L702.
- Merge HPM metrics into step_metrics — at ~L709-L733, add hpm_gate_mean, hpm_z_abs_mean, hpm_z_abs_max, hpm_write_mag, hpm_force_unlocks_step, hpm_mu, hpm_sigma, hpm_open_frac, hpm_locked_frac.
- Add MetricSpec entries in score_logging.py build_default_metric_specs() for color-coded HPM metrics.
- Add force-unlock write decomposition (Critique #5) — in hpm.py forward loop, track hpm_write_regular_frac vs hpm_write_forced_frac to determine if force-unlock is doing too much load-bearing work.

## Phase B: Surprise Subspace + Adaptive σ Floor (Critiques #1, #6) — parallel steps

**Learned surprise projection** — New HPMConfig.surprise_dim: int = 64. Add per-slot surprise_proj = SlotLinear(n_slots, d_model, surprise_dim). Prediction error computed in projected subspace instead of raw h-space. The predictor still predicts in full d_model (preserving gradient flow through write_encoder); only the surprise scalar moves to the learned subspace. This is what enables unsupervised slot specialization — each slot's surprise_proj can learn a different task-relevant feature detector.

**Adaptive σ floor + absolute MSE gate** — Dual mechanism to prevent the convergence-collapse that killed AB1 ep3. (a) σ floor scales with batch_mu so it can't shrink to 1e-3 when the model is near-perfect. (b) Force-unlock requires e_t > min_surprise_threshold AND |z_t| > critical_z, preventing z-reinflation on trivially small absolute errors. New config: sigma_floor_adaptive: bool = True, sigma_floor_scale: float = 0.1, min_surprise_threshold: float = 0.01.

## Phase C: Content-Conditional Writing + Retroactive Buffer (Critiques #2, #3) — parallel steps

**Content-conditional write gating** — Replaces the unused competitive flag with content-key matching. Each slot gets a learnable key slot_key ∈ ℝ^{slot_dim}; input emits a query q(h_t) via new content_query projection. Write strength becomes g_t * softmax(q · k / √d), so slots develop content preferences through gradient pressure — the missing ingredient for Phase 4 slot specialization. New config: HPMConfig.content_gating: bool = True.

**Activate retroactive binding for C3 experiments** — Already implemented in hpm.py L394-416 but disabled (retroactive_window=0). Just needs retroactive_window=4 in C3 branch config. No code changes in hpm.py.

## Phase D: Learnable Gains + Dual-Bank Split (Critique #4, deeper question)

**Learnable state gains** — Replace fixed 1.0/0.5/0.1 with softplus(gain_logits) initialized to the same values. New config: HPMConfig.learnable_gains: bool = True. The optimizer can collapse states toward continuous if that's better, while preserving the state machine as a debugging primitive.

**Dual-bank HPM (fast + slow)** — Two HPM modules with different hyperparams run in parallel on the same input, outputs concatenated. Fast bank: n_slots=2, critical_z=2.5, ema_decay=0.95 (moment-to-moment state). Slow bank: n_slots=2, critical_z=4.0, ema_decay=0.995, gain_locked=0.05 (rule/structural facts). Would likely have prevented AB1 ep3 collapse because the slow bank resists surprise-driven overwriting. New config: WorldModelConfig.hpm_slow_config: Optional[HPMConfig] = None in homeostatic_multimodal_world_model_chunked.py. Note: doubles HPM output width — need to verify holder_feature_dim is computed dynamically.

## Relevant Files

- hpm.py — HPMConfig (L58-104), __init__ (L153-210), forward() loop (L280-490), diagnostics (L492-503)
- tmew1_run.py — train_one_step() (L221), step_metrics (L709), log_training_snapshot calls (~L733)
- score_logging.py — build_default_metric_specs()
- homeostatic_multimodal_world_model_chunked.py — ForwardOutput (L1329), HPM instantiation (L2649), WorldModelConfig (L397)

## Verification

- **Phase A:** Short 1-tier run — confirm HPM metrics appear in colored log output; check hpm_write_forced_frac vs hpm_write_regular_frac ratio
- **Phase B:** Re-run AB1 config for 3+ T3 epochs — latent_acc must NOT collapse at ep3; hpm_sigma should stabilize above adaptive floor
- **Phase C:** Run C1 branch with content_gating — per-slot hpm_write_mag should diverge (specialization); run C3 with retroactive_window=4 — retroactive binding probe should flip from negative to positive
- **Phase D:** Train 50 steps with learnable gains — print softplus(gain_logits) to see if they diverge from initial values; dual-bank run should show fast_hpm_gate_mean > slow_hpm_gate_mean

## Decisions

- Content gating subsumes competitive — old flag preserved but deprecated (never activated in any run)
- State machine kept, not replaced — learnable gains are the right middle ground; full replacement is too much churn
- Surprise projection is additive — predictor still predicts in full d_model; only the scalar surprise signal moves to projected subspace
- All changes config-gated — default configs reproduce current behavior exactly

## Further Considerations

- **Experiment sequencing:** Phase A should be merged and a diagnostics-only baseline run performed before any architectural changes. This gives you HPM telemetry on the current architecture first.
- **Dual-bank output width:** With dual_bank=True, HPM output doubles. Need to confirm holder_feature_dim and query head input dim are computed dynamically (not hardcoded). If hardcoded, that's an additional change site.
- **competitive deprecation:** Should we remove competitive entirely or keep it as a third option alongside content_gating? Recommendation: deprecate, since it was never used.
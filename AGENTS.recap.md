# TMEW1 Training Debug — Context Recap

## Project

**TMEW1** = a homeostatic multimodal world model with a curriculum-staged training pipeline. Key components:

- **`homeostatic_multimodal_world_model_chunked.py`** — core model, `MultimodalPredictionLoss`, `LossWeights`, `ForwardOutput`, the controller with `emergency_stabilize()`, block-level `residual_gate` / `forget_lambda`.
- **`pnn.py`** — Perineuronal Net layers with OPEN/LOCKED/CLOSED states and `force_unlock()`.
- **`hpm.py`** — homeostatic plasticity module, force-unlock logic.
- **`rd_stress.py`** — per-layer-group stress field (CV-based), immune cache for threat signatures.
- **`tmew1_run.py`** — training step, NaN guards, controller_step, evaluation loop.
- **`tmew1_train.py`** — curriculum tiers, loss weighting, driver.
- **`tmew1_queries.py`** / **`tmew1_diagnostics.py`** — evaluation and handoff/rule/temporal-ordering probes.

## Curriculum structure

- **Tier 1:** modalities = `('vision', 'numeric')`, T=24, template = `trigger_delay`.
- **Tier 2:** adds `audio`, T=48, adds `occlusion_identity`, `multi_chain`, `handoff`, `false_cue` templates. Holder head becomes active.
- **Tier 3:** adds `text` (vocab 64), T=64. **This is where things broke.**

## The bug (original run: `training_logs_6_NAN_explosion.txt`)

At **tier 3 ep0 s0020 → s0030**, training NaN'd catastrophically and never recovered. Signature:

```
t3 s0010: next_step = 1.22  (vs ~0.05 at end of t2)
t3 s0020: next_step = 0.60
NaN detected in model weights after optimizer.step — zeroed out
NaN/Inf loss detected (total=nan) — skipping backward, running emergency_stabilize
[... forever ...]
val: next_step = 4.540645  (≈ ln(64), uniform prior — model collapsed to constant output)
```

Every subsequent tier-3 val produced bit-identical losses, confirming the model was fully dead.

## Root cause diagnosed

The text head is **fresh at tier 3** (vocab=64, cross-entropy, weight=1.0 in `LossWeights`). First-step symptoms:

1. Untrained softmax over 64 classes → raw CE in the 4–20 range.
2. AdamW's second-moment buffer `v` was near-zero for these never-exercised params, so first large gradient produced `update = lr * g / (sqrt(v) + eps)` ≈ huge.
3. That step pushed weights to inf/NaN.
4. Existing guard `p.data.nan_to_num_(0.0, 1.0, -1.0)` zeroed the weights but **left Adam's `exp_avg` / `exp_avg_sq` still poisoned with inf/NaN** → next step reproduced NaN immediately.
5. `emergency_stabilize()` only touches `residual_gate` / `forget_lambda`, not weights or optimizer state — it couldn't save this.
6. Model permanently stuck outputting zero-logit distribution; controller state corrupted by NaN loss feedback.

Grad clip norm = 1.0 doesn't help when the gradient itself is `inf` (clipping `inf` still gives `inf`).

## Fixes applied (these worked)

Per prior advice, Wes implemented:

1. **Text weight warmup** — ramp `weights.text` from 0 → 1 over the first N tier-3 steps rather than cold-starting at 1.0.
2. **Pre-step finite-grad check** — before `optimizer.step()`, verify all grads are finite; if not, zero grads and skip the step (no weight corruption).
3. **Adam state reset on NaN repair** — when a param hits NaN, `nan_to_num_` the weight AND clear `optimizer.state[p]` so Adam moments start fresh.
4. **Holder loss upweighted** — from `0.3 * holder_loss` → stronger weight, because holder tracking was the hardest task and needed more gradient signal.

## Current run results (`training_logs_5_5.txt` and later excerpts)

**Tier 1 val (clean):** next_step=0.041, latent_acc=1.00, most qacc metrics at 0.9+.

**Tier 2 val (dramatically improved):**
- `next_step = 0.049`, `holder_acc = 0.625`, `what_was_true_rule = 0.979` (was 0.770 in old run).
- `did_alarm_fire = 0.951`, `did_chain2_fire = 0.953`, `which_entity_first_occluded = 0.863`.
- `who_holds_token` aggregate still weak at 0.299 — handoff tracking still the soft spot.

**Tier 3 ep1 val (the big win — no NaN, actually learning):**
- `next_step = 0.205` (vs 4.54 / NaN before).
- `holder_acc = 0.983`, `what_was_true_rule = 1.000`, `did_chain2_fire = 0.996`, `did_trigger_before_alarm = 0.996`.
- `who_holds_token = 0.469` aggregate — still the weakest metric.

**Tier 3 ep2 training (latest excerpts):**
- Total loss in 92–94% score band.
- `holder_acc` pinned at 99.4–100% — handoff tracking genuinely fixed during training.
- `next_step` stuck at 0.18–0.23 ("medium" band) — this is the remaining question.
- `entity_acc` climbing into 80–95% range (was 50–70%).
- PNN states cycling normally (O/L/C transitions, force-unlocks working).

## Current status / open questions

**What we know is fixed:**
- NaN explosion at tier transitions — solved.
- Holder/handoff training accuracy — solved (99%+).
- Belief revision (`what_was_true_rule`) — solved (near 1.0).
- Chain2, alarm, trigger ordering, occlusion — solved.

**What might still be off:**

1. **`next_step` plateau at 0.18–0.23 on tier 3.** Two hypotheses:
   - *Benign:* text CE near its irreducible entropy floor given the actual per-step token distribution; the "medium" score band is an artifact of thresholds calibrated for MSE-scale losses.
   - *Non-benign:* text head still underweighted due to warmup, needs longer tier 3 or higher final text weight.
   - **Action needed:** log `losses.parts["text"]`, `losses.parts["vision"]`, etc. separately instead of just summed `next_step`. If text is ~0.2 and others are ~0.01, it's the text floor; if all are ~0.05 each summing to 0.2, it's real and benign.

2. **Gap between training holder_acc (99%+) and val `who_holds_token` (~0.47).** Could mean:
   - The per-step holder head memorizes "last entity referenced" rather than composing handoff semantics.
   - Val distribution is genuinely harder (longer chains, more handoffs).
   - **Action needed:** look at next tier 3 val's `who_holds_token by handoffs` breakdown table. If `handoffs≥3` buckets are above 0.5, genuinely solid. If stuck near chance, holder head learned a shortcut.

3. **`color_change` zero-shot** flat at ~0.24 across lags — expected (never trained), just noting for future-tier planning.

## Recommended next actions

1. **Let tier 3 finish at least ep3–ep5** before declaring convergence — compare val `next_step` trajectory.
2. **Split the `next_step` metric into per-modality parts** in logging.
3. **Check val `who_holds_token by handoffs` breakdown** at the next checkpoint — this is the decisive metric for whether handoff tracking generalizes.
4. **If text head is truly plateaued and that's the ceiling:** consider label smoothing (0.05), smaller output proj init std (0.02), or accept it as the modality's entropy floor.
5. **If val/train handoff gap persists:** add a handoff-conditional loss term that only penalizes holder CE on steps where a handoff just fired (currently the mean is diluted by trivial "same holder as last step" predictions).

## Standing preferences / notes for continuation

- Wes wants **honest trusted-advisor tone**, devil's advocate posture, no cliffhangers, concrete actionable next steps.
- No "REALITY CHECK" framing.
- Full production-ready code, no placeholders.
- Preferred stack: Python, PyTorch, React, Flask, JS, HTML.
- Code blocks **only** for actual code (responses are often listened to aloud).
- This is a side-project AI research thread (TMEW1 / evolutionary homeostatic models), distinct from the Glassbox Bio work.

## Files to have on hand in next session

- `tmew1_run.py` (training step, NaN guards, controller_step)
- `tmew1_train.py` (curriculum, loss weights, warmup logic)
- `homeostatic_multimodal_world_model_chunked.py` (`MultimodalPredictionLoss`, `LossWeights`)
- Latest training log (tier 3 ep3+ val with handoff breakdown)
- Optionally: `tmew1_diagnostics.py`, `tmew1_queries.py`, `rd_stress.py`, `pnn.py`, `hpm.py`

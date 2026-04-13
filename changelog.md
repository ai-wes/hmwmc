## [2026-04-11 14:08]

### Summary

Ran `python tmew1_run.py --epochs 4 --batch-size 8` to execute the requested training command. The run failed during the first curriculum tier with a PyTorch autograd error caused by an in-place tensor modification during backpropagation.

### Changes

- Executed `python tmew1_run.py --epochs 4 --batch-size 8`.
- Created `changelog.md` and logged the run result.

### Decisions

- Did not modify training code because the request was to run the command only.
- Recorded the failure details so the next turn can debug the in-place autograd mutation if needed.

### Next Steps

- Inspect `tmew1_run.py` around `train_one_step` to find the in-place operation affecting backward().
- If desired, rerun with anomaly detection enabled to pinpoint the offending tensor update.
## [2026-04-11 14:18]

### Summary

Fixed the PyTorch autograd versioning failure by deferring homeostatic controller parameter mutations until after backward and optimizer step. Verified the fix with `python -u tmew1_run.py --smoke`, which completed successfully. Re-ran `python tmew1_run.py --epochs 4 --batch-size 8`; that full run is still in progress.

### Changes

- Updated `tmew1_run.py` to call `model.controller_step(...)` after `total.backward()` and `optimizer.step()`.
- Updated `tmew1_train.py` with the same ordering fix for the base training loop.
- Updated the wiring example in `tmew1_queries.py` to match the new safe ordering.
- Ran `python -u tmew1_run.py --smoke` successfully.
- Started `python tmew1_run.py --epochs 4 --batch-size 8` again after the fix.

### Decisions

- Kept the controller logic unchanged and fixed the execution order in the training loops, because the root cause was in-place mutation of live parameters before autograd completed.
- Used the smoke test for immediate verification because the full run is long and stdout is buffered.

### Next Steps

- Let the full `python tmew1_run.py --epochs 4 --batch-size 8` rerun finish and inspect its final metrics.
- If needed, rerun the full command with unbuffered output for live progress visibility.

## [2026-04-12 20:45]

### Summary

Shipped the durable false-cue correction fix that resolved belief revision in Tier 2/3 diagnostics, and followed it with handoff-targeted training support after verifying that the ambient holder audio signal was present in generated episodes.

### Changes

- Marked the false-cue correction pulse fix as complete after `what_was_true_rule` with-cue reached 1.00 in Tier 2 and held through Tier 3.
- Increased `WorldConfig.audio_dim` to 16 and kept the holder identity channels at `audio[8 + holder_id]` with a continuous ambient signal.
- Extended handoff episode generation to keep forced-handoff audio consistent from the forced transfer step onward.
- Added `holder_per_step` targets to diagnostic episodes and tensors.
- Added an auxiliary current-holder head and loss in `tmew1_run.py` so the model is explicitly trained to decode the holder identity signal when audio is enabled.
- Hardened controller diagnostics against non-finite values to prevent NaN crashes in signature quantization and score computation.

### Decisions

- Did not add permanent training-loop debug prints because a direct episode-generation check confirmed the ambient holder channel was already present in the tensors.
- Kept backward-compatible aliases in the controller refactor so the neural path uses score/layer/step semantics without breaking older call sites.

### Next Steps

- Re-run Tier 2 and Tier 3 to measure whether the holder auxiliary loss breaks the handoff parity shortcut.
- Check the final `who_holds_token by handoffs` buckets with emphasis on the 1-handoff and 2-handoff cases.

## [2026-04-13 HPM integration]

### Summary

Shipped Level 1 Homeostatic Predictive Memory (HPM) as a new architectural module and wired it through the main model and query pipeline. HPM is a surprise-gated working-memory channel that runs parallel to the main sequence representation, with writes governed by z-scored prediction error and slot state managed by a PNN-style OPEN/CLOSING/LOCKED state machine. This replaces attention as the retention primitive for "most recent event" queries and directly extends the PNN/RDStress homeostasis pattern from MOEA to per-timestep memory plasticity.

### Changes

- Added `hpm.py` containing `HomeostaticPredictiveMemory`, `HPMConfig`, and a vectorized `SlotLinear` helper.
  - Per-slot one-step-ahead predictor provides an intrinsic, task-agnostic surprise signal.
  - Z-score gate uses detached per-slot EMA of error mean and variance (proper two-moment statistics, not absolute deviation).
  - Slot state machine gated by gate EMA with configurable transition thresholds; state-dependent gate gain (OPEN 1.0, CLOSING 0.5, LOCKED 0.1).
  - Force-unlock on |z| > critical_z, PNN `force_unlock` lineage preserved.
  - 50-step warmup suppresses force-unlock while sigma stabilizes.
  - Read modes: concat (default), mean, attn.
  - Competitive gating available via config flag for Level 2 without rewrite.
- Patched `homeostatic_multimodal_world_model_chunked.py`:
  - Added `hpm_config: HPMConfig` to `WorldModelConfig`.
  - Added `hpm_sequence` and `hpm_diagnostics` fields to `ForwardOutput`.
  - Instantiated `self.hpm` when config is enabled; kept optional so ablations flip `enabled=False`.
  - `sequence` is passed through HPM after the main loop; HPM output is returned alongside, never concatenated into `sequence` itself, so next-step prediction heads stay on pure block output.
- Patched `tmew1_run.py`:
  - New `build_query_input` helper concatenates base sequence, audio holder channels, and `hpm_sequence` before QueryHead.
  - `QueryHead` input dim bumped by `model.hpm.output_dim`.
  - `per_qtype_accuracy` takes the full `ForwardOutput` (was `output_sequence`).
  - Step log now reports `hpm_gate_mean`, `hpm_z_abs_mean`, `hpm_z_abs_max`, `hpm_sigma`, `hpm_locked_frac`, `hpm_force_unlocks_step`, and a per-slot state string.

### Decisions

- Kept surprise source intrinsic (self-prediction of hidden state) rather than hooking into `MultimodalPredictionLoss`. Decouples HPM from task losses and lets each slot develop its own expectation under Level 2.
- Attributed surprise to the step that was surprising (e_t measures h_t against prediction from h_{t-1}, with e_0 = 0), so g_t writes at the same step where the event registers.
- Wrote slot state in a local loop variable and committed to the buffer after the loop, because in-place mutation of `slot_state` during the forward broke autograd via the `nn.Embedding` index dependency.
- Kept `sequence` unchanged for backward compatibility of anything downstream that doesn't know about HPM. HPM rides as a separate field.

### Next Steps

- First training run: Tier 2 only with `HPMConfig(n_slots=1, read_mode="concat", competitive=False)` to validate Level 1 behavior matches the handoff-doc specification.
- Watch `hpm_gate_mean` (should stabilize in 0.1–0.3), `hpm_sigma` (should grow from 1.0 then slowly shrink as predictor improves), and `hpm_z_abs_max` (should exceed 3.0 at handoff/collision events once warmup ends).
- Full curriculum run once Tier 2 looks healthy. Target on Tier 3 diagnostics: handoff=2 >= 0.70, handoff=3+ >= 0.60.
- Contingent on first result: bump `n_slots` to 4 for Level 2 and add an information-theoretic slot-differentiation loss.
- Publishable ablation: disable the audio holder one-hot (set indices to zero) and retrain to test whether HPM recovers multi-hop from vision alone.

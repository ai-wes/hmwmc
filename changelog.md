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

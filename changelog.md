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

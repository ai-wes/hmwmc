# Legacy raw-run provenance

The release branch removes `.history/` because it is editor state, not a stable research-artifact location. Several non-empty raw logs in that directory still carry useful provenance, so this manifest retains immutable links to their blobs in the verified repair commit `78121b697a54806907feaf7d7bd022fcd419927c`.

These logs predate the v2 runner. They are working records, not current benchmark evidence; their commands, configurations, and metric definitions should be read from the individual files before comparison.

| Historical path | Git blob | Provenance note |
|---|---|---|
| [`training_logs_20260411150614.txt`](https://github.com/ai-wes/hmwmc/blob/78121b697a54806907feaf7d7bd022fcd419927c/.history/training_logs_20260411150614.txt) | `3d1d7c427d7bf0b9b3495ffe0c5cff3571972433` | early legacy runner snapshot |
| [`training_logs_20260411151229.txt`](https://github.com/ai-wes/hmwmc/blob/78121b697a54806907feaf7d7bd022fcd419927c/.history/training_logs_20260411151229.txt) | `e0fd201edcb84f62ada843f1b855824f93da33d0` | early legacy runner snapshot |
| [`training_logs_20260411151519.txt`](https://github.com/ai-wes/hmwmc/blob/78121b697a54806907feaf7d7bd022fcd419927c/.history/training_logs_20260411151519.txt) | `b40c403e5d53b9e6d612111fa4d3a71374693657` | early legacy runner snapshot |
| [`training_logs_20260411162210.txt`](https://github.com/ai-wes/hmwmc/blob/78121b697a54806907feaf7d7bd022fcd419927c/.history/training_logs_20260411162210.txt) | `6b84dea75752c0752867edcd8bfebefcb5e2c4da` | complete legacy command and Tier-3 diagnostics; begins with `python tmew1_run.py --epochs 4 --batch-size 4` |
| [`training_logs_20260411164430.txt`](https://github.com/ai-wes/hmwmc/blob/78121b697a54806907feaf7d7bd022fcd419927c/.history/training_logs_20260411164430.txt) | `6081f18cede9918077f624f9cb0cf70db9065f27` | later legacy runner snapshot |

The empty `training_logs_20260411150611.txt` snapshot is intentionally omitted. The original paths and blobs remain available through Git history; this manifest exists so removing editor debris does not make the research provenance undiscoverable.

## v2 runs preserved in this directory

| File | Code state | Note |
|---|---|---|
| `baseline_v2_2026-11-03.log` | commit `82ab1e1`, unmodified | 544 of 768 optimizer steps skipped for non-finite gradients; negative baseline. Multi-line gradient-skip parameter dumps are truncated |
| `fixed_v2_2026-11-03.log` | `82ab1e1` plus the 2026-11-03 holder-label and episodic-init fixes | 0 skipped steps; the run reported in the README results table |
| `honest_v2_2026-11-03b.log` + `honest_v2_2026-11-03b_json/` | `6db1b34`, benchmark revision 2026-11-03b | default curriculum with the per-epoch probe monitor; holder query 0.293 once the ambient cue is removed. JSON directory holds `run_config.json`, `val_tier*_ep*.json`, `diagnostics_tier*.json`, `probes_tier*_ep*.json` |
| `honest_A1_2026-11-03b.log` | `6db1b34` | branch A1 (6–8 entities), verdict kill on belief-revision regression |
| `honest_A2_2026-11-03b.log` | `6db1b34` | branch A2 (`T=96`), CUDA out of memory in Tier 3 at batch 32 on an 8 GB card; no verdict |

Each file begins with a header giving the exact command, Python, PyTorch, and GPU used.

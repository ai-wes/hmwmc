# Memory ablations on the honest benchmark (2026-11-04)

Colab, NVIDIA A100 40 GB, notebook section 5b (job runner, several runs in
parallel), code at `a5ec217` on `codex/research-release-readme-ci`, benchmark
revision 2026-11-03b, `--size default`, `--probe-every-epoch`, seeds 0, 1, 2.
Source JSONs: Drive `hmwmc_runs/abl_{none,hpm,episodic,structured}_seed{0,1,2}/`
and the per-condition `abl_*_summary.json` written by `tmew1_aggregate_runs.py`.
The large-model reference row is `v2_a100_large_summary_tier3.json`
(`--size large`, A100, 3 seeds, same benchmark).

`abl_none_seed2` died of CUDA out-of-memory at the first Tier 3 step while
sharing the GPU with other jobs, so the full-model row has 2 seeds. Everything
else completed all three tiers.

## Tier 3 final validation, mean ± std over seeds

| condition | seeds | who_holds_token | holder_acc | who_was_first_tagged | what_was_true_rule | latent | alarm | trigger<alarm | chain2 | occluded | first_occluded |
|---|---|---|---|---|---|---|---|---|---|---|---|
| full model | 2 | 0.338 ± 0.028 | 0.481 | 0.629 ± 0.067 | 0.954 ± 0.025 | 1.000 | 1.000 | 0.996 | 0.974 | 0.850 | 0.873 |
| no HPM | 3 | 0.299 ± 0.042 | 0.474 | 0.739 ± 0.122 | 0.859 ± 0.197 | 0.932 | 1.000 | 1.000 | 0.972 | 0.852 | 0.855 |
| no episodic | 3 | 0.325 ± 0.048 | 0.466 | 0.716 ± 0.119 | 0.783 ± 0.132 | 0.993 | 0.996 | 0.994 | 0.981 | 0.845 | 0.860 |
| no structured | 3 | 0.322 ± 0.029 | 0.470 | 0.671 ± 0.038 | 0.948 ± 0.061 | 0.991 | 1.000 | 1.000 | 0.983 | 0.850 | 0.863 |
| large (33.5M), ref | 3 | 0.374 ± 0.081 | 0.552 | 0.698 ± 0.058 | 0.779 ± 0.092 | 0.991 | 1.000 | 0.998 | 0.987 | 0.848 | 0.856 |

## who_holds_token by handoffs before the query (Tier 3, mean over seeds)

| condition | 0 | 1 | 2 | 3 | 4 | 5 | 6+ |
|---|---|---|---|---|---|---|---|
| full model | 0.83 | 0.22 | 0.41 | 0.27 | 0.31 | 0.20 | 0.16 |
| no HPM | 0.78 | 0.27 | 0.47 | 0.18 | 0.27 | 0.13 | 0.20 |
| no episodic | 0.87 | 0.10 | 0.54 | 0.19 | 0.15 | 0.18 | 0.22 |
| no structured | 0.89 | 0.05 | 0.59 | 0.18 | 0.19 | 0.13 | 0.23 |
| large (33.5M), ref | 0.91 | 0.32 | 0.33 | 0.26 | 0.33 | 0.27 | 0.28 |

Query counts per bucket (same validation episodes in every run): 18, 36, 27,
24, 16, 15, 29.

## Reading

1. **No memory module is load-bearing for the holder query.** The four rows
   are within seed noise of each other on every bucket. Removing HPM, the
   episodic bank, or the whole structured-state path (state table, typed event
   log, checkpoint bank) leaves the overall query accuracy at 0.30 to 0.34.
2. **Without structured state the model is a near-constant predictor.** At
   Tier 2 the three `no structured` seeds produced bit-identical holder tables
   (0.926 / 0.026 / 0.548 / 0.190 / 0.125 / 0.100 / 0.308). At Tier 3 they are
   within 0.05 of each other with 0.03 to 0.08 at one handoff. That is the
   signature of always answering entity 0, the benchmark's initial holder.
   Treat "always entity 0" as the baseline row: it scores about 0.9 at zero
   handoffs and about 0.55 at two handoffs (the token often returns to entity 0)
   without tracking anything.
3. **With structured state the model beats that baseline only at one handoff**
   (0.22 versus 0.05) and is worse at two handoffs (0.41 versus 0.59). The
   structured holder logits are the only thing moving the answer, and they
   move it very little.
4. **`which_entity_changed_color` is at chance in every condition**, 0.20 to
   0.23, including the constant-predictor condition. 240 of 254 probes fall in
   the 31+ step lag bucket.
5. **Observable queries are unaffected by any ablation**: alarm, trigger-order,
   chain-2, and occlusion queries stay at 0.85 to 1.0 in every row. The
   backbone answers these from the current frame.
6. **The only candidate signal is belief revision.** `what_was_true_rule` is
   0.95 with the full model and with structured state removed, 0.86 ± 0.20
   without HPM, and 0.78 ± 0.13 without the episodic bank. The query has 31
   validation instances, the stds overlap, and the large reference model sits
   at 0.78 with all modules present. Not a result; worth one targeted check
   later, not now.
7. **Emergent probes do not track task performance.** `thread_separation` falls
   from 0.21 (full) to 0.07 without structured state, and the HPM probes
   (`slot_specialization`, `surprise_event_corr`) go to null without HPM, as
   expected. `retroactive_binding` and `state_reactivation` are slightly
   negative in every row. `event_chunking` is 7 to 9 everywhere, so it is not
   an HPM property.

## Conclusion

The ablations confirm the premise of
`docs/superpowers/specs/2026-11-03-transient-cue-write-path-design.md`: the
honest benchmark's one-step cues are not being written anywhere they survive,
and no existing memory module changes that. The next experiment is the
event-latched write path, not more capacity or more epochs.

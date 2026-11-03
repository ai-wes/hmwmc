# HMWMC / TMEW-1

**A synthetic benchmark and PyTorch research prototype for multimodal temporal reasoning, structured memory, and homeostatic adaptation.**

## Abstract

HMWMC studies whether a stateful multimodal world model can retain and query event structure across delayed effects, entity handoffs, occlusion, parallel causal chains, and misleading cues. The accompanying TMEW-1 environment generates short synthetic episodes with vision, numeric, audio, and text observations under latent rules. The model combines modality-specific encoders, gated fusion, adaptive temporal blocks, surprise-gated Homeostatic Predictive Memory (HPM), episodic memory, and explicit structured state, event, and checkpoint stores. Evaluation asks the model to predict the next observation and answer entity, binary, temporal-ordering, belief-revision, and limited counterfactual questions. The repository is a research prototype, not evidence of general-world reasoning: the environment is synthetic, the default validation split is seed-separated rather than compositionally held out, the preserved results are primarily single-seed runs, and the ablation record is incomplete. The current v2 contract repair is preserved at commit `78121b6`.

## Evidence status

| Surface | Status | Evidence boundary |
|---|---|---|
| Synthetic simulator and curriculum | Implemented | Deterministic procedural episodes; no external or real-user data |
| Multimodal world model and HPM | Implemented | PyTorch research code with diagnostic instrumentation |
| Structured state, typed events, checkpoints, and query routing | Implemented and regression-tested | Contract tests cover tensor shapes, routing, gradients, empty sequences, and checkpoint boundaries |
| End-to-end training path | Smoke-tested | The smoke run checks execution only; its tiny metrics are not benchmark results |
| Full v2 curriculum benchmark | Rerun once on this release (2026-11-03, seed 0, RTX 3070) | Single-seed run after the holder-label and episodic-init fixes; see Results. The unmodified release was also run and preserved as a negative baseline |
| Seed-separated validation | Implemented | Separate procedural seed ranges; not a compositional or cross-domain holdout |
| Multi-seed estimates and run-level confidence intervals | Not completed | The table below identifies single-run evidence and only computes binomial intervals where logged counts exist |
| Canonical ablation report | Incomplete | Several branch outcomes are recorded, but the full suite has not been rerun and normalized on this release |
| External or real-world generalization | Not evaluated | No claim is made beyond the synthetic benchmark |

## Benchmark specification

TMEW-1 is a procedurally generated grid-world benchmark. An episode contains two to six moving entities, one of six latent rules, delayed events, optional occlusion, and one or two causal chains. Observations are rendered into modality-specific tensors; query targets are derived directly from simulator state.

### Curriculum

| Tier | Length | Modalities | Default templates |
|---|---:|---|---|
| 1 | 24 | vision, numeric | delayed trigger |
| 2 | 48 | vision, numeric, audio | delayed trigger, occlusion, multi-chain; the runner boosts handoff and false-cue examples |
| 3 | 64 | vision, numeric, audio, text | the full boosted template pool |

The default world uses a `16 x 16` grid, at most six entities, six latent rules, and deterministic episode seeds. The full configuration lives in `WorldConfig`, `CurriculumTier`, and `TrainConfig` in [`tmew1_train_v2.py`](tmew1_train_v2.py).

### Core query families

| Query | Answer type | Capability under test |
|---|---|---|
| `who_holds_token` | entity | state tracking across handoffs |
| `who_was_first_tagged` | entity | temporal-order recall |
| `did_alarm_fire` | binary | delayed causal-chain completion |
| `which_entity_occluded` | entity | object permanence |
| `did_trigger_before_alarm` | binary | temporal ordering |
| `which_entity_first_occluded` | entity | ordering under occlusion |
| `did_chain2_fire` | binary | parallel causal tracking |
| `what_was_true_rule` | categorical diagnostic | belief revision under a false cue |

Each query carries a `time_asked` in the back half of the episode, and every stateful target is defined as of that step. In particular `who_holds_token` is the holder at `time_asked`, not the holder at the end of the episode. Earlier v2 releases labelled it with the end-of-episode holder; because the backbone is a causal scan, roughly a third of Tier 2/3 holder queries were then unanswerable by construction (on the Tier 3 validation set only 67.6% of those labels matched the holder at `time_asked`). `_test_query_labels.py` pins the corrected semantics.

Additional branch configurations introduce relational, negation, and bounded counterfactual questions. Those additions should not be interpreted as open-ended counterfactual reasoning.

## Model architecture

The current v2 path is assembled in [`homeostatic_multimodal_world_model_chunked_v2.py`](homeostatic_multimodal_world_model_chunked_v2.py), [`hpm_v2.py`](hpm_v2.py), [`tmew1_queries_v2.py`](tmew1_queries_v2.py), and [`tmew1_run_v2.py`](tmew1_run_v2.py).

1. **Modality encoders and gated fusion.** Text, vision, audio, and numeric streams are encoded separately and fused at each timestep.
2. **Adaptive temporal backbone.** Sparse recurrent world blocks maintain hidden and slot state. An online controller can alter plasticity in response to diagnostics and stability signals.
3. **Episodic and homeostatic memory.** Episodic memory stores event-boundary representations. HPM uses prediction-error statistics, gated writes, and configurable slot dynamics to retain surprising state.
4. **Structured memory.** The v2 configuration adds an entity-indexed state table, typed event log, and time-indexed checkpoint bank. The final timestep is retained even in the one-checkpoint edge case.
5. **Prediction and query heads.** Modality heads optimize next-step prediction. Auxiliary heads predict the latent rule and current holder; the structured query head combines sequence state with entity, event, and checkpoint context.

The default v2 builder uses `d_model=256`, six temporal layers, 32 per-layer memory slots, and 64 episodic slots. These settings are a research configuration, not a claim of scaling performance.

## Dataset generation

All data are generated locally by the simulator; there is no downloaded dataset. For the standard training path:

- each tier generates 2,048 training episodes and 256 validation episodes by default;
- each episode is deterministic given its tier, configuration, and integer seed;
- the training loop uses global seed `0` unless overridden;
- query labels, event labels, holder state, and the latent rule are derived from simulator state rather than human annotation.

The procedural generator is implemented in [`tmew1_train_v2.py`](tmew1_train_v2.py); query augmentation is implemented in [`tmew1_queries_v2.py`](tmew1_queries_v2.py).

## Held-out split methodology

The repository currently provides a **seed-separated procedural validation split**:

- training episodes use `base_seed = 1000 * tier`;
- validation episodes use `base_seed = 9000 + 1000 * tier`;
- the episode index is added to the base seed.

Within a tier, the default training and validation seed ranges do not overlap. The validation set therefore contains newly generated episodes from the same simulator distribution. Templates, rule families, and generator code are shared between training and validation. A template-compositional holdout, cross-generator holdout, and real-data evaluation remain future work; “held out” in this repository means seed-disjoint validation only.

## Baselines

The current experiment program uses a frozen HPM-enabled reference run and branch-specific promotion/kill rubrics in [`tmew1_experiments_v2.py`](tmew1_experiments_v2.py). It does not yet include a standardized comparison against published external architectures.

Useful nominal floors are:

- binary questions: `0.50` under a balanced uniform guess;
- latent-rule prediction: `1/6 = 0.167` under a uniform guess;
- entity questions: approximately `1/6 = 0.167` at the six-class output width, although the effective floor changes with the number of active entities and masking.

The stored “baseline” values are run artifacts from this codebase, not population estimates. Comparisons should name the branch, metric, split, and seed.

## Results and confidence intervals

The table below reports the latest complete Tier-3 validation snapshot preserved in [`docs/new_results.txt`](docs/new_results.txt). Git history binds that log to commit `d01d87b` (2026-04-17), when the executable path was the legacy [`tmew1_run.py`](tmew1_run.py); `tmew1_run_v2.py` did not yet exist. The log records one run and the source defaulted to seed `0`, but the exact command, dependency versions, and hardware are not embedded in the artifact. The validation loader contains 256 generated episodes, while each diagnostic has its own eligible subset. Wilson intervals are shown only where the log records both an accuracy and an eligible count; they quantify binomial sampling uncertainty for that subset, not variation across training seeds.

| Metric | Validation value | Eligible `n` | 95% Wilson CI | Notes |
|---|---:|---:|---:|---|
| latent-rule accuracy | 1.000 | not logged | not available | episode-level auxiliary probe |
| holder auxiliary accuracy | 0.886 | not logged | not available | per-timestep holder head; not the same as query accuracy |
| `who_holds_token` query accuracy | 0.539 | not logged | not available | entity query on the validation loader |
| belief revision with false cue | 1.000 | 31 | [0.890, 1.000] | diagnostic subset |
| trigger-before-alarm ordering | 0.992 | 127 | [0.957, 0.999] | 126 of 127 eligible diagnostics |
| second-chain completion | 1.000 | 131 | [0.972, 1.000] | diagnostic subset; query-loader accuracy in the same snapshot is 0.986 |

These numbers are legacy-run provenance, not a benchmark of the repaired v2 release. They show that the earlier code path learned several synthetic tasks in one run; they are not multi-seed estimates and do not establish external generalization.

### v2 curriculum run (2026-11-03)

Two runs of `python tmew1_run_v2.py --epochs 4 --batch-size 32 --train-episodes 2048` (seed 0, Python 3.12.3, PyTorch 2.9.0+cu128, one RTX 3070) are preserved with their commands and environment in [`docs/raw_runs/`](docs/raw_runs/README.md). The first is the unmodified release; the second adds the holder-at-query-time label fix and the non-zero episodic-slot initialization (see the 2026-11-03 changelog entry). Values are final-epoch Tier 3 validation on the 256-episode seed-separated split.

| Metric | Unmodified release | With fixes | Notes |
|---|---:|---:|---|
| optimizer steps skipped for non-finite gradients | 544 / 768 | 0 / 768 | the unmodified run is numerically broken, not merely weak |
| latent-rule accuracy | 0.582 | 1.000 | same metric definition in both runs |
| holder auxiliary accuracy (per timestep) | 0.467 | 0.968 | same definition; supervised by the per-step simulator holder |
| belief revision with false cue (diagnostic, `n=31`) | 0.226 | 1.000, 95% Wilson [0.890, 1.000] | same definition |
| trigger-before-alarm ordering (`n=127`) | 0.976 | 1.000, [0.971, 1.000] | same definition |
| second-chain completion (`n=131`) | 0.939 | 1.000, [0.972, 1.000] | same definition |
| first occluded entity (`n=135`) | 0.896 | 0.896, [0.833, 0.937] | unchanged |
| `who_holds_token` query accuracy | 0.269 | 0.963 | **not comparable**: the label definition changed between runs |
| `who_holds_token` by handoffs before `time_asked` (`n=165`) | not available | 0.964, [0.923, 0.983]; 0.733 in the five-handoff bucket (`n=15`) | diagnostic subset under the corrected labels |
| `who_was_first_tagged` query accuracy | 0.634 | 0.648 | unchanged; still the weakest entity query |

Tier 1, which has no audio holder cue, remains weak on the holder query in both runs (0.512 and 0.591) and does not pass its promotion gate; Tier 2 passes its gate in the fixed run. These are single-seed numbers from one machine and should be read as a working baseline for the repaired code path, not as a multi-seed estimate.

## Ablations

The branch framework covers harder worlds (A), harder query families (B), and architecture changes (C). The available records largely come from the legacy experiment path and include useful negative results:

| Branch | Change | Recorded outcome | Boundary |
|---|---|---|---|
| A1 | denser entity worlds | kill | holder-query accuracy fell to 0.217 in the recorded final validation |
| A2 | longer horizon (`T=96`) | kill | holder-query accuracy was 0.375 versus a 0.459 reference |
| A4 | more consequential dynamic rules | kill by regression guard | rule-state metrics improved, but temporal ordering fell to 0.923 from about 0.992 |
| B4 | bounded counterfactual queries | promote under its configured rubric | one target saturated at 1.000 and may be too easy; `holder_if_handoff2_absent` reached 0.437 |

The full A/B/C matrix is not a finished ablation result. Some working notes are incomplete or use earlier thresholds, the C-family architecture comparison has not been consolidated into a canonical multi-seed table, and these outcomes have not been rerun on v2. Use [`docs/experiment_plan.md`](docs/experiment_plan.md) and the branch runner as raw provenance, not as a polished paper appendix.

## Reproduction

Python 3.12 is the currently tested interpreter. Dependencies are intentionally listed in a small, unpinned [`requirements.txt`](requirements.txt); there is not yet a locked, repository-owned environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the contract and end-to-end smoke suite:

```bash
bash scripts/run_ci_checks.sh
```

Run only the v2 contract smoke:

```bash
python _smoke_test_v2.py
```

Run a full default curriculum:

```bash
python tmew1_run_v2.py --epochs 4 --batch-size 32 --train-episodes 2048
```

The full run uses CUDA when available and otherwise falls back to CPU. It writes tier checkpoints in the current directory. Exact reproducibility across dependency releases is not guaranteed until versions and hardware settings are locked.

### Regression inventory

CI executes:

- `_smoke_test_v2.py` — 12 explicit v2 contract checks;
- `_test_hpmv2.py` — 12 HPM/structured-memory checks;
- `_test_config_override.py` — configuration isolation;
- `_test_et_only.py` — entity-routing modes and gradient flow;
- `_test_hpm_changes.py` — legacy HPM paths and gradients;
- `_test_query_labels.py` — query labels are answerable at `time_asked` (holder-at-query-time semantics in both v2 generators);
- `tmew1_run_v2.py --smoke` — tiny end-to-end CPU pipeline.

The end-to-end check also fails if the runner emits a standalone `nan`/`inf`, `non-finite`, or skipped-backward signal. This prevents a numerically collapsed smoke step from being reported as green merely because the process reached the end.

`_test_mp.py`, `_test_mp2.py`, and `_test_log_groups.py` are assert-free demonstrations and are not counted as regression tests.

## Limitations

- TMEW-1 is a synthetic, low-entity-count environment with short horizons and handcrafted event families.
- The current validation set is seed-disjoint but distribution-matched; no compositional holdout is implemented in the default runner.
- Most preserved results are single-seed runs. Multi-seed means, run-level confidence intervals, and statistical comparisons are not available.
- The `who_holds_token` label definition changed on 2026-11-03 (holder at `time_asked` instead of end-of-episode holder); holder-query accuracies logged before that date are not comparable with later ones.
- Tier 1 exposes no holder cue (vision and numeric only), so its holder query depends on inferring stochastic handoff rules from positions; it has not passed its promotion gate in any preserved v2 run.
- The preserved Tier-3 result log predates the v2 runner and does not embed its exact command, environment, or hardware.
- The full ablation program has not been rerun against one immutable environment and normalized rubric.
- Some historical working notes remain exploratory and should not be treated as peer-reviewed claims.
- The dependency set is not pinned. CI constrains Python and CPU execution, but it does not create a fully locked research environment.
- Smoke-test metrics are execution diagnostics, not performance claims.
- No real-world, embodied, or user-facing benefit has been evaluated.

## Repository map

| Path | Purpose |
|---|---|
| `tmew1_train_v2.py` | simulator, curriculum, data generation, model builder |
| `tmew1_run_v2.py` | training/evaluation runner, query heads, checkpoints, diagnostics |
| `tmew1_queries_v2.py` | query generation and structured query routing |
| `hpm_v2.py` | HPM, structured state, typed events, checkpoints |
| `homeostatic_multimodal_world_model_chunked_v2.py` | multimodal backbone and controller |
| `tmew1_experiments_v2.py` | branch configurations and promotion/kill rubrics |
| `run_ablation_suite.py` | ablation orchestration |
| `docs/new_results.txt` | preserved raw run log used in the results table |
| `docs/raw_runs/README.md` | immutable pointers to legacy raw logs removed from editor-history paths |
| `scripts/run_ci_checks.sh` | local and CI regression entrypoint |

# Transient-cue write path: event-latched relational registers

Date: 2026-11-03. Status: design, pre-implementation. Applies to the v2 stack
(`homeostatic_multimodal_world_model_chunked_v2.py`, `hpm_v2.py`,
`tmew1_run_v2.py`, `tmew1_queries_v2.py`, `tmew1_train_v2.py`).

This document assumes the memory ablations (`--ablate hpm|episodic|structured`)
come back flat: removing any one memory module leaves the holder-by-handoffs
table unchanged. If the ablations instead show one module is load-bearing, stop
and re-plan; this design still applies but its success criteria change.

## 1. Why

On the honest benchmark (revision 2026-11-03b) the large A100 run, 3 seeds,
Tier 3 final epoch, gives:

| handoffs before query | 0 | 1 | 2 | 3 | 4 | 5 | 6+ |
|---|---|---|---|---|---|---|---|
| who_holds_token acc | 0.91 | 0.32 | 0.33 | 0.26 | 0.33 | 0.27 | 0.28 |

`which_entity_changed_color` is 0.22 at lag 31+. The emergent probes for
retroactive binding and state reactivation never leave zero. Scaling from
13M to 33M params moved the holder query from 0.29 to 0.37.

The benchmark now renders a state change as a one-step cue (audio channel
`8 + new_holder_id` on the handoff step, numeric `tagged_this_step` on the tag
step, a one-step vision change for colour) and never renders the standing
state. So a query asked k steps after the cue is answerable only if the model
(a) detected the cue on the step it fired and (b) kept the written value
intact across k steps of unrelated input. The handoff table says the model
does neither reliably.

Reading the current write path shows why. Nothing in it is built to latch a
one-step cue:

1. **The cue is diluted before it reaches memory.** Audio goes through
   `AudioEncoder`, then `GatedModalityFusion` takes a softmax over the four
   modalities per step, then RMSNorm. A one-hot spike in one of 16 audio
   channels competes with full vision and numeric frames for a share of one
   vector. Then it passes through 6 to 8 recurrent blocks whose hidden state
   decays by `forget_lambda` every step and whose slot memory is a shift
   register of 32 to 48 slots. Everything downstream (`StructuredStateTable`,
   `TypedEventLog`, HPM, episodic) consumes `sequence`, the output of that
   stack, not the cue.
2. **The structured state writes every step, unconditionally.**
   `StructuredStateTable.forward` routes `h_t` to entities with a softmax over
   entity keys, so the routing weights sum to one on every step, and every
   entity GRU is stepped on every step with that routed input. There is no
   "nothing happened, hold" option. A GRU can learn to approximate a hold, but
   it has to do so against 60 steps of nonzero input at every entity, and the
   result is the erosion the table shows.
3. **Holder is decoded per entity, not as one relation.** Each entity's GRU
   emits its own scalar holder logit. "A lost the token" is only learnable
   from "B gained it" through the shared `h_t`, and the GRUs have no
   cross-entity path. A one-of-N relation is being stored as N independent
   bits.
4. **The query-time shortcut only works on the cue step.**
   `augment_sequence_with_holder_audio` appends the raw holder audio channels
   to the query input at the query step. That is only informative when the
   query lands on the handoff step itself, which is the handoffs=0 bucket that
   already scores 0.91.
5. **The typed event log is a reader, not a writer.** `TypedEventLog` selects
   entries from `sequence` by a supervised event-type head, and the query head
   attends over them. This is the right shape for historical queries, but its
   entries are projections of the same eroded `sequence`, so they carry
   whatever the backbone kept, not what the cue said.

The fix is a write path that sees the cue before dilution, writes only when an
event fires, stores a relation as one register, and holds that register
exactly until the next write.

## 2. Goals, non-goals, anti-goals

Goals

- Holder query accuracy after one or more handoffs reaches the same level as
  with zero handoffs, on the honest benchmark, default size, 3 seeds.
- `which_entity_changed_color` and `who_was_first_tagged` at long lag come up
  the same way, from the same mechanism, not from per-query special cases.
- A diagnostic that separates "did not detect the cue" from "detected it but
  lost it", so the next failure is localizable in one run.
- The existing modules (HPM, episodic bank, structured state, typed event log,
  checkpoint bank) stay in place and stay ablatable. The new path is one more
  `--ablate` target.

Non-goals

- Changing the benchmark or the observation contract. The contract tests in
  `_test_observation_contract.py` must pass unchanged.
- Scaling. This is tested at `--size default` first.
- Replacing the backbone, fusion, or next-step prediction heads.
- Counterfactual queries (`holder_if_handoff2_absent` and friends). They are
  out of scope until the factual ones work.

Anti-goals (what would make this a failure even if the numbers go up)

- Leaking standing state into the model through the training targets or the
  query input. The latch must be written from the observation stream, not
  from `holder_per_step`. Supervision on the latch output is fine, because it
  is the same supervision the structured holder head already gets; feeding
  ground-truth events into the write gate is allowed only under the explicit
  `oracle_gate` diagnostic flag and never in a reported result.
- A holder query that is answered by the query head re-reading the whole
  sequence with attention. That would be a transformer in disguise and would
  tell us nothing about the recurrent memory.

## 3. Design

### 3.1 Shape

Add one module, `EventLatchBank`, that runs alongside the existing structured
state. Per step it does three things:

1. **Detect**: from the raw per-modality encoder outputs (before fusion), a
   small cue encoder produces, for each register r, a write gate
   `g[t, r] in (0, 1)` and a write value `v[t, r]`, a distribution over
   entities.
2. **Latch**: `L[t, r] = (1 - g[t, r]) * L[t-1, r] + g[t, r] * v[t, r]`.
   With `g = 0` the register is bit-for-bit unchanged. There is no decay term
   and no per-step input when nothing fires.
3. **Log**: when `g[t, r]` crosses a threshold, append `(t, r, v[t, r])` to a
   fixed-capacity write log. The log is what historical queries read.

`L[-1, r]` (the value before any write) is a learned per-register logit
vector, `L0[r]`, supervised like every other step (3.5). This matters because
the benchmark never cues the initial holder: `HandoffState` starts at
`state.entities[0].id`, which is entity 0, and the handoffs=0 bucket depends
on the model knowing that convention. A learned prior captures it; a uniform
prior would regress that bucket.

Registers are typed, one per relation the benchmark asks about:

| register | written on | value | read by |
|---|---|---|---|
| `holder` | handoff cue (audio 8+id) | new holder | who_holds_token |
| `first_tagged` | tag cue (numeric per-entity `tagged_this_step`), write-once | first tagged entity | who_was_first_tagged |
| `color_changed` | colour change (vision frame diff) | entity whose colour changed | which_entity_changed_color |

`first_tagged` is write-once: its effective gate is `g * (1 - written_before)`,
where `written_before` is a per-episode scalar that saturates after the first
write. This is the only register-specific rule. The register set is a config
list so more can be added without touching the module.

### 3.2 Where the cue encoder reads from

`_encode_modalities` currently returns `(fused, fusion_weights)` and discards
the per-modality `features` dict it builds. Change its return signature to
also return that dict; it has one call site in `forward`. The cue encoder takes:

- the raw `audio` input slice `[8 : 8 + n_entities]` and the raw numeric
  per-entity block (`NUMERIC_ENTITY_OFFSET`, `NUMERIC_ENTITY_STRIDE`), passed
  through as-is, and
- the per-modality encoder outputs, concatenated.

Passing the raw channels through is deliberate. The observation contract
guarantees the cue is present in those channels on exactly the event step; the
cue encoder should not have to rediscover that through a learned encoder and a
fusion softmax. This is not a leak: the raw channels are already model input.
What the model may not see is standing state, and those channels do not carry
it (that is what `test_holder_cue_is_transient` pins).

The cue encoder sees two consecutive steps of raw input, `t-1` and `t`
(zeros for `t-1` at `t=0`). This is what makes the colour register possible:
a colour change is rendered as a change in the vision planes and in the
numeric per-entity `color_norm` channel, so the per-step cue is the delta,
and the delta identifies the entity. A one-step lag of the observation is
still observation, not standing state. Builder: the gate input is the raw
channels at `t` and `t-1` plus the per-modality encoder features at `t`;
no `sequence`.

The value head additionally gets `sequence[t]` as context so it can use what
the backbone knows about entity identity. The gate never sees `sequence`.

Register-specific value sources (the builder may add an MLP on top of each):
`holder` reads the audio one-hot at `t`; `first_tagged` reads the numeric
`tagged_this_step` block at `t`; `color_changed` reads the per-entity
`color_norm` delta between `t-1` and `t`.

### 3.3 Gate behaviour in training and eval

The gate is a sigmoid. Training uses the soft gate as written in 3.1; there is
no straight-through estimator and no hard threshold in the latch update. The
log append uses a threshold (`log_threshold`, default 0.5) and is not
differentiated through.

The gate gets direct supervision (3.5), so it does not have to discover the
cue from the query loss alone. That is the point: the query loss alone has
not been enough.

### 3.4 Reads

`StructuredQueryHeadV2` gains three inputs: `latch_values` (B, T, R, E),
`latch_log_entries`, `latch_log_mask`, `latch_log_times`, `latch_log_register`.

- `who_holds_token`: gather `latch_values[:, t_q, holder]` and use it in place
  of the current structured holder blend. Only `StructuredQueryHeadV2` is
  updated; `IterativeQueryHead` ignores the new kwargs (it already swallows
  unknown context) and is not a supported configuration for these results. Builder's call whether to keep the
  structured holder logits as a second term; if kept, the latch term gets the
  larger weight and both weights are config, not literals.
- `who_was_first_tagged`: gather `latch_values[:, t_q, first_tagged]`.
- `which_entity_changed_color`: this query type is eval-only and never
  trained (`_EVAL_ONLY` in `tmew1_diagnostics_v2.py`), and the zero-shot
  probe in `evaluate_with_diagnostics` builds a synthetic query at `t_max`
  using the trained `which_entity_occluded` embedding. Keep it eval-only and
  keep that probe as it is, reported under its existing name. Add a second
  line, `which_entity_changed_color_latch`, that reads
  `argmax latch_values[:, t_max, color_changed]` directly, with the same lag
  buckets. The success criterion in section 4 is on the latch line. The
  query head does not special-case this type.
- All other historical queries: the latch log entries are appended to the
  memory the existing `_run_history_attention` already attends over, with the
  same time-distance penalty. Each log entry is `entry_proj([v, one_hot(r)])
  + time_embed(t)`, mirroring `TypedEventLog`.

The raw-audio augmentation in `build_query_input` stays for now so results are
comparable; it may be removed once the latch path is shown to work.

### 3.5 Losses

Added to `train_one_step`, each behind a weight in `TrainConfig`:

- **Gate loss**: BCE of `g[t, r]` against a per-register, per-step binary
  target aligned to the **input step the cue appears on**. Do not reuse
  `build_typed_event_targets`: it labels a handoff at the step before the
  cue (`holder[t] != holder[t+1]`), labels colour at the cue step, and
  stores one label per step so coincident events overwrite each other. Build
  a new `build_latch_targets(batch, seq_len)` returning gate targets
  `(B, T, R)` and value targets `(B, T, R)`:
  - `holder` gate: `holder_per_step[t] != holder_per_step[t-1]`, zero at
    `t=0` (the cue is rendered on the step `holder_per_step` changes, see
    `generate_episode_with_diagnostics`).
  - `first_tagged` gate: the numeric `tagged_this_step` block at `t` summed
    over entities, clipped to 1, and only on the first such step.
  - `color_changed` gate: `t == color_change_step`.
  Positive steps are rare (one or a few per episode), so use `pos_weight`
  from the observed positive rate per register, recomputed per batch.
- **Latch loss**: cross-entropy of `L[t, r]` against the per-step register
  target at every step, including steps before the first write, so the
  learned prior `L0` is trained. Pre-write targets are the values the query
  generator itself uses: `holder` before any handoff is `holder_per_step[t]`
  (entity 0); `first_tagged` and `color_changed` before their event are 0,
  matching `max(0, id)` in the query generator. This is a benchmark
  convention, not world knowledge, and the write-up should say so.
  Per-step targets: `holder` uses `holder_per_step` (in the batch).
  `color_changed` is derived in the collate from `color_change_entity_id` and
  `color_change_step`, both already in the batch. `first_tagged` needs two
  new episode fields, `first_tagged_id` and `first_tagged_step`, taken from
  `handoff.first_tagged_id` (already computed as `trigger_source_id`) and the
  first step whose `tagged_ids` event is non-empty; add them to
  `EpisodeWithDiagnostics` and the collate. No new simulator state.
- The existing `holder_loss` on structured state logits stays. Nothing is
  removed.

Weights: gate 0.5, latch 1.0 to start. These are starting points for the
builder to tune on the smoke run, not decisions.

### 3.6 Diagnostics (required, not optional)

Added to `diagnostics_tier{N}.json` and to `tmew1_aggregate_runs.py`:

- `latch_gate/{register}`: precision, recall, and F1 of `g > 0.5` against the
  event target. This is the "did it detect" number.
- `latch_retention/{register}`: accuracy of `argmax L[t, r]` bucketed by steps
  since the register's last true write (0, 1-5, 6-15, 16-30, 31+). This is
  the "did it keep it" number. If gate F1 is high and retention falls with
  lag, the fault is in the latch, which by construction it cannot be unless
  the gate is firing spuriously; so check `latch_gate` false-positive rate
  per step as a third line.
- `who_holds_token_by_handoffs` already exists and is the headline.

And one flag:

- `--latch-oracle-gate`: replace `g` with the ground-truth event target. This
  is a diagnostic upper bound, reported separately and never as a result. If
  the oracle run is near 1.0 and the learned run is not, the problem is
  detection; if the oracle run is also low, the problem is the value head or
  the read.

### 3.7 Config and flags

- `LatchConfig(enabled: bool, registers: tuple[str, ...], n_entities: int,
  audio_holder_offset: int = 8, numeric_entity_offset: int = 10,
  numeric_entity_stride: int = 5, gate_hidden: int = 64,
  log_capacity: int = 32, log_threshold: float = 0.5,
  oracle_gate: bool = False)` in `hpm_v2.py`, wired as
  `WorldModelConfig.latch_config: Optional[LatchConfig]`. The offsets are
  config fields because the model file cannot import them from
  `tmew1_train_v2.py` (that file imports the model).
- `build_model` enables it by default with the three registers above and
  fills the offsets from `NUMERIC_ENTITY_OFFSET` / `NUMERIC_ENTITY_STRIDE`.
- `ABLATIONS["latch"] = {"latch_config": None}` in `tmew1_run_v2.py`.
- `--latch-oracle-gate` CLI flag sets `oracle_gate=True`.
- `run_config.json` records all of it (it already serializes the config).

### 3.8 Interaction with the existing modules

- The latch output is **not** concatenated into `sequence`, so next-step
  prediction heads are untouched and the ablation stays clean.
- `StructuredStateTable` is unchanged. A later change could feed `L[t]` into
  its routing so entity GRUs see the relation; that is deliberately out of
  scope here because it would blur the ablation.
- The latch log and the typed event log are both offered to the history
  attention. They may be redundant; the ablation will say which.

## 4. Experiments that decide whether it worked

All at `--size default`, 3 seeds, `--probe-every-epoch`, on the honest
benchmark, run through the notebook's job runner.

1. `latch` on, learned gate (the result).
2. `latch` on, `--latch-oracle-gate` (upper bound, diagnostic only).
3. `--ablate latch` (should reproduce today's table; confirms the comparison
   is apples to apples).
4. `--ablate structured` with latch on (does the latch alone carry the holder
   query, or does it need the structured state).

Success is run 1 giving who_holds_token at handoffs >= 1 within 0.05 of
handoffs = 0, mean over 3 seeds, and `which_entity_changed_color_latch` at
lag 31+ above 0.8. Failure with oracle near 1.0 means iterate on the gate; failure
with oracle also low means iterate on the value head or the read.

Expected to stay flat: the emergent probes. This design makes the relation
explicit, so `retroactive_binding` and `state_reactivation` are not expected
to move; the claim being tested is "can the architecture hold a transient
fact", not "does it do so emergently". Say so in the write-up.

## 5. Implementation order

1. `_encode_modalities` returns per-modality features alongside `fused`.
   No behaviour change. Smoke test passes.
2. `EventLatchBank` module plus a unit test: write at t=5 with g=1, feed 60
   steps of random input with g=0, assert `L[65] == L[5]` exactly, and that
   `first_tagged` ignores a second write.
3. Wire into `forward` and `ForwardOutput`. The latch runs after the chunk
   loop on the full sequence, exactly like `StructuredStateTable` does, so it
   needs no `WorldModelState` fields; the training and eval loops never pass
   a `state` in. Smoke test passes with latch enabled and disabled.
4. Per-step targets for `first_tagged` and `color_changed` in the generator
   and collate, with a test in `_test_query_labels.py` that they agree with the
   query targets at `time_asked`.
5. `build_latch_targets` and the losses in `train_one_step`. Gate and
   latch metrics in both `evaluate` functions (`tmew1_run_v2.py` and
   `tmew1_train_v2.py`), in `evaluate_with_diagnostics`, and in the
   aggregate script. Both context builders change: `get_retrieval_context`
   in `tmew1_run_v2.py` and `_get_retrieval_context_v2` in
   `tmew1_diagnostics_v2.py`, plus `_build_query_input_v2` if the latch is
   appended to the query input.
6. Query head reads.
7. `--ablate latch`, `--latch-oracle-gate`, notebook section for the four
   runs, changelog entry, README results table row.

Steps 1 to 3 are one PR; 4 to 7 a second.

## 6. Left to the builder

- Exact widths (`gate_hidden`, value head size, log entry projection).
- Whether the value head for `holder` uses the raw audio one-hot directly as
  a prior (it is the answer on the cue step) or only as input to an MLP.
  Either is acceptable; the test in 3.6 will show whether detection is the
  bottleneck.
- Whether to keep the structured holder logits in the `who_holds_token` blend.
- Whether `log_capacity` of 32 is enough at T=64; it should be, since events
  per episode are in the single digits.
- Module file placement (`hpm_v2.py` next to the other v2 memories, or a new
  `latch_memory.py`).

## 7. Calls made here, for the reader to check

- The gate is supervised directly rather than learned from the query loss.
  Reason: the query loss has had three tiers and 33M parameters to find the
  cue and has not.
- The latch has no decay term. Reason: the benchmark's relations do not decay;
  a holder stays the holder until the next handoff. Decay is exactly the
  failure mode being removed.
- Registers are typed per relation rather than a generic slot bank. Reason:
  it makes the diagnostic in 3.6 possible and keeps the ablation
  interpretable. A generic bank is a reasonable follow-up once a typed one is
  shown to work.
- The raw cue channels bypass fusion. Reason: item 1 in section 1; the fusion
  softmax is a plausible single point of loss and there is no reason to make
  the detector fight it.
- The existing modules are left untouched. Reason: this change must be
  ablatable against today's numbers.
- The cue encoder sees one step of lag. Reason: the colour cue is a change,
  and a change needs two frames. One frame of observation history is not
  standing state.
- Pre-write register values are supervised to the benchmark's conventions
  (holder 0, others 0). Reason: the queries use those conventions, and an
  unsupervised prior would lose the handoffs=0 bucket the result is measured
  against.
- `which_entity_changed_color` stays eval-only and is read from the latch by
  a new probe line rather than through the query head. Reason: training it
  would change the benchmark; reading it through an untrained query embedding
  would measure the wrong thing.

A reviewer who had not seen the conversation read a draft of this document;
the calls above under "lag", "pre-write", "eval-only", the gate target
alignment in 3.5, the config offsets in 3.7, and the post-hoc placement in
section 5 are answers to that review.

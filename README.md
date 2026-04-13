How to actually run it
From the directory containing all three Python files plus the chunked model:

python tmew1_run.py --smoke

If that prints loss numbers and per-query accuracies without traceback, run the full thing:

python tmew1_run.py --epochs 4 --batch-size 8

The first real signal you're looking for is on the Tier 1 validation line. next_step should drop within the first epoch, latent_acc should climb above 0.5 (chance is 0.25 with four rules), and qacc/did_alarm_fire should be the first query type to lift off because it only requires recent state. If qacc/who_was_first_tagged is still at chance after Tier 2, that's your episodic memory failing — and that's the experiment the whole benchmark exists to run.
What's still missing for the full benchmark spec
This is honestly enough to start training and learn something real about the architecture. The remaining items from your spec — false-cue / belief-revision template, multi-entity interaction template, counterfactual queries, longer episode lengths, compositional held-out splits — all slot into the same generator and runner without architectural changes. I'd hold off on building them until you've seen Tier 1–2 learning curves, because those will tell you whether the problem is benchmark difficulty or model capacity. There's no point building Tier 5 if Tier 1 isn't moving.
When you're ready, the next thing I'd build is a held-out compositional eval set — same generator, but with template/delay combinations that never appear in training. That's the test that tells you whether the model is learning temporal abstraction or memorizing episode shapes, and it costs maybe 30 lines on top of what's already here.


Core losses (lower is better):


total — Weighted sum of all losses below. The single number the optimizer minimizes.
next_step — Next-timestep prediction loss. The primary world-model objective: given frames 0…t, predict frame t+1. Started bad (1.22) and quickly dropped to ~0.05 — the model learned the world dynamics fast.
aux_latent — Auxiliary cross-entropy loss for predicting which of the 6 latent rules is active. Started at ~1.79 (near chance for 6 classes = ln(6) ≈ 1.79) and dropped to ~0.12 by step 200 — the model learned to identify the hidden rule.
q_loss — Cross-entropy loss on query answers (who holds token, did alarm fire, etc.). Dropped from 0.60 → 0.26.
holder_loss — Per-timestep loss for predicting which entity holds the token right now. Shows 0.000 in Tier 1 because Tier 1 only has trigger_delay template (no handoffs). Jumps to ~2.0 in Tier 2 when handoff episodes appear, then drops to ~1.0 as it learns.
stress — Homeostatic stress from the RD (reaction-diffusion) module. Measures how far the internal state is from equilibrium. Low = stable. It's consistently good (95%+), which means the model isn't being destabilized.


Accuracies (higher is better):

latent_acc — What fraction of the batch the model correctly identified the hidden rule. Reached 100% by step 150.
holder_acc — Per-timestep accuracy: at each frame, does the model correctly predict who holds the token? 0% in Tier 1 (no handoffs), climbing to ~72% in Tier 2.
entity_acc — Accuracy on entity-answer queries (who_holds_token, who_was_first_tagged, which_entity_occluded, which_entity_first_occluded). Ranges 45-95% — these are hard because there are 6 possible entities.
binary_acc — Accuracy on yes/no queries (did_alarm_fire, did_trigger_before_alarm, did_chain2_fire). Quickly hit 100%.
PNN status (the pnn=O/C/O tag):

Progressive Neural Network gate states for the 3 layers. O=open (plastic, still learning), C=closed (converging), L=locked (frozen, fully learned). The gates cycling between O/C/L means the network is selectively consolidating learned knowledge.

---

## Query Types (Evaluation Tasks)

These are the questions posed to the model after it watches an episode unfold. They test different aspects of temporal reasoning, memory, and causal understanding. Each query is asked at a specific timestep and the model must answer using only information available up to that point in the episode.

### Entity-answer queries (model picks one of 6 entities)

| Query | What it tests | How it works |
|---|---|---|
| **who_holds_token** | Working memory + tracking through handoffs | A "token" is passed between entities when they collide. The model must remember the full chain of transfers and report who currently holds it. Bucketed by handoff count (0–6+) in diagnostics because difficulty scales with the number of transfers. |
| **who_was_first_tagged** | Episodic recall of temporal order | Multiple entities get "tagged" (triggered) during an episode. The model must remember which entity was tagged earliest — not the most recent one. Tests whether the model encodes event ordering, not just recency. |
| **which_entity_occluded** | Object permanence under occlusion | An entity is hidden (its vision pixels go dark). The model must identify which entity disappeared. Relatively easy when the model can compare consecutive frames. |
| **which_entity_first_occluded** | Temporal ordering + object permanence | When multiple entities are occluded at different times, the model must report which one was occluded first. Combines occlusion tracking with temporal memory. |

### Binary queries (model answers yes=1 / no=0)

| Query | What it tests | How it works |
|---|---|---|
| **did_alarm_fire** | Causal chain detection | Each episode has a trigger event that starts a countdown; when the countdown expires, an "alarm" fires. The model must detect whether this causal chain completed. Usually the first query to reach high accuracy because it only requires recent-state awareness. |
| **did_trigger_before_alarm** | Temporal ordering of causal events | The model must determine whether the trigger event occurred before the alarm fired — sounds trivial, but it requires the model to encode the relative order of two events in time, not just their existence. Tests temporal sequencing explicitly. |
| **did_chain2_fire** | Multi-chain causal reasoning | In `multi_chain` episodes, a second independent causal chain runs alongside the first. The model must track whether chain 2's alarm fired, independent of chain 1. Tests parallel causal tracking. |

### Diagnostics-only query (not used during training)

| Query | What it tests | How it works |
|---|---|---|
| **what_was_true_rule** | Belief revision under deception | Only generated in episodes with a "false cue" — a misleading signal injected to suggest the wrong latent rule. The model must see through the false cue and identify the actual active rule. Reported as `with_cue` accuracy in diagnostics. This is the hardest test: it requires the model to override a strong perceptual signal with deeper causal reasoning. |

### Diagnostic Breakdowns

- **who_holds_token by handoffs (0, 1, 2, 3, 4, 5, 6+)**: Accuracy bucketed by how many times the token changed hands. More handoffs = longer dependency chain = harder.
- **who_was_first_tagged by handoffs**: Same bucketing — more handoffs means more events competing in memory.
- **what_was_true_rule with_cue / without_cue**: Accuracy when a false cue was present vs. absent. The gap measures vulnerability to deception.
- **temporal ordering (trigger<alarm)**: Aggregate accuracy on `did_trigger_before_alarm`.
- **first occluded entity**: Aggregate accuracy on `which_entity_first_occluded`.
- **chain2 fire**: Aggregate accuracy on `did_chain2_fire`.

### Episode Templates

Each episode is generated from one of these templates, which determine what events occur:

| Template | What it generates |
|---|---|
| **trigger_delay** | A single entity triggers a countdown → alarm fires after a delay. The simplest causal chain. |
| **handoff** | The token is passed between entities on collision. Tests working memory across multiple transfers. |
| **multi_chain** | Two independent trigger→alarm causal chains run in parallel, each with its own trigger pair. |
| **occlusion_identity** | An entity is hidden partway through the episode. Tests object permanence. |
| **false_cue** | A misleading audio/visual signal suggests the wrong latent rule. Tests belief revision. |

### Curriculum Tiers

| Tier | Episode Length | Modalities | Key additions |
|---|---|---|---|
| **Tier 1** | T=24 | vision, numeric | Basic world model learning — trigger_delay only |
| **Tier 2** | T=48 | + audio | Handoffs, multi_chain, occlusion, false_cue |
| **Tier 3** | T=64 | + text | All templates, longest episodes, maximum difficulty |

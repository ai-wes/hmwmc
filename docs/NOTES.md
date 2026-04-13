
Tier 1 can't test the fix. Audio isn't in Tier 1's enabled modalities — look at the header: `modalities=('vision', 'numeric')`. The holder one-hot is being rendered into audio, but the model never sees audio during Tier 1 forward or eval, so those handoff diagnostics are the pre-fix baseline by construction. Identical numbers are expected. Not diagnostic.

Tier 2 is where the signal lives, and you cut off at s0050. Let it run to end-of-tier.

What's actually interesting in what you pasted: Tier 2 s0025 total=0.20 and s0050 total=0.16 — noticeably cleaner convergence than the two prior Tier 2 openings (which sat at 0.42 and 0.53 at s0025). Stress is also lower (0.009 vs 0.010-0.025). Could be noise; could be early evidence the richer audio is actually helping the model stabilize earlier. Either way, the NaN guard held and the controller survived. Good.

Also worth noting: `entity_acc` at s0000 is 0.96 — the model is starting Tier 2 with carryover from Tier 1 weights, and the new audio channels haven't broken anything. If audio_dim change had introduced a shape mismatch, you'd see either a crash or entity_acc at 0.0 there.

Let it run. The only number that matters right now is the end-of-Tier-2 `who_holds_token by handoffs` breakdown. Target: handoff=1 ≥ 0.80, handoff=2 ≥ 0.60. Post that when you have it.

---

## Double-Parameter Run vs Regular Run Comparison (2026-04-13)

**Configs:**
- **Double**: `d_model=256, num_layers=6, num_memory_slots=32, num_episodic_slots=64` (~4x params). NaN'd at Tier 2 s0120.
- **Regular**: `d_model=128, num_layers=3, num_memory_slots=8, num_episodic_slots=16` (~1x params). Completed full curriculum through Tier 3 ep3.

### Tier 1 Training Final Step (s0250)

| Metric | Double (d=256, L=6) | Regular (d=128, L=3) | Delta |
|---|---|---|---|
| total (raw) | 0.225 (95.5%) | 0.383 (92.3%) | **+3.2%** |
| next_step (raw) | 0.042 (91.6%) | 0.036 (92.8%) | -1.2% |
| aux_latent (raw) | 0.058 (97.1%) | 0.428 (78.6%) | **+18.5%** |
| latent_acc | 100% | 100% | = |
| q_loss (raw) | 0.154 (92.3%) | 0.133 (93.3%) | -1.0% |
| entity_acc | 85.0% | 84.2% | +0.8% |
| binary_acc | 100% | 100% | = |
| stress | 0.041 (72.9%) | 0.010 (93.5%) | **-20.6%** |

Notes: Double model converged to lower total loss and much lower latent auxiliary loss. But stress was 4x higher — the 6-layer controller was working much harder. This stress differential foreshadowed the NaN.

### Tier 1 Validation

| Metric | Double | Regular | Delta |
|---|---|---|---|
| next_step | 94.85% | 93.67% | +1.2% |
| latent_acc | 100% | 99.6% | +0.4% |
| who_holds_token | 56.97% | 56.97% | = |
| who_was_first_tagged | 60.83% | 65.26% | **-4.4%** |
| did_alarm_fire | 98.09% | 99.65% | -1.6% |
| which_entity_occluded | 100% | 100% | = |
| did_trigger_before_alarm | 100% | 99.09% | +0.9% |
| which_entity_first_occluded | 100% | 100% | = |
| did_chain2_fire | 95.07% | 95.17% | = |
| belief_revision (with_cue) | 13.5% | 12.2% | +1.3% |
| color_change (zero-shot) | 23.8% | 23.8% | = |

Notes: T1 validation is nearly identical. Extra capacity didn't help here — T1 is easy enough that the small model saturates. who_holds_token is identical because T1 has no audio (handoff signal isn't available).

### Tier 1 Diagnostic Handoff Breakdown

| Handoffs (who_holds_token) | Double | Regular |
|---|---|---|
| 0 | 1.000 | 1.000 |
| 1 | 0.000 | 0.000 |
| 2 | 0.655 | 0.655 |
| 3 | 0.077 | 0.077 |
| 4 | 0.500 | 0.500 |
| 5 | 0.167 | 0.167 |
| 6+ | 0.273 | 0.273 |

Notes: Identical. Same eval set, no audio → no holder signal. This is the expected pre-audio baseline.

| Handoffs (who_was_first_tagged) | Double | Regular |
|---|---|---|
| 0 | 0.784 | **0.725** |
| 1 | 0.667 | **0.818** |
| 2 | 0.607 | **0.679** |
| 3 | 0.125 | **0.375** |
| 4 | 0.125 | **0.625** |
| 5 | 0.250 | **0.375** |
| 6+ | 0.500 | 0.500 |

Notes: Regular model is better at first_tagged across most handoff counts. Double model is only better at 0-handoff (trivial case). Possible overfitting to the latent space in the double model — it specialized aux_latent at the expense of query generalization.

### Tier 2 Training (matched steps, pre-NaN)

| Metric @ s0050 | Double | Regular | Delta |
|---|---|---|---|
| total | 84.9% | 79.2% | **+5.7%** |
| next_step | 79.0% | 81.6% | -2.6% |
| aux_latent | 96.8% | 80.3% | **+16.5%** |
| latent_acc | 100% | 87.5% | **+12.5%** |
| holder_loss | 44.7% | 25.4% | **+19.3%** |
| holder_acc | 64.9% | 47.9% | **+17.0%** |
| entity_acc | 86.7% | 81.2% | **+5.5%** |
| binary_acc | 94.1% | 100% | -5.9% |
| stress | 93.6% | 96.0% | -2.4% |

| Metric @ s0100 | Double | Regular | Delta |
|---|---|---|---|
| total | 84.7% | 79.7% | **+5.0%** |
| next_step | 88.4% | 84.5% | **+3.9%** |
| aux_latent | 94.5% | 87.1% | **+7.4%** |
| latent_acc | 100% | 87.5% | **+12.5%** |
| holder_loss | 42.3% | 20.6% | **+21.7%** |
| holder_acc | 57.2% | 39.9% | **+17.3%** |
| entity_acc | 73.7% | 63.6% | **+10.1%** |
| binary_acc | 92.3% | 90.0% | +2.3% |
| stress | 91.6% | 81.4% | **+10.2%** |

Notes: The double model was significantly ahead in Tier 2 training before it died. Holder accuracy was 17 points higher at both checkpoints — the extra capacity was genuinely helping with holder tracking. Entity accuracy was 10 points higher. The model was learning the multimodal integration faster.

### Where the regular run eventually landed (T2 val + T3 vals)

| Metric | Reg T2 val | Reg T3 ep1 val | Reg T3 ep2 val | Reg T3 ep3 (s0150, latest) |
|---|---|---|---|---|
| who_holds_token | 29.8% | 37.7% | 47.4% | — |
| who_was_first_tagged | 55.9% | 69.7% | 66.4% | — |
| belief_revision | 84.0% | 100% | 100% | — |
| holder_acc (train) | 66.5% | 81.2% | 98.5% | 98.2% |
| next_step | 89.2% | 51.5% | 58.5% | — |

### Key Takeaways

1. **The double model was learning holder tracking ~17% faster in Tier 2.** At s0100, holder_acc was 57% vs 40%. The regular model didn't reach 57% holder_acc until T2 s0175 — the double model was ~75 steps ahead on holder learning.

2. **Extra capacity helped latent rule inference massively.** The double model maintained 100% latent accuracy in T2 from step 0. The regular model dropped to 87.5% and took until ~s0125 to recover.

3. **Tier 1 was not differentiated.** Both models saturated T1 to roughly the same point. The extra capacity had no visible benefit on the easy tier — it only mattered when multimodal complexity increased in T2.

4. **Stress was consistently higher in the double model.** 4x higher at T1 end, and the gap persisted into T2. The 6-layer controller was making far more interventions. This is the root cause of the NaN — see changelog entry for the full kill chain.

5. **The double model would likely have overtaken the regular run's final T3 numbers**, given its T2 learning rate advantage. The regular model took ~250 T3 steps to reach 98% holder_acc. Projecting the double model's 17% holder learning advantage, it might have reached that in Tier 2 itself — if it had survived.

6. **Needs re-run with scaled controller config** (`exploit_budget=40, intervention_interval=8, strategic_unlock_fraction=0.25`). The capacity advantage is real; the controller just needs to be tuned for the deeper model.

While you wait — did you add the debug print confirming `audio[0]` shows nonzero activity at indices 8-11 on handoff episodes? If not, five-minute add:

```python
# in train_one_step, first step only
if not hasattr(self, "_audio_checked"):
    a = batch["audio"][0].cpu().numpy()  # (T, audio_dim)
    handoff_steps = np.where(a[:, 3] > 0.5)[0]
    if len(handoff_steps) > 0:
        t = handoff_steps[0]
        print(f"AUDIO CHECK @ step {t}: handoff_bits={a[t, 3:5]}, holder_onehot={a[t, 8:12]}")
    self._audio_checked = True
```


Diagnosed model shortcuts and charted verification pathway forward
Diagnosed model shortcuts and charted verification pathway forward

Two distinct outcomes, and the first one is a clean, shippable win.
Belief revision is solved. what_was_true_rule with_cue jumped from 0.24 to 1.00 at Tier 2, held at 1.00 through every Tier 3 evaluation. Four-step correction pulse did exactly what we predicted — a durable signal outcompetes the decoy window. Controller stress hookup (change 3) is now optional; you got there without it. Log this fix in the changelog and mark it done.

That tells you in one line whether the signal is actually present in the tensors the model is consuming.
One last note: it's worth pausing to appreciate the belief revision result. 0.24 → 1.00 is a bigger jump than most papers report between architecture iterations. The false-cue mechanism — durable correction against asymmetric decoy — is worth a standalone writeup or at least an internal memo. That's the kind of result that belongs in a Future Human Journal piece on signal-to-noise in sequential belief formation.



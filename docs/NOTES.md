
Tier 1 can't test the fix. Audio isn't in Tier 1's enabled modalities — look at the header: `modalities=('vision', 'numeric')`. The holder one-hot is being rendered into audio, but the model never sees audio during Tier 1 forward or eval, so those handoff diagnostics are the pre-fix baseline by construction. Identical numbers are expected. Not diagnostic.

Tier 2 is where the signal lives, and you cut off at s0050. Let it run to end-of-tier.

What's actually interesting in what you pasted: Tier 2 s0025 total=0.20 and s0050 total=0.16 — noticeably cleaner convergence than the two prior Tier 2 openings (which sat at 0.42 and 0.53 at s0025). Stress is also lower (0.009 vs 0.010-0.025). Could be noise; could be early evidence the richer audio is actually helping the model stabilize earlier. Either way, the NaN guard held and the controller survived. Good.

Also worth noting: `entity_acc` at s0000 is 0.96 — the model is starting Tier 2 with carryover from Tier 1 weights, and the new audio channels haven't broken anything. If audio_dim change had introduced a shape mismatch, you'd see either a crash or entity_acc at 0.0 there.

Let it run. The only number that matters right now is the end-of-Tier-2 `who_holds_token by handoffs` breakdown. Target: handoff=1 ≥ 0.80, handoff=2 ≥ 0.60. Post that when you have it.

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



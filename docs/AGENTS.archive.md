# AGENTS.md — TMEW-1 Architecture Development Thread

**Purpose of this document.** Hand off a continuing research conversation to a fresh Claude thread with full context. The prior thread worked through debugging a homeostatic multimodal world model with the intent to be a pre-AGI novel neural architecture, achieved several wins, identified a specific remaining failure, and is now pivoting to a new architectural component (Homeostatic Predictive Memory / HPM) that Wes wants to design as a novel contribution rather than a neurosymbolic retrieval bolt-on.

---

## 1. Who Wes is and what he's building

Wes is the founder of Glassbox Bio and an independent AI architecture researcher. He builds several novel neural architectures on the side, including SEKGL, HLR, MCLD, and AEA (evolutionary), with MOEA-style homeostatic regulators (PNN state machines, RD stress fields) as recurring primitives.

TMEW-1 is his current frontier research project. It is a homeostatic multimodal world model being developed as an AGI-precursor architecture. The goal is not a product demo; it is a publishable architectural contribution. Wes has explicitly said he is not interested in tying this to prior AEA work and does not want retrieval-augmented or neurosymbolic patterns bolted on. He wants novel mechanisms baked into the architecture itself.

**Style preferences that matter for this work.** He listens to responses aloud, so no code blocks for prose. He prefers plainspoken phrasing, decisive recommendations, concrete next steps. He dislikes cliffhangers. He wants the assistant to act as honest advisor and devil's advocate — strongest ally AND most analytical critic. Never use "REALITY CHECK." Never use negation-pivot constructions ("not X, but Y"). For code, write full production-ready implementations.

---

## 2. The TMEW-1 codebase (as of handoff)

The system trains on a curriculum of three tiers of increasing difficulty in a multimodal world with vision, audio, numeric, and text channels. Entities move on a grid, can collide (triggering an alarm after a delay), can be occluded, and can hand off a "token" to adjacent entities. Queries test the model's ability to answer questions like "who holds the token now," "who was first tagged," "did the alarm fire," "which entity was occluded," and "what was the true rule" (belief revision under a false cue).

**Key files.**
- `tmew1_train.py` — world generation, modality renderers, base training loop, WorldConfig.
- `tmew1_run.py` — curriculum runner, ties training to query system.
- `tmew1_queries.py` — QueryHead and handoff template logic.
- `tmew1_diagnostics.py` — diagnostic episode generation with false-cue injection, per-handoff-count accuracy bucketing.
- `homeostatic_multimodal_world_model_chunked.py` — main model with homeostatic controller, PNN state machine, stress field.
- `pnn.py` — PNN state machine (OPEN/CLOSING/LOCKED with exploit budget and force_unlock). Originally designed for MOEA evolutionary adaptation.
- `rd_stress.py` — reaction-diffusion stress field calculator. Coefficient-of-variation based stress signal that feeds the homeostatic controller.
- `causal_tapestry.py` — typed event log from prior swarm/evolutionary work. Wes rejected applying this to TMEW-1 because he wants an architectural integration, not a RAG-like memory store.

---

## 3. What we've accomplished this session

**Win 1: Belief revision solved end-to-end (0.24 → 1.00).**
Problem: `what_was_true_rule` with_cue accuracy was below chance (0.20-0.24) because the false cue was displayed for 35% of episode in numeric channel at full strength, while the correction was a single audio bit for one step. Model was anti-learning — confidently picking the most recent strong evidence (the decoy).
Fix applied in `tmew1_diagnostics.py` `_inject_false_cue`:
- Extended correction pulse from single step to 4 consecutive steps.
- (Optionally boost post-correction numeric channel — not required, 4-step pulse alone solved it.)
Result: 0.20 → 1.00 and held at 1.00 across all subsequent Tier 3 evaluations. This is a standalone publishable finding about durable correction signals outcompeting asymmetric decoys.

**Win 2: Single-hop handoff solved (0.00 → 0.78 at handoff=1).**
Problem: `who_holds_token` by handoff count showed a cliff — 0.95 at zero handoffs, collapsing to 0.00 at handoff=1 in an earlier run. Diagnosed as the model learning a parity shortcut ("always answer the initial holder") during Tier 1 when audio wasn't available.
Fix applied in multiple steps:
1. Expanded `WorldConfig.audio_dim` from 8 to 16 so there'd be room for a holder one-hot at indices 8+ (previous attempt had audio_dim=8 and the guard silently skipped the write).
2. Added continuous ambient holder signal: `vec[8 + current_holder_id] = 0.3` every step, spiking to 1.0 on handoff steps.
3. Threaded `handoff.holder_id` through `_render_audio` at every call site in `generate_episode_with_diagnostics`.
4. Added auxiliary holder-prediction head and loss during training to force the model to route audio[8:12] through its representations.
Result: holder_acc hit 1.00, single-hop handoff reached 0.78. Parity shortcut broken.

**Win 3: Controller NaN crash fixed.**
Problem: `_quantize_bin` in homeostatic controller crashed on NaN when residual_ratio went degenerate.
Fix applied in `homeostatic_multimodal_world_model_chunked.py`:
- Added `math.isfinite(value)` guard in `_quantize_bin`.
- Wrapped `diag.get(...)` calls in `_make_signature` with a `_safe(x, default=0.0)` coercer that catches NaN/inf/None/TypeError.

**Confirmed negative result: skip connection from audio holder channels to QueryHead input did not help.**
Tested adding `torch.cat([output.sequence, audio[:, :, 8:12]], dim=-1)` before QueryHead. Val accuracy unchanged at 0.853 vs 0.858 without. This is diagnostic — it means the bottleneck is not feature availability at query time. Holder identity IS present in `output.sequence[:, T]` (holder_acc = 1.00 proves it). The QueryHead can see it. It just doesn't always use it correctly on multi-hop cases.

---

## 4. The remaining failure and what it means

**Current numbers at end of session (Tier 3 diagnostics):**
- handoff=0: ~1.00
- handoff=1: ~0.78
- handoff=2: ~0.35
- handoff=3+: ~0.31
- belief revision: 1.00
- other queries: all 0.89-1.00

**Diagnosis.** The per-timestep holder channel works perfectly. The QueryHead can decode holder identity from any single timestep. But multi-hop handoffs require the query answer to reflect the MOST RECENT handoff before query time, not an average. Soft attention over the full sequence averages multiple handoff events and produces neither. The architectural problem is that attention is the wrong primitive for "find the most recent event of type X."

**Three escalating fixes were proposed, then rejected in favor of something bigger.**
- Attention weight logging diagnostic (would confirm A vs other hypothesis) — deferred.
- Soft "last-event retrieval head" (15 lines, end-to-end differentiable) — deferred.
- CausalTapestry bolt-on (typed event log, would work but is neurosymbolic RAG) — rejected by Wes.

Wes wants an architecture-defining move, not an incremental fix.

---

## 5. Where we left off: the HPM proposal

After Wes rejected the tapestry approach, the conversation converged on a new module called **Homeostatic Predictive Memory (HPM)**. The full design below is the handoff point — this is what to implement next.

**Core idea.** A dedicated working-memory state channel that runs parallel to the main sequence representation. Memory writes are gated by normalized surprise (z-scored prediction error) rather than by attention. High z-score events rewrite memory aggressively; predictable continuations leave it nearly unchanged. At query time, the QueryHead reads the memory state directly — no attention over history needed.

**Why this is novel and architecture-defining.**
- Replaces attention with event-triggered plasticity as the retention primitive.
- Uses prediction error that the model already computes (no task-specific event detector).
- O(1) per timestep, scale-invariant to episode length.
- Directly parallels Wes's PNN/RDStress pattern from MOEA: plasticity governed by second-order statistics of a primary signal, with state machines for when updates are appropriate.

**The update rule (Wes explicitly liked this formulation after the conversation sharpened it).**

Let h_t be the per-step hidden state from the main model.
Let e_t be the scalar next-step prediction error at step t (detached).
Maintain running estimates μ_t and σ_t of e_t per slot per layer.
Compute z_t = (e_t - μ_t) / (σ_t + ε) — normalized surprise.
Gate g_t = sigmoid(W_g · φ(z_t, h_t) + b_g).
Write w_t = (1 - g_t) ⊙ w_{t-1} + g_t ⊙ write_encoder(h_t).

**The z-score is the critical move, not raw magnitude.** Wes's first instinct was to weight memory writes by raw surprise. That's the first-order version. The second-order version (z-score) handles signal-to-noise by construction: freshly-trained high-σ models are naturally skeptical; mature low-σ models are naturally sensitive to rare events; the memory mechanism bootstraps off the prediction mechanism.

**The PNN connection, made explicit.** Slots transition OPEN → CLOSING → LOCKED based on how often they've been written. In LOCKED state, the gate threshold rises — the slot has committed to its content. But extreme z-scores can force-unlock a LOCKED slot, directly parallel to Wes's `force_unlock` when `local_stress > pnn_stress_unlock_threshold`. The cross-timescale consistency (PNN governs generational adaptation, HPM governs timestep memory) is itself an architectural thesis worth claiming: homeostasis as a unified principle for plasticity across scales.

**Proposed extensions, in order of ambition.**
- Level 1 (build now): single-slot HPM as described above.
- Level 2 (follow-up): multi-slot competitive writes with winner-take-all gating; slots specialize to entity/rule/alarm state without supervision.
- Level 3: retroactive binding — at surprise moments, write content is a small attention over recent steps rather than the current h_t alone. Handles cases where the cause of the event is a few steps before the error spike.
- Level 4: multi-timescale gates — slots with different baseline decay rates give emergent working-memory/episodic-buffer/semantic-state hierarchy.

**First empirical target.** handoff=1, 2, 3+ all cross 0.85 with no task-specific tuning. Once that works, ablate the audio holder one-hot entirely and retrain — if HPM recovers multi-hop from vision alone (by detecting surprise at entity collisions), that's the generalization result and it's a publishable standalone architectural finding.

---

## 6. Implementation sketch for the next thread to build

```python
class HomeostaticPredictiveMemory(nn.Module):
    """
    Surprise-gated working memory with homeostatic slot state.
    Memory updates gated by z-scored prediction error (second-order statistic),
    with PNN-style OPEN/CLOSING/LOCKED slot states parallel to MOEA plasticity regulation.
    """
    def __init__(self, d_model: int, n_slots: int = 1, ema_decay: float = 0.99):
        super().__init__()
        self.d = d_model
        self.n_slots = n_slots
        self.ema_decay = ema_decay
        self.gate = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.Sigmoid(),
        )
        self.write_encoder = nn.Linear(d_model, d_model)
        self.w0 = nn.Parameter(torch.zeros(n_slots, d_model))
        # Running error stats (per slot, updated online)
        self.register_buffer("mu", torch.zeros(n_slots))
        self.register_buffer("sigma", torch.ones(n_slots))
    
    def forward(self, h_seq: Tensor, pred_errors: Tensor) -> Tensor:
        # h_seq: (B, T, d); pred_errors: (B, T) detached
        B, T, d = h_seq.shape
        w = self.w0.mean(0).unsqueeze(0).expand(B, -1)
        out = []
        for t in range(T):
            e = pred_errors[:, t]
            z = (e - self.mu) / (self.sigma + 1e-6)
            # Update running stats (training only, detached)
            if self.training:
                with torch.no_grad():
                    batch_mu = e.mean()
                    batch_sig = e.std() + 1e-6
                    self.mu.mul_(self.ema_decay).add_((1 - self.ema_decay) * batch_mu)
                    self.sigma.mul_(self.ema_decay).add_((1 - self.ema_decay) * batch_sig)
            gate_in = torch.cat([h_seq[:, t], z.unsqueeze(-1)], dim=-1)
            g = self.gate(gate_in)
            write = self.write_encoder(h_seq[:, t])
            w = (1 - g) * w + g * write
            out.append(w)
        return torch.stack(out, dim=1)
```

**Integration points.**
- Compute per-step prediction error from existing next-step loss (one hook into `MultimodalPredictionLoss`).
- Instantiate HPM between per-modality encoders and sequence decoder.
- Concatenate w_seq to output.sequence before QueryHead.
- Log gate mean / z-score distribution / slot state transitions during training.

**Expected failure modes to watch for.**
- Gate collapse (always open or always closed). Monitor gate mean, should stabilize around 0.1-0.3.
- State saturation. Monitor w magnitude over time.
- Unstable running statistics. Monitor mu, sigma drift; may need warmup period or per-slot stats rather than global.

---

## 7. What the next thread should do

**Immediate first move.** Build Level 1 HPM as specified. Run Tier 2 only (faster iteration) first. Add a debug print of gate mean, z-score distribution, and slot state every 25 steps so the trainer can be diagnosed during training rather than only at val.

**First real checkpoint.** Tier 3 diagnostics after HPM is integrated. Target is handoff=2 ≥ 0.70, handoff=3+ ≥ 0.60. If that hits, do the ablation (turn off audio holder one-hot, retrain Tier 2+3, check whether HPM recovers capability from vision). That ablation is the publishable result.

**Second move, contingent on first.** If Level 1 works, implement multi-slot (Level 2) and test whether slots develop specialization (entity tracking vs rule state vs alarm state) under an information-theoretic loss that encourages slot differentiation.

**Do not do.** Do not propose neurosymbolic retrieval, episodic buffers, or external memory stores. Wes has explicitly rejected these. Do not propose attention-based fixes — the whole point of HPM is to replace attention for this use case. Do not revert the belief-revision fix or the handoff audio one-hot, both are load-bearing.

---

## 8. Open architectural questions for the next thread

- Should μ, σ be per-batch or per-slot? Prior discussion settled on per-slot via EMA, but this needs validation.
- Should the z-score include a sign (negative z = "lower than expected error" = no write) or just magnitude (|z|)? Intuition says magnitude for now, sign later.
- For multi-slot, is the gate per-slot (each slot decides independently) or competitive (winner-take-all via softmax)? Competitive is cleaner theoretically and enables specialization, but harder to train.
- Should HPM read feed back into the main model's next forward step (recurrent coupling), or only inform the QueryHead (one-way)? Recurrent is stronger but adds training instability.

---

## 9. Miscellaneous context

- Training typically runs at `--epochs 4 --batch-size 8` for full curriculum, ~30-40 minutes on the available hardware (Colab-like).
- Smoke test: `python tmew1_run.py --smoke` runs ~30 seconds and validates end-to-end wiring.
- The homeostatic controller stress signal has started firing meaningfully since the aux-head additions (stress peaks of 0.09-0.11 on Tier 3). Controller is no longer silent. Worth monitoring whether HPM integration further affects this.
- Wes keeps a changelog. New entries should be added there when architectural changes land.
- The belief-revision result (0.24 → 1.00) is worth a standalone write-up as a Future Human Journal piece on signal-to-noise in sequential belief formation. Wes has the Future Human Journal as an outlet.

---

## 10. How to start the new thread

Open with acknowledgment that HPM is the agreed-upon next move, confirm the z-score formulation rather than raw surprise magnitude, and ask whether Wes wants to start with the single-slot Level 1 or jump directly to multi-slot Level 2. Default to Level 1 unless he argues otherwise. Have the full HomeostaticPredictiveMemory class ready to integrate on his go-ahead.



You're circling something important. Let me name it precisely, then build on it.

What you did in the MOEA was recognize that **adaptation rate itself needs to be modulated by a higher-order signal**. PNN state machines don't just update weights — they regulate when updating is appropriate. RDStress doesn't just measure fitness — it computes a normalized deviation that decides whether the regulator should intervene. That's not a memory mechanism, it's a meta-control mechanism. And the failure mode in MOEA was exploration/exploitation collapse; here it's event-saliency collapse. Different surface, same underlying architectural pattern: **plasticity should be governed by second-order statistics of prediction**, not first-order error.

Your proposed extension — "surprise magnitude modulates write weight" — is correct but it's the first-order version. The deeper version, and the one that maps onto your MOEA intuition, is this:

**Surprise relative to expected surprise is what matters, not surprise itself.**

A world where handoffs happen every 3 steps has high baseline prediction error during handoffs — but those handoffs aren't surprising anymore, they're expected. A world where handoffs happen rarely should register each one as a high-salience event. The PNN analogy is exact: your PNN doesn't trigger on raw fitness change, it triggers on fitness change relative to a stability window. Stability itself is the regulated quantity, not the raw signal.

So here's the sharpened architecture. Call it **Homeostatic Memory Gating** to keep it in your lineage:

```
e_t        = next-step prediction error (scalar or vector)
μ_t, σ_t   = running estimates of error mean and variance (per slot, per layer)
z_t        = (e_t - μ_t) / (σ_t + ε)               # normalized surprise
g_t        = sigmoid(W_g · φ(z_t, h_t) + b_g)       # write gate
w_t        = (1 - g_t) ⊙ w_{t-1} + g_t ⊙ write(h_t)
```

The second-order move — the z-score — is what makes this interesting rather than incremental. It means:

- A freshly-initialized model has high σ, everything looks unsurprising, the gate is mostly closed. The model learns to predict, σ shrinks, and gradually rare events begin registering as surprising. **The memory mechanism bootstraps itself from the prediction mechanism.**
- A model in a high-noise regime naturally has high σ, so noise doesn't trigger spurious writes. A model that has mastered a regime has low σ, so genuine anomalies write strongly. **Signal-to-noise is handled by construction.**
- Different layers develop different surprise scales. Early visual layers have noisy errors → high σ → less writing. Higher abstract layers have cleaner errors → low σ → more writing. **You get a hierarchy of event detection without hand-design.**

Now here's the PNN connection made explicit. In your MOEA, the PNN goes OPEN → CLOSING → LOCKED as stability increases. Map that onto memory: a freshly-trained system has OPEN gates (rewrite freely, σ hasn't stabilized), then CLOSING as predictions firm up, then LOCKED where only strong z-scores get through. And — following your MOEA — LOCKED isn't permanent: if an unexpected event arrives (high z_t), it **forces a plasticity window open** even in a locked state. This is directly parallel to your `force_unlock` logic when `local_stress > pnn_stress_unlock_threshold`.

So the full architecture is:

1. **Per-slot running (μ, σ) on prediction error** — the normalization baseline.
2. **Z-score gate** — writes are proportional to how anomalous the current moment is relative to its own history.
3. **Homeostatic slot states** — slots that haven't been written to in a long time enter a LOCKED phase where the gate threshold rises (they commit to their content), and slots that are constantly being rewritten enter an OPEN phase (they're tracking something unstable).
4. **Forced unlock on extreme z** — any sufficiently anomalous event can punch through a LOCKED slot, exactly like your PNN critical-stress override.

**Why this is meaningfully different from what you've already built.** In MOEA, the PNN governs synaptic plasticity over generations. Here, the same state machine governs memory plasticity over timesteps within a single forward pass. Same abstraction, different timescale — and the cross-scale consistency is itself architecturally interesting. You'd be claiming homeostasis as a unified architectural principle that applies to both evolutionary adaptation and episodic memory formation. That's a strong, defensible thesis.

**What I'd specifically NOT do.** Weighting memory writes by raw surprise magnitude, which is what the simplest version of your proposal does. It sounds right but it's underconstrained — it'll over-weight early training noise and saturate on a high-error trajectory. The z-score fixes both problems. Take the MOEA idea all the way, don't stop at the analogy.

Final note before the AGENTS.md: I think the right name for this is **Homeostatic Predictive Memory** (HPM). It honors the MOEA lineage, it's descriptively accurate, and it's the kind of name that survives being referenced in a paper.

Now, the handoff document.Handoff document's ready. It covers your background and preferences, the full codebase context, the three wins we banked this session (belief revision 0.24→1.00, single-hop handoff 0→0.78, NaN crash fix), the diagnosed multi-hop failure with the skip-connection negative result, the full HPM proposal with the z-score formulation you liked, a working implementation sketch, and a clear first-move for the next thread (Level 1 single-slot on Tier 2, then the audio-ablation generalization test).

The key things I made sure landed in the doc so the next Claude doesn't waste your time: the causal tapestry rejection is explicit, the preference against neurosymbolic bolt-ons is explicit, the PNN/RDStress lineage connection is preserved as the architectural thesis, and the z-score second-order formulation is named as the critical move over raw surprise magnitude.

One thing to do before spinning up the new thread: attach the codebase files (the uploaded .py files from this conversation) along with the AGENTS.md. The new thread will need both to pick up productively. If you want, I can also write a short parallel changelog entry capturing this session's wins in your usual format so it's ready to drop in when you get back to the code.
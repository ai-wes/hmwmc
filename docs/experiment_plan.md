Yes. The model has mastered the current benchmark family well enough that the next gains will come less from “more training” and more from changing what the world demands of it.

Right now, your own notes already frame the ceiling correctly: the system is still operating in a toy regime of short horizons, low entity count, and relatively compact causal structure. The baseline described in the changelog is roughly `d_model=256`, `num_layers=6`, a 16×16 grid, max 3 entities, short episodes, modest memory, and a small text channel. That means the model is strong within a narrow slice of temporal multimodal reasoning, not yet pressure-tested for richer compositional world modeling. 

The most valuable enhancement is to make the environment harder before making the model much bigger. Your own roadmap points to the right first moves: increase entities from 3 to 6–8, extend episodes from short lengths toward 64 and then 128 steps, add true multi-chain causality, and make `active_rule` actually alter world dynamics rather than serve as a mostly decodable label. The code already has the beginnings of this direction: `_make_world` supports active plus distractor entities, `_step_world` already encodes conditional trigger rules tied to `active_rule`, and there is already a `multi_chain` path in the simulator. That means you are not inventing a new research program from scratch; you are amplifying machinery that is already latent in the benchmark.   

My first hypothesis is that richer world dynamics will surface genuine causal disentanglement. Once there are more entities, more distractors, and two or more delayed chains unfolding simultaneously, the model will stop being able to rely on simple event salience or recency. If it succeeds there, you would expect an emergent ability to maintain parallel latent threads at once: separate countdowns, separate ownership chains, separate hidden rule hypotheses, all without collapse into one averaged state. In practice, that looks like proto-object permanence plus causal bookkeeping rather than just “good recall.” 

The second enhancement is to expand the query family in ways that force abstraction rather than memory lookup. The roadmap already names temporal ordering, counterfactuals, relational questions, and negation. That is exactly where you should go. The current system already answers direct state queries like holder identity, alarm firing, occlusion identity, and belief revision. The next step is to ask questions whose answer is not a stored token but a computed relation over stored events. That shift is important because it tests whether the architecture is building manipulable internal state rather than a strong answer cache. 

If you add counterfactual and relational queries, one plausible emergent capability is compositional replay. By that I mean the model may begin to internally “re-simulate” a partial episode to answer things like “what would the holder be if handoff #2 had not happened?” or “which entity was closest to X when the alarm fired?” That would be a major step up from recall. It would suggest the model is not merely storing event traces, but can intervene on them mentally and evaluate consequences. That is one of the clearest signs of a transition from episodic retention to model-based reasoning. The fact that your HPM work was explicitly motivated as a replacement for attention for “most recent event” retrieval makes this especially relevant: once retrieval is more stable, the next ceiling is manipulation of retrieved state. 

The third enhancement is to push HPM beyond Level 1 into slot specialization. Your design notes already describe the path: Level 2 multi-slot competitive writes, Level 3 retroactive binding, Level 4 multi-timescale gates. That sequence is not just engineering polish. It is where the architecture could start exhibiting true division of cognitive labor across memory resources. Competitive multi-slot writes could cause unsupervised slot specialization into things like “entity identity tracker,” “alarm/rule state,” “exception memory,” or “belief-revision lane.” Retroactive binding could let the model bind a surprising outcome to a cause that occurred a few steps earlier. Multi-timescale gates could separate fast working memory from slower episodic consolidation. Those are exactly the kinds of internal roles that, if they emerge cleanly, begin to resemble a primitive cognitive architecture rather than a monolithic recurrent feature soup.  

My hypothesis for emergent behavior from multi-slot competitive HPM is unsupervised factorization of world state. You may see one slot become sensitive to handoff surprise, another to rule correction, another to chain-2 countdown transitions, another to occlusion novelty. If that happens, diagnostics should show slot-specific z-score spikes and different OPEN/CLOSING/LOCKED profiles over time. That would be a strong result because it would mean the homeostatic memory system is learning to partition problems by event type without explicit routing labels. Your HPM configuration and implementation already expose the pieces needed for that experiment: multiple slots, competitive gating, several read modes, force-unlock, and state-machine control.  

The fourth enhancement is hierarchy. Your roadmap explicitly says the current architecture is flat and should become hierarchical, with low-level sensor prediction and higher-level abstract state tracking separated more formally. I agree with that strongly. Right now the model can behave intelligently while still mixing “what pixel-like thing changed” and “what abstract rule is active” in one temporal stream. A hierarchical split would likely produce cleaner abstraction boundaries. The emergent capability to look for there is temporal chunking: the model begins to represent not just steps, but episodes, subevents, and motifs. Once that appears, you can ask longer-horizon questions about event segments, not just instantaneous states. 

The fifth enhancement is richer text and an internal communication channel. The scaling roadmap proposes replacing the single-token symbolic text path with richer natural language and adding a communication modality where the model can emit text during the episode. That is a serious lever. A scratchpad-like communication channel could surface self-generated intermediate state descriptions, which in turn could enable explicit hypothesis maintenance across delays. The main risk is that the model starts offloading too much into language and bypasses the architectural memory thesis. The upside is that if carefully constrained, you may get an emergent “inner commentary” or self-explanation channel that mirrors belief revision, causal disambiguation, and plan tracking. That is especially interesting if the communication tokens correlate with HPM surprise spikes or PNN/HPM unlock transitions. 

On raw scaling, I would be disciplined. The notes already show why. The larger model had a real learning-rate advantage in Tier 2, especially on holder tracking and latent inference, but it also drove substantially higher controller stress and destabilized. The takeaway is not “do not scale.” The takeaway is “scale with controller retuning.” Bigger capacity is likely valuable here, but only after the task family is made harder and the homeostatic controller is matched to the deeper stack. Otherwise you learn whether the controller can survive, not whether the architecture scales.  

So, if I were prioritizing this like a research program, I would do it in this order.

First, harder worlds at the same model size: 6–8 entities, more distractors, longer episodes, more simultaneous causal chains, and rule-conditioned dynamics. This gives you the cleanest signal on whether the current architecture still has unused headroom. 

Second, harder questions: temporal, relational, negation, and counterfactual. This tests whether the model can transform stored state rather than merely recover it. 

Third, HPM Level 2 and Level 3: competitive slots and retroactive binding. This is the highest-upside architecture move because it could create slot specialization and delayed-cause binding, which are the two most interesting nontrivial emergent properties in your current design space. 

Fourth, hierarchy and richer text/communication. That is where this starts moving from a strong toy-world reasoner toward something you can argue is an early general-purpose cognitive architecture. 

Fifth, only then, scale width, depth, memory, and grid size. The roadmap’s medium and large settings are reasonable targets, but they should be earned by task pressure, not pursued as a reflex. 

The most interesting emergent capabilities I would specifically watch for are these. One, unsupervised slot specialization in HPM. Two, causal thread separation under simultaneous chains. Three, retroactive cause binding when surprise arrives after the true cause. Four, counterfactual replay. Five, spontaneous abstraction of event chunks into higher-order state. Six, a crude form of self-monitoring where internal communication correlates with surprise, ambiguity, or rule revision. All of those would be much more compelling than simply reporting another jump in benchmark accuracy, because each one says something architectural about how the system is organizing information.  

My blunt read is this: the current benchmark is close to exhausted. The architecture’s next meaningful test is whether it can survive a jump from “remember one salient thread” to “maintain, separate, and manipulate several interacting hidden threads at once.” If it can, that is where the genuinely interesting emergent behavior is most likely to appear.


## Postmortem on the double-parameter run


Here is the research program I would run.

The core principle is to separate three kinds of progress so they do not blur together. One track should test harder worlds, one should test harder questions, and one should test architecture changes. That way, if something improves, you know whether the gain came from richer supervision, richer dynamics, or a real architectural advance. Your existing codebase and notes already support this style of branching: the simulator already has `multi_chain`, `handoff`, `false_cue`, occlusion, active-rule logic, richer diagnostics, and an HPM path designed to scale from Level 1 to Level 2 without rewrite.   

I would organize this as a staged parallel program with shared baselines and explicit stop conditions.

Phase 0 is setup and measurement hardening. Do this once before branching. The goal is to make every later run comparable.

Step 0.1 is to freeze one canonical baseline. Use the current HPM-enabled configuration that completed all three tiers cleanly and produced the best holder and belief-revision results. Your changelog says that run completed without regressions and set new highs for `what_was_true_rule` and all handoff buckets. That should be the reference point for every experiment branch. 

Step 0.2 is to standardize what every run must log. At minimum, log overall validation, `who_holds_token by handoffs`, `what_was_true_rule`, per-query-type accuracy, holder auxiliary accuracy, episodic read entropy, and HPM diagnostics including gate mean, z-score magnitude, sigma, locked fraction, force-unlocks, and per-slot state summary. Those fields are already described in your HPM integration notes and existing diagnostics.  

Step 0.3 is to define a promotion rubric for experiments. A branch counts as promising only if it clears both capability and stability bars. Capability means it beats baseline on its target metric by a real margin. Stability means no NaN path, no catastrophic controller collapse, and no regression larger than a preset threshold on the already-solved tasks. Your earlier runs make clear why this matters: bigger models could learn faster and still destabilize from controller stress.  

The capability thresholds I would use are these. For targeted branches, demand at least a 10-point absolute gain on the new target metric, or clear transfer to a harder bucket without harming solved buckets by more than 3 points. For architecture branches, require at least one genuinely new qualitative behavior, not just a small aggregate bump. For world-difficulty branches, require maintained performance under increased entity count or horizon, rather than improvement on the old easier evals.

Phase 1 is the parallel branch structure. Run three families at once.

Branch family A is harder worlds with the same architecture.

This branch asks whether the current model still has unused headroom. You should not change HPM, QueryHead, or controller behavior here beyond whatever is already in the stable baseline.

Experiment A1 should raise entity count and distractor load. Increase the active-plus-distractor regime from the current small world toward 6–8 entities. The world generator already supports active and distractor entities and max entity count in `WorldConfig`, so this is a natural continuation rather than a rewrite.  

What to look for in A1 is whether handoff tracking degrades gracefully or collapses sharply once there are more possible owners and more spurious collisions. If performance drops only gradually, the architecture is likely representing entities distinctly. If it collapses suddenly, the representation is probably overcompressing identity.

Promote A1 if `who_holds_token` at 1 and 2 handoffs stays above baseline minus 10 points despite the harder world, and if holder auxiliary accuracy remains high. Kill A1 if handoff metrics collapse while all easier binary queries stay strong, because that pattern would suggest the world has outgrown identity tracking specifically, not global learning capacity.

Experiment A2 should extend horizon length. Move from the current curriculum horizon to 64 and then 128 steps while keeping everything else fixed. Your roadmap already identifies this as one of the main scaling levers. 

What to look for in A2 is whether performance degrades mainly with lag or mainly with overwrite count. If lag is the main failure mode, you need stronger temporal persistence. If overwrite count is the main failure mode, you need better slot separation or retroactive binding.

Promote A2 if long-lag recall and belief revision remain strong and HPM gate statistics do not saturate. Kill A2 if HPM gate mean collapses toward always-open or always-closed, or if sigma drifts badly, because that means the memory controller is not calibrated for the longer timescale.

Experiment A3 should activate truly concurrent multi-chain worlds. The simulator already includes `multi_chain` and delayed secondary chains. Increase the frequency and independence of chain-2 events, and ensure chains can overlap temporally rather than occurring as mostly separate events.  

What to look for in A3 is whether the model can keep separate latent threads alive. The success signature is stable `did_chain2_fire`, preserved ordering queries, and little degradation on holder tracking. The failure signature is “thread blending,” where the model answers a merged average of events rather than one chain or the other.

Promote A3 if multi-chain accuracy stays high and false-cue belief revision remains near perfect. Kill A3 if chain-2 and belief revision both worsen at once, because that pattern would suggest the model is running out of latent partitioning capacity.

Experiment A4 should make `active_rule` genuinely dynamic and consequential. Right now the simulator already has rule-conditioned triggers in `_step_world`, but you should make rules diverge more sharply in downstream consequences, not just trigger conditions. 

What to look for in A4 is whether latent-rule probes remain clean and whether the model starts using rule-state as a real hidden variable rather than a classifier label. Success here would likely precede more abstract reasoning later.

Branch family B is harder question families with the same world and same architecture.

This branch asks whether the current latent state is manipulable, not just decodable.

Experiment B1 should add temporal ordering queries beyond the existing trigger-before-alarm family. Ask whether event X happened before Y, whether the first chain fired before the second, and whether an occlusion began before a handoff. The current diagnostics already include one temporal ordering query, so you can generalize that pattern. 

What to look for is whether the model can compare timestamps implicitly or whether it only knows event presence. If event-presence stays strong but ordering stays weak, that tells you the world model is storing “what happened” without enough sequence structure.

Experiment B2 should add relational queries. Examples are which entity was closest to holder at alarm fire, which entity was visible during correction, or which tagged entity shared color with the trigger source. This directly tests whether stored state supports relational recombination.

What to look for is whether relational errors correlate with object count. If yes, entity binding is the bottleneck. If not, relation formation itself is the bottleneck.

Experiment B3 should add negation-style queries such as which entity was never occluded or which chain never fired. This is deceptively important because negation demands global episode bookkeeping rather than salience detection.

What to look for is whether the model overanswers salient entities. If negation fails by selecting the most memorable entity, then the memory system is still too event-driven and insufficiently exhaustive.

Experiment B4 should add counterfactual queries in a limited, controlled form. Do not begin with open-ended “what would happen.” Start with one intervention at a known event index, such as “who would hold the token if handoff #2 had not occurred?” or “would the alarm have fired if the correction had not happened?” This is the most ambitious query branch and should remain evaluation-focused at first.

What to look for is whether success on counterfactuals appears after relational and temporal queries improve, or whether it emerges early once HPM is strong. That sequencing will tell you whether counterfactual reasoning is built from stable episodic retrieval or from a more general internal simulator.

Promote B-branch experiments if they show improvement without hurting the base recall tasks. Kill a query family if it causes training to overfit to its own supervision and degrade world-modeling signals. The rule here is that query expansion should reveal latent competence, not replace it with answer-head specialization.

Branch family C is architecture change with the same benchmark.

This branch is where the most novel gains could happen, so keep it clean. Change one architecture idea at a time.

Experiment C1 should be HPM Level 2: multi-slot competitive writes with concat readout. Your HPM config already supports multi-slot and competitive gating. This is the cleanest next architecture move because it requires no conceptual rewrite of the module’s role. 

What to look for is slot specialization. A good result is that different slots show different z-spike profiles, different lock dynamics, and different usefulness to different query types. A weak result is that all slots behave identically, meaning extra capacity became redundancy.

Promote C1 if you observe differentiated slot dynamics and improved multi-hop handoff or concurrent-chain performance. Kill C1 if slots collapse to identical behavior or if competitive gating causes sparse starvation, where one slot dominates and the others never meaningfully update.

Experiment C2 should be HPM read-mode ablation. Compare `concat`, `mean`, and `attn` without changing the rest. The config already supports all three. 

What to look for is whether concat helps because it preserves slot identity, or whether attention readout helps because the query can target the right slot dynamically. If concat wins on recall and attn wins on relational questions later, that would be a very useful architectural clue.

Experiment C3 should be retroactive binding. This is the first architecture change that is more than a config flip. At a surprise step, write not only the current hidden state but a short learned mixture over the recent window. Your notes already framed this as Level 3. 

What to look for is improvement specifically on delayed-cause tasks. If retroactive binding helps handoff and false-cue correction equally, that means surprise events are acting as general anchors. If it only helps one class, it may be too specialized.

Promote C3 only if it beats C1 on longer-lag or delayed-cause tasks. Kill it quickly if it introduces instability or if gains can already be obtained by simple multi-slot HPM.

Experiment C4 should be multi-timescale HPM. Give different slots different default update persistence or gain schedules, effectively creating fast and slow memory lanes. Your notes already point toward this as Level 4. 

What to look for is natural specialization by timescale: one slot should track immediate holder changes, another should hold episode-level rule state, another may retain belief-correction context. If that appears, it is one of the strongest architectural signals you could publish.

Phase 2 is resource allocation and run cadence.

Do not run everything at full curriculum immediately. Use a funnel.

For each branch, first do a smoke tier focused on the branch’s target. For world branches, run only the tier where the added difficulty actually matters. For query branches, run evaluation-only first where possible. For HPM branches, run Tier 2-focused training first because that is where holder and multimodal integration are stressed most clearly. Your own notes already recommended Tier 2 first for validating HPM Level 1. 

Then only promote the top 30–40% of branches into a full three-tier run. This matters because some branches will fail quickly and informatively. You do not need to pay the full training cost to learn that a query family is poorly posed or that an HPM mode collapses.

A practical parallel layout would be this.

Group 1, world-only runs: A1 entity scaling, A2 horizon scaling, A3 multi-chain concurrency.

Group 2, query-only runs: B1 temporal ordering expansion, B2 relational queries, B3 negation queries.

Group 3, architecture-only runs: C1 multi-slot competitive HPM, C2 read-mode ablation, C4 multi-timescale HPM.

Keep B4 counterfactual and C3 retroactive binding in reserve until one earlier branch shows the need and the promise for them. Those two are more expensive and more ambiguous if introduced too early.

Phase 3 is decision rules after the first parallel batch.

If world branches succeed and query branches fail, the architecture likely has representational headroom but insufficient manipulation capacity. In that case, prioritize relational and counterfactual queries next, not larger models.

If query branches succeed on the old world but world branches fail on larger worlds, the model is overfit to the benchmark regime. In that case, prioritize entity scaling, horizon scaling, and slot-structured HPM.

If architecture branches improve harder worlds without needing extra supervision, that is your strongest signal that the architecture itself is advancing.

If only bigger task supervision improves results, you are mainly building a better benchmark-trained specialist, not yet uncovering a new mechanism.

Phase 4 is what counts as an emergent capability.

You asked specifically what emergent behaviors to look for. I would treat the following as milestone phenomena rather than mere metric wins.

The first is slot specialization. If different HPM slots become tied to different event classes or temporal roles without explicit supervision, that is a meaningful emergent internal structure. 

The second is parallel causal thread separation. If the model can answer about chain 1 and chain 2 independently in overlapping episodes, you have evidence of nontrivial latent partitioning rather than monolithic salience tracking. 

The third is retroactive cause binding. If the model succeeds on questions whose answer depends on a cause several steps before the surprise point, you are beginning to see event binding rather than simple recency memory.

The fourth is counterfactual replay. If a model can correctly answer a restricted “what if event k did not happen” query, that is a clear step above retrieval.

The fifth is temporal chunking. If performance improves more than expected once events are naturally grouped into motifs or phases, you may be seeing the beginnings of hierarchical internal segmentation.

Phase 5 is notes and guardrails.

Do not change world difficulty and architecture in the same run unless you are in a confirmatory stage. Early on, that destroys attribution.

Do not add many new query families at once. Add one family, because each one teaches you something different about the latent state.

Do not interpret aggregate query accuracy as enough. Your current diagnostics are already better than that because they bucket by handoffs and cue conditions. Keep that philosophy. 

Do not judge HPM branches only by end metrics. Judge them by internal behavior too: gate mean, slot lock fractions, force-unlock spikes, sigma behavior, and whether slot roles differentiate at all. 

Do not scale model size as the next reflex. Your earlier double-parameter run showed real capability gains and real stress costs. Capacity scaling is a later-stage lever once the task family tells you exactly what additional capacity is needed. 



Design these experiments 
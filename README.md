How to actually run it
From the directory containing all three Python files plus the chunked model:

python tmew1_run.py --smoke

If that prints loss numbers and per-query accuracies without traceback, run the full thing:

python tmew1_run.py --epochs 4 --batch-size 8

The first real signal you're looking for is on the Tier 1 validation line. next_step should drop within the first epoch, latent_acc should climb above 0.5 (chance is 0.25 with four rules), and qacc/did_alarm_fire should be the first query type to lift off because it only requires recent state. If qacc/who_was_first_tagged is still at chance after Tier 2, that's your episodic memory failing — and that's the experiment the whole benchmark exists to run.
What's still missing for the full benchmark spec
This is honestly enough to start training and learn something real about the architecture. The remaining items from your spec — false-cue / belief-revision template, multi-entity interaction template, counterfactual queries, longer episode lengths, compositional held-out splits — all slot into the same generator and runner without architectural changes. I'd hold off on building them until you've seen Tier 1–2 learning curves, because those will tell you whether the problem is benchmark difficulty or model capacity. There's no point building Tier 5 if Tier 1 isn't moving.
When you're ready, the next thing I'd build is a held-out compositional eval set — same generator, but with template/delay combinations that never appear in training. That's the test that tells you whether the model is learning temporal abstraction or memorizing episode shapes, and it costs maybe 30 lines on top of what's already here.

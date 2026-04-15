from score_logging import ScoreLogger, log_training_snapshot

logger = ScoreLogger("test")
metrics = {
    "total": 1.032, "next_step": 0.017, "aux_latent": 1.316, "latent_acc": 0.375,
    "q_loss": 0.356, "holder_loss": 0.0, "holder_acc": 0.0, "entity_acc": 0.726,
    "binary_acc": 0.818, "loss/vision": 0.018, "loss/numeric": 0.100, "stress": 0.013,
    "pnn_open": 4.0, "pnn_unlocks": 2.0,
    "hpm_gate_mean": 0.558, "hpm_z_abs_mean": 1.681, "hpm_z_abs_max": 2.161,
    "hpm_err_mean": 1.882, "hpm_write_mag": 0.971, "hpm_open_frac": 1.0,
    "hpm_closing_frac": 0.0, "hpm_locked_frac": 0.0, "hpm_force_unlocks_step": 0.0,
    "hpm_mu": 0.515, "hpm_sigma": 0.843, "hpm_write_regular_frac": 1.0,
    "hpm_write_forced_frac": 0.0,
}
log_training_snapshot(
    logger,
    step_label="t1 ep0 s0030 | pnn=O/O/C/C/O open=4 (+2 unlocked)",
    metrics=metrics,
)

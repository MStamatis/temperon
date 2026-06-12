import json

from eos_switch.report.logging import MilestoneTracker, RunLogger


def test_milestones_first_hit_only():
    mt = MilestoneTracker(targets=(0.5, 0.6))
    mt.update(epoch=0, wall_clock_s=10.0, val_acc=0.55)
    mt.update(epoch=1, wall_clock_s=20.0, val_acc=0.65)
    mt.update(epoch=2, wall_clock_s=30.0, val_acc=0.70)
    flat = mt.as_flat_dict()
    assert flat["hit_0.50_epoch"] == 0 and flat["hit_0.50_s"] == 10.0
    assert flat["hit_0.60_epoch"] == 1 and flat["hit_0.60_s"] == 20.0


def test_milestones_unreached_are_none():
    mt = MilestoneTracker(targets=(0.5,))
    mt.update(epoch=0, wall_clock_s=1.0, val_acc=0.4)
    assert mt.as_flat_dict()["hit_0.50_epoch"] is None


def test_run_logger_files(tmp_path):
    logger = RunLogger(tmp_path / "run")
    logger.log_step({"step": 0, "loss": 1.0})
    logger.log_step({"step": 1, "loss": 0.9})
    logger.log_probe({"step": 0, "lambda_max": 2.0})
    logger.log_epoch({"epoch": 0, "train_loss": 1.0, "val_acc": 0.5, "epoch_time_s": 1.0})
    logger.write_meta({"config": {"x": 1}})
    logger.finalize({"final_val_acc": 0.5})

    out = tmp_path / "run"
    steps = [json.loads(line) for line in (out / "steps.jsonl").read_text().splitlines()]
    assert len(steps) == 2 and steps[1]["loss"] == 0.9
    assert (out / "probes.jsonl").exists()
    assert (out / "epochs.csv").exists()
    assert (out / "summary.csv").exists()
    assert json.loads((out / "meta.json").read_text())["config"] == {"x": 1}

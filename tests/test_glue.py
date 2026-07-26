"""Phase 8 GLUE pieces: metrics, task table, SAM gate alignment to the decay."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "experiments"))

from glue.train_glue import TASKS, compute_metric  # noqa: E402
from lm.train_lm import sam_start_step, wsd_mult  # noqa: E402


def test_task_table_shapes():
    assert set(TASKS) == {"rte", "mrpc", "stsb", "cola"}
    assert TASKS["stsb"]["n_labels"] == 1          # regression
    assert TASKS["cola"]["keys"][1] is None        # single-sentence task
    for spec in TASKS.values():
        assert spec["metric"] in {"acc", "acc_f1", "pearson", "mcc"}


def test_metric_accuracy_and_f1():
    preds = np.array([1, 1, 0, 0])
    labels = np.array([1, 0, 0, 0])
    m = compute_metric("acc", preds, labels)
    assert m["acc"] == pytest.approx(0.75) and m["score"] == pytest.approx(0.75)
    m2 = compute_metric("acc_f1", preds, labels)
    assert m2["f1"] == pytest.approx(2 / 3)
    assert m2["score"] == pytest.approx((0.75 + 2 / 3) / 2)  # GLUE MRPC avg


def test_metric_mcc_and_pearson():
    perfect = np.array([0, 1, 0, 1])
    assert compute_metric("mcc", perfect, perfect)["mcc"] == pytest.approx(1.0)
    x = np.array([1.0, 2.0, 3.0, 4.0])
    m = compute_metric("pearson", x, 2 * x + 1)
    assert m["pearson"] == pytest.approx(1.0)
    assert m["score"] == m["pearson"]


def test_sam_tail_starts_exactly_at_decay():
    """The tail arm must own the whole anneal: gate step == decay start."""
    total, decay_frac = 1000, 0.3
    start = sam_start_step("tail", 1.0 - decay_frac, total)
    assert start == 700
    # lr is still at the stable plateau one step before, and decaying after.
    warmup = 60
    assert wsd_mult(start - 1, total, warmup, decay_frac) == 1.0
    assert wsd_mult(start, total, warmup, decay_frac) == pytest.approx(1.0)
    assert wsd_mult(start + 150, total, warmup, decay_frac) == pytest.approx(0.5)
    assert wsd_mult(total, total, warmup, decay_frac) == pytest.approx(0.0)


def test_arm_off_never_gates_on():
    total = 1000
    start = sam_start_step("off", 0.7, total)
    assert all(s < start for s in range(total))

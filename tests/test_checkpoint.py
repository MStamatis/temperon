"""Checkpoint/resume: atomic single-file save + full-state round-trip."""

import torch
import torch.nn as nn

from eos_switch.controllers import build_controller
from eos_switch.report.logging import MilestoneTracker
from eos_switch.train.loop import _load_checkpoint, _real_opt, _save_checkpoint


def _ctrl(model):
    c = build_controller({"type": "cyclic_catapult", "base_optimizer": "sgd_momentum",
                          "lr": 0.1, "n_cycles": 4, "t_mult": 2.0})
    c.setup(model, steps_per_epoch=10, total_epochs=8)
    return c


class _FakeLogger:
    _epoch_rows: list = []


def test_milestone_state_roundtrip():
    m = MilestoneTracker()
    m.update(3, 120.0, 0.66)
    m2 = MilestoneTracker()
    m2.load_state_dict(m.state_dict())
    assert m2.hits[0.65]["epoch"] == 3
    assert m2.as_flat_dict() == m.as_flat_dict()


def test_checkpoint_atomic_and_full_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = nn.Linear(8, 4)
    ctrl = _ctrl(model)
    opt = _real_opt(ctrl)
    ctrl.begin_step(5)
    model(torch.randn(4, 8)).sum().backward()
    opt.step()                                  # populate momentum state
    ctrl._last_cycle, ctrl._loss_ema = 2, 1.23
    mile = MilestoneTracker(); mile.update(3, 100.0, 0.66)
    gen = torch.Generator().manual_seed(7)

    p = tmp_path / "checkpoint.pt"
    _save_checkpoint(p, next_epoch=4, global_step=37, best_val_acc=0.66, model=model, opt=opt,
                     controller=ctrl, milestones=mile, epoch_rows=[{"epoch": 0}, {"epoch": 1}],
                     gen=gen, is_cuda=False)
    assert p.exists() and not (tmp_path / "checkpoint.pt.tmp").exists()  # atomic, single file

    model2 = nn.Linear(8, 4)
    ctrl2 = _ctrl(model2)
    opt2 = _real_opt(ctrl2)
    mile2, gen2, logger = MilestoneTracker(), torch.Generator(), _FakeLogger()
    ne, gs, bva = _load_checkpoint(p, model=model2, opt=opt2, controller=ctrl2,
                                   milestones=mile2, logger=logger, gen=gen2, is_cuda=False)
    assert (ne, gs, bva) == (4, 37, 0.66)
    assert ctrl2._last_cycle == 2 and abs(ctrl2._loss_ema - 1.23) < 1e-9
    assert mile2.hits[0.65]["epoch"] == 3
    assert logger._epoch_rows == [{"epoch": 0}, {"epoch": 1}]
    assert torch.allclose(model2.weight, model.weight)         # weights restored
    assert gen2.get_state().equal(gen.get_state())             # RNG restored
    # optimizer momentum restored
    assert opt2.state_dict()["state"].keys() == opt.state_dict()["state"].keys()

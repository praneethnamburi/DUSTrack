"""The producer-side-H2D patch that fixes DLC 3's async-inference leak.

The patch itself needs DLC + a GPU to exercise end to end; what is unit
testable (and what would silently break on a DLC refactor) is the
wrapper's behaviour: it must move only the batch tensor of a
``(batch, model_kwargs)`` pair, leave everything else untouched, and
always delegate to the original ``_safe_put``.
"""

import pytest

from dustrack import dlcpatch

torch = pytest.importorskip("torch")


class _FakeRunner:
    """Stands in for ``InferenceRunner``: just a device + a recorder."""

    def __init__(self, device="cpu"):
        self.device = device
        self.seen = []

    def _orig_safe_put(self, item):
        self.seen.append(item)
        return True


class _RecordingTensor:
    """Tracks whether ``.to`` was called and with what."""

    def __init__(self):
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


@pytest.fixture
def patched_put():
    return dlcpatch._make_device_safe_put(_FakeRunner._orig_safe_put)


def test_batch_is_moved_to_device_before_queueing(patched_put, monkeypatch):
    runner = _FakeRunner(device="cuda:0")
    batch, kwargs = _RecordingTensor(), {"k": 1}
    monkeypatch.setattr(torch, "is_tensor", lambda obj: obj is batch)

    patched_put(runner, (batch, kwargs))

    assert batch.moved_to == "cuda:0", "batch must be moved on the producer thread"
    queued_batch, queued_kwargs = runner.seen[0]
    assert queued_batch is batch
    assert queued_kwargs is kwargs, "model_kwargs must pass through untouched"


def test_cpu_device_is_left_alone(patched_put, monkeypatch):
    """No device copy to make -- and no reason to touch the tensor."""
    runner = _FakeRunner(device="cpu")
    batch = _RecordingTensor()
    monkeypatch.setattr(torch, "is_tensor", lambda obj: obj is batch)

    patched_put(runner, (batch, {}))

    assert batch.moved_to is None
    assert runner.seen == [(batch, {})]


@pytest.mark.parametrize(
    "item",
    [None, "sentinel", (1, 2, 3)],
    ids=["none-sentinel", "non-tuple", "wrong-arity"],
)
def test_non_batch_items_pass_through_unchanged(patched_put, item):
    """``_safe_put(None)`` is the queue sentinel -- mangling it would hang
    the consumer."""
    runner = _FakeRunner(device="cuda:0")

    patched_put(runner, item)

    assert runner.seen == [item]


def test_patch_is_idempotent(monkeypatch):
    """Double-patching would nest the wrapper and move the batch twice."""
    monkeypatch.setattr(dlcpatch, "_H2D_PATCHED", True)
    assert dlcpatch.patch_dlc_async_h2d(verbose=False) is True


def test_patch_opts_out_via_env(monkeypatch):
    monkeypatch.setattr(dlcpatch, "_H2D_PATCHED", False)
    monkeypatch.setenv("DUSTRACK_DISABLE_DLC_H2D_PATCH", "1")
    assert dlcpatch.patch_dlc_async_h2d(verbose=False) is False

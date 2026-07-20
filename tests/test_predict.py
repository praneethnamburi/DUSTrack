"""Tests for :mod:`dustrack.predict` -- range-restricted DLC inference.

Split in two:

* **No-DLC tests** (the bulk) exercise everything this module actually
  *adds* -- frame-list iteration, bounds validation, cancellation,
  chunking, and the DataFrame/annotation conversion -- against a fake
  base iterator and a fake runner. They need no GPU, no model and no
  torch, so they run on the standalone CI matrix.
* **A parity test**, skipped unless pointed at a real project, pins the
  claim that makes this module trustworthy: predictions for frames
  [a, b] must match the ``analyze_videos`` h5 rows for the same frames.
  Set ``DUSTRACK_PARITY_PROJECT`` (a config.yaml) and
  ``DUSTRACK_PARITY_VIDEO`` to run it.
"""

from __future__ import annotations

import os
import threading

import numpy as np
import pandas as pd
import pytest

from dustrack import predict as dpredict
from dustrack.predict import (
    SCORER,
    PredictionCancelled,
    RangePredictor,
    make_frame_list_iterator_class,
)


# --------------------------------------------------------------------- #
# Fakes                                                                 #
# --------------------------------------------------------------------- #
class FakeVideoIterator:
    """Stands in for DLC's ``VideoIterator``.

    Reproduces the surface :class:`FrameListIterator` builds on: a
    cursor-based ``read_frame``, ``set_to_frame``, ``_n_frames``,
    ``_crop`` / ``_context``. Each "frame" is a 1x1x3 array whose value
    is the frame index, so a test can assert *which* frames were read.
    """

    def __init__(self, video_path, cropping=None, n_frames=1000):
        self.video_path = str(video_path)
        self._n_frames = n_frames
        self._crop = cropping is not None
        self._context = None
        self._cursor = 0
        self.reads: list[int] = []
        self.seeks: list[int] = []
        self.closed = False

    def set_to_frame(self, ind):
        # Mirrors the patched dnav reader: clamp rather than raise. The
        # point of FrameListIterator's own validation is that this
        # clamping must never be reached with a bad index.
        self.seeks.append(int(ind))
        self._cursor = min(int(ind), self._n_frames - 1)

    def read_frame(self, shrink=1, crop=False):
        if self._cursor >= self._n_frames:
            return None
        self.reads.append(self._cursor)
        frame = np.full((1, 1, 3), self._cursor, dtype=np.uint8)
        self._cursor += 1
        return frame

    def close(self):
        self.closed = True


class FakeRunner:
    """Stands in for ``PoseInferenceRunner``.

    Consumes the iterable and emits one prediction per frame, encoding
    the frame index into the x coordinate so ordering is checkable.
    """

    def __init__(self, n_bodyparts=2):
        self.n_bodyparts = n_bodyparts
        self.calls = 0

    def inference(self, images, shelf_writer=None):
        self.calls += 1
        out = []
        for frame in images:
            idx = float(frame[0, 0, 0])
            bodyparts = np.array(
                [[[idx, idx + 100.0, 0.9] for _ in range(self.n_bodyparts)]],
                dtype=float,
            )
            out.append({"bodyparts": bodyparts})
        return out


def make_iterator(n_frames=1000, frames=(), cropping=None):
    cls = make_frame_list_iterator_class(base=FakeVideoIterator)
    # Construct empty, size the fake video, *then* set frames -- so the
    # bounds validation in set_frames sees the intended length.
    it = cls("fake.mp4", frames=(), cropping=cropping)
    it._n_frames = n_frames
    return it.set_frames(frames)


def make_predictor(n_frames=1000, n_bodyparts=2, runner=None):
    """A predictor with the model + reader layers stubbed out."""
    p = RangePredictor("fake/config.yaml")
    p._runner = runner or FakeRunner(n_bodyparts=n_bodyparts)
    p._bodyparts = [f"point{i}" for i in range(n_bodyparts)]
    it = make_iterator(n_frames=n_frames)
    p._iterators["fake.mp4"] = it
    # No real file behind "fake.mp4"; skip the utils.Video open so the
    # conversion path is exercised without a video asset.
    p._videos["fake.mp4"] = None
    return p, it


# --------------------------------------------------------------------- #
# Standalone-import invariant                                           #
# --------------------------------------------------------------------- #
def test_no_module_level_dlc_or_torch_import():
    """``import dustrack.predict`` must not require the torch stack.

    Same invariant 1.3.1 restored for ``dustrack`` as a whole: every DLC
    import here is deferred into a function body. Asserted by walking the
    module's own top-level statements, which holds regardless of whether
    the running env happens to have deeplabcut installed.

    Deliberately *not* done with ``importlib.reload``: reloading a module
    under test rebinds its classes, so anything holding a reference to
    the pre-reload ``PredictionCancelled`` (this test module, at import
    time) then compares against a stale object. That made
    ``test_predict_frames_cancel_before_start_raises`` fail on CI under
    random test ordering while passing locally in file order.
    """
    import ast
    import pathlib

    src = pathlib.Path(dpredict.__file__).read_text(encoding="utf-8")
    banned = ("deeplabcut", "torch")
    offenders = []
    for node in ast.parse(src).body:  # top level only
        if isinstance(node, ast.Import):
            offenders += [
                a.name for a in node.names if a.name.split(".")[0] in banned
            ]
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in banned:
                offenders.append(node.module)
    assert not offenders, f"module-level heavy imports: {offenders}"


def test_constructing_predictor_does_not_load_model():
    """The model is built on first use, not in ``__init__``."""
    p = RangePredictor("nonexistent/config.yaml")
    assert p._runner is None
    assert "not loaded" in repr(p)


# --------------------------------------------------------------------- #
# FrameListIterator                                                     #
# --------------------------------------------------------------------- #
def test_iterator_yields_exactly_the_requested_frames():
    it = make_iterator(frames=[10, 20, 30])
    frames = [int(f[0, 0, 0]) for f in it]
    assert frames == [10, 20, 30]


def test_iterator_preserves_arbitrary_order():
    """Frames come back in the order asked for, not sorted."""
    it = make_iterator(frames=[30, 10, 20])
    assert [int(f[0, 0, 0]) for f in it] == [30, 10, 20]


def test_iterator_handles_duplicates():
    it = make_iterator(frames=[5, 5, 7])
    assert [int(f[0, 0, 0]) for f in it] == [5, 5, 7]


def test_iterator_rejects_out_of_range_high():
    """Out-of-range must raise, never silently clamp.

    The patched ``set_to_frame`` clamps to the last frame with a
    warning; letting that happen would return a prediction for a
    different frame than the one requested.
    """
    with pytest.raises(IndexError, match=r"video has 100 frames"):
        make_iterator(n_frames=100, frames=[99, 100])


def test_iterator_rejects_negative_frame():
    with pytest.raises(IndexError):
        make_iterator(n_frames=100, frames=[-1, 0])


def test_iterator_accepts_last_frame():
    """The upper bound is inclusive of n_frames - 1."""
    it = make_iterator(n_frames=100, frames=[99])
    assert [int(f[0, 0, 0]) for f in it] == [99]


def test_iterator_is_reusable_across_passes():
    """``set_frames`` re-points one open reader -- no re-open per call."""
    it = make_iterator(frames=[1, 2])
    assert [int(f[0, 0, 0]) for f in it] == [1, 2]
    it.set_frames([7, 8, 9])
    assert [int(f[0, 0, 0]) for f in it] == [7, 8, 9]


def test_iterator_restarts_on_reiteration():
    it = make_iterator(frames=[3, 4])
    assert [int(f[0, 0, 0]) for f in it] == [3, 4]
    assert [int(f[0, 0, 0]) for f in it] == [3, 4]


def test_iterator_empty_frame_list():
    it = make_iterator(frames=[])
    assert list(it) == []


def test_iterator_cancel_stops_iteration():
    evt = threading.Event()
    it = make_iterator(frames=list(range(100)))
    it.set_cancel_event(evt)
    collected = []
    for frame in it:
        collected.append(int(frame[0, 0, 0]))
        if len(collected) == 3:
            evt.set()
    assert collected == [0, 1, 2]
    assert it.n_consumed == 3


def test_contiguous_range_seeks_only_once():
    """A contiguous range must decode sequentially after one seek.

    Seeking per frame makes PyAV re-decode from the preceding keyframe
    every time. Measured on the interosseous s001 model, that was 56 fps
    vs 165 fps sequential -- a ~3x regression hiding behind identical
    output, which is exactly the kind of thing only a test will hold.
    """
    it = make_iterator(frames=list(range(100, 120)))
    frames = [int(f[0, 0, 0]) for f in it]
    assert frames == list(range(100, 120))
    assert it.seeks == [100]


def test_non_contiguous_range_seeks_per_jump():
    """Only genuine discontinuities pay for a seek."""
    it = make_iterator(frames=[10, 11, 12, 500, 501])
    list(it)
    assert it.seeks == [10, 500]


def test_reiteration_reseeks():
    """A second pass must not assume the reader is still positioned."""
    it = make_iterator(frames=[10, 11])
    list(it)
    list(it)
    assert it.seeks == [10, 10]


def test_backward_range_seeks_every_frame():
    """Descending order is honoured even though it defeats the fast path."""
    it = make_iterator(frames=[30, 29, 28])
    assert [int(f[0, 0, 0]) for f in it] == [30, 29, 28]
    assert it.seeks == [30, 29, 28]


def test_iterator_reports_frames_and_consumed():
    it = make_iterator(frames=[2, 4, 6])
    assert it.frames == [2, 4, 6]
    assert it.n_consumed == 0
    next(iter(it))
    assert it.n_consumed == 1


# --------------------------------------------------------------------- #
# predict_frames                                                        #
# --------------------------------------------------------------------- #
def test_predict_frames_indexes_by_requested_frames():
    """The DataFrame index must be real frame numbers, not 0..n-1.

    This is the difference that makes results mergeable into an existing
    trace; a positional index would silently mis-attribute every point.
    """
    p, _ = make_predictor()
    df = p.predict_frames("fake.mp4", [100, 101, 102])
    assert list(df.index) == [100, 101, 102]
    assert df.index.name == "frame"


def test_predict_frames_column_layout_matches_dlc():
    p, _ = make_predictor(n_bodyparts=3)
    df = p.predict_frames("fake.mp4", [0, 1])
    assert df.columns.names == ["scorer", "bodyparts", "coords"]
    assert list(df.columns.levels[0]) == [SCORER]
    assert list(df.columns.levels[2]) == ["likelihood", "x", "y"]
    assert df.shape == (2, 3 * 3)


def test_predict_frames_values_track_frames():
    """Each row's coordinates come from the frame actually decoded."""
    p, _ = make_predictor()
    df = p.predict_frames("fake.mp4", [7, 3, 9])
    xs = df.loc[:, (SCORER, "point0", "x")].tolist()
    assert xs == [7.0, 3.0, 9.0]


def test_predict_frames_empty_list_returns_empty_frame():
    p, _ = make_predictor()
    df = p.predict_frames("fake.mp4", [])
    assert len(df) == 0
    assert df.columns.names == ["scorer", "bodyparts", "coords"]


def test_predict_frames_out_of_range_raises_before_inference():
    """A bad index fails before any GPU work happens."""
    runner = FakeRunner()
    p, _ = make_predictor(n_frames=50, runner=runner)
    with pytest.raises(IndexError):
        p.predict_frames("fake.mp4", [10, 999])
    assert runner.calls == 0


def test_predict_frames_chunks_the_request():
    """Chunking is what makes progress + cancellation observable."""
    runner = FakeRunner()
    p, _ = make_predictor(runner=runner)
    df = p.predict_frames("fake.mp4", list(range(20)), chunk_size=5)
    assert runner.calls == 4
    assert len(df) == 20
    assert list(df.index) == list(range(20))


def test_predict_frames_progress_callback():
    seen: list[tuple[int, int]] = []
    p, _ = make_predictor()
    p.predict_frames(
        "fake.mp4",
        list(range(20)),
        chunk_size=5,
        progress_callback=lambda done, total: seen.append((done, total)),
    )
    assert seen == [(5, 20), (10, 20), (15, 20), (20, 20)]


def test_predict_frames_cancel_returns_partial():
    """A mid-run cancel keeps the frames already finished."""
    evt = threading.Event()
    p, _ = make_predictor()

    def on_progress(done, total):
        if done >= 10:
            evt.set()

    df = p.predict_frames(
        "fake.mp4",
        list(range(100)),
        chunk_size=5,
        progress_callback=on_progress,
        cancel_event=evt,
    )
    assert len(df) == 10
    assert list(df.index) == list(range(10))


def test_predict_frames_cancel_before_start_raises():
    """Cancelled with nothing to return is an error, not an empty frame."""
    evt = threading.Event()
    evt.set()
    p, _ = make_predictor()
    with pytest.raises(PredictionCancelled):
        p.predict_frames("fake.mp4", list(range(10)), cancel_event=evt)


def test_predict_frames_clears_cancel_event_after_use():
    """The iterator must not stay armed for the next caller."""
    evt = threading.Event()
    p, it = make_predictor()
    p.predict_frames("fake.mp4", [1, 2], cancel_event=evt)
    assert it._cancel_evt is None


# --------------------------------------------------------------------- #
# predict_range                                                         #
# --------------------------------------------------------------------- #
def test_predict_range_is_inclusive_of_both_ends():
    """Matches ``lucas_kanade_rstc``'s convention."""
    p, _ = make_predictor()
    df = p.predict_range("fake.mp4", 10, 14, annotation=False)
    assert list(df.index) == [10, 11, 12, 13, 14]


def test_predict_range_clamps_end_to_video():
    p, _ = make_predictor(n_frames=50)
    df = p.predict_range("fake.mp4", 45, 999, annotation=False)
    assert list(df.index) == [45, 46, 47, 48, 49]


def test_predict_range_step():
    p, _ = make_predictor()
    df = p.predict_range("fake.mp4", 0, 10, step=5, annotation=False)
    assert list(df.index) == [0, 5, 10]


def test_predict_range_rejects_negative_start():
    p, _ = make_predictor()
    with pytest.raises(ValueError, match="start must be >= 0"):
        p.predict_range("fake.mp4", -1, 10)


def test_predict_range_rejects_inverted_range():
    p, _ = make_predictor()
    with pytest.raises(ValueError, match="empty range"):
        p.predict_range("fake.mp4", 20, 10)


def test_predict_range_returns_annotation_by_default():
    p, _ = make_predictor()
    ann = p.predict_range("fake.mp4", 10, 12)
    assert hasattr(ann, "data")
    assert set(ann.data["0"].keys()) == {10, 11, 12}


# --------------------------------------------------------------------- #
# Conversion to VideoAnnotation                                         #
# --------------------------------------------------------------------- #
def test_to_annotation_is_sparse():
    """Only requested frames appear -- the interpolate_blips contract."""
    p, _ = make_predictor()
    ann = p.predict_range("fake.mp4", 100, 102)
    for label in ann.data:
        assert set(ann.data[label].keys()) == {100, 101, 102}


def test_to_annotation_strips_point_prefix_like_the_h5_path():
    """Label naming must match ``_dlc_trace_to_annotation_dict``.

    Bodyparts named ``point0``/``point1`` become labels ``0``/``1`` --
    reusing DUSTrack's own mapper is what guarantees a range-predicted
    layer lines up with an analyze_videos one.
    """
    p, _ = make_predictor(n_bodyparts=2)
    ann = p.predict_range("fake.mp4", 0, 1)
    assert set(ann.data.keys()) == {"0", "1"}


def test_to_annotation_coordinates_drop_likelihood():
    """Annotation values are [x, y]; likelihood stays in the DataFrame."""
    p, _ = make_predictor()
    ann = p.predict_range("fake.mp4", 42, 42)
    assert ann.data["0"][42] == [42.0, 142.0]


def test_to_annotation_inherits_source_labels():
    """Labels the model doesn't predict still appear, empty."""

    class FakeSource:
        labels = ["0", "1", "99"]
        video = None

    p, _ = make_predictor(n_bodyparts=2)
    df = p.predict_frames("fake.mp4", [1, 2])
    ann = p.to_annotation(df, "fake.mp4", source_annotation=FakeSource())
    assert ann.data["99"] == {}
    assert set(ann.data["0"].keys()) == {1, 2}


def test_to_annotation_shares_the_source_reader():
    """The GUI path must not open the video a second time."""
    class FakeVideo:
        """Minimal reader stand-in (VideoAnnotation probes ``len``)."""

        def __len__(self):
            return 1000

    sentinel = FakeVideo()

    class FakeSource:
        labels = ["0", "1"]
        video = sentinel

    p, _ = make_predictor(n_bodyparts=2)
    df = p.predict_frames("fake.mp4", [1, 2])
    ann = p.to_annotation(df, "fake.mp4", source_annotation=FakeSource())
    assert ann.video is sentinel


def test_to_annotation_does_not_write_to_disk(tmp_path):
    """Nothing lands on disk -- no contention with the h5 folder."""
    p, _ = make_predictor()
    before = set(os.listdir(tmp_path))
    p.predict_range("fake.mp4", 0, 5)
    assert set(os.listdir(tmp_path)) == before


# --------------------------------------------------------------------- #
# Lifecycle                                                             #
# --------------------------------------------------------------------- #
def test_close_releases_readers():
    p, it = make_predictor()
    p.close()
    assert it.closed
    assert p._iterators == {}
    assert p._runner is None


def test_context_manager_closes():
    p, it = make_predictor()
    with p:
        pass
    assert it.closed


def test_reader_is_cached_across_calls():
    """Repeated ranges on one video must not re-open it."""
    p, it = make_predictor()
    p.predict_range("fake.mp4", 0, 2)
    p.predict_range("fake.mp4", 10, 12)
    assert p._iterators["fake.mp4"] is it


# --------------------------------------------------------------------- #
# Parity with analyze_videos (needs a real project + GPU)               #
# --------------------------------------------------------------------- #
_PARITY_PROJECT = os.environ.get("DUSTRACK_PARITY_PROJECT")
_PARITY_VIDEO = os.environ.get("DUSTRACK_PARITY_VIDEO")
_PARITY_H5 = os.environ.get("DUSTRACK_PARITY_H5")


#: Parity tolerance, in pixels. Agreement is *not* bit-exact and should
#: not be asserted as such: ``dlcpatch`` forces AMP autocast on, and a
#: chunked frame list composes batches differently from a whole-video
#: pass, so float reduction order differs. Measured drift on the
#: interosseous s001 model was max 0.0036 px over 50 frames (~1.3e-05
#: relative). This bound sits an order of magnitude above that and still
#: two orders below any real defect: an off-by-one frame, a wrong
#: snapshot, or missing preprocessing all move points by whole pixels or
#: more, not hundredths.
PARITY_ATOL_PX = 0.05


@pytest.mark.skipif(
    not (_PARITY_PROJECT and _PARITY_VIDEO and _PARITY_H5),
    reason=(
        "set DUSTRACK_PARITY_PROJECT / _VIDEO / _H5 to run the "
        "analyze_videos parity check"
    ),
)
def test_range_prediction_matches_analyze_videos():
    """The load-bearing claim: same frames in, same coordinates out.

    Compares a range prediction against the h5 an ``analyze_videos`` run
    already wrote for that video. Any drift in preprocessing, decode or
    snapshot resolution shows up here as a coordinate mismatch well
    beyond :data:`PARITY_ATOL_PX`.
    """
    start, end = 100, 149
    reference = pd.read_hdf(_PARITY_H5)
    if hasattr(reference, "to_frame"):
        reference = pd.DataFrame(reference)

    with RangePredictor(_PARITY_PROJECT) as p:
        got = p.predict_frames(_PARITY_VIDEO, list(range(start, end + 1)))

    ref = reference.loc[start:end]
    assert len(got) == len(ref)

    ref_scorer = ref.columns.levels[0][0]
    for bodypart in p.bodyparts:
        for coord in ("x", "y"):
            a = got.loc[:, (SCORER, bodypart, coord)].to_numpy()
            b = ref.loc[:, (ref_scorer, bodypart, coord)].to_numpy()
            np.testing.assert_allclose(
                a,
                b,
                rtol=0,
                atol=PARITY_ATOL_PX,
                err_msg=f"{bodypart}/{coord} drift",
            )

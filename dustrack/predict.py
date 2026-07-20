"""Range-restricted DLC inference with a cached model.

``DLCProject.analyze_videos`` is whole-video only: its sole subsetting
axis is *which video files*, never *which frames*. It also rebuilds the
pose model on every call (and, via ``process()`` / ``train_iteration()``,
runs a full ``evaluate()`` over every checkpoint first), so the cost of
asking "what does the model predict right here?" is a multi-minute
whole-video pass. That is the wrong shape for the incremental refine
loop, where the useful question is about the few hundred frames the user
is actually looking at.

This module provides the other shape: **a persistent predictor that holds
a loaded model and runs inference on an arbitrary frame list.** Measured
on the interosseous s001 model (ResNet-50 BU, 706x558, RTX 4090,
``batchsize=4``): a one-time ~4.5 s model load, then **~165 fps** -- so a
200-frame neighbourhood comes back in **~1.2 s**. A range prediction is
an interactive operation rather than a coffee break.

Design -- deliberately thin
---------------------------
The one thing this module must not do is reimplement DLC's preprocessing.
Hand-fed tensors can silently drift from the training-time transforms and
produce predictions that quietly disagree with ``analyze_videos``. So the
only thing that changes is **which frames get yielded**:

* :class:`FrameListIterator` subclasses DLC's own ``VideoIterator`` and
  overrides iteration to walk a caller-supplied frame list. Everything
  below it -- decode, crop, preprocessing, batching, the model forward
  pass -- is DLC's, unmodified, reached through the same
  ``PoseInferenceRunner`` that ``analyze_videos`` builds.
* ``VideoIterator`` inherits from ``auxfun_videos.VideoReader``, so
  :func:`dustrack.dlcpatch.patch_dlc_decoder`'s dnav PyAV+TOC reader
  applies here too. Random frame access is a TOC seek, not a linear scan
  -- which is what makes an arbitrary frame list affordable at all.

A seek is still far from free (PyAV decodes forward from the preceding
keyframe), so the iterator seeks only on a genuine discontinuity: a
contiguous ascending range pays one seek and then reads sequentially.
That single detail is worth ~3x -- 56 fps seeking per frame vs 165 fps
sequential -- and ``test_contiguous_range_seeks_only_once`` guards it,
since the outputs are identical either way.

Parity with the whole-video path is therefore *structural* rather than
hoped-for, and ``tests/test_predict.py`` pins it: predictions for frames
[a, b] must equal the ``analyze_videos`` h5 rows for those same frames.

Results come back **in memory** as a sparse :class:`VideoAnnotation`,
mirroring :func:`dustrack.blip.interpolate_blips`. Nothing is written to
``videos/iteration-{N}/``, so a range prediction cannot collide with the
h5 files that ``_refresh_dlc_layers`` globs. It also never touches the
project ``config.yaml``: ``analyze_videos`` mutates ``snapshotindex`` and
restores it afterwards, which is not re-entrant and would corrupt model
selection under any concurrency. Resolving the snapshot once, up front,
sidesteps that entirely.

Nothing here touches Qt. GUI integration is a separate layer; this module
is usable from a plain console session.

Example:
    >>> from dustrack.predict import RangePredictor
    >>> p = RangePredictor("path/to/project/config.yaml")
    >>> ann = p.predict_range("video.mp4", 4200, 4400)   # sparse layer
    >>> ann.save("video_annotations_range.json")
    >>> p.close()
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from dustrack import dlcloader as _dlcloader
from dustrack.annotations import VideoAnnotation

__all__ = [
    "RangePredictor",
    "PredictionCancelled",
    "make_frame_list_iterator_class",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CHUNK_BATCHES",
]

#: Matches the default in ``DLCProject.analyze_videos``: DLC's PyTorch
#: backend defaults to 1, which leaves an RTX-class GPU idle. The
#: throughput knee measured 2026-05-20 (ResNet-50 BU, 706x558, DLC
#: 3.0.0rc14, RTX 4090) is ~4.
DEFAULT_BATCH_SIZE = 4

#: Frames are pushed through the runner in chunks of
#: ``DEFAULT_CHUNK_BATCHES * batch_size`` so progress can be reported and
#: cancellation observed between chunks -- the same "poll between units"
#: shape as the batch modal's ``QThread`` worker. The chunk is the
#: preemption unit, which is what lets a foreground request jump ahead of
#: a background sweep.
DEFAULT_CHUNK_BATCHES = 8

#: The scorer level of the returned DataFrame's column MultiIndex.
#: ``VideoAnnotation._dlc_trace_to_annotation_dict`` reads this level
#: positionally (``levels[0].values[0]``), so the value is cosmetic --
#: but it makes a range-predicted frame distinguishable from an
#: ``analyze_videos`` one when inspecting a DataFrame by eye.
SCORER = "DUSTrackRangePredict"


class PredictionCancelled(RuntimeError):
    """Raised when a prediction was cancelled before any frame completed.

    A cancellation that lands *after* some frames are done is not an
    error -- those frames are returned. This is raised only when there is
    nothing to return.
    """


def make_frame_list_iterator_class(base=None):
    """Build the :class:`FrameListIterator` class.

    Constructed lazily because it subclasses a DeepLabCut type, and
    ``import dustrack`` must work on a standalone install with no torch
    stack (the invariant restored in 1.3.1).

    Args:
        base: Optional base class to subclass instead of DLC's
            ``VideoIterator``. Only for tests -- it lets the frame-list
            logic (bounds validation, ordering, cancellation), which is
            the whole of what this module adds, be exercised without a
            GPU, a model, or the torch stack.
    """
    if base is None:
        _dlcloader._ensure_dlc_loaded()
        from deeplabcut.pose_estimation_pytorch.apis.videos import (  # noqa: F401
            VideoIterator,
        )
    else:
        VideoIterator = base

    class FrameListIterator(VideoIterator):
        """A ``VideoIterator`` that yields an explicit list of frames.

        DLC's ``VideoIterator.__next__`` reads sequentially off an
        internal cursor. This yields ``frames`` in the given order
        instead, seeking per frame. Everything else -- crop handling,
        the RGB contract, the reader itself -- is inherited.

        The frame list is validated against the video length at
        assignment: the patched ``set_to_frame`` *clamps* an
        out-of-range index to the last frame (with a warning), which
        would silently return a prediction for the wrong frame. An
        explicit error is the only safe reading of that request.

        One instance is reusable across calls via :meth:`set_frames`,
        which matters because construction opens the video and builds a
        PyAV TOC -- a cost the interactive loop should pay once per
        video, not once per request.
        """

        def __init__(self, video_path, frames=(), cropping=None):
            super().__init__(str(video_path), cropping=cropping)
            self._frames: list[int] = []
            self._pos = 0
            self._next_expected = -1  # frame the reader is positioned at
            self._cancel_evt: threading.Event | None = None
            self.set_frames(frames)

        def set_frames(self, frames: Sequence[int]) -> "FrameListIterator":
            """Point the iterator at a new frame list. Validates bounds."""
            frames = [int(f) for f in frames]
            if frames:
                n = int(self._n_frames)
                lo, hi = min(frames), max(frames)
                if lo < 0 or hi >= n:
                    raise IndexError(
                        f"frame list spans [{lo}, {hi}] but the video has "
                        f"{n} frames (valid: [0, {n - 1}])"
                    )
            self._frames = frames
            self._pos = 0
            self._next_expected = -1
            return self

        def set_cancel_event(self, evt: threading.Event | None) -> None:
            """Set the cooperative-cancellation flag, polled per frame.

            Checked in ``__next__``, so a cancel takes effect within one
            frame's decode rather than at the end of the chunk. Stopping
            iteration makes the runner return the frames completed so
            far; the caller reconciles the count.
            """
            self._cancel_evt = evt

        @property
        def frames(self) -> list[int]:
            return list(self._frames)

        @property
        def n_consumed(self) -> int:
            """How many frames this pass actually yielded."""
            return self._pos

        def __iter__(self):
            self._pos = 0
            self._next_expected = -1
            return self

        def __next__(self):
            if self._pos >= len(self._frames):
                raise StopIteration
            if self._cancel_evt is not None and self._cancel_evt.is_set():
                raise StopIteration
            target = self._frames[self._pos]
            # Only seek when the reader isn't already in position.
            # ``read_frame`` advances by one, so a contiguous ascending
            # range -- the common "predict here" case -- decodes
            # sequentially after a single initial seek. Seeking every
            # frame instead costs a keyframe re-decode per frame, which
            # measured ~3x slower end to end.
            #
            # Position is tracked here rather than read off the reader:
            # ``_cursor`` exists only on the dnav reader installed by
            # ``patch_dlc_decoder``, not on the stock cv2 ``VideoReader``.
            if target != self._next_expected:
                self.set_to_frame(target)
            frame = self.read_frame(crop=self._crop)
            self._next_expected = target + 1
            if frame is None:
                raise StopIteration
            self._pos += 1
            # Copy for the same reason DLC's VideoIterator does: a
            # negative-stride view can't become a torch tensor.
            frame = frame.copy()
            if self._context is None:
                return frame
            return frame, self._context[self._pos - 1]

    return FrameListIterator


class RangePredictor:
    """Runs DLC inference on arbitrary frame ranges, holding the model open.

    The model is built on first use and reused for the lifetime of the
    instance -- the single largest win available here, since the stock
    path reloads it per call. Video readers are cached per path too, so
    repeated requests against the video being refined pay no re-open or
    TOC-rebuild cost.

    Not thread-safe by itself: one predictor is meant to be owned by one
    worker thread. That thread may serve a priority queue (foreground
    range requests preempting a background sweep) -- the serialisation
    lives in the queue, not here.

    Args:
        config_path: Path to the DLC project ``config.yaml``.
        snapshot_index: Which snapshot to load. ``None`` defers to the
            project config's ``snapshotindex``, matching what
            ``analyze_videos`` would resolve. Note DUSTrack's
            ``_config.DLC3_USE_LAST_SNAPSHOT`` convention means "best"
            generally resolves to *last* elsewhere in the codebase.
        batch_size: Inference batch size. Defaults to
            :data:`DEFAULT_BATCH_SIZE`.
        device: Torch device override, e.g. ``"cuda:0"``. ``None`` uses
            the model config's choice.
        shuffle / trainingsetindex / modelprefix: Standard DLC model
            selectors, forwarded to ``DLCLoader``.
    """

    def __init__(
        self,
        config_path: str | Path,
        *,
        snapshot_index: int | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str | None = None,
        shuffle: int = 1,
        trainingsetindex: int = 0,
        modelprefix: str = "",
    ) -> None:
        self.config_path = str(config_path)
        self.snapshot_index = snapshot_index
        self.batch_size = int(batch_size)
        self.device = device
        self.shuffle = shuffle
        self.trainingsetindex = trainingsetindex
        self.modelprefix = modelprefix

        self._runner = None
        self._loader = None
        self._bodyparts: list[str] | None = None
        self._iterators: dict[str, object] = {}
        self._videos: dict[str, object] = {}
        self._iterator_cls = None

    # ------------------------------------------------------------------ #
    # Model                                                              #
    # ------------------------------------------------------------------ #
    def _ensure_runner(self):
        """Build (once) the pose runner, mirroring ``analyze_videos``.

        The construction sequence is deliberately the same as DLC's own:
        ``DLCLoader`` -> ``parse_snapshot_index_for_analysis`` ->
        ``get_model_snapshots`` -> ``get_pose_inference_runner``. Diverging
        here would be exactly the kind of drift the parity test exists to
        catch.
        """
        if self._runner is not None:
            return self._runner

        _dlcloader._ensure_dlc_loaded()
        import deeplabcut.pose_estimation_pytorch.apis.utils as dlc_api_utils
        from deeplabcut.pose_estimation_pytorch.data import DLCLoader

        loader = DLCLoader(
            self.config_path,
            trainset_index=self.trainingsetindex,
            shuffle=self.shuffle,
            modelprefix=self.modelprefix,
        )

        individuals = loader.model_cfg["metadata"]["individuals"]
        if len(individuals) > 1:
            raise NotImplementedError(
                "RangePredictor targets single-animal DUSTrack projects; "
                f"this model declares {len(individuals)} individuals. The "
                "multi-animal column layout (an extra 'individuals' level) "
                "is unhandled -- use analyze_videos."
            )

        snapshot_index, _ = dlc_api_utils.parse_snapshot_index_for_analysis(
            loader.project_cfg, loader.model_cfg, self.snapshot_index, None
        )
        snapshot = dlc_api_utils.get_model_snapshots(
            snapshot_index, loader.model_folder, loader.pose_task
        )[0]

        if self.device is not None:
            loader.model_cfg["device"] = self.device

        self._runner = dlc_api_utils.get_pose_inference_runner(
            model_config=loader.model_cfg,
            snapshot_path=snapshot.path,
            max_individuals=len(individuals),
            batch_size=self.batch_size,
        )
        self._loader = loader
        self._bodyparts = list(loader.model_cfg["metadata"]["bodyparts"])
        self.snapshot_path = str(snapshot.path)
        return self._runner

    @property
    def bodyparts(self) -> list[str]:
        """Bodypart names in model order (builds the model if needed)."""
        self._ensure_runner()
        return list(self._bodyparts or [])

    # ------------------------------------------------------------------ #
    # Video readers                                                      #
    # ------------------------------------------------------------------ #
    def _iterator_for(self, video_path: str | Path):
        """Get (or open) the cached iterator for a video."""
        key = str(video_path)
        it = self._iterators.get(key)
        if it is None:
            if self._iterator_cls is None:
                self._iterator_cls = make_frame_list_iterator_class()
            it = self._iterator_cls(key)
            self._iterators[key] = it
        return it

    def n_frames(self, video_path: str | Path) -> int:
        """Frame count of a video, via the cached reader."""
        return int(self._iterator_for(video_path)._n_frames)

    # ------------------------------------------------------------------ #
    # Prediction                                                         #
    # ------------------------------------------------------------------ #
    def predict_frames(
        self,
        video_path: str | Path,
        frames: Sequence[int],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
        cancel_event: threading.Event | None = None,
        chunk_size: int | None = None,
    ) -> pd.DataFrame:
        """Predict an arbitrary list of frames.

        Args:
            video_path: Video to run against.
            frames: Frame indices, in the order they should be returned.
            progress_callback: Optional ``(done, total)``, fired once per
                chunk. Mirrors :func:`dustrack.blip.interpolate_blips`.
            cancel_event: Optional cooperative cancel, polled per frame.
                A cancel mid-run returns the frames already completed.
            chunk_size: Frames per runner call. Defaults to
                ``DEFAULT_CHUNK_BATCHES * batch_size``.

        Returns:
            A DLC-format DataFrame -- columns
            ``(scorer, bodypart, {x, y, likelihood})``, **indexed by the
            requested frame numbers** (not 0..n-1). This is the same
            layout ``analyze_videos`` writes to h5, so it can be handed
            straight to
            :meth:`VideoAnnotation._dlc_trace_to_annotation_dict`.

        Raises:
            PredictionCancelled: If cancelled before any frame finished.
            IndexError: If any frame is out of range for the video.
        """
        frames = [int(f) for f in frames]
        runner = self._ensure_runner()
        it = self._iterator_for(video_path)
        if chunk_size is None:
            chunk_size = DEFAULT_CHUNK_BATCHES * self.batch_size
        chunk_size = max(1, int(chunk_size))

        if not frames:
            return self._empty_frame()

        # Validate the whole request up front so a bad index fails before
        # any GPU work rather than partway through.
        it.set_frames(frames)
        it.set_cancel_event(cancel_event)

        predictions: list[dict] = []
        done_frames: list[int] = []
        total = len(frames)
        try:
            for start in range(0, total, chunk_size):
                if cancel_event is not None and cancel_event.is_set():
                    break
                chunk = frames[start : start + chunk_size]
                it.set_frames(chunk)
                out = runner.inference(images=it)
                predictions.extend(out)
                done_frames.extend(chunk[: len(out)])
                if progress_callback is not None:
                    progress_callback(len(done_frames), total)
                if len(out) < len(chunk):
                    # Short chunk == cancelled (or a decode stop) mid-way.
                    break
        finally:
            it.set_cancel_event(None)

        if not predictions:
            if cancel_event is not None and cancel_event.is_set():
                raise PredictionCancelled(
                    "cancelled before any frame completed"
                )
            return self._empty_frame()

        return self._to_dataframe(predictions, done_frames)

    def predict_range(
        self,
        video_path: str | Path,
        start: int,
        end: int,
        *,
        step: int = 1,
        annotation: bool = True,
        source_annotation: VideoAnnotation | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        cancel_event: threading.Event | None = None,
    ):
        """Predict a contiguous frame range, inclusive of both endpoints.

        The inclusive-both-ends convention matches
        :func:`dustrack.lk_opticalflow.lucas_kanade_rstc`, so a range
        means the same thing across DUSTrack's two local-tracking paths.

        Args:
            video_path: Video to run against.
            start / end: Inclusive frame bounds. ``end`` is clamped to the
                last frame; ``start`` below 0 is an error.
            step: Frame stride. ``1`` predicts every frame.
            annotation: If True (default) return a sparse
                :class:`VideoAnnotation`; if False return the raw
                DataFrame.
            source_annotation: Optional annotation to inherit the label
                set and the open video reader from, avoiding a second
                ``av.open``. Its data is never read or modified.
            progress_callback / cancel_event: As :meth:`predict_frames`.

        Returns:
            A sparse ``VideoAnnotation`` (or a DataFrame if
            ``annotation=False``) covering only the requested frames.
        """
        if start < 0:
            raise ValueError(f"start must be >= 0, got {start}")
        n = self.n_frames(video_path)
        end = min(int(end), n - 1)
        if end < start:
            raise ValueError(
                f"empty range: start={start} > end={end} (video has {n} frames)"
            )
        frames = list(range(int(start), int(end) + 1, max(1, int(step))))
        df = self.predict_frames(
            video_path,
            frames,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        if not annotation:
            return df
        return self.to_annotation(
            df, video_path, source_annotation=source_annotation
        )

    # ------------------------------------------------------------------ #
    # Conversion                                                         #
    # ------------------------------------------------------------------ #
    def _empty_frame(self) -> pd.DataFrame:
        cols = pd.MultiIndex.from_product(
            [[SCORER], self.bodyparts, ["x", "y", "likelihood"]],
            names=["scorer", "bodyparts", "coords"],
        )
        return pd.DataFrame(columns=cols, index=pd.Index([], dtype=int))

    def _to_dataframe(
        self, predictions: list[dict], frames: Sequence[int]
    ) -> pd.DataFrame:
        """Stack raw runner output into the DLC DataFrame layout.

        Mirrors ``deeplabcut...videos.create_df_from_prediction`` for the
        single-animal case, minus the h5/pickle writes.
        """
        pred = np.stack([p["bodyparts"][..., :3] for p in predictions])
        pred = pred[:, :1]  # single individual (enforced in _ensure_runner)
        cols = pd.MultiIndex.from_product(
            [[SCORER], self.bodyparts, ["x", "y", "likelihood"]],
            names=["scorer", "bodyparts", "coords"],
        )
        return pd.DataFrame(
            pred.reshape((len(pred), -1)),
            columns=cols,
            index=pd.Index([int(f) for f in frames], name="frame"),
        )

    def _video_for(self, video_path: str | Path):
        """A cached ``utils.Video`` reader for annotation postprocessing.

        Distinct from the DLC iterator's reader: with the decoder patch
        the iterator holds a bare dnav ``VideoReader``, but
        ``VideoAnnotation`` consumers (LK-RSTC in particular) need
        ``utils.Video``'s additions -- ``fname``, ``name``, ``gray()``.
        Sharing the iterator's reader would hand back an annotation that
        silently can't postprocess.

        Returns ``None`` if the path isn't openable as a video, which
        leaves a coordinate-only annotation -- valid, just not
        postprocessable.
        """
        key = str(video_path)
        if key in self._videos:
            return self._videos[key]
        from datanavigator import utils as _dnav_utils

        video = None
        try:
            if _dnav_utils.is_video(key):
                video = _dnav_utils.Video(key)
        except Exception:
            video = None
        self._videos[key] = video
        return video

    def to_annotation(
        self,
        df: pd.DataFrame,
        video_path: str | Path,
        *,
        source_annotation: VideoAnnotation | None = None,
        video=None,
        name: str | None = None,
    ) -> VideoAnnotation:
        """Wrap a prediction DataFrame as a sparse :class:`VideoAnnotation`.

        Label naming goes through
        ``VideoAnnotation._dlc_trace_to_annotation_dict`` -- the same
        function the h5 path uses -- so a range-predicted layer names its
        labels identically to an ``analyze_videos`` one. Reimplementing
        that mapping is precisely how the two paths would drift.

        The result is sparse in the ``interpolate_blips`` sense: only the
        predicted frames are populated, and nothing is written to disk.

        Args:
            df: A DataFrame from :meth:`predict_frames`.
            video_path: The video the prediction came from.
            source_annotation: Optional layer to inherit the label set
                and the open reader from. Sharing the reader avoids a
                second ``av.open`` -- the same reason
                :func:`dustrack.blip.interpolate_blips` does it.
            video: Explicit reader override, ahead of both of the above.
            name: Layer name; defaults to ``range_predict``.
        """
        data = VideoAnnotation._dlc_trace_to_annotation_dict(df)
        if source_annotation is not None:
            if video is None:
                video = source_annotation.video
            # Expose every label the source carries, even those the model
            # doesn't predict, so the layers line up.
            for label in source_annotation.labels:
                data.setdefault(label, {})
        if video is None:
            video = self._video_for(video_path)

        stem = Path(video_path).stem
        fname = f"{stem}_annotations_{name or 'range_predict'}.json"
        # ``vname=None`` + an explicit reader, mirroring interpolate_blips:
        # it keeps ``_parse_inp``'s ``utils.is_video`` filesystem probe off
        # the interactive path. Video identity travels with the reader.
        return VideoAnnotation(
            fname=fname,
            vname=None,
            n_labels=max(1, len(data)),
            preloaded_json=data,
            video=video,
        )

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        """Release cached readers and drop the model.

        The GPU memory the runner holds is reclaimed when it is garbage
        collected; this drops the reference and clears the torch cache if
        torch is loaded.
        """
        for it in self._iterators.values():
            try:
                it.close()
            except Exception:
                pass
        self._iterators.clear()
        self._videos.clear()
        self._runner = None
        self._loader = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def __enter__(self) -> "RangePredictor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "loaded" if self._runner is not None else "not loaded"
        return (
            f"RangePredictor(config={self.config_path!r}, "
            f"batch_size={self.batch_size}, model={state})"
        )

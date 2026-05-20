"""Runtime patches for DeepLabCut PyTorch inference.

Targets the throughput ceiling identified by the 2026-05-20 sweep
(see ``S:/_corpus/dustrack/dlc_inference_bench_2026-05-20/``):
DLC's ``InferenceRunner`` runs a single preprocessing thread feeding a
small prefetch queue, which caps fps around ~140 on a 4090 with a
ResNet-50 BU model on 706x558 frames -- the GPU is starved on the CPU.

Patches applied:
    * ``_preprocessing_worker``: single-threaded -> dedicated reader
      thread + N preprocess worker threads + in-order batcher
    * ``PoseInferenceRunner.predict``: ``inputs.to(device)`` -> ``...,
      non_blocking=True``

NOT patched in this prototype:
    * GPU-side normalize / ToTensor (correctness against training-time
      transforms needs validation; deferred to a follow-up)
    * PyAV / NVDEC video decode (decode is only the limit once
      preprocessing stops being one)

Two DLC versions are supported via feature detection (no version
pinning):
    * The pre-refactor API used in ``deeplabcut`` <= 3.0.0rc10:
      ``_batch`` is a single concatenated tensor; ``_input_queue`` is
      built in ``__init__`` directly.
    * The post-refactor API in ``deeplabcut`` >= 3.0.0rc13:
      ``_batch_list`` is a list of per-frame tensors stacked at flush
      time; ``_safe_put`` / ``_safe_get`` helpers; ``InferenceConfig``
      drives multithreading settings.

Detection key: presence of ``_safe_put``.

Opt-out: ``DUSTRACK_DISABLE_DLC_PATCH=1``.

A separate decoder patch (``patch_dlc_decoder``) replaces DLC's
``cv2.VideoCapture``-backed ``VideoReader`` with a dnav PyAV+TOC reader
so that annotation, training-frame extraction, and inference all go
through one decode path. The 2026-05-20 parity test
(``S:/_corpus/dustrack/dlc_inference_bench_2026-05-20/parity_decoder.json``)
confirmed bit-exact agreement with cv2 on a 500-frame CFR clip;
decode-speed difference is irrelevant because decode is not the
inference bottleneck. Opt-out: ``DUSTRACK_DISABLE_DLC_DECODER_PATCH=1``.
"""
from __future__ import annotations

import os
import threading
from queue import Empty, Queue

import numpy as np
import torch


_PATCHED = False
_DECODER_PATCHED = False

DEFAULT_NUM_WORKERS = 4
DEFAULT_RAW_QUEUE_SIZE = 16

# Force-enable torch.autocast on CUDA, regardless of DLC's
# ``inference_cfg.autocast.enabled``. rc13+ defaults this to False, which
# leaves FP16-capable GPUs running FP32 inference. For pose estimation
# the prediction is a heatmap argmax, so FP16 produces effectively
# identical pixel-precision keypoints; flip to ``False`` if a future
# model class shows accuracy drift.
FORCE_AUTOCAST_CUDA = True


# --------------------------------------------------------------------------- #
# rc10 worker -- uses ``self._batch`` (single concatenated tensor).
# --------------------------------------------------------------------------- #


def _worker_rc10(self, images):
    SENTINEL = object()
    N = DEFAULT_NUM_WORKERS
    raw_q: Queue = Queue(maxsize=DEFAULT_RAW_QUEUE_SIZE)
    preproc_q: Queue = Queue(maxsize=DEFAULT_RAW_QUEUE_SIZE)

    def reader():
        try:
            for idx, data in enumerate(images):
                if self._stop_event.is_set():
                    break
                raw_q.put((idx, data))
        except Exception as e:  # noqa: BLE001
            self._exception = e
            self._stop_event.set()
        finally:
            for _ in range(N):
                raw_q.put(SENTINEL)

    def preprocess_one(data):
        if isinstance(data, tuple):
            inputs, context = data
        else:
            inputs, context = data, {}
        if self.preprocessor is not None:
            inputs, context = self.preprocessor(inputs, context)
        else:
            inputs = torch.as_tensor(inputs)
        return inputs, context

    def preproc_worker():
        try:
            while not self._stop_event.is_set():
                item = raw_q.get()
                if item is SENTINEL:
                    preproc_q.put(SENTINEL)
                    return
                idx, data = item
                preproc_q.put((idx, preprocess_one(data)))
        except Exception as e:  # noqa: BLE001
            self._exception = e
            self._stop_event.set()
            preproc_q.put(SENTINEL)

    workers = [threading.Thread(target=preproc_worker, daemon=True) for _ in range(N)]
    reader_t = threading.Thread(target=reader, daemon=True)
    reader_t.start()
    for w in workers:
        w.start()

    next_idx = 0
    buffer: dict = {}
    sentinels_seen = 0
    try:
        while True:
            if sentinels_seen >= N and not buffer:
                break
            try:
                item = preproc_q.get(timeout=self.timeout)
            except Empty:
                if self._stop_event.is_set():
                    break
                continue
            if item is SENTINEL:
                sentinels_seen += 1
                continue
            idx, payload = item
            buffer[idx] = payload
            while next_idx in buffer:
                inputs, context = buffer.pop(next_idx)
                next_idx += 1
                _append_rc10(self, inputs, context)
                while self._batch is not None and len(self._batch) >= self.batch_size:
                    batch = self._batch[: self.batch_size]
                    model_kwargs = {
                        mk: v[: self.batch_size] for mk, v in self._model_kwargs.items()
                    }
                    self._input_queue.put((batch, model_kwargs), timeout=self.timeout)
                    if len(self._batch) <= self.batch_size:
                        self._batch = None
                        self._model_kwargs = {}
                    else:
                        self._batch = self._batch[self.batch_size :]
                        self._model_kwargs = {
                            mk: v[self.batch_size :]
                            for mk, v in self._model_kwargs.items()
                        }
        if self._batch is not None and len(self._batch) > 0:
            self._input_queue.put((self._batch, self._model_kwargs), timeout=self.timeout)
    except Exception as e:  # noqa: BLE001
        self._exception = e
        self._stop_event.set()
    finally:
        self._input_queue.put(None, timeout=self.timeout)
        reader_t.join(timeout=self.timeout)
        for w in workers:
            w.join(timeout=self.timeout)


def _append_rc10(self, inputs, context):
    model_kwargs = context.pop("model_kwargs", {})
    for k, v in model_kwargs.items():
        curr_v = self._model_kwargs.get(k)
        if curr_v is None or len(curr_v) == 0:
            curr_v = v
        elif len(v) == 0:
            continue
        elif isinstance(curr_v, np.ndarray):
            curr_v = np.concatenate([curr_v, v], axis=0)
        elif isinstance(curr_v, torch.Tensor):
            curr_v = torch.cat([curr_v, v], dim=0)
        else:
            raise ValueError(f"unexpected model_kwargs type for {k}: {type(v)}")
        self._model_kwargs[k] = curr_v

    self._contexts.append(context)
    self._image_batch_sizes.append(len(inputs))
    if len(inputs) == 0:
        return
    if self._batch is None:
        self._batch = inputs
    else:
        self._batch = torch.cat([self._batch, inputs], dim=0)


# --------------------------------------------------------------------------- #
# rc13 worker -- uses ``self._batch_list`` and ``self._safe_put``.
# --------------------------------------------------------------------------- #


def _worker_rc13(self, images):
    SENTINEL = object()
    N = DEFAULT_NUM_WORKERS
    raw_q: Queue = Queue(maxsize=DEFAULT_RAW_QUEUE_SIZE)
    preproc_q: Queue = Queue(maxsize=DEFAULT_RAW_QUEUE_SIZE)
    timeout = getattr(self.inference_cfg.multithreading, "timeout", 30.0)

    def reader():
        try:
            for idx, data in enumerate(images):
                if self._stop_event.is_set():
                    break
                raw_q.put((idx, data))
        except BaseException as e:  # noqa: BLE001
            self._exception = e
            self._stop_event.set()
        finally:
            for _ in range(N):
                raw_q.put(SENTINEL)

    def preprocess_one(data):
        if isinstance(data, tuple):
            inputs, context = data
        else:
            inputs, context = data, {}
        if self.preprocessor is not None:
            inputs, context = self.preprocessor(inputs, context)
        else:
            inputs = torch.as_tensor(inputs)
        return inputs, context

    def preproc_worker():
        try:
            while not self._stop_event.is_set():
                item = raw_q.get()
                if item is SENTINEL:
                    preproc_q.put(SENTINEL)
                    return
                idx, data = item
                preproc_q.put((idx, preprocess_one(data)))
        except BaseException as e:  # noqa: BLE001
            self._exception = e
            self._stop_event.set()
            preproc_q.put(SENTINEL)

    workers = [threading.Thread(target=preproc_worker, daemon=True) for _ in range(N)]
    reader_t = threading.Thread(target=reader, daemon=True)
    reader_t.start()
    for w in workers:
        w.start()

    next_idx = 0
    buffer: dict = {}
    sentinels_seen = 0
    try:
        while True:
            if sentinels_seen >= N and not buffer:
                break
            try:
                item = preproc_q.get(timeout=timeout)
            except Empty:
                if self._stop_event.is_set():
                    break
                continue
            if item is SENTINEL:
                sentinels_seen += 1
                continue
            idx, payload = item
            buffer[idx] = payload
            while next_idx in buffer:
                inputs, context = buffer.pop(next_idx)
                next_idx += 1
                _append_rc13(self, inputs, context)
                while len(self._batch_list) >= self.batch_size:
                    batch = torch.stack(self._batch_list[: self.batch_size], dim=0)
                    model_kwargs = {
                        mk: v[: self.batch_size] for mk, v in self._model_kwargs.items()
                    }
                    self._safe_put((batch, model_kwargs))
                    if len(self._batch_list) <= self.batch_size:
                        self._batch_list, self._model_kwargs = [], {}
                    else:
                        self._batch_list = self._batch_list[self.batch_size :]
                        self._model_kwargs = {
                            mk: v[self.batch_size :]
                            for mk, v in self._model_kwargs.items()
                        }
        if len(self._batch_list) > 0:
            batch = torch.stack(self._batch_list, dim=0)
            self._safe_put((batch, self._model_kwargs))
    except BaseException as e:  # noqa: BLE001
        self._exception = e
        self._stop_event.set()
    finally:
        self._safe_put(None)
        reader_t.join(timeout=timeout)
        for w in workers:
            w.join(timeout=timeout)


def _append_rc13(self, inputs, context):
    model_kwargs = context.pop("model_kwargs", {})
    for k, v in model_kwargs.items():
        curr_v = self._model_kwargs.get(k)
        if curr_v is None or len(curr_v) == 0:
            curr_v = v
        elif len(v) == 0:
            continue
        elif isinstance(curr_v, np.ndarray):
            curr_v = np.concatenate([curr_v, v], axis=0)
        elif isinstance(curr_v, torch.Tensor):
            curr_v = torch.cat([curr_v, v], dim=0)
        else:
            raise ValueError(f"unexpected model_kwargs type for {k}: {type(v)}")
        self._model_kwargs[k] = curr_v

    self._contexts.append(context)
    self._image_batch_sizes.append(len(inputs))
    if len(inputs) == 0:
        return
    self._batch_list.extend(list(inputs))


# --------------------------------------------------------------------------- #
# Shared: non-blocking H2D for PoseInferenceRunner.predict.
# --------------------------------------------------------------------------- #


def _make_patched_pose_predict(_orig):
    def predict(self, inputs, **kwargs):
        batch_size = len(inputs)
        if self.dynamic is not None:
            inputs = self.dynamic.crop(inputs)
        # non_blocking=True is harmless for unpinned CPU tensors (just
        # blocks anyway). The pipeline-parallel gain materialises when
        # we later land pinned-memory preprocessing.
        device = self.device
        cuda = device and "cuda" in str(device)
        # rc10 always uses autocast for cuda; rc13 makes it opt-in via
        # inference_cfg.autocast.enabled. The patch forces it on by
        # default because that lifts the rc13+ peak on FP16-capable GPUs
        # (the 2026-05-20 sweep showed multi-thread preprocessing alone
        # only delivered +6% peak fps on a 4090 because rc13 was running
        # FP32). Flip ``FORCE_AUTOCAST_CUDA`` to False to defer to
        # inference_cfg.
        if FORCE_AUTOCAST_CUDA:
            use_autocast = cuda
        elif hasattr(self, "inference_cfg"):
            use_autocast = bool(
                cuda and getattr(self.inference_cfg.autocast, "enabled", False)
            )
        else:
            use_autocast = cuda
        if cuda and use_autocast:
            with torch.autocast(device_type=str(device)):
                outputs = self.model(inputs.to(device, non_blocking=True), **kwargs)
                raw_predictions = self.model.get_predictions(outputs)
        else:
            outputs = self.model(inputs.to(device, non_blocking=True), **kwargs)
            raw_predictions = self.model.get_predictions(outputs)

        if self.dynamic is not None:
            raw_predictions["bodypart"]["poses"] = self.dynamic.update(
                raw_predictions["bodypart"]["poses"]
            )

        return [
            {
                head: {
                    pred_name: pred[b].cpu().numpy()
                    for pred_name, pred in head_outputs.items()
                }
                for head, head_outputs in raw_predictions.items()
            }
            for b in range(batch_size)
        ]
    return predict


# --------------------------------------------------------------------------- #
# Entry point.
# --------------------------------------------------------------------------- #


def patch_dlc(
    verbose: bool = True,
    multithread: bool | None = None,
    autocast: bool | None = None,
) -> bool:
    """Apply runtime patches. Idempotent.

    Args:
        verbose: print a one-line notice on success / skip.
        multithread: install the multi-threaded preprocessing worker.
            ``None`` -> read ``DUSTRACK_PATCH_MULTITHREAD`` env var
            (default True if patch is otherwise active).
        autocast: install the non_blocking H2D + force-on-autocast
            predict patch. ``None`` -> read
            ``DUSTRACK_PATCH_AUTOCAST`` env var (default True).

    Returns True if at least one component was applied, False if
    skipped (DUSTRACK_DISABLE_DLC_PATCH set, or DLC missing, or both
    components disabled, or an unrecognised DLC API).
    """
    global _PATCHED
    if _PATCHED:
        return True
    if os.environ.get("DUSTRACK_DISABLE_DLC_PATCH"):
        return False

    def _envflag(name: str, default: bool) -> bool:
        v = os.environ.get(name)
        if v is None:
            return default
        return v not in ("", "0", "false", "False", "FALSE", "no", "No")

    if multithread is None:
        multithread = _envflag("DUSTRACK_PATCH_MULTITHREAD", True)
    if autocast is None:
        autocast = _envflag("DUSTRACK_PATCH_AUTOCAST", True)
    if not (multithread or autocast):
        return False

    try:
        import deeplabcut
        from deeplabcut.pose_estimation_pytorch.runners import inference as _inf
    except ImportError:
        return False

    version = getattr(deeplabcut, "__version__", "?")
    worker = None
    if multithread:
        if hasattr(_inf.InferenceRunner, "_safe_put"):
            worker = _worker_rc13
            flavor = "rc13+"
        elif "_batch" in getattr(
            _inf.InferenceRunner.__init__,
            "__code__",
            type("", (), {"co_names": ()})(),
        ).co_names:
            worker = _worker_rc10
            flavor = "<=rc10"
        else:
            if verbose:
                print(
                    f"dustrack: skipping DLC inference patch -- unrecognised "
                    f"InferenceRunner API in deeplabcut {version}."
                )
            return False
    else:
        flavor = "predict-only"

    if worker is not None:
        _inf.InferenceRunner._preprocessing_worker = worker
    if autocast:
        _inf.PoseInferenceRunner.predict = _make_patched_pose_predict(
            _inf.PoseInferenceRunner.predict
        )
    _PATCHED = True
    if verbose:
        parts = []
        if worker is not None:
            parts.append(f"multithread={DEFAULT_NUM_WORKERS}w")
        if autocast:
            parts.append("autocast+non_blocking")
        print(f"dustrack: patched DLC {version} ({flavor}) -- {', '.join(parts)}")
    return True


# --------------------------------------------------------------------------- #
# Decoder patch: replace DLC's cv2.VideoCapture-backed VideoReader with a
# dnav PyAV+TOC reader so annotation, training-frame extraction, and
# inference all go through one decode path.
# --------------------------------------------------------------------------- #


def _make_dnav_videoreader_init(_orig_init):
    """Replacement for ``deeplabcut.utils.auxfun_videos.VideoReader.__init__``
    that opens via dnav's PyAV+TOC reader instead of ``cv2.VideoCapture``.

    Preserves the original public surface: ``video_path``, ``video``,
    ``_bbox``, ``_n_frames``, ``_width``, ``_height``, ``_fps``,
    ``_n_frames_robust``. The ``video`` attribute is the dnav reader
    itself (it exposes ``__len__`` / ``__getitem__`` / ``get_avg_fps``
    instead of cv2's prop API; downstream code that calls
    ``self.video.get(cv2.CAP_PROP_*)`` would need to be patched too,
    but the in-tree DLC code accesses cv2 via the wrapper methods we
    also override, so the only at-risk caller is third-party).
    """
    def __init__(self, video_path):
        import os
        from datanavigator.video_reader import VideoReader as _DnavReader

        if not os.path.isfile(video_path):
            raise ValueError(
                f'Video path "{video_path}" does not point to a file.'
            )
        self.video_path = video_path
        self.video = _DnavReader(video_path)
        self._bbox = 0, 1, 0, 1
        self._n_frames_robust = None
        self._cursor = 0
        self._dnav_frame_count = len(self.video)
        self._dnav_fps = round(float(self.video.get_avg_fps()), 2)
        # Probe one frame for dimensions (PyAV TOC build already ran).
        frame0 = np.asarray(self.video[0])
        self._dnav_height, self._dnav_width = int(frame0.shape[0]), int(frame0.shape[1])
        self.parse_metadata()
    return __init__


def _dnav_parse_metadata(self):
    import warnings as _w
    self._n_frames = int(self._dnav_frame_count)
    if self._n_frames >= 1e9:
        _w.warn(
            "The video has more than 10^9 frames, we recommend chopping it up."
        )
    self._width = int(self._dnav_width)
    self._height = int(self._dnav_height)
    self._fps = float(self._dnav_fps)


def _dnav_set_to_frame(self, ind):
    import warnings as _w
    if ind < 0:
        raise ValueError("Index must be a positive integer.")
    last_frame = len(self) - 1
    if ind > last_frame:
        _w.warn(
            "Index exceeds the total number of frames. "
            "Setting to last frame instead."
        )
        ind = last_frame
    self._cursor = int(ind)


def _dnav_reset(self):
    self._cursor = 0


def _dnav_read_frame(self, shrink=1, crop=False):
    import cv2 as _cv2
    if self._cursor >= self._n_frames:
        return None
    frame = np.asarray(self.video[int(self._cursor)])  # already RGB
    self._cursor += 1
    if crop:
        x1, x2, y1, y2 = self.get_bbox(relative=False)
        frame = frame[y1:y2, x1:x2]
    if shrink > 1:
        h, w = frame.shape[:2]
        frame = _cv2.resize(
            frame,
            (w // shrink, h // shrink),
            fx=0,
            fy=0,
            interpolation=_cv2.INTER_AREA,
        )
    return frame


def _dnav_close(self):
    # PyAV containers held by dnav close on GC; explicit drop here.
    self.video = None


def _dnav_check_integrity_robust(self):
    # Walk every frame to exercise the decode path. dnav's __getitem__
    # raises on a bad seek; surface as a warning to match cv2 behaviour.
    import warnings as _w
    for fr in range(self._n_frames):
        try:
            _ = self.video[fr]
        except Exception:
            _w.warn(
                f"PyAV failed to load frame {fr}. Use ffmpeg to re-encode "
                f"video file"
            )


def patch_dlc_decoder(verbose: bool = True) -> bool:
    """Replace DLC's ``cv2.VideoCapture``-backed ``VideoReader`` with a
    ``dnav.VideoReader`` (PyAV+TOC) adapter. Idempotent.

    Affects:
        * ``deeplabcut.utils.auxfun_videos.VideoReader`` and its
          ``deeplabcut.utils`` re-export
        * Subclasses inherit the patch automatically:
          ``deeplabcut.utils.auxfun_videos.VideoWriter`` (used by
          ``extract_frames``) and
          ``deeplabcut.pose_estimation_pytorch.apis.videos.VideoIterator``
          (used by ``analyze_videos``).

    Opt-out: ``DUSTRACK_DISABLE_DLC_DECODER_PATCH=1``.

    Returns True if the patch was applied, False if skipped.
    """
    global _DECODER_PATCHED
    if _DECODER_PATCHED:
        return True
    if os.environ.get("DUSTRACK_DISABLE_DLC_DECODER_PATCH"):
        return False
    try:
        from deeplabcut.utils import auxfun_videos as _av
        # Trigger an import error early if dnav is missing.
        from datanavigator.video_reader import VideoReader as _DnavReader  # noqa: F401
    except ImportError:
        return False

    _av.VideoReader.__init__ = _make_dnav_videoreader_init(_av.VideoReader.__init__)
    _av.VideoReader.parse_metadata = _dnav_parse_metadata
    _av.VideoReader.set_to_frame = _dnav_set_to_frame
    _av.VideoReader.reset = _dnav_reset
    _av.VideoReader.read_frame = _dnav_read_frame
    _av.VideoReader.close = _dnav_close
    _av.VideoReader.check_integrity_robust = _dnav_check_integrity_robust

    _DECODER_PATCHED = True
    if verbose:
        try:
            import deeplabcut
            version = getattr(deeplabcut, "__version__", "?")
        except ImportError:
            version = "?"
        print(
            f"dustrack: patched DLC {version} VideoReader -- dnav PyAV+TOC "
            f"(unified decoder)"
        )
    return True

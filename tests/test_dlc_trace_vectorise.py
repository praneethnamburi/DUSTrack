"""
Parity tests for the vectorised ``_dlc_trace_to_annotation_dict``
(1.2.0 cold-open optimisation).

The pre-1.2.0 implementation walked every frame with a pandas
``.loc[frame]`` call, hitting ~73 k cross-section lookups per DLC
trace on a 36 k-frame video. The vectorised replacement does a single
column-slice + ``.to_numpy()`` + NaN-mask per label.

These tests pin the contract: for every DLC-h5-shaped DataFrame we
care about, the two implementations must produce identical
dictionaries.

The reference (legacy) implementation lives inline below as
``_dlc_trace_to_annotation_dict_legacy`` so the parity check has both
versions available without going through the git log.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datanavigator import utils
from dustrack.pointtracking import VideoAnnotation


# ---------------------------------------------------------------------------
# Reference implementation (pre-1.2.0 -- preserved for the parity check).
# Keep this verbatim other than the function name; do not "improve" it --
# it's the contract we're testing against.
# ---------------------------------------------------------------------------


def _dlc_trace_to_annotation_dict_legacy(
    df: pd.DataFrame, remove_label_prefix: str = "point",
) -> dict:
    if False in [
        utils.removeprefix(x, remove_label_prefix).isdigit()
        for x in df.columns.levels[1]
    ]:
        label_orig_to_internal = {
            x: str(xcnt) for xcnt, x in enumerate(df.columns.levels[1].tolist())
        }
    else:
        label_orig_to_internal = {
            x: utils.removeprefix(x, remove_label_prefix)
            for x in df.columns.levels[1].tolist()
        }
    frames = df.index.values

    data = {label: {} for label in label_orig_to_internal.values()}
    scorer = df.columns.levels[0].values[0]
    for label_orig, label_internal in label_orig_to_internal.items():
        coords = df.loc[:, (scorer, label_orig, ["x", "y"])]
        for frame in frames:
            if frame in coords.index:
                coord_val = coords.loc[frame].values
                if np.all(np.isnan(coord_val)):
                    continue
                data[label_internal][frame] = coord_val.tolist()

    return data


# ---------------------------------------------------------------------------
# DataFrame builders shaped like real DLC analyze_videos output.
# ---------------------------------------------------------------------------


def _build_dlc_df(
    n_frames: int,
    labels: list[str],
    scorer: str = "DLC_resnet50_test",
    nan_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build a DLC-shaped multi-index DataFrame.

    Columns are ``(scorer, label, coord)`` triples; ``coord`` cycles
    through ``("x", "y", "likelihood")`` to match the real DLC h5
    schema. Values are deterministic-pseudo-random based on frame +
    label index so tests are reproducible without hardcoded fixtures.

    ``nan_mask`` (shape ``(n_frames, n_labels)``, optional) marks
    rows-to-NaN per label -- ``True`` flips both x and y to NaN for
    that frame/label, matching the "absent frame" pattern the legacy
    implementation skipped.
    """
    rng = np.random.default_rng(0)
    columns = pd.MultiIndex.from_tuples(
        [(scorer, label, coord)
         for label in labels
         for coord in ("x", "y", "likelihood")],
        names=("scorer", "bodyparts", "coords"),
    )
    arr = rng.uniform(0.0, 1000.0, size=(n_frames, len(columns)))
    df = pd.DataFrame(arr, columns=columns)

    if nan_mask is not None:
        for i_label, label in enumerate(labels):
            absent = nan_mask[:, i_label]
            df.loc[absent, (scorer, label, "x")] = np.nan
            df.loc[absent, (scorer, label, "y")] = np.nan

    return df


# ---------------------------------------------------------------------------
# Parity tests
# ---------------------------------------------------------------------------


def _assert_dicts_equal(d_legacy: dict, d_vec: dict) -> None:
    """Deep-equality check that survives float ndarray-vs-list and key-type
    differences between the two paths.
    """
    assert set(d_legacy.keys()) == set(d_vec.keys()), (
        f"label sets differ: legacy={list(d_legacy.keys())} "
        f"vec={list(d_vec.keys())}"
    )
    for label in d_legacy:
        legacy_frames = d_legacy[label]
        vec_frames = d_vec[label]
        # Frame keys may differ in numpy int vs python int -- coerce.
        legacy_keys = {int(k): v for k, v in legacy_frames.items()}
        vec_keys = {int(k): v for k, v in vec_frames.items()}
        assert set(legacy_keys.keys()) == set(vec_keys.keys()), (
            f"frame sets differ for label {label!r}: "
            f"legacy minus vec={set(legacy_keys) - set(vec_keys)}, "
            f"vec minus legacy={set(vec_keys) - set(legacy_keys)}"
        )
        for frame, legacy_val in legacy_keys.items():
            vec_val = vec_keys[frame]
            np.testing.assert_allclose(
                np.asarray(legacy_val), np.asarray(vec_val),
                err_msg=f"value mismatch at label={label!r} frame={frame}",
                rtol=0, atol=0,
            )


class TestParityNumericLabels:
    """Labels named ``"point0"`` ... ``"pointN"`` -- digit-stripped to
    canonical internal names ``"0"`` ... ``"N"``."""

    def test_dense_no_nans(self):
        df = _build_dlc_df(n_frames=100, labels=["point0", "point1"])
        d_legacy = _dlc_trace_to_annotation_dict_legacy(df)
        d_vec = VideoAnnotation._dlc_trace_to_annotation_dict(df)
        _assert_dicts_equal(d_legacy, d_vec)
        # Sanity: all 100 frames recorded for both labels.
        assert len(d_vec["0"]) == 100
        assert len(d_vec["1"]) == 100

    def test_partial_nans(self):
        n_frames = 200
        rng = np.random.default_rng(42)
        nan_mask = rng.random((n_frames, 3)) < 0.3  # ~30% drop rate
        df = _build_dlc_df(
            n_frames=n_frames,
            labels=["point0", "point1", "point2"],
            nan_mask=nan_mask,
        )
        d_legacy = _dlc_trace_to_annotation_dict_legacy(df)
        d_vec = VideoAnnotation._dlc_trace_to_annotation_dict(df)
        _assert_dicts_equal(d_legacy, d_vec)

    def test_all_nans_for_one_label(self):
        n_frames = 50
        nan_mask = np.zeros((n_frames, 2), dtype=bool)
        nan_mask[:, 1] = True  # label 1 totally absent
        df = _build_dlc_df(
            n_frames=n_frames,
            labels=["point0", "point1"],
            nan_mask=nan_mask,
        )
        d_legacy = _dlc_trace_to_annotation_dict_legacy(df)
        d_vec = VideoAnnotation._dlc_trace_to_annotation_dict(df)
        _assert_dicts_equal(d_legacy, d_vec)
        assert d_vec["1"] == {}, "label 1 should be empty dict"
        assert len(d_vec["0"]) == 50

    def test_single_frame(self):
        df = _build_dlc_df(n_frames=1, labels=["point0"])
        d_legacy = _dlc_trace_to_annotation_dict_legacy(df)
        d_vec = VideoAnnotation._dlc_trace_to_annotation_dict(df)
        _assert_dicts_equal(d_legacy, d_vec)

    def test_pia02_shape_2_labels(self):
        """Mimic a pia02 DLC trace: 36715-frame video, 2 labels, sparse
        NaNs from low-confidence DLC predictions. Asserts parity at the
        production shape that's actually hitting the slow path on disk.
        """
        n_frames = 36715
        rng = np.random.default_rng(123)
        nan_mask = rng.random((n_frames, 2)) < 0.05
        df = _build_dlc_df(
            n_frames=n_frames, labels=["point0", "point1"], nan_mask=nan_mask,
        )
        d_legacy = _dlc_trace_to_annotation_dict_legacy(df)
        d_vec = VideoAnnotation._dlc_trace_to_annotation_dict(df)
        _assert_dicts_equal(d_legacy, d_vec)


class TestParityNonNumericLabels:
    """Labels that don't fit the ``pointN`` pattern -- map to ordinal
    internal names by enumeration index, not by digit-stripping."""

    def test_named_labels(self):
        df = _build_dlc_df(
            n_frames=80,
            labels=["finger_distal", "wrist_dorsal"],
        )
        d_legacy = _dlc_trace_to_annotation_dict_legacy(df)
        d_vec = VideoAnnotation._dlc_trace_to_annotation_dict(df)
        _assert_dicts_equal(d_legacy, d_vec)
        # Internal naming should be "0" / "1" (enumeration index).
        assert set(d_vec.keys()) == {"0", "1"}

    def test_mixed_naming_falls_back(self):
        """If ANY label fails the ``stripped-prefix.isdigit()`` test,
        the whole batch falls back to enumeration. Documenting + parity-
        testing the all-or-nothing behaviour."""
        df = _build_dlc_df(
            n_frames=20,
            labels=["point0", "finger_distal"],
        )
        d_legacy = _dlc_trace_to_annotation_dict_legacy(df)
        d_vec = VideoAnnotation._dlc_trace_to_annotation_dict(df)
        _assert_dicts_equal(d_legacy, d_vec)
        assert set(d_vec.keys()) == {"0", "1"}


class TestEdgeCases:
    def test_value_type_is_list_of_floats(self):
        """Downstream consumers index ``data[label][frame]`` and expect
        a Python list ``[x, y]`` (not a numpy array). The legacy code
        called ``coord_val.tolist()``; the vectorised version must
        preserve that contract."""
        df = _build_dlc_df(n_frames=5, labels=["point0"])
        d_vec = VideoAnnotation._dlc_trace_to_annotation_dict(df)
        for frame, val in d_vec["0"].items():
            assert isinstance(val, list), (
                f"frame {frame}: expected list, got {type(val).__name__}"
            )
            assert len(val) == 2
            assert all(isinstance(x, float) for x in val)

    def test_frame_keys_are_python_ints(self):
        """Legacy used ``df.index.values`` which yields numpy ints when
        the index is an integer range; ``frame in dict`` lookups elsewhere
        in DUSTrack key on plain ints. The vectorised version coerces."""
        df = _build_dlc_df(n_frames=10, labels=["point0"])
        d_vec = VideoAnnotation._dlc_trace_to_annotation_dict(df)
        for frame in d_vec["0"]:
            assert isinstance(frame, int), (
                f"expected python int, got {type(frame).__name__}"
            )


# ---------------------------------------------------------------------------
# Speedup smoke -- not a hard assertion, just an informational benchmark.
# Confirms the vectorised path is at least as fast as legacy; the real
# numbers come from tests/qt_learning/24_benchmark_cold_open.py.
# ---------------------------------------------------------------------------


def test_vectorised_at_least_as_fast(benchmark_pia02_shape: pd.DataFrame):
    """Sanity check: on a pia02-shaped frame, the vectorised version is
    not slower than the legacy. Not a hard guarantee (single sample,
    noisy wall-clock), but a regression alarm if the vectorise ever
    breaks."""
    import time
    df = benchmark_pia02_shape

    t0 = time.perf_counter()
    d_legacy = _dlc_trace_to_annotation_dict_legacy(df)
    t1 = time.perf_counter()
    d_vec = VideoAnnotation._dlc_trace_to_annotation_dict(df)
    t2 = time.perf_counter()

    legacy_ms = (t1 - t0) * 1000.0
    vec_ms = (t2 - t1) * 1000.0
    print(f"\nlegacy={legacy_ms:.1f} ms  vec={vec_ms:.1f} ms  "
          f"speedup={legacy_ms / max(vec_ms, 1e-9):.1f}x")
    # Parity must hold.
    _assert_dicts_equal(d_legacy, d_vec)
    # Allow some slack (CI noise); fail only on egregious regression.
    assert vec_ms < legacy_ms * 1.5, (
        f"vectorised path is unexpectedly slow: "
        f"{vec_ms:.1f} ms vs legacy {legacy_ms:.1f} ms"
    )


@pytest.fixture
def benchmark_pia02_shape() -> pd.DataFrame:
    """Pia02-shaped DLC trace: 36715 frames x 2 labels with sparse NaNs."""
    rng = np.random.default_rng(2026)
    n_frames = 36715
    nan_mask = rng.random((n_frames, 2)) < 0.05
    return _build_dlc_df(
        n_frames=n_frames,
        labels=["point0", "point1"],
        nan_mask=nan_mask,
    )

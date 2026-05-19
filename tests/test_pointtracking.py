import pytest
import os
from pathlib import Path
import numpy as np
import pandas as pd
import json
from unittest.mock import patch, MagicMock

import matplotlib.pyplot as plt

import datanavigator
from datanavigator.examples import get_example_video
from tests.conftest import (
    simulate_key_press,
    simulate_key_press_at_xy,
    simulate_mouse_click,
)


@pytest.fixture(scope="module")
def video_fname(tmp_path_factory):
    return get_example_video(dest_folder=str(tmp_path_factory.getbasetemp()))


@pytest.fixture(scope="module")
def ann_fname(video_fname):
    """Fixture to create a temporary JSON file with annotations in the first 10 frames, 3 labels per frame."""
    video = datanavigator.Video(video_fname)
    height, width = video[0].shape[:2]
    ann = datanavigator.VideoAnnotation(vname=video_fname)
    for frame_count in range(10):
        for label in range(3):
            ann.add(
                location=[
                    width * ((frame_count + 10) / 30) + 5 * label,
                    height * ((frame_count + 10) / 30),
                ],
                label=str(label),
                frame_number=frame_count,
            )
    ann.save()
    # load it again to get rid of the empty labels
    ann = datanavigator.VideoAnnotation(fname=ann.fname)
    ann.save()
    return ann.fname


@pytest.fixture(scope="module")
def ann2_fname(video_fname):
    """Fixture to create a second temporary JSON file with 9 annotated frames, 3 labels per frame."""
    video = datanavigator.Video(video_fname)
    height, width = video[0].shape[:2]
    ann2 = datanavigator.VideoAnnotation()
    for frame_count in range(9):
        for label in range(3):
            ann2.add(
                location=[
                    width * ((frame_count - 10) / 30) + 5 * label,
                    height * ((frame_count - 10) / 30),
                ],
                label=str(label),
                frame_number=frame_count * 2 + 50,
            )
    ann2.fname = os.path.join(
        Path(video_fname).parent, Path(video_fname).stem + "_annotations_pn.json"
    )
    ann2.save()
    # loading removes empty labels
    ann2 = datanavigator.VideoAnnotation(fname=ann2.fname)
    ann2.save()
    return ann2.fname


@pytest.fixture(scope="module")
def ann_h5_fname(ann_fname):
    """Fixture to create a temporary HDF5 file of ann."""
    ann = datanavigator.VideoAnnotation(fname=ann_fname)
    ann.to_dlc(save=True)
    return str(Path(ann.fname).with_suffix(".h5"))


@pytest.fixture(scope="function")
def ann_object(ann_fname):
    """Fixture returning a VideoAnnotation object."""
    return datanavigator.VideoAnnotation(fname=ann_fname)


@pytest.fixture(scope="module")
def ann_object_no_video(ann_fname):
    """Fixture returning a VideoAnnotation object without an associated video."""
    # Create a dummy json that doesn't match video name pattern
    data = datanavigator.VideoAnnotation._load_json(ann_fname)
    dummy_fname = Path(ann_fname).parent / "dummy_no_video_annotations.json"
    with open(dummy_fname, "w") as f:
        json.dump(data, f)
    ann = datanavigator.VideoAnnotation(fname=str(dummy_fname))
    yield ann
    os.remove(dummy_fname)  # Clean up dummy file


@pytest.fixture(scope="function")
def ann_object_overlapping(video_fname):
    """Fixture with annotations where all labels overlap for some frames."""
    video = datanavigator.Video(video_fname)
    height, width = video[0].shape[:2]
    ann = datanavigator.VideoAnnotation(vname=video_fname, name="overlapping")
    # Frames 5, 6, 7 have all 3 labels
    for frame_count in range(5, 8):
        for label in range(3):
            ann.add(
                location=[
                    width * ((frame_count + 10) / 30) + 5 * label,
                    height * ((frame_count + 10) / 30),
                ],
                label=str(label),
                frame_number=frame_count,
            )
    # Frame 8 has only label 0
    ann.add(location=[10, 10], label="0", frame_number=8)
    # Frame 9 has labels 0 and 1
    ann.add(location=[10, 10], label="0", frame_number=9)
    ann.add(location=[20, 20], label="1", frame_number=9)
    ann.save()
    # load it again to get rid of the empty labels and ensure correct state
    ann = datanavigator.VideoAnnotation(fname=ann.fname)
    yield ann
    os.remove(ann.fname)  # Clean up


def test_video_annotation_init_empty(tmp_path):
    # test empty initialization
    ann = datanavigator.VideoAnnotation()
    assert ann.fname is None
    assert ann.fstem is None
    assert ann.name == "noname"
    assert ann.video is None
    assert len(ann.data) == len(ann.palette)
    assert all(
        x in ann.plot_handles
        for x in ["ax_list_scatter", "ax_list_trace_x", "ax_list_trace_y"]
    )
    with pytest.raises(AssertionError):
        ann.save()


def test_video_annotation_init_with_video(video_fname):
    video_folder = str(Path(video_fname).parent)
    vstem = Path(video_fname).stem

    # vname only
    ann = datanavigator.VideoAnnotation(vname=video_fname)
    assert ann.fname == os.path.join(video_folder, f"{vstem}_annotations.json")
    assert ann.fstem == f"{vstem}_annotations"
    assert ann.name == "noname"  # default name
    assert ann.video.fname == video_fname


def test_video_annotation_init_with_video_2(video_fname):
    """Supply fname and vname, and omit name"""
    video_folder = str(Path(video_fname).parent)
    vstem = Path(video_fname).stem
    ann_fname = os.path.join(video_folder, f"{vstem}_annotations_pn.json")

    # fname and vname
    ann = datanavigator.VideoAnnotation(fname=ann_fname, vname=video_fname)
    assert ann.fname == os.path.join(video_folder, f"{vstem}_annotations_pn.json")
    assert ann.fstem == f"{vstem}_annotations_pn"
    assert ann.name == "pn"
    assert ann.video.fname == video_fname

    # fname only - find the video
    ann = datanavigator.VideoAnnotation(fname=ann_fname)
    assert ann.fname == os.path.join(video_folder, f"{vstem}_annotations_pn.json")
    assert ann.fstem == f"{vstem}_annotations_pn"
    assert ann.name == "pn"
    assert ann.video.fname == video_fname

    # fname only without "_annotations" in it - this should not be allowed in the future
    ann_fname = os.path.join(video_folder, f"{vstem}_myann.json")
    ann = datanavigator.VideoAnnotation(fname=ann_fname)
    assert ann.fname == ann_fname
    assert ann.fstem == Path(ann_fname).stem
    assert ann.name == "noname"
    assert ann.video is None


def test_video_annotation_init_with_video_3(video_fname):
    """Supply vname and name, and omit fname"""
    video_folder = str(Path(video_fname).parent)
    vstem = Path(video_fname).stem

    # vname and name
    ann = datanavigator.VideoAnnotation(vname=video_fname, name="pn")
    assert ann.fname == os.path.join(video_folder, f"{vstem}_annotations_pn.json")
    assert ann.fstem == f"{vstem}_annotations_pn"
    assert ann.name == "pn"
    assert ann.video.fname == video_fname


def test_video_annotation_load(ann_fname, ann2_fname, ann_h5_fname):
    ann2 = datanavigator.VideoAnnotation(fname=ann2_fname)
    assert ann2.fname == ann2_fname
    assert ann2.fstem == Path(ann2_fname).stem
    assert ann2.name == "pn"
    assert len(ann2.data) == 3
    assert len(ann2.data["0"]) == 9

    # test loading from h5 file
    ann = datanavigator.VideoAnnotation(fname=ann_fname)
    ann_h5 = datanavigator.VideoAnnotation(fname=ann_h5_fname)
    assert ann_h5.fname == ann_h5_fname
    assert ann_h5.fstem == Path(ann_h5_fname).stem
    assert ann_h5.name == "noname"
    assert all([np.allclose(ann.data["0"][i], ann_h5.data["0"][i]) for i in range(10)])


def test_video_annotation_from_multiple_files(
    video_fname, ann_fname, ann2_fname, ann_h5_fname
):
    """Test loading annotations from multiple files."""
    fname_merged = os.path.join(
        Path(video_fname).parent, Path(video_fname).stem + "_annotations_merged.json"
    )
    ann = datanavigator.VideoAnnotation.from_multiple_files(
        fname_list=[ann_fname, ann2_fname],
        vname=video_fname,
        name="merged",
        fname_merged=fname_merged,
    )
    assert ann.fname == fname_merged
    assert len(ann.data) == 3
    assert len(ann.data["0"]) == 19  # 10 + 9 frames

    # test loading from h5 file
    ann_2 = datanavigator.VideoAnnotation.from_multiple_files(
        fname_list=[ann_h5_fname, ann2_fname],
        vname=video_fname,
        name="merged2",
        fname_merged=fname_merged,
    )
    assert ann_2.fname == fname_merged
    assert ann_2.name == "merged2"
    assert len(ann_2.data) == 3
    assert len(ann_2.data["0"]) == 19  # 10 + 9 frames
    assert all([np.allclose(ann.data["0"][i], ann_2.data["0"][i]) for i in range(10)])


def test_video_annotation_properties(
    ann_object, ann_object_no_video, ann_object_overlapping
):
    # n_frames
    assert ann_object.n_frames == len(ann_object.video)
    assert ann_object_no_video.n_frames == 10  # max frame number + 1

    # n_annotations (number of labels)
    assert ann_object.n_annotations == 3
    assert len(ann_object) == 3

    # labels
    assert ann_object.labels == ["0", "1", "2"]

    # frames
    assert ann_object.frames == list(range(10))
    assert ann_object_overlapping.frames == [5, 6, 7, 8, 9]

    # frames_overlapping
    assert ann_object_overlapping.frames_overlapping == [5, 6, 7]
    # Test case with no overlapping frames
    ann_no_overlap = datanavigator.VideoAnnotation()
    ann_no_overlap.data = {}
    ann_no_overlap.add_label("0")
    ann_no_overlap.add_label("1")
    ann_no_overlap.add([1, 1], "0", 0)
    ann_no_overlap.add([1, 1], "1", 1)
    assert ann_no_overlap.frames_overlapping == []


def test_video_annotation_get_frames(ann_object):
    assert ann_object.get_frames("0") == list(range(10))
    assert ann_object.get_frames("1") == list(range(10))
    assert ann_object.get_frames("2") == list(range(10))
    with pytest.raises(AssertionError):
        ann_object.get_frames("nonexistent")


def test_video_annotation_save_errors(ann_object, tmp_path):
    with pytest.raises(ValueError, match="Supply a json file name."):
        ann_object.save(fname=tmp_path / "test.h5")


def test_video_annotation_get_values_cv(ann_object):
    vals = ann_object.get_values_cv(5)
    assert isinstance(vals, np.ndarray)
    assert vals.dtype == np.float32
    assert vals.shape == (3, 1, 2)
    assert not np.isnan(vals).any()

    # Test frame with missing annotations (should not happen in ann_object)
    ann = datanavigator.VideoAnnotation()
    ann.data = {}
    ann.add_label("0")
    ann.add_label("1")
    ann.add([10, 10], "0", 0)
    vals_missing = ann.get_values_cv(0)
    assert vals_missing.shape == (2, 1, 2)
    assert np.isnan(vals_missing[1]).all()


def test_video_annotation_frame_num_str(ann_object, ann_object_no_video):
    assert ann_object._n_digits_in_frame_num() == str(len(str(ann_object.n_frames)))
    assert (
        ann_object._frame_num_as_str(5) == f"{5:0{ann_object._n_digits_in_frame_num()}}"
    )
    assert ann_object_no_video._n_digits_in_frame_num() == str(
        len(str(ann_object_no_video.n_frames))
    )
    assert (
        ann_object_no_video._frame_num_as_str(5)
        == f"{5:0{ann_object_no_video._n_digits_in_frame_num()}}"
    )


def test_video_annotation_add_get_at_frame(ann_object):
    frame_num = 15
    values = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    ann_object.add_at_frame(frame_num, values)
    retrieved = ann_object.get_at_frame(frame_num)
    assert retrieved == values
    assert ann_object.data["0"][frame_num] == values[0]
    assert ann_object.data["1"][frame_num] == values[1]
    assert ann_object.data["2"][frame_num] == values[2]

    # Test get_at_frame with missing values
    retrieved_missing = ann_object.get_at_frame(frame_num + 1)
    assert len(retrieved_missing) == 3
    assert all(np.isnan(p).all() for p in retrieved_missing)


def test_video_annotation_getitem(ann_object):
    # Get label data
    assert isinstance(ann_object["0"], dict)
    assert 5 in ann_object["0"]

    # Get frame data
    frame_data = ann_object[5]
    assert isinstance(frame_data, list)
    assert len(frame_data) == 3
    assert frame_data[0] == ann_object.data["0"][5]

    # Get invalid key
    with pytest.raises(ValueError):
        _ = ann_object["invalid_key"]
    with pytest.raises(ValueError):
        _ = ann_object[ann_object.n_frames + 100]  # Non-existent frame


def test_video_annotation_to_dlc_options(ann_object, tmp_path):
    # Test save=False
    df_no_save = ann_object.to_dlc(save=False)
    assert isinstance(df_no_save, pd.DataFrame)
    assert len(df_no_save) == 10  # 10 frames

    # Test custom output path and prefix
    custom_path = tmp_path / "custom_dlc"
    if not os.path.exists(custom_path):
        os.mkdir(custom_path)
    custom_prefix = "my_prefix"
    df_custom = ann_object.to_dlc(
        output_path=custom_path, file_prefix=custom_prefix, save=True
    )
    assert (custom_path / f"{custom_prefix}.h5").exists()
    assert (custom_path / f"{custom_prefix}.csv").exists()
    pd.testing.assert_frame_equal(df_no_save, df_custom)

    # Test dlc prefix
    df_dlc_prefix = ann_object.to_dlc(
        output_path=custom_path, file_prefix="dlc", scorer="tester", save=True
    )
    assert (custom_path / "CollectedData_tester.h5").exists()
    assert (custom_path / "CollectedData_tester.csv").exists()

    # Test internal_to_dlc_labels
    label_map = {"0": "head", "1": "tail", "2": "mid"}
    df_mapped = ann_object.to_dlc(internal_to_dlc_labels=label_map, save=False)
    assert "head" in df_mapped.columns.get_level_values("bodyparts")
    assert "tail" in df_mapped.columns.get_level_values("bodyparts")
    assert "mid" in df_mapped.columns.get_level_values("bodyparts")
    assert "point0" not in df_mapped.columns.get_level_values("bodyparts")


def test_video_annotation_to_dlc_populates_values(ann_object):
    """Regression guard: to_dlc must populate values for every annotation.

    Catches the pandas 3.0 Copy-on-Write failure mode where the old
    chained-assignment ``df.loc[row][col] = val`` silently no-ops and
    leaves the dataframe full of NaN. With the single-call ``df.loc[row, col]``
    fix shipped in 1.2.0, every annotated (label, frame) must surface as
    finite x and y values in the DataFrame.
    """
    df = ann_object.to_dlc(save=False)
    scorer = df.columns.levels[0].values[0]
    index_length = ann_object._n_digits_in_frame_num()
    for label in ann_object.labels:
        for frame, xy in ann_object.data[label].items():
            img_stem = f"img{frame:0{index_length}}.png"
            x_val = df.loc[
                ("labeled-data", ann_object.video.name, img_stem),
                (scorer, f"point{label}", "x"),
            ]
            y_val = df.loc[
                ("labeled-data", ann_object.video.name, img_stem),
                (scorer, f"point{label}", "y"),
            ]
            assert not np.isnan(x_val), f"label={label} frame={frame} x was NaN"
            assert not np.isnan(y_val), f"label={label} frame={frame} y was NaN"
            assert x_val == xy[0]
            assert y_val == xy[1]


REAL_DLC_H5 = Path(
    r"S:\2201000537 - Operator\data_opr02\005_02\ml_models\dlc"
    r"\bi_4pt_t009-pnlk-2024-03-27\labeled-data\opr02_s005_t009_us_b_007"
    r"\CollectedData_pnlk.h5"
)
REAL_DLC_VIDEO = Path(
    r"S:\2201000537 - Operator\data_opr02\005_02\ml_models\dlc"
    r"\bi_4pt_t009-pnlk-2024-03-27\videos\opr02_s005_t009_us_b_007.mp4"
)


@pytest.mark.skipif(
    not (REAL_DLC_H5.exists() and REAL_DLC_VIDEO.exists()),
    reason="Real DLC fixture from the OPR02 study is not reachable on this host",
)
def test_video_annotation_to_dlc_real_data_roundtrip(tmp_path):
    """Real-data DLC h5 round-trip: load -> to_dlc -> reload -> compare.

    Uses the OPR02 4-point DLC project as a load-bearing real-world
    fixture (438 frames × 4 labels = 1752 annotations). Skipped on
    hosts without S: drive access; meaningful only locally + on
    Praneeth's machine, but it pins the 1.2.0 to_dlc fix against a
    real-shape DLC labeled-data file (multi-bodypart, full coverage)
    that the synthetic fixture above can't replicate.
    """
    ann_orig = datanavigator.VideoAnnotation(
        fname=str(REAL_DLC_H5), vname=str(REAL_DLC_VIDEO)
    )
    n_orig = sum(len(ann_orig.data[lbl]) for lbl in ann_orig.labels)
    assert n_orig > 0, "real DLC fixture loaded zero annotations"
    df_orig = ann_orig.to_dlc(save=False)
    # Spot-check: no NaN where the source dict has values.
    scorer = df_orig.columns.levels[0].values[0]
    index_length = ann_orig._n_digits_in_frame_num()
    for label in ann_orig.labels:
        for frame in ann_orig.data[label]:
            img_stem = f"img{frame:0{index_length}}.png"
            x_val = df_orig.loc[
                ("labeled-data", ann_orig.video.name, img_stem),
                (scorer, f"point{label}", "x"),
            ]
            assert not np.isnan(x_val), f"chained-assignment regression: ({label}, {frame}) is NaN"
    # Full round-trip via a temporary h5.
    out_dir = tmp_path / "roundtrip"
    out_dir.mkdir()
    ann_orig.to_dlc(output_path=str(out_dir), file_prefix="roundtrip", save=True)
    ann_back = datanavigator.VideoAnnotation(
        fname=str(out_dir / "roundtrip.h5"), vname=str(REAL_DLC_VIDEO)
    )
    n_back = sum(len(ann_back.data[lbl]) for lbl in ann_back.labels)
    assert n_back == n_orig, f"round-trip lost annotations: {n_orig} -> {n_back}"
    for label in ann_orig.labels:
        assert label in ann_back.data
        for frame, xy in ann_orig.data[label].items():
            np.testing.assert_allclose(ann_back.data[label][frame], xy)


def test_video_annotation_to_traces(ann_object):
    traces = ann_object.to_traces()
    assert isinstance(traces, dict)
    assert set(traces.keys()) == set(ann_object.labels)
    assert traces["0"].shape == (ann_object.n_frames, 2)
    assert traces["1"].shape == (ann_object.n_frames, 2)
    assert traces["2"].shape == (ann_object.n_frames, 2)
    assert not np.isnan(traces["0"][:10]).any()  # First 10 frames are annotated
    assert np.isnan(traces["0"][10:]).all()  # Rest should be NaN


@patch(
    "datanavigator.pointtracking.pysampled", create=True
)  # Mock pysampled if not installed or for isolation
def test_video_annotation_to_signal(mock_pysampled, ann_object, ann_object_no_video):
    mock_data = MagicMock()
    mock_pysampled.Data.return_value = mock_data

    # Test with video
    signal_0 = ann_object.to_signal("0")
    mock_pysampled.Data.assert_called_once()
    call_args, call_kwargs = mock_pysampled.Data.call_args
    np.testing.assert_array_equal(call_args[0], ann_object.to_trace("0"))
    assert call_kwargs["sr"] == ann_object.video.get_avg_fps()
    assert signal_0 == mock_data

    # Test multiple signals
    mock_pysampled.Data.reset_mock()
    signals = ann_object.to_signals()
    assert mock_pysampled.Data.call_count == 3
    assert set(signals.keys()) == set(ann_object.labels)
    assert signals["0"] == mock_data
    assert signals["1"] == mock_data
    assert signals["2"] == mock_data

    # Test without video (should raise error)
    mock_pysampled.Data.reset_mock()
    with pytest.raises(AssertionError):
        ann_object_no_video.to_signal("0")
    with pytest.raises(AssertionError):
        ann_object_no_video.to_signals()


def test_video_annotation_add_label():
    ann = datanavigator.VideoAnnotation()
    assert ann.labels == [str(i) for i in range(10)]  # Starts with 10 empty labels

    # Add data to existing label
    ann.add([1, 1], "0", 0)
    assert ann.data["0"] == {0: [1, 1]}

    # Create a new instance to test adding labels beyond initial empty ones
    ann_new = datanavigator.VideoAnnotation()
    ann_new.data = {}  # Start truly empty for this test

    # Add label automatically
    ann_new.add_label()
    assert ann_new.labels == ["0"]
    assert ann_new.data["0"] == {}
    assert ann_new.palette[list(ann_new.data.keys()).index("0")] is not None  # Color assigned

    # Add specific label
    ann_new.add_label("5")
    assert ann_new.labels == ["0", "5"]
    assert ann_new.data["5"] == {}
    assert ann_new.palette[list(ann_new.data.keys()).index("0")] is not None

    # Add with specific color
    ann_new.add_label("2", color=(0.1, 0.2, 0.3))
    assert ann_new.labels == ["0", "2", "5"]
    assert ann_new.data["2"] == {}

    # Add existing label (no longer fails)
    ann_new.add_label("0")

    # Add invalid label (non-digit string)
    with pytest.raises(AssertionError):
        ann_new.add_label("a")

    # Add more than 10 labels (fill up first)
    for i in [1, 3, 4, 6, 7, 8, 9]:
        ann_new.add_label(str(i))
    assert len(ann_new.labels) == 10
    ann_new.add_label()  # Cannot add 11th label automatically
    assert "10" in ann_new.labels  # Check that it was added as a new label


def test_video_annotation_remove(ann_object):
    assert 5 in ann_object.data["0"]
    ann_object.remove("0", 5)
    assert 5 not in ann_object.data["0"]
    # Remove non-existent frame (should not fail)
    ann_object.remove("0", 500)
    # Remove non-existent label (should fail)
    with pytest.raises(AssertionError):
        ann_object.remove("nonexistent", 0)


def test_video_annotation_revision_bumps_on_mutation(video_fname):
    """``_revision`` must bump on every operation that changes data.

    VideoPointAnnotator.update_frame_marker /
    VideoAnnotation.update_display_trace key per-frame caches on this
    counter; an under-counted mutation would cause the cache to serve
    stale arrays. Test each public mutation method explicitly.
    """
    ann = datanavigator.VideoAnnotation(vname=video_fname)
    ann.add_label("0")
    ann.add_label("1")
    base = ann._revision

    ann.add(location=[10.0, 20.0], label="0", frame_number=5)
    assert ann._revision > base, "add() must bump _revision"
    r1 = ann._revision

    ann.remove(label="0", frame_number=5)
    assert ann._revision > r1, "remove() must bump _revision"
    r2 = ann._revision

    ann.add(location=[1.0, 1.0], label="0", frame_number=1)
    ann.add(location=[1.0, 1.0], label="1", frame_number=1)
    r3 = ann._revision

    ann.sort_labels()
    assert ann._revision > r3, "sort_labels() must bump _revision"
    r4 = ann._revision

    ann.sort_data()
    assert ann._revision > r4, "sort_data() must bump _revision"
    r5 = ann._revision

    ann.add_at_frame(2, np.full((ann.n_annotations, 2), 3.0))
    assert ann._revision > r5, "add_at_frame() must bump _revision"
    r6 = ann._revision

    ann.clip_labels(start_frame=0, end_frame=10)
    assert ann._revision > r6, "clip_labels() must bump _revision"


def test_video_annotation_clip_labels(ann_object_overlapping):
    # Original frames: [5, 6, 7, 8, 9]
    ann_object_overlapping.clip_labels(start_frame=6, end_frame=8)
    assert ann_object_overlapping.frames == [6, 7, 8]
    assert 5 not in ann_object_overlapping.data["0"]
    assert 9 not in ann_object_overlapping.data["0"]
    assert 6 in ann_object_overlapping.data["0"]
    assert 7 in ann_object_overlapping.data["0"]
    assert 8 in ann_object_overlapping.data["0"]
    assert 6 in ann_object_overlapping.data["1"]  # Check other labels too
    assert 7 in ann_object_overlapping.data["1"]
    assert 8 not in ann_object_overlapping.data["1"]  # Label 1 didn't exist at frame 8


def test_video_annotation_keep_overlapping_continuous_frames(ann_object_overlapping):
    # Original frames: [5, 6, 7, 8, 9]
    # Overlapping frames: [5, 6, 7]
    # Continuous overlapping: [5, 6], [6, 7] -> keep 5, 6, 7
    assert set(ann_object_overlapping.frames) == set([5, 6, 7, 8, 9])
    ann_object_overlapping.keep_overlapping_continuous_frames()
    assert set(ann_object_overlapping.frames) == set([5, 6, 7])
    assert 8 not in ann_object_overlapping.data["0"]
    assert 9 not in ann_object_overlapping.data["0"]
    assert 5 in ann_object_overlapping.data["0"]
    assert 6 in ann_object_overlapping.data["1"]
    assert 7 in ann_object_overlapping.data["2"]

    # Test case with no continuous overlapping frames
    ann = datanavigator.VideoAnnotation()
    ann.data = {}
    ann.add_label("0")
    ann.add_label("1")
    ann.add([1, 1], "0", 0)
    ann.add([1, 1], "1", 0)
    ann.add([1, 1], "0", 2)
    ann.add([1, 1], "1", 2)
    ann.keep_overlapping_continuous_frames()  # Should print warning and do nothing
    assert ann.frames == [0, 2]


def test_video_annotation_keep_overlapping_frames(ann_object_overlapping):
    # Original frames: [5, 6, 7, 8, 9]; overlapping: [5, 6, 7]
    assert set(ann_object_overlapping.frames) == {5, 6, 7, 8, 9}
    assert ann_object_overlapping.frames_overlapping == [5, 6, 7]
    ann_object_overlapping.keep_overlapping_frames()
    assert set(ann_object_overlapping.frames) == {5, 6, 7}
    for label in ann_object_overlapping.labels:
        assert 8 not in ann_object_overlapping.data[label]
        assert 9 not in ann_object_overlapping.data[label]


def test_video_annotation_keep_overlapping_frames_keeps_non_continuous():
    """Distinguishes keep_overlapping_frames from the continuous variant:
    isolated fully-labeled frames survive here, but are dropped by
    keep_overlapping_continuous_frames."""
    ann = datanavigator.VideoAnnotation()
    ann.data = {}
    ann.add_label("0")
    ann.add_label("1")
    # Frames 5 and 10 fully labeled but non-consecutive
    ann.add([1, 1], "0", 5)
    ann.add([2, 2], "1", 5)
    ann.add([3, 3], "0", 10)
    ann.add([4, 4], "1", 10)
    # Frame 7 incomplete (label 0 only)
    ann.add([5, 5], "0", 7)
    assert ann.frames == [5, 7, 10]
    ann.keep_overlapping_frames()
    assert ann.frames == [5, 10]


def test_video_annotation_keep_overlapping_frames_no_overlap_aborts():
    """No fully-labeled frame -> abort, data untouched (mirrors the
    continuous variant's safeguard)."""
    ann = datanavigator.VideoAnnotation()
    ann.data = {}
    ann.add_label("0")
    ann.add_label("1")
    ann.add([1, 1], "0", 0)
    ann.add([1, 1], "1", 5)
    ann.keep_overlapping_frames()
    assert ann.frames == [0, 5]


@patch("datanavigator.pointtracking.pysampled", create=True)  # Mock pysampled
def test_video_annotation_get_area(mock_pysampled, ann_object, ann_object_no_video):
    mock_data = MagicMock()
    mock_pysampled.Data.return_value = mock_data

    # Test with video -> returns pysampled.Data
    area_signal = ann_object.get_area(labels=["0", "1", "2"])
    assert area_signal == mock_data
    mock_pysampled.Data.assert_called_once()
    call_args, call_kwargs = mock_pysampled.Data.call_args
    assert call_args[0].shape == (ann_object.n_frames,)
    assert call_kwargs["sr"] == ann_object.video.get_avg_fps()

    # Test with string labels
    mock_pysampled.Data.reset_mock()
    area_signal_str = ann_object.get_area(labels="012")
    assert area_signal_str == mock_data
    mock_pysampled.Data.assert_called_once()

    # Test with lowpass
    mock_pysampled.Data.reset_mock()
    mock_signal = MagicMock()
    mock_signal.lowpass.return_value.return_value = ann_object.to_trace(
        "0"
    )  # Mock lowpass output
    with patch.object(
        ann_object,
        "to_signals",
        return_value={"0": mock_signal, "1": mock_signal, "2": mock_signal},
    ):
        area_signal_lp = ann_object.get_area(labels="012", lowpass=5)
        assert area_signal_lp == mock_data
        mock_signal.lowpass.assert_called_with(5)
        mock_pysampled.Data.assert_called_once()

    # Test without video -> returns np.ndarray
    mock_pysampled.Data.reset_mock()
    area_array = ann_object_no_video.get_area(labels=["0", "1", "2"])
    assert isinstance(area_array, np.ndarray)
    assert area_array.shape == (ann_object_no_video.n_frames,)
    mock_pysampled.Data.assert_not_called()

    # Test invalid label
    with pytest.raises(AssertionError):
        ann_object.get_area(labels=["0", "invalid"])


def test_video_annotation_init_preloaded(ann_fname):
    # Load data first
    preloaded_data = datanavigator.VideoAnnotation._load_json(ann_fname)
    # Init with preloaded data
    ann = datanavigator.VideoAnnotation(preloaded_json=preloaded_data)
    assert ann.fname is None  # No fname provided
    assert ann.video is None  # No vname provided
    assert ann.name == "noname"
    assert ann.data == preloaded_data
    assert len(ann.labels) == 3


def test_video_annotation_display_setup(video_fname):
    # Use spec=matplotlib.axes.Axes to make isinstance checks pass
    fig, (ax_img, ax_x, ax_y) = plt.subplots(3, 1)
    ann = datanavigator.VideoAnnotation(vname=video_fname, name="test_display_setup")

    # Setup scatter
    ann.setup_display_scatter(ax_list_scatter=[ax_img])
    assert "labels_in_ax0" in ann.plot_handles

    # Setup trace
    ann.setup_display_trace(ax_list_trace_x=[ax_x], ax_list_trace_y=[ax_y])
    # Called 10 times for x and 10 times for y (for labels 0-9)
    assert "trace_in_axx0_label0" in ann.plot_handles
    assert "trace_in_axy0_label0" in ann.plot_handles

    plt.close(fig)


def test_setup_display_trace_xlim_guard(video_fname):
    """`setup_display_trace` must claim xlim only when no one else has.

    Bug B in the 2026-05-19 trace-scaling audit: pre-fix, every
    `VideoAnnotation` constructed on the same axes ran
    `ax_x.set_xlim(0, n_frames)`, wiping any user pan/zoom or any
    prior layer's xlim claim. Each in-session `_adopt_layer` call
    (DUSTrack Reduce jitter, post-train DLC refresh, Apply manual
    corrections first-time) blew away the trace time-axis.

    The fix guards the call with ``ax_x.get_autoscalex_on()``: the
    first annotation claims xlim (flipping autoscale off); subsequent
    setups are a no-op.
    """
    # First annotation claims xlim.
    fig, (ax_img, ax_x, ax_y) = plt.subplots(3, 1)
    ann1 = datanavigator.VideoAnnotation(vname=video_fname, name="ann1")
    ann1.setup_display(
        ax_list_scatter=[ax_img], ax_list_trace_x=[ax_x], ax_list_trace_y=[ax_y]
    )
    assert ax_x.get_xlim() == (0, ann1.n_frames), (
        f"First annotation should claim xlim=(0, n_frames); got {ax_x.get_xlim()}"
    )
    assert ax_x.get_autoscalex_on() is False, (
        "After claiming xlim, autoscalex_on should be False"
    )

    # User pans the trace x-axis after first setup.
    ax_x.set_xlim(20, 50)

    # Second annotation on the same axes must NOT clobber the user's pan.
    ann2 = datanavigator.VideoAnnotation(vname=video_fname, name="ann2")
    ann2.setup_display(
        ax_list_scatter=[ax_img], ax_list_trace_x=[ax_x], ax_list_trace_y=[ax_y]
    )
    assert ax_x.get_xlim() == (20, 50), (
        f"Second annotation must preserve user's xlim; got {ax_x.get_xlim()}"
    )

    plt.close(fig)


def test_video_point_annotator_xlim_preserved_across_layer_add(video_fname):
    """End-to-end: `_adopt_layer`-shaped flow preserves user's trace x-zoom.

    Concrete repro of the DUSTrack-side complaint that drove Bug B:
    open the annotator, pan/zoom the trace time-axis, then add a new
    annotation layer via `add_annotation_layers` (the exact call used
    by DUSTrack's `_adopt_layer` after Reduce jitter / DLC refresh).
    Pre-fix, the trace xlim resets to (0, n_frames). Post-fix, it
    stays where the user put it.
    """
    v = datanavigator.VideoPointAnnotator(vid_name=video_fname)
    n_frames = v.ann.n_frames

    # Initial setup leaves xlim at (0, n_frames).
    assert v._ax_trace_x.get_xlim() == (0, n_frames)

    # User pans to a sub-range.
    v._ax_trace_x.set_xlim(15, 40)

    # Mid-session layer add (the path DUSTrack._adopt_layer follows).
    v.add_annotation_layers({"smoothed": None})

    assert v._ax_trace_x.get_xlim() == (15, 40), (
        f"Mid-session layer add must preserve user's xlim; "
        f"got {v._ax_trace_x.get_xlim()}"
    )
    plt.close(v.figure)


def test_video_point_annotator_ylim_manual_policy(video_fname):
    """Manual y-policy: ylim refits once, then stays put across mutations.

    Bug A in the 2026-05-19 trace-scaling audit: pre-fix,
    `update_frame_marker`'s cache-miss branch unconditionally
    re-applied ``set_ylim(nanlim(trace_data))`` on every mutation,
    every label switch, every FOI toggle. The user's hand-zoomed
    trace y-view was wiped whenever they touched the data.

    Post-fix, the first cache miss with real data fits ylim (claiming
    autoscaley_on=False as a side effect); subsequent cache misses
    are no-ops. Pressing `r` (= `reset_axes("both")`) re-enables
    autoscale, restoring a one-shot refit.
    """
    v = datanavigator.VideoPointAnnotator(
        vid_name=video_fname, annotation_names=["test"]
    )
    # The annotator's __init__ runs update() once with the empty annotation;
    # all-NaN trace data leaves ylim at the matplotlib default.

    # First real annotation -> first cache miss with non-NaN data ->
    # one-shot fit (autoscale flips off).
    v.ann.add([100.0, 200.0], "0", 0)
    v.update()
    ylim_after_first_fit_x = v._ax_trace_x.get_ylim()
    ylim_after_first_fit_y = v._ax_trace_y.get_ylim()
    assert v._ax_trace_x.get_autoscaley_on() is False
    assert v._ax_trace_y.get_autoscaley_on() is False
    # Fit should bracket the data (allow margin from nanlim's *0.9 / *1.1).
    assert ylim_after_first_fit_x[0] <= 100.0 <= ylim_after_first_fit_x[1]
    assert ylim_after_first_fit_y[0] <= 200.0 <= ylim_after_first_fit_y[1]

    # User manually zooms y after the auto-fit.
    v._ax_trace_x.set_ylim(0, 500)
    v._ax_trace_y.set_ylim(0, 500)

    # Adding more annotations must NOT clobber the user's zoom, even
    # though the cache misses on _revision bump.
    v.ann.add([800.0, 900.0], "0", 1)  # Wildly outside (0, 500)
    v.update()
    assert v._ax_trace_x.get_ylim() == (0, 500), (
        f"Manual y-policy: user's ylim_x must survive mutation; "
        f"got {v._ax_trace_x.get_ylim()}"
    )
    assert v._ax_trace_y.get_ylim() == (0, 500), (
        f"Manual y-policy: user's ylim_y must survive mutation; "
        f"got {v._ax_trace_y.get_ylim()}"
    )

    # Label switch also causes a cache miss; ylim must still survive.
    v.statevariables["annotation_label"].set_state("1")
    v.update()
    assert v._ax_trace_x.get_ylim() == (0, 500)
    assert v._ax_trace_y.get_ylim() == (0, 500)

    # FOI toggle: another cache-miss path; ylim must still survive.
    v.frames_of_interest.append(0)
    v.update()
    assert v._ax_trace_x.get_ylim() == (0, 500)
    assert v._ax_trace_y.get_ylim() == (0, 500)

    # Pressing `r` re-enables autoscale, so the NEXT cache miss refits.
    v.reset_axes(axis="both")
    assert v._ax_trace_x.get_autoscaley_on() is True
    # Trigger a cache miss by mutating again.
    v.ann.add([50.0, 60.0], "0", 2)
    v.update()
    # ylim should have been refit (no longer (0, 500)).
    assert v._ax_trace_x.get_ylim() != (0, 500), (
        f"After `r`, next mutation should refit ylim; got {v._ax_trace_x.get_ylim()}"
    )
    plt.close(v.figure)


def test_r_keybinding_cursor_aware_dispatch(video_fname):
    """Cursor-aware ``r``: image-pane / trace / fallback, traces to full video.

    The Tier 2 ``r`` binding (``_reset_view_all``) dispatches on
    ``event.inaxes`` set by :meth:`_patch_event_for_image_pane`:

    - Cursor over ``_ax_image`` (Tier 2: aliased to ``_image_pane``) →
      ``_image_pane.reset_view()`` only; trace axes untouched.
    - Cursor over ``_ax_trace_x`` or ``_ax_trace_y`` → trace x reset to
      ``(0, n_frames)``, y autoscaled (via scoped ``reset_axes(axis="y",
      axes=[trace_x, trace_y])``); image pane untouched. Setting x to
      the full video range keeps frames outside the current annotation
      extent visible — the usual case when extending annotations to a
      new region. ``alt+r`` (the ``_reset_view_to_data_extent`` sibling)
      covers the "shrink to data extent" use case.
    - Cursor undetectable (``event is None`` / ``event.inaxes is None``)
      → reset everything: image pane reset AND the same trace treatment
      (x = full-video, y = autoscale).

    Agg-headless tests don't actually build a Qt image pane, so this
    test stubs ``_image_pane`` and ``_ax_image`` onto the annotator
    and intercepts ``reset_axes`` to record the scope kwarg per call.
    Trace ``set_xlim`` is NOT intercepted; we read back the real xlim
    to verify the full-video pin.
    """
    v = datanavigator.VideoPointAnnotator(vid_name=video_fname)
    n_frames = v.ann.n_frames

    mock_pane = MagicMock()
    v._image_pane = mock_pane
    # In fast_render, _ax_image is mirrored to _image_pane after super().__init__.
    v._ax_image = mock_pane

    reset_axes_calls = []
    orig_reset_axes = v.reset_axes

    def record_reset(axis="both", event=None, axes=None):
        reset_axes_calls.append({"axis": axis, "axes": axes})

    v.reset_axes = record_reset

    try:
        # Branch 1: cursor over image pane -> reset_view only, traces untouched.
        v._ax_trace_x.set_xlim(15, 40)
        event_img = simulate_key_press(v.figure, key="r", inaxes=mock_pane)
        v._reset_view_all(event_img)
        assert mock_pane.reset_view.call_count == 1
        assert reset_axes_calls == [], (
            f"Image-pane branch must not call reset_axes; got {reset_axes_calls}"
        )
        assert v._ax_trace_x.get_xlim() == (15, 40), (
            "Image-pane branch must NOT touch trace xlim"
        )

        # Branch 2a: cursor over _ax_trace_x -> xlim=(0, n_frames), reset_axes(y).
        event_tx = simulate_key_press(v.figure, key="r", inaxes=v._ax_trace_x)
        v._reset_view_all(event_tx)
        assert mock_pane.reset_view.call_count == 1, (
            "Trace branch must not touch the image pane"
        )
        assert len(reset_axes_calls) == 1
        assert reset_axes_calls[-1]["axes"] == [v._ax_trace_x, v._ax_trace_y]
        assert reset_axes_calls[-1]["axis"] == "y"
        assert v._ax_trace_x.get_xlim() == (0, n_frames), (
            f"Trace branch must pin xlim to (0, {n_frames}); "
            f"got {v._ax_trace_x.get_xlim()}"
        )

        # Branch 2b: cursor over _ax_trace_y -> same treatment.
        v._ax_trace_x.set_xlim(10, 20)
        event_ty = simulate_key_press(v.figure, key="r", inaxes=v._ax_trace_y)
        v._reset_view_all(event_ty)
        assert mock_pane.reset_view.call_count == 1
        assert len(reset_axes_calls) == 2
        assert reset_axes_calls[-1]["axes"] == [v._ax_trace_x, v._ax_trace_y]
        assert reset_axes_calls[-1]["axis"] == "y"
        assert v._ax_trace_x.get_xlim() == (0, n_frames)

        # Branch 3a: event is None -> reset everything, traces to full video.
        v._ax_trace_x.set_xlim(10, 20)
        v._reset_view_all(None)
        assert mock_pane.reset_view.call_count == 2
        assert len(reset_axes_calls) == 3
        assert reset_axes_calls[-1]["axes"] == [v._ax_trace_x, v._ax_trace_y]
        assert reset_axes_calls[-1]["axis"] == "y"
        assert v._ax_trace_x.get_xlim() == (0, n_frames)

        # Branch 3b: event with inaxes=None -> same fallback.
        v._ax_trace_x.set_xlim(10, 20)
        event_none = simulate_key_press(v.figure, key="r", inaxes=None)
        v._reset_view_all(event_none)
        assert mock_pane.reset_view.call_count == 3
        assert len(reset_axes_calls) == 4
        assert reset_axes_calls[-1]["axes"] == [v._ax_trace_x, v._ax_trace_y]
        assert reset_axes_calls[-1]["axis"] == "y"
        assert v._ax_trace_x.get_xlim() == (0, n_frames)
    finally:
        v.reset_axes = orig_reset_axes
        plt.close(v.figure)


def test_alt_r_keybinding_cursor_aware_data_extent_dispatch(video_fname):
    """Cursor-aware ``alt+r``: traces autoscale to data extent (sibling of ``r``).

    ``_reset_view_to_data_extent`` mirrors ``_reset_view_all``'s
    dispatch structure but the trace branch and the fallback autoscale
    both x and y to the data extent instead of pinning x to the full
    video range. Useful when the annotated region is much narrower than
    the full video and the user wants to zoom in to it.
    """
    v = datanavigator.VideoPointAnnotator(vid_name=video_fname)
    n_frames = v.ann.n_frames

    mock_pane = MagicMock()
    v._image_pane = mock_pane
    v._ax_image = mock_pane

    reset_axes_calls = []
    orig_reset_axes = v.reset_axes

    def record_reset(axis="both", event=None, axes=None):
        reset_axes_calls.append({"axis": axis, "axes": axes})

    v.reset_axes = record_reset

    try:
        # Branch 1: cursor over image pane -> reset_view only.
        v._ax_trace_x.set_xlim(15, 40)
        event_img = simulate_key_press(v.figure, key="alt+r", inaxes=mock_pane)
        v._reset_view_to_data_extent(event_img)
        assert mock_pane.reset_view.call_count == 1
        assert reset_axes_calls == []
        assert v._ax_trace_x.get_xlim() == (15, 40), (
            "Image-pane branch must NOT touch trace xlim"
        )

        # Branch 2a: cursor over _ax_trace_x -> reset_axes(both) scoped, no pin.
        event_tx = simulate_key_press(v.figure, key="alt+r", inaxes=v._ax_trace_x)
        v._reset_view_to_data_extent(event_tx)
        assert mock_pane.reset_view.call_count == 1
        assert len(reset_axes_calls) == 1
        assert reset_axes_calls[-1]["axes"] == [v._ax_trace_x, v._ax_trace_y]
        assert reset_axes_calls[-1]["axis"] == "both"
        # alt+r must NOT pin xlim to (0, n_frames) — the whole point is to
        # let reset_axes autoscale x to the data extent.
        assert v._ax_trace_x.get_xlim() != (0, n_frames)

        # Branch 2b: cursor over _ax_trace_y -> same.
        event_ty = simulate_key_press(v.figure, key="alt+r", inaxes=v._ax_trace_y)
        v._reset_view_to_data_extent(event_ty)
        assert mock_pane.reset_view.call_count == 1
        assert len(reset_axes_calls) == 2
        assert reset_axes_calls[-1]["axes"] == [v._ax_trace_x, v._ax_trace_y]
        assert reset_axes_calls[-1]["axis"] == "both"

        # Branch 3: event is None -> reset everything, traces autoscale both.
        v._reset_view_to_data_extent(None)
        assert mock_pane.reset_view.call_count == 2
        assert len(reset_axes_calls) == 3
        assert reset_axes_calls[-1]["axes"] is None
        assert reset_axes_calls[-1]["axis"] == "both"

        # Branch 3b: event with inaxes=None -> same fallback.
        event_none = simulate_key_press(v.figure, key="alt+r", inaxes=None)
        v._reset_view_to_data_extent(event_none)
        assert mock_pane.reset_view.call_count == 3
        assert len(reset_axes_calls) == 4
        assert reset_axes_calls[-1]["axes"] is None
        assert reset_axes_calls[-1]["axis"] == "both"
    finally:
        v.reset_axes = orig_reset_axes
        plt.close(v.figure)


def test_video_annotation_display_update_visibility(video_fname):
    fig, (ax_img, ax_x, ax_y) = plt.subplots(3, 1)

    ann = datanavigator.VideoAnnotation(
        vname=video_fname, name="test_display_update_visibility"
    )
    ann.setup_display(
        ax_list_scatter=[ax_img], ax_list_trace_x=[ax_x], ax_list_trace_y=[ax_y]
    )
    ann.add([1, 1], "0", 0)  # Add some data

    # Update scatter
    ann.update_display_scatter(0)

    # Update trace
    ann.update_display_trace("0")
    # Called for x and y trace of label 0

    # Update display calls both
    ann.update_display(0, "0")

    # Test visibility/alpha/plot_type
    ann.hide(draw=False)

    ann.show(draw=False)

    ann.set_alpha(0.5, draw=False)

    ann.set_plot_type("line", draw=False)

    ann.set_plot_type("dot", draw=False)

    # Test show/hide one trace
    ann.hide_trace("0", draw=False)

    ann.show_one_trace("1", draw=False)
    plt.close(fig)
    assert True  # If no exceptions, test passed


def test_set_plot_type_syncs_plot_type_attribute(video_fname):
    """``set_plot_type`` and the ``plot_type`` setter must leave the
    annotation in the same state.

    Regression guard: pre-fix, ``set_plot_type("line")`` only
    updated trace handles' linestyle/marker, leaving
    ``self._plot_type`` at its ``__init__`` default ``"dot"``. A
    subsequent ``re_setup_display`` (which fires inside
    ``add_annotation_layers`` for every existing layer) would call
    ``setup_display`` → ``set_plot_type(self.plot_type)``, reading
    the stale ``"dot"`` and reverting the visual. Caused the
    DUSTrack ``dlccorr`` "renders as dots after Reduce jitter"
    regression.
    """
    fig, (ax_img, ax_x, ax_y) = plt.subplots(3, 1)
    ann = datanavigator.VideoAnnotation(
        vname=video_fname, name="plot_type_sync"
    )
    ann.setup_display(
        ax_list_scatter=[ax_img], ax_list_trace_x=[ax_x], ax_list_trace_y=[ax_y]
    )
    # init default is "dot"
    assert ann.plot_type == "dot"
    # Method call must update the attribute, not just the visuals.
    ann.set_plot_type("line", draw=False)
    assert ann.plot_type == "line"
    assert ann._plot_type == "line"
    ann.set_plot_type("dot", draw=False)
    assert ann.plot_type == "dot"
    # Property setter still works and delegates to the same machinery.
    ann.plot_type = "line"
    assert ann.plot_type == "line"
    assert ann._plot_type == "line"
    plt.close(fig)


def test_set_plot_type_survives_re_setup_display(video_fname):
    """After ``set_plot_type("line")``, re-running
    ``re_setup_display`` (which fires for every existing layer
    inside ``add_annotation_layers``) must preserve the line
    styling. Direct verification of the regression mechanism.
    """
    fig, (ax_img, ax_x, ax_y) = plt.subplots(3, 1)
    ann = datanavigator.VideoAnnotation(
        vname=video_fname, name="plot_type_after_resetup"
    )
    ann.setup_display(
        ax_list_scatter=[ax_img], ax_list_trace_x=[ax_x], ax_list_trace_y=[ax_y]
    )
    ann.add([1, 1], "0", 0)
    ann.set_plot_type("line", draw=False)
    ann.re_setup_display()
    # After re_setup_display, the styling must still be line.
    assert ann.plot_type == "line"
    # Spot-check the actual artist: a line plot has linestyle "-".
    sample_handle = next(iter(ann._trace_handles.values()))
    assert sample_handle.get_linestyle() == "-"
    plt.close(fig)


def test_video_point_annotator_init(video_fname):
    # simple initialization with video only.
    v = datanavigator.VideoPointAnnotator(vid_name=video_fname)
    assert len(v.annotations) == 2
    assert v.annotations[0].name == ""
    assert v.annotations[0].fstem == "example_video_annotations"
    assert v.annotations[-1].name == "buffer"
    plt.close(v.figure)

    v = datanavigator.VideoPointAnnotator(vid_name=video_fname, annotation_names=["pn"])
    assert v.annotations.names == [
        "pn",
        "buffer",
    ]  # the _annotations.json file will not be loaded
    plt.close(v.figure)

    v = datanavigator.VideoPointAnnotator(
        vid_name=video_fname, annotation_names=["", "pn"]
    )
    assert v.annotations.names == ["", "pn", "buffer"]
    plt.close(v.figure)

    # explicitly adding a "buffer" name will not make a difference
    v = datanavigator.VideoPointAnnotator(
        vid_name=video_fname, annotation_names=["", "pn", "buffer"]
    )
    assert v.annotations.names == ["", "pn", "buffer"]
    plt.close(v.figure)


def test_video_point_annotator_from_annotations(ann_object):
    v = datanavigator.VideoPointAnnotator.from_annotations(ann_object)
    assert len(v.ann.frames) == 10


def test_video_point_annotator_add_new_label(ann_object):
    v = datanavigator.VideoPointAnnotator.from_annotations(ann_object)
    assert "5" not in v.ann.labels
    assert "5" not in v.annotations["buffer"].labels
    v(simulate_key_press(v.figure, key="5"))
    assert "5" in v.ann.labels
    assert "5" in v.annotations["buffer"].labels
    assert len(v.ann.data["5"]) == 0

    # test "place" mode
    v(simulate_key_press(v.figure, key="`"))  # toggle what number keys do
    assert v.statevariables["number_keys"].current_state == "place"
    assert "6" not in v.ann.labels
    v(simulate_key_press_at_xy((v.figure, v._ax_image), key="6", xdata=645, ydata=360))
    assert "6" in v.ann.labels
    assert len(v.ann.data["6"]) == 1
    assert np.allclose(v.ann["6"][v._current_idx], [645, 360])
    assert (
        len(v.annotations["buffer"].frames) == 0
    )  # buffer should still be empty if we added the annotation to the current layer


def test_video_point_annotator_key_bindings(video_fname):
    v = datanavigator.VideoPointAnnotator(
        vid_name=video_fname, annotation_names=["", "pn"]
    )
    assert len(v.annotations[""].frames) == 10
    assert len(v.annotations["pn"].frames) == 9
    assert len(v.annotations["buffer"].frames) == 0

    # check state variables
    assert v.statevariables.names == ['annotation_layer', 'annotation_overlay', 'annotation_label', 'label_range', 'number_keys']

    # check the current state
    assert v._current_idx == 0
    assert v._current_label == v.statevariables["annotation_label"].current_state == "0"
    assert v._current_layer == v.statevariables["annotation_layer"].current_state == ""
    assert (
        v._current_overlay is None
        and v.statevariables["annotation_overlay"].current_state is None
    )

    # cycle through the annotation layers
    assert v.ann.name == ""
    v(simulate_key_press(v.figure, key="="))
    assert v.ann.name == "pn"
    v(simulate_key_press(v.figure, key="="))
    assert v.ann.name == "buffer"
    v(simulate_key_press(v.figure, key="="))
    assert v.ann.name == ""
    v(simulate_key_press(v.figure, key="="))
    assert v.ann.name == "pn"

    # in layer pn, go to frame 18
    v(simulate_key_press(v.figure, key="'"))  # go to annotation label 1
    assert v._current_label == "1"
    event = simulate_mouse_click(
        (v.figure, v._ax_trace_x),
        xdata=18,
        ydata=np.mean(v._ax_trace_x.get_ylim()),
        button=3,  # Right click
    )
    v.figure.canvas.callbacks.process("button_press_event", event)
    assert v._current_idx == 18

    # annotate a point with label "1" at frame 18 in layer "pn"
    assert v._current_idx not in v.ann[v._current_label]
    v(simulate_key_press_at_xy((v.figure, v._ax_image), key="t", xdata=645, ydata=360))
    assert np.allclose(v.ann[v._current_label][v._current_idx], [645, 360])

    # move this point with right mouse click on the image
    event = simulate_mouse_click(
        (v.figure, v._ax_image),
        xdata=700,
        ydata=400,
        button=3,  # Right click
    )
    v.figure.canvas.callbacks.process("button_press_event", event)
    assert np.allclose(v.ann[v._current_label][v._current_idx], [700, 400])

    # check that the pn layer has 1 annotation at frame 18, and 9 existing annotations
    assert v.ann.frames == list(np.r_[18, 50 : 50 + 18 : 2])

    # make buffer the annotation overlay
    v(simulate_key_press(v.figure, key="["))
    assert v._current_overlay == "buffer"

    # interpolate with lk (check labels with lk - puts interpolated points in a buffer layer)
    # test1 - minimal mode - two previous labeled frames to two next frames (or 1 next frame if current frame is annotated)
    assert (
        len(v.annotations["buffer"].frames) == 0
    )  # check that the buffer layer is empty
    v(simulate_key_press(v.figure, key="v"))
    assert v.annotations["buffer"].frames == list(np.r_[18:51])

    v._current_idx = 19
    v.update()
    v(simulate_key_press(v.figure, key="v"))
    assert v.annotations["buffer"].frames == list(np.r_[18:53])

    v._current_idx = 51
    v.update()
    v(simulate_key_press(v.figure, key="v"))
    assert v.annotations["buffer"].frames == list(np.r_[18:55])
    assert len(v.annotations["buffer"].frames) == 37  # 18 to 54, including 18 and 54

    # clear the buffer layer
    # bring the buffer layer to the foreground
    v(simulate_key_press(v.figure, key="="))
    assert v._current_layer == "buffer"
    # select an event using the 'z' key
    for xdata in (4, 85):
        v(
            simulate_key_press_at_xy(
                (v.figure, v._ax_trace_y),
                key="z",
                xdata=xdata,
                ydata=400,
            )
        )
    v(simulate_key_press(v.figure, key="alt+a"))  # clear the buffer layer
    assert (
        len(v.annotations["buffer"].frames) == 0
    )  # check that the buffer layer is empty
    plt.close(v.figure)


def test_video_point_annotator_add_remove_annotation(video_fname):
    v = datanavigator.VideoPointAnnotator(
        vid_name=video_fname, annotation_names=["pn2"]
    )
    assert len(v.annotations) == 2
    assert v.annotations.names == ["pn2", "buffer"]

    assert v.statevariables["number_keys"].current_state == "select"
    v(simulate_key_press(v.figure, key="`"))  # toggle what number keys do
    assert v.statevariables["number_keys"].current_state == "place"

    # place 3 pointso on the current frame (addubg with "t" is already covered in the previous test)
    v._current_idx = 45  # go to frame 45
    v.update()
    assert v._current_layer == "pn2"
    for label in ("0", "1", "2"):
        v(
            simulate_key_press_at_xy(
                (v.figure, v._ax_image),
                key=label,
                xdata=100 + int(label) * 10,
                ydata=200 - int(label) * 10,
            )
        )
    assert v.ann.frames == [45]
    assert np.allclose(v.ann["0"][v._current_idx], [100, 200])
    assert np.allclose(v.ann["1"][v._current_idx], [110, 190])
    assert np.allclose(v.ann["2"][v._current_idx], [120, 180])
    assert v._current_label == "2"  # last label used is the current label

    # remove annotation for point 2
    assert len(v.ann["2"]) == 1  # check that the annotation for point 2 exists
    v(simulate_key_press(v.figure, key="y"))
    assert len(v.ann["2"]) == 0  # check that the annotation for point 2 is removed
    plt.close(v.figure)


def test_video_point_annotator_conditional_move(video_fname):
    """Test the feature that allows to change frame only if there is no annotation at the current frame."""
    v = datanavigator.VideoPointAnnotator(
        vid_name=video_fname, annotation_names=["pn2"]
    )
    assert len(v.annotations) == 2
    assert v.annotations.names == ["pn2", "buffer"]

    # add a point at frame 10
    v._current_idx = 10  # go to frame 0
    v.update()
    v(
        simulate_key_press_at_xy(
            (v.figure, v._ax_image),
            key="t",
            xdata=100,
            ydata=200,
        )
    )
    assert np.allclose(v.ann["0"][v._current_idx], [100, 200])

    v(simulate_key_press(v.figure, key="left"))
    assert v._current_idx == 9  # go to frame 9
    v(simulate_key_press(v.figure, key="f"))
    assert v._current_idx == 10
    v(simulate_key_press(v.figure, key="f"))
    assert (
        v._current_idx == 10
    )  # shouldn not move because there is an annotation at frame 10

    v(simulate_key_press(v.figure, key="g"))
    assert v._current_idx == 11  # go to frame 9
    v(simulate_key_press(v.figure, key="d"))
    assert v._current_idx == 10  # go to frame 10
    v(simulate_key_press(v.figure, key="d"))
    assert (
        v._current_idx == 10
    )  # shouldn't move because there is an annotation at frame 10


def test_video_point_annotator_state_variables(video_fname):
    """Test cycling through the state variables."""
    v = datanavigator.VideoPointAnnotator(
        vid_name=video_fname, annotation_names=["", "pn"]
    )
    assert len(v.annotations) == 3
    assert v.annotations.names == ["", "pn", "buffer"]

    # check state variables
    assert v.statevariables.names == ['annotation_layer', 'annotation_overlay', 'annotation_label', 'label_range', 'number_keys']

    # check the current state
    assert v._current_idx == 0
    assert v._current_label == v.statevariables["annotation_label"].current_state == "0"
    assert v._current_layer == v.statevariables["annotation_layer"].current_state == ""
    assert (
        v._current_overlay is None
        and v.statevariables["annotation_overlay"].current_state is None
    )

    assert v._current_layer == ""
    v(simulate_key_press(v.figure, key="="))
    assert v._current_layer == "pn"
    v(simulate_key_press(v.figure, key="-"))
    assert v._current_layer == ""
    v(simulate_key_press(v.figure, key="-"))
    assert v._current_layer == "buffer"
    v(simulate_key_press(v.figure, key="="))
    assert v._current_layer == ""

    assert v._current_overlay is None
    v(simulate_key_press(v.figure, key="["))
    assert v._current_overlay == "buffer"
    v(simulate_key_press(v.figure, key="]"))
    assert v._current_overlay is None
    v(simulate_key_press(v.figure, key="]"))
    assert v._current_overlay == ""
    v(simulate_key_press(v.figure, key="]"))
    assert v._current_overlay == "pn"

    assert v.ann.labels == ["0", "1", "2"]
    assert v._current_label == "0"
    v(simulate_key_press(v.figure, key="'"))
    assert v._current_label == "1"
    v(simulate_key_press(v.figure, key=";"))
    assert v._current_label == "0"
    v(simulate_key_press(v.figure, key=";"))
    assert v._current_label == "2"

    v(simulate_key_press(v.figure, key="="))
    assert v._current_layer == "pn"
    assert v.ann.frames == list(
        np.r_[50 : 50 + 18 : 2]
    )  # check that the annotation layer is empty
    assert v._current_idx == 0
    v(simulate_key_press(v.figure, key="."))
    assert v._current_idx == 50
    v(simulate_key_press(v.figure, key="."))
    assert v._current_idx == 52
    v(simulate_key_press(v.figure, key=","))
    assert v._current_idx == 50
    v(simulate_key_press(v.figure, key=","))
    assert v._current_idx == 50

    plt.close(v.figure)


def test_video_point_annotator_frames_of_interest(video_fname):
    """Test the frames of interest feature."""
    v = datanavigator.VideoPointAnnotator(
        vid_name=video_fname, annotation_names=["pn3"]
    )
    assert len(v.annotations) == 2
    assert v.annotations.names == ["pn3", "buffer"]

    # check initial state
    assert v.frames_of_interest == []

    # add a frame of interest
    assert v._current_layer == "pn3"
    # test navigation with empty frames of interest
    v(simulate_key_press(v.figure, key="alt+,"))  # should not throw an error
    v(simulate_key_press(v.figure, key="alt+."))  # should not throw an error

    v._current_idx = 4
    v.update()
    v(simulate_key_press(v.figure, key="m", inaxes=v._ax_trace_x))
    assert v.frames_of_interest == [4]
    v(simulate_key_press(v.figure, key="m", inaxes=v._ax_image))  # this should not work
    assert v.frames_of_interest == [4]
    v(simulate_key_press(v.figure, key="m"))  # this should also not work
    assert v.frames_of_interest == [4]

    # add and remove a frame of interest
    v._current_idx = 7
    v.update()
    v(simulate_key_press(v.figure, key="m", inaxes=v._ax_trace_x))
    assert v.frames_of_interest == [4, 7]
    v(simulate_key_press(v.figure, key="alt+,"))
    assert v._current_idx == 4
    v(simulate_key_press(v.figure, key="alt+,"))
    assert v._current_idx == 4
    v(simulate_key_press(v.figure, key="alt+."))
    assert v._current_idx == 7
    v(simulate_key_press(v.figure, key="alt+."))
    assert v._current_idx == 7

    v(simulate_key_press(v.figure, key="m", inaxes=v._ax_trace_y))
    assert v.frames_of_interest == [4]

    # label two frames, interpolate between them into the buffer, and copy frames of interest from the buffer
    # label two frames
    event = simulate_mouse_click(
        (v.figure, v._ax_trace_x),
        xdata=27.1,
        ydata=np.mean(v._ax_trace_x.get_ylim()),
        button=3,  # Right click
    )
    v.figure.canvas.callbacks.process("button_press_event", event)
    assert v._current_idx == 27
    assert v._current_idx not in v.ann[v._current_label]
    v(simulate_key_press_at_xy((v.figure, v._ax_image), key="t", xdata=441, ydata=380))
    assert np.allclose(v.ann[v._current_label][v._current_idx], [441, 380])

    event = simulate_mouse_click(
        (v.figure, v._ax_trace_x),
        xdata=60.1,
        ydata=np.mean(v._ax_trace_x.get_ylim()),
        button=3,  # Right click
    )
    v.figure.canvas.callbacks.process("button_press_event", event)
    assert v._current_idx == 60
    assert v._current_idx not in v.ann[v._current_label]
    v(simulate_key_press_at_xy((v.figure, v._ax_image), key="t", xdata=468, ydata=487))
    assert np.allclose(v.ann[v._current_label][v._current_idx], [468, 487])

    # bring buffer to overlay
    v(simulate_key_press(v.figure, key="["))
    assert v._current_overlay == "buffer"

    # interpolate with lk (check labels with lk - puts interpolated points in a buffer layer)
    assert (
        len(v.annotations["buffer"].frames) == 0
    )  # check that the buffer layer is empty
    v(simulate_key_press(v.figure, key="v"))
    assert v.annotations["buffer"].frames == list(np.r_[27:61])

    # mark frames of interest as 31, 42, 52
    v.frames_of_interest = [31, 42, 52]
    v.update()

    # copy frames of interest from the buffer to the annotation layer
    v(simulate_key_press(v.figure, key="alt+c"))

    assert v.ann.frames == [27, 31, 42, 52, 60]

    plt.close(v.figure)


def test_video_point_annotator_copy_annotations(video_fname):
    """Test the feature of copying annotations between layers."""
    v = datanavigator.VideoPointAnnotator(
        vid_name=video_fname, annotation_names=["pn4", "pn5"]
    )
    # testing c - copy current annotaion from overlay to the current annotation layer
    # add a label to pn4
    v.annotations["pn4"].add([1, 1], "1", 5)

    # go to frame 5
    event = simulate_mouse_click(
        (v.figure, v._ax_trace_x),
        xdata=5,
        ydata=np.mean(v._ax_trace_x.get_ylim()),
        button=3,  # Right click
    )
    v.figure.canvas.callbacks.process("button_press_event", event)
    assert v._current_idx == 5

    # set current layer to pn5
    v(simulate_key_press(v.figure, key="="))
    assert v._current_layer == "pn5"

    # set current overlay to pn4
    v(simulate_key_press(v.figure, key="]"))
    assert v._current_overlay == "pn4"

    # check current state of the data
    assert v.ann.frames == []
    assert v.annotations["pn4"].frames == [5]

    # copy annotation from overlay to current layer
    v(
        simulate_key_press(v.figure, key="c")
    )  # nothing should happen because current label is 0
    assert v.ann.frames == []

    # set current label to 1
    v(simulate_key_press(v.figure, key="'"))

    # copy annotation from overlay to current layer
    v(
        simulate_key_press(v.figure, key="c")
    )  # copy annotation from overlay to current layer
    assert v.ann.frames == [5]
    assert len(v.ann.data["1"]) == 1
    assert all([len(v.ann.data[x]) == 0 for x in v.ann.labels if x != "1"])

    # remove annotation from the current layer
    v(simulate_key_press(v.figure, key="y"))  # remove annotation from the current layer
    assert all([len(v.ann.data[x]) == 0 for x in v.ann.labels])

    # testing alt+c - copy frames of interest from overlay to current layer
    # add data into the buffer layer from frames 7 to 50
    for i in range(7, 51):
        v.annotations["buffer"].add([i, i], "1", i)
    assert (
        len(v.annotations["buffer"].frames) == 44
    )  # check that the buffer layer has 44 frames

    # add frames of interest
    v.frames_of_interest = [2, 20, 23]  # there is no data in 2
    v.update()

    # copy frames of interest from buffer to current layer
    assert v._current_layer == "pn5"
    assert list(v.annotations["pn5"].data["1"].keys()) == []
    assert v._current_overlay != "buffer"
    v(
        simulate_key_press(v.figure, key="alt+c")
    )  # copy frames of interest from buffer to current layer
    assert list(v.annotations["pn5"].data["1"].keys()) == []

    # should only work after changing the overlay to buffer
    v(simulate_key_press(v.figure, key="]"))
    v(simulate_key_press(v.figure, key="]"))
    assert v._current_overlay == "buffer"
    v(
        simulate_key_press(v.figure, key="alt+c")
    )  # copy frames of interest from buffer to current layer
    assert list(v.annotations["pn5"].data["1"].keys()) == [20, 23]

    # ctrl+alt+c - copy frames in interval from overlay to current layer.
    # change current layer to pn4 and overlay to pn5
    v(simulate_key_press(v.figure, key="="))
    v(simulate_key_press(v.figure, key="="))
    assert v._current_layer == "pn4"

    # add interval to the current layer - select an event using the 'z' key
    for xdata in (30, 70):
        v(
            simulate_key_press_at_xy(
                (v.figure, v._ax_trace_y),
                key="z",
                xdata=xdata,
                ydata=v._ax_trace_y.get_ylim()[0],
            )
        )
    assert v.annotations[v._current_overlay].frames == list(np.r_[7:51])
    v(
        simulate_key_press(v.figure, key="ctrl+alt+c")
    )  # copy interval from overlay to current layer
    assert v.ann.frames == list(np.r_[5, 30:51])

    # copy all annotation from overlay to current layer in the current frame (no shortcut)
    # go to frame 12
    event = simulate_mouse_click(
        (v.figure, v._ax_trace_x),
        xdata=12,
        ydata=np.mean(v._ax_trace_x.get_ylim()),
        button=3,  # Right click
    )
    v.figure.canvas.callbacks.process("button_press_event", event)
    assert v._current_idx == 12
    assert v._current_layer == "pn4"
    assert v._current_overlay == "buffer"
    assert v._current_label == "1"

    # add data to labels 3 and 4 in the buffer layer
    v.annotations["buffer"].add([100, 13], "3", 12)
    v.annotations["buffer"].add([100, 123], "4", 12)

    assert 12 not in v.ann.frames
    v.copy_annotations_from_overlay()
    assert 12 in v.ann.frames
    assert list(v.ann.data["3"].keys()) == [12]
    assert list(v.ann.data["4"].keys()) == [12]
    assert np.allclose(v.ann["3"][12], [100, 13])
    assert np.allclose(v.ann["4"][12], [100, 123])
    assert all(
        [12 not in v.ann.data[x] for x in v.ann.labels if x not in ["1", "3", "4"]]
    )

    plt.close(v.figure)


def test_video_point_annotator_lucas_kanade(video_fname):
    # -- predict_labels_with_lucas_kanade --
    v = datanavigator.VideoPointAnnotator(
        vid_name=video_fname, annotation_names=["pn6"]
    )
    v.ann.data = {}
    v.ann.add_label("0")
    v.ann.add_label("1")
    v.ann.add_label("2")
    v.ann.add_label("3")

    # add some annotations in frame 5
    v.annotations["pn6"].add([10, 10], "0", 5)
    v.annotations["pn6"].add([10, 11], "1", 5)
    v.annotations["pn6"].add([10, 12], "2", 5)

    # go to frame 8
    event = simulate_mouse_click(
        (v.figure, v._ax_trace_x),
        xdata=8,
        ydata=np.mean(v._ax_trace_x.get_ylim()),
        button=3,  # Right click
    )
    v.figure.canvas.callbacks.process("button_press_event", event)
    assert v._current_idx == 8

    # alt+b - predict current point with lucas kanade and add to the current layer
    assert v._current_label == "0"
    assert 8 not in v.annotations["pn6"].frames
    v(simulate_key_press(v.figure, key="alt+b"))
    assert 8 in v.annotations["pn6"]["0"]
    assert 8 not in v.annotations["pn6"]["1"]
    assert 8 not in v.annotations["pn6"]["2"]
    assert 8 not in v.annotations["pn6"]["3"]

    # ctrl+b - predict all points with lucas kanade and add to the current layer
    # the nearest annotated frame for ALL labels should be the same, otherwise this will not work
    with pytest.raises(ValueError):
        # because there is no data for label 3 in frame 5
        v(simulate_key_press(v.figure, key="ctrl+b"))
    v.annotations["pn6"].add([10, 13], "3", 5)

    with pytest.raises(AssertionError):
        # because different labels have different nearest annotated frames
        v(simulate_key_press(v.figure, key="ctrl+b"))
    v(simulate_key_press(v.figure, key="y"))  # otherwise the next test will fail
    v.update()

    v(simulate_key_press(v.figure, key="ctrl+b"))
    assert 8 in v.annotations["pn6"]["1"]
    assert 8 in v.annotations["pn6"]["2"]
    assert 8 in v.annotations["pn6"]["3"]

    # change overlay to pn6 and current layer to buffer
    v(simulate_key_press(v.figure, key="]"))
    assert v._current_overlay == "pn6"
    v(simulate_key_press(v.figure, key="="))
    assert v._current_layer == "buffer"

    # add an interval to the current layer - select an event using the 'z' key
    for xdata in (6, 40):
        v(
            simulate_key_press_at_xy(
                (v.figure, v._ax_trace_y),
                key="z",
                xdata=xdata,
                ydata=v._ax_trace_y.get_ylim()[0],
            )
        )
    with pytest.raises(KeyError):
        # doesn't work if the starting point of the interval is not in the overlay
        v(simulate_key_press(v.figure, key="ctrl+d"))

    for xdata in (8, 40):
        v(
            simulate_key_press_at_xy(
                (v.figure, v._ax_trace_y),
                key="z",
                xdata=xdata,
                ydata=v._ax_trace_y.get_ylim()[0],
            )
        )
    v(simulate_key_press(v.figure, key="ctrl+d"))

    plt.close(v.figure)


def test_video_point_annotator_prev_next_frames_with_current_label(video_fname):
    v = datanavigator.VideoPointAnnotator(
        vid_name=video_fname, annotation_names=["pn7"]
    )
    # add annotation in frames 5 and 8
    v.annotations["pn7"].add([10, 10], "0", 5)
    v.annotations["pn7"].add([10, 11], "0", 8)
    v.annotations["pn7"].add([10, 11], "1", 12)

    assert v._current_idx == 0
    v(simulate_key_press(v.figure, key="p"))  # previous frame - none
    assert v._current_idx == 0
    v(simulate_key_press(v.figure, key="n"))  # next frame
    assert v._current_idx == 5
    v(simulate_key_press(v.figure, key="n"))
    assert v._current_idx == 8
    v(simulate_key_press(v.figure, key="n"))
    assert (
        v._current_idx == 8
    )  # should not move because there are no more annotations for this label

    plt.close(v.figure)


def test_save(video_fname):
    v = datanavigator.VideoPointAnnotator(
        vid_name=video_fname, annotation_names=["pn7"]
    )
    assert not os.path.exists(v.annotations["pn7"].fname)
    v.save()
    assert os.path.exists(v.annotations["pn7"].fname)
    plt.close(v.figure)


def test_video_point_annotator_keep_overlapping_continuous_frames(video_fname):
    v = datanavigator.VideoPointAnnotator(
        vid_name=video_fname, annotation_names=["pn8"]
    )
    v.ann.data = {}
    v.ann.add_label("0")
    v.ann.add_label("1")
    v.ann.add_label("2")

    # add annotations in frame 10-50 for label 0
    for i in range(10, 51):
        v.annotations["pn8"].add([i, i], "0", i)

    # add annotations in frame 20-60 for label 1
    for i in range(20, 61):
        v.annotations["pn8"].add([i, i], "1", i)

    # add annotations in frame 5-55 for label 2
    for i in range(5, 56):
        v.annotations["pn8"].add([i, i], "2", i)

    v(simulate_key_press(v.figure, key="alt+q"))  # keep overlapping continuous frames
    assert len(v.annotations["pn8"].frames) == 31  # all frames should be kept
    plt.close(v.figure)


def test_video_point_annotator_keep_overlapping_frames(video_fname):
    """Wrapper drops incomplete frames + bumps the revision via update().

    Unlike :meth:`keep_overlapping_continuous_frames` (alt+q), no
    keybinding is wired -- DUSTrack's "Train DLC model" pre-flight
    is the only caller.
    """
    v = datanavigator.VideoPointAnnotator(
        vid_name=video_fname, annotation_names=["pn_kof"]
    )
    v.ann.data = {}
    v.ann.add_label("0")
    v.ann.add_label("1")
    v.ann.add_label("2")

    # Three non-consecutive fully-labeled frames
    for frame in (5, 10, 15):
        for label in ("0", "1", "2"):
            v.annotations["pn_kof"].add([frame, frame], label, frame)
    # One incomplete frame (label 0 only)
    v.annotations["pn_kof"].add([20, 20], "0", 20)
    assert set(v.annotations["pn_kof"].frames) == {5, 10, 15, 20}

    v.keep_overlapping_frames()
    assert set(v.annotations["pn_kof"].frames) == {5, 10, 15}
    plt.close(v.figure)


def test_video_point_annotator_render(video_fname):
    v = datanavigator.VideoPointAnnotator(
        vid_name=video_fname, annotation_names=["pn9"]
    )
    v.annotations["pn9"].add([10, 100], "0", 0)  # add some data
    v.annotations["pn9"].add([50, 200], "0", 60)  # add some data
    out_vid_name = os.path.join(
        os.path.dirname(video_fname), "test_video_point_annotator_render.mp4"
    )
    assert not os.path.exists(out_vid_name)
    v.render(10, 20, out_vid_name)
    assert os.path.exists(out_vid_name)
    # check that the video is rendered correctly
    assert len(datanavigator.Video(out_vid_name)) == 11


# --- _TrackedFrameDict mutation-guard tests -----------------------------------
# Cover the "direct ann.data[label][frame] = ..." bug class that bit
# datanavigator 1.4.0rc1 via check_labels_with_lk and DUSTrack's
# copy_existing_annotations_from_overlay. Both sites were fixed via the
# public add() API in 1.4.0rc2; the data-structure-level guard below
# makes the next bypass unrepresentable: any write to an inner per-label
# dict bumps _revision regardless of which API path the caller took.

from datanavigator.pointtracking import _TrackedFrameDict


def test_inner_dict_is_tracked_frame_dict(ann_object):
    """Each per-label inner dict is a _TrackedFrameDict bound to its parent."""
    for label in ann_object.labels:
        inner = ann_object.data[label]
        assert isinstance(inner, _TrackedFrameDict)
        assert inner._parent_ref is not None
        assert inner._parent_ref() is ann_object


def test_direct_setitem_bumps_revision(ann_object):
    """Direct ann.data[label][frame] = ... bumps _revision (the bypass we close)."""
    before = ann_object._revision
    label = ann_object.labels[0]
    ann_object.data[label][9999] = [1.0, 2.0]
    assert ann_object._revision > before
    assert ann_object.data[label][9999] == [1.0, 2.0]


def test_direct_delitem_bumps_revision(ann_object):
    """``del ann.data[label][frame]`` bumps _revision."""
    label = ann_object.labels[0]
    frame = next(iter(ann_object.data[label]))
    before = ann_object._revision
    del ann_object.data[label][frame]
    assert ann_object._revision > before
    assert frame not in ann_object.data[label]


def test_pop_bumps_revision_only_when_key_existed(ann_object):
    """pop() bumps only on actual removal, not on default-fallback."""
    label = ann_object.labels[0]
    frame = next(iter(ann_object.data[label]))
    rev0 = ann_object._revision
    ann_object.data[label].pop(frame)
    assert ann_object._revision > rev0
    rev1 = ann_object._revision
    ann_object.data[label].pop(999999, None)  # not present
    assert ann_object._revision == rev1


def test_update_bumps_revision(ann_object):
    """dict.update() on an inner dict bumps _revision."""
    label = ann_object.labels[0]
    before = ann_object._revision
    ann_object.data[label].update({12345: [3.0, 4.0], 12346: [5.0, 6.0]})
    assert ann_object._revision > before


def test_setdefault_bumps_revision_only_on_insert(ann_object):
    """setdefault() bumps when inserting a new key, not when reading existing."""
    label = ann_object.labels[0]
    frame_existing = next(iter(ann_object.data[label]))
    rev0 = ann_object._revision
    ann_object.data[label].setdefault(frame_existing, [0.0, 0.0])  # existing
    assert ann_object._revision == rev0
    ann_object.data[label].setdefault(99998, [7.0, 8.0])  # new
    assert ann_object._revision > rev0


def test_wholesale_data_reassignment_rewraps(ann_object):
    """Setting ann.data = {...} rewraps inner dicts; subsequent direct writes bump."""
    ann_object.data = {"0": {1: [1.0, 2.0]}, "1": {2: [3.0, 4.0]}}
    for label, inner in ann_object._data.items():
        assert isinstance(inner, _TrackedFrameDict)
        assert inner._parent_ref() is ann_object
    before = ann_object._revision
    ann_object.data["0"][42] = [9.0, 9.0]
    assert ann_object._revision > before


def test_add_label_wraps_new_inner_dict(ann_object):
    """add_label() places a _TrackedFrameDict, not a bare dict, at the new key."""
    new_label = str(int(max(ann_object.labels)) + 1) if ann_object.labels else "0"
    if new_label in ann_object.labels:
        new_label = "9"
    if new_label in ann_object.labels:
        pytest.skip("no free label slot")
    ann_object.add_label(label=new_label)
    inner = ann_object.data[new_label]
    assert isinstance(inner, _TrackedFrameDict)
    assert inner._parent_ref() is ann_object
    before = ann_object._revision
    inner[5] = [1.0, 1.0]
    assert ann_object._revision > before


def test_sort_data_preserves_wrapping(ann_object):
    """sort_data() rebuilds inner dicts but the result is rewrapped via the setter."""
    ann_object.sort_data()
    for label in ann_object.labels:
        inner = ann_object.data[label]
        assert isinstance(inner, _TrackedFrameDict)
        assert inner._parent_ref() is ann_object


def test_clip_labels_preserves_wrapping(ann_object):
    """clip_labels() reassigns via setter; inner dicts stay wrapped."""
    frames_present = sorted(ann_object.data[ann_object.labels[0]].keys())
    if len(frames_present) < 2:
        pytest.skip("need >=2 annotated frames")
    ann_object.clip_labels(frames_present[0], frames_present[-1])
    for label in ann_object.labels:
        inner = ann_object.data[label]
        assert isinstance(inner, _TrackedFrameDict)
        assert inner._parent_ref() is ann_object
    # And mutation of the post-clip inner still bumps:
    label = ann_object.labels[0]
    before = ann_object._revision
    ann_object.data[label][77777] = [0.5, 0.5]
    assert ann_object._revision > before


def test_keep_overlapping_frames_preserves_wrapping(ann_object):
    """keep_overlapping_frames() reassigns via setter; inner dicts stay wrapped."""
    ann_object.keep_overlapping_frames()
    for label in ann_object.labels:
        inner = ann_object.data[label]
        assert isinstance(inner, _TrackedFrameDict)


def test_clear_bumps_revision_only_when_nonempty(ann_object):
    """clear() bumps only when the dict had contents."""
    label = ann_object.labels[0]
    # First clear: bump expected.
    rev0 = ann_object._revision
    ann_object.data[label].clear()
    assert ann_object._revision > rev0
    # Second clear: no-op, no bump.
    rev1 = ann_object._revision
    ann_object.data[label].clear()
    assert ann_object._revision == rev1


def test_revision_consumers_invalidate_on_direct_mutation(video_fname, ann_fname):
    """Trace-display cache key is (labels, _revision); direct mutation invalidates."""
    v = datanavigator.VideoPointAnnotator(
        vid_name=video_fname, annotation_names=[Path(ann_fname).stem]
    )
    ann = v.annotations[Path(ann_fname).stem]
    v.update()  # populate cache
    cache_before = getattr(ann, "_trace_display_cache_key", None)
    if cache_before is None:
        pytest.skip("trace cache not exercised on this build")
    # Direct mutation (the bug pattern the guard closes). Use a frame
    # within the video's n_frames so to_trace's ndarray write succeeds.
    label = ann.labels[0]
    target_frame = min(100, ann.n_frames - 1)
    ann.data[label][target_frame] = [10.0, 10.0]
    v.update()
    cache_after = ann._trace_display_cache_key
    assert cache_after != cache_before
    plt.close(v.figure)


def test_tracked_dict_survives_pickle_roundtrip():
    """_TrackedFrameDict round-trips through pickle as a parent-less instance."""
    import pickle
    d = _TrackedFrameDict({1: [1.0, 2.0], 2: [3.0, 4.0]})
    restored = pickle.loads(pickle.dumps(d))
    assert isinstance(restored, _TrackedFrameDict)
    assert dict(restored) == {1: [1.0, 2.0], 2: [3.0, 4.0]}
    assert restored._parent_ref is None  # weakref dropped, as designed
    # Restored dict still mutable, just doesn't bump anything (no parent):
    restored[3] = [5.0, 6.0]
    assert restored[3] == [5.0, 6.0]


def test_video_annotation_reload_with_file_on_disk(ann_fname):
    """``reload()`` drops in-memory edits and re-syncs from disk."""
    ann = datanavigator.VideoAnnotation(fname=ann_fname)
    rev_before = ann._revision
    disk_label0 = dict(ann.data["0"])
    # In-memory mutation that diverges from disk.
    ann.add(location=[999.0, 999.0], label="0", frame_number=42)
    assert ann.data["0"].get(42) == [999.0, 999.0]
    rev_after_mutate = ann._revision
    assert rev_after_mutate > rev_before

    ann.reload()
    assert 42 not in ann.data["0"]
    assert dict(ann.data["0"]) == disk_label0
    assert ann._revision > rev_after_mutate


def test_video_annotation_reload_without_file(tmp_path):
    """``reload()`` with no file restores the empty per-label shape."""
    missing_fname = str(tmp_path / "nonexistent_annotations.json")
    ann = datanavigator.VideoAnnotation(fname=missing_fname)
    # Empty fallback shape from load(): one inner dict per label, default 10.
    n_labels_before = len(ann.labels)
    assert n_labels_before > 0
    ann.add(location=[10.0, 20.0], label=ann.labels[0], frame_number=3)
    assert ann.data[ann.labels[0]].get(3) == [10.0, 20.0]
    rev_after_mutate = ann._revision

    ann.reload()
    # After reload, every label maps to an empty dict and the count is preserved.
    assert all(len(ann.data[label]) == 0 for label in ann.labels)
    assert len(ann.labels) == n_labels_before
    assert ann._revision > rev_after_mutate


def test_asset_container_remove_happy_path_and_keyerror():
    """``AssetContainer.remove(name)`` pops by name, preserves order, then raises."""
    from datanavigator.assets import AssetContainer

    class _Named:
        def __init__(self, name):
            self.name = name

    container = AssetContainer(parent=None)
    a, b, c = _Named("a"), _Named("b"), _Named("c")
    container.add(a)
    container.add(b)
    container.add(c)
    assert container.names == ["a", "b", "c"]

    popped = container.remove("b")
    assert popped is b
    assert container.names == ["a", "c"]

    with pytest.raises(KeyError):
        container.remove("b")


def test_remove_annotation_layer_swaps_active_and_clears_overlay(video_fname):
    """Removing the active layer auto-selects a survivor; overlay clears.

    Three named layers + the implicit ``buffer`` -> remove the middle
    named layer (which is the primary on construction default
    ``annotation_layer`` state) and the overlay (set to the removed
    layer beforehand). After: primary points at a survivor, overlay
    is ``None``, and both statevars' rotation lists no longer contain
    the removed name.
    """
    v = datanavigator.VideoPointAnnotator(
        vid_name=video_fname, annotation_names=["alpha", "beta", "gamma"],
    )
    # Sanity: rotation includes all three plus the implicit buffer slot.
    assert set(v.statevariables["annotation_layer"].states) >= {
        "alpha", "beta", "gamma", "buffer",
    }

    v.statevariables["annotation_layer"].set_state("beta")
    v.statevariables["annotation_overlay"].set_state("beta")
    assert v._current_layer == "beta"

    v.remove_annotation_layer("beta")

    assert "beta" not in v.annotations.names
    assert "beta" not in v.statevariables["annotation_layer"].states
    assert "beta" not in v.statevariables["annotation_overlay"].states
    # Primary handed off to a survivor (the rotation-previous, "alpha").
    assert v._current_layer == "alpha"
    # Overlay was pointing at the removed layer -> cleared to None.
    assert v.statevariables["annotation_overlay"].current_state is None
    plt.close(v.figure)


def test_remove_annotation_layer_refuses_only_layer(video_fname):
    """``remove_annotation_layer`` raises ValueError when one layer remains.

    Starts with one user layer + the implicit ``"buffer"`` (n=2); after
    removing the user layer the container holds only ``"buffer"``, and
    a second remove must refuse. The dnav layer treats every named
    entry the same -- the buffer-exclusion is a DUSTrack UI concern.
    """
    v = datanavigator.VideoPointAnnotator(
        vid_name=video_fname, annotation_names=["solo"],
    )
    assert set(v.annotations.names) == {"solo", "buffer"}
    v.remove_annotation_layer("solo")
    assert v.annotations.names == ["buffer"]
    with pytest.raises(ValueError):
        v.remove_annotation_layer("buffer")
    plt.close(v.figure)


def test_remove_annotation_layer_preserves_overlay_when_unrelated(video_fname):
    """Overlay stays put when a non-overlay, non-primary layer is removed."""
    v = datanavigator.VideoPointAnnotator(
        vid_name=video_fname, annotation_names=["alpha", "beta", "gamma"],
    )
    v.statevariables["annotation_layer"].set_state("alpha")
    v.statevariables["annotation_overlay"].set_state("gamma")

    v.remove_annotation_layer("beta")

    assert "beta" not in v.annotations.names
    assert v._current_layer == "alpha"
    assert v.statevariables["annotation_overlay"].current_state == "gamma"
    plt.close(v.figure)


def test_video_annotations_reorder_permutes_in_place():
    """``VideoAnnotations.reorder(names)`` rewrites _list order; idempotent."""
    from datanavigator.pointtracking import VideoAnnotations

    class _Named:
        def __init__(self, name):
            self.name = name

    container = VideoAnnotations(parent=None)
    a, b, c, d = _Named("a"), _Named("b"), _Named("c"), _Named("d")
    for x in (a, b, c, d):
        container._list.append(x)
    assert container.names == ["a", "b", "c", "d"]

    container.reorder(["c", "a", "d", "b"])
    assert container.names == ["c", "a", "d", "b"]
    # Same objects, not copies.
    assert container["a"] is a and container["b"] is b
    assert container["c"] is c and container["d"] is d

    # Idempotent no-op when already in target order.
    before = list(container._list)
    container.reorder(["c", "a", "d", "b"])
    assert container._list == before


def test_video_annotations_reorder_rejects_non_permutation():
    """``reorder`` rejects missing / extra / duplicate names."""
    from datanavigator.pointtracking import VideoAnnotations

    class _Named:
        def __init__(self, name):
            self.name = name

    container = VideoAnnotations(parent=None)
    for n in ("a", "b", "c"):
        container._list.append(_Named(n))

    # Missing one.
    with pytest.raises(ValueError):
        container.reorder(["a", "b"])
    # Extra unknown name.
    with pytest.raises(ValueError):
        container.reorder(["a", "b", "c", "d"])
    # Duplicate (same length, wrong set).
    with pytest.raises(ValueError):
        container.reorder(["a", "a", "b"])
    # State unchanged on all rejections.
    assert container.names == ["a", "b", "c"]

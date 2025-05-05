import pytest
import os
from pathlib import Path
import numpy as np
import pandas as pd
import json
from unittest.mock import patch, MagicMock

import matplotlib.lines
import matplotlib.figure
import matplotlib.collections
import matplotlib.axes
import matplotlib.pyplot as plt

import datanavigator
from datanavigator.examples import get_example_video
from datanavigator.pointtracking import VideoAnnotation


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
                location=[width*((frame_count+10)/30) + 5*label, height*((frame_count+10)/30)],
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
                location=[width*((frame_count-10)/30) + 5*label, height*((frame_count-10)/30)],
                label=str(label),
                frame_number=frame_count*2+50,
            )
    ann2.fname = os.path.join(Path(video_fname).parent, Path(video_fname).stem + "_annotations_pn.json")
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
                location=[width*((frame_count+10)/30) + 5*label, height*((frame_count+10)/30)],
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
    assert all(x in ann.plot_handles for x in ["ax_list_scatter", "ax_list_trace_x", "ax_list_trace_y"])
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


def test_video_annotation_from_multiple_files(video_fname, ann_fname, ann2_fname, ann_h5_fname):
    """Test loading annotations from multiple files."""
    fname_merged = os.path.join(Path(video_fname).parent, Path(video_fname).stem + "_annotations_merged.json")
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


def test_video_annotation_properties(ann_object, ann_object_no_video, ann_object_overlapping):
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
    ann_no_overlap.add([1,1], "0", 0)
    ann_no_overlap.add([1,1], "1", 1)
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
    ann.add([10,10], "0", 0)
    vals_missing = ann.get_values_cv(0)
    assert vals_missing.shape == (2, 1, 2)
    assert np.isnan(vals_missing[1]).all()


def test_video_annotation_frame_num_str(ann_object, ann_object_no_video):
    assert ann_object._n_digits_in_frame_num() == str(len(str(ann_object.n_frames)))
    assert ann_object._frame_num_as_str(5) == f"{5:0{ann_object._n_digits_in_frame_num()}}"
    assert ann_object_no_video._n_digits_in_frame_num() == str(len(str(ann_object_no_video.n_frames)))
    assert ann_object_no_video._frame_num_as_str(5) == f"{5:0{ann_object_no_video._n_digits_in_frame_num()}}"


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
    df_custom = ann_object.to_dlc(output_path=custom_path, file_prefix=custom_prefix, save=True)
    assert (custom_path / f"{custom_prefix}.h5").exists()
    assert (custom_path / f"{custom_prefix}.csv").exists()
    pd.testing.assert_frame_equal(df_no_save, df_custom)

    # Test dlc prefix
    df_dlc_prefix = ann_object.to_dlc(output_path=custom_path, file_prefix="dlc", scorer="tester", save=True)
    assert (custom_path / "CollectedData_tester.h5").exists()
    assert (custom_path / "CollectedData_tester.csv").exists()

    # Test internal_to_dlc_labels
    label_map = {"0": "head", "1": "tail", "2": "mid"}
    df_mapped = ann_object.to_dlc(internal_to_dlc_labels=label_map, save=False)
    assert "head" in df_mapped.columns.get_level_values("bodyparts")
    assert "tail" in df_mapped.columns.get_level_values("bodyparts")
    assert "mid" in df_mapped.columns.get_level_values("bodyparts")
    assert "point0" not in df_mapped.columns.get_level_values("bodyparts")


def test_video_annotation_to_traces(ann_object):
    traces = ann_object.to_traces()
    assert isinstance(traces, dict)
    assert set(traces.keys()) == set(ann_object.labels)
    assert traces["0"].shape == (ann_object.n_frames, 2)
    assert traces["1"].shape == (ann_object.n_frames, 2)
    assert traces["2"].shape == (ann_object.n_frames, 2)
    assert not np.isnan(traces["0"][:10]).any()  # First 10 frames are annotated
    assert np.isnan(traces["0"][10:]).all()  # Rest should be NaN


@patch('datanavigator.pointtracking.pysampled', create=True)  # Mock pysampled if not installed or for isolation
def test_video_annotation_to_signal(mock_pysampled, ann_object, ann_object_no_video):
    mock_data = MagicMock()
    mock_pysampled.Data.return_value = mock_data

    # Test with video
    signal_0 = ann_object.to_signal("0")
    mock_pysampled.Data.assert_called_once()
    call_args, call_kwargs = mock_pysampled.Data.call_args
    np.testing.assert_array_equal(call_args[0], ann_object.to_trace("0"))
    assert call_kwargs['sr'] == ann_object.video.get_avg_fps()
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
    ann.add([1,1], "0", 0)
    assert ann.data["0"] == {0: [1,1]}

    # Create a new instance to test adding labels beyond initial empty ones
    ann_new = datanavigator.VideoAnnotation()
    ann_new.data = {}  # Start truly empty for this test

    # Add label automatically
    ann_new.add_label()
    assert ann_new.labels == ["0"]
    assert ann_new.data["0"] == {}
    assert ann_new.palette[0] is not None  # Color assigned

    # Add specific label
    ann_new.add_label("5")
    assert ann_new.labels == ["0", "5"]
    assert ann_new.data["5"] == {}
    assert ann_new.palette[5] is not None

    # Add with specific color
    ann_new.add_label("2", color=(0.1, 0.2, 0.3))
    assert ann_new.labels == ["0", "2", "5"]
    assert ann_new.data["2"] == {}

    # Add existing label (should fail)
    with pytest.raises(AssertionError):
        ann_new.add_label("0")

    # Add invalid label (non-digit string)
    with pytest.raises(AssertionError):
        ann_new.add_label("a")

    # Add more than 10 labels (fill up first)
    for i in [1, 3, 4, 6, 7, 8, 9]:
         ann_new.add_label(str(i))
    assert len(ann_new.labels) == 10
    with pytest.raises(AssertionError):
         ann_new.add_label()  # Cannot add 11th label automatically
    with pytest.raises(AssertionError):
         ann_new.add_label("10")  # Cannot add specific label > 9


def test_video_annotation_remove(ann_object):
    assert 5 in ann_object.data["0"]
    ann_object.remove("0", 5)
    assert 5 not in ann_object.data["0"]
    # Remove non-existent frame (should not fail)
    ann_object.remove("0", 500)
    # Remove non-existent label (should fail)
    with pytest.raises(AssertionError):
        ann_object.remove("nonexistent", 0)


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
    ann.add([1,1], "0", 0)
    ann.add([1,1], "1", 0)
    ann.add([1,1], "0", 2)
    ann.add([1,1], "1", 2)
    ann.keep_overlapping_continuous_frames()  # Should print warning and do nothing
    assert ann.frames == [0, 2]


@patch('datanavigator.pointtracking.pysampled', create=True)  # Mock pysampled
def test_video_annotation_get_area(mock_pysampled, ann_object, ann_object_no_video):
    mock_data = MagicMock()
    mock_pysampled.Data.return_value = mock_data

    # Test with video -> returns pysampled.Data
    area_signal = ann_object.get_area(labels=["0", "1", "2"])
    assert area_signal == mock_data
    mock_pysampled.Data.assert_called_once()
    call_args, call_kwargs = mock_pysampled.Data.call_args
    assert call_args[0].shape == (ann_object.n_frames,)
    assert call_kwargs['sr'] == ann_object.video.get_avg_fps()

    # Test with string labels
    mock_pysampled.Data.reset_mock()
    area_signal_str = ann_object.get_area(labels="012")
    assert area_signal_str == mock_data
    mock_pysampled.Data.assert_called_once()

    # Test with lowpass
    mock_pysampled.Data.reset_mock()
    mock_signal = MagicMock()
    mock_signal.lowpass.return_value.return_value = ann_object.to_trace("0")  # Mock lowpass output
    with patch.object(ann_object, 'to_signals', return_value={"0": mock_signal, "1": mock_signal, "2": mock_signal}):
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
    assert 'labels_in_ax0' in ann.plot_handles

    # Setup trace
    ann.setup_display_trace(ax_list_trace_x=[ax_x], ax_list_trace_y=[ax_y])
    # Called 10 times for x and 10 times for y (for labels 0-9)
    assert 'trace_in_axx0_label0' in ann.plot_handles
    assert 'trace_in_axy0_label0' in ann.plot_handles

    plt.close(fig)


def test_video_annotation_display_update_visibility(video_fname):
    fig, (ax_img, ax_x, ax_y) = plt.subplots(3, 1)

    ann = datanavigator.VideoAnnotation(vname=video_fname, name="test_display_update_visibility")
    ann.setup_display(ax_list_scatter=[ax_img], ax_list_trace_x=[ax_x], ax_list_trace_y=[ax_y])
    ann.add([1,1], "0", 0)  # Add some data

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

# TODO: Add tests for loading different H5 formats (_dlc_df_to_annotation_dict, _dlc_trace_to_annotation_dict)
# This would require creating fixture H5 files representing labeled data and trace data.


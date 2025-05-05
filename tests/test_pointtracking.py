import pytest
import os
from pathlib import Path

import datanavigator
from datanavigator.examples import get_example_video


@pytest.fixture(scope="module")
def video_fname(tmp_path_factory):
    return get_example_video(dest_folder=str(tmp_path_factory.getbasetemp()))


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
    assert ann.name == "noname" # default name
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

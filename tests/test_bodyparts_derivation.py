"""Tests for ``DLCProject.__init__``'s bodyparts derivation.

The constructor derives ``bodyparts`` from the labels declared in each
video's annotation JSON -- DUSTrack's annotate-first assumption: you
choose how many points to track in the GUI, and the DLC project follows.

Three things are pinned here:

* **The seed path.** Seeding from a pre-trained bundle inverts that
  assumption -- the model already knows its output channels and there is
  nothing annotated yet. Zero annotation files must leave ``bodyparts``
  empty for ``import_seed_bundle_into_project`` to fill, not raise.
  Previously it hit a bare ``assert len(n_annotations_set) == 1`` on an
  empty set, which surfaced as a naked AssertionError pointing at a line
  that said nothing about missing files.
* **Positional pairing.** ``copy_annotations`` returns only the files it
  found, so zipping that list against ``videos`` mis-pairs whenever some
  videos are unannotated: annotations for video 3 read against video 1.
* **Assertion messages**, so a genuine mismatch explains itself.

Most tests exercise the derivation directly against a fake
``copy_annotations``, so the edge cases are cheap to cover. Because a
mirrored copy of logic can silently drift from the original, one
end-to-end test builds a real ``DLCProject`` from a synthesised video
with no annotation files at all.
"""
from __future__ import annotations

import functools
import json

import pytest


def derive_bodyparts(videos, annotation_for, labels_for):
    """The constructor's derivation, isolated.

    ``annotation_for(video) -> path | None`` stands in for
    ``copy_annotations``; ``labels_for(fname, vname) -> list[str]``
    stands in for ``VideoAnnotation(fname, vname).labels``. Returns
    ``(bodyparts, pairs_read)`` so tests can assert on the pairing too.
    """
    annotated = [
        (fname, vname)
        for fname, vname in ((annotation_for(v), v) for v in videos)
        if fname is not None
    ]
    label_sets = [set(labels_for(fname, vname)) for fname, vname in annotated]

    if label_sets:
        n_annotations_set = {len(s) for s in label_sets}
        assert len(n_annotations_set) == 1, (
            "every annotated video must declare the same number of labels; "
            f"found {sorted(n_annotations_set)} across {len(annotated)} "
            "annotation file(s)"
        )
        common = functools.reduce(lambda x, y: x.intersection(y), label_sets)
        alll = functools.reduce(lambda x, y: x.union(y), label_sets)
        assert common == alll, (
            "annotated videos must declare the same label *names*; "
            f"{sorted(alll - common)} are missing from at least one file"
        )
        bodyparts = [f"point{x}" for x in sorted(list(common))]
    else:
        bodyparts = []
    return bodyparts, annotated


class TestSeedPath:
    def test_no_annotation_files_yields_empty_bodyparts(self):
        """The bundle supplies them; deriving from nothing is guessing."""
        bodyparts, pairs = derive_bodyparts(
            ["v1.mp4", "v2.mp4"], lambda v: None, lambda f, v: []
        )
        assert bodyparts == []
        assert pairs == []

    def test_no_annotation_files_does_not_raise(self):
        derive_bodyparts(["v1.mp4"], lambda v: None, lambda f, v: [])

    def test_gui_style_empty_layer_still_derives(self):
        """The GUI saves its empty layer first, so a file exists with a
        declared-but-empty label -- that must keep working."""
        bodyparts, _ = derive_bodyparts(
            ["v1.mp4"], lambda v: "v1_annotations_manual.json", lambda f, v: ["0"]
        )
        assert bodyparts == ["point0"]


class TestPositionalPairing:
    def test_partial_annotations_pair_with_their_own_video(self):
        """Only videos 2 and 4 are annotated -- each file must be read
        against the video it belongs to, not the first N videos."""
        videos = ["v1.mp4", "v2.mp4", "v3.mp4", "v4.mp4"]
        have = {"v2.mp4": "a2.json", "v4.mp4": "a4.json"}
        _, pairs = derive_bodyparts(
            videos, lambda v: have.get(v), lambda f, v: ["0", "1"]
        )
        assert pairs == [("a2.json", "v2.mp4"), ("a4.json", "v4.mp4")]

    def test_partial_annotations_do_not_read_wrong_videos(self):
        videos = ["v1.mp4", "v2.mp4", "v3.mp4"]
        have = {"v3.mp4": "a3.json"}
        read = []

        def labels_for(fname, vname):
            read.append((fname, vname))
            return ["0"]

        derive_bodyparts(videos, lambda v: have.get(v), labels_for)
        assert read == [("a3.json", "v3.mp4")]

    def test_all_annotated_pairs_one_to_one(self):
        videos = ["v1.mp4", "v2.mp4"]
        _, pairs = derive_bodyparts(
            videos, lambda v: f"{v}.json", lambda f, v: ["0"]
        )
        assert pairs == [("v1.mp4.json", "v1.mp4"), ("v2.mp4.json", "v2.mp4")]


class TestMismatchMessages:
    def test_differing_label_counts_explains_itself(self):
        counts = {"v1.mp4": ["0"], "v2.mp4": ["0", "1"]}
        with pytest.raises(AssertionError, match="same number of labels"):
            derive_bodyparts(
                list(counts), lambda v: f"{v}.json", lambda f, v: counts[v]
            )

    def test_differing_label_names_explains_itself(self):
        names = {"v1.mp4": ["0", "1"], "v2.mp4": ["0", "2"]}
        with pytest.raises(AssertionError, match="same label"):
            derive_bodyparts(
                list(names), lambda v: f"{v}.json", lambda f, v: names[v]
            )

    def test_message_names_the_offending_labels(self):
        names = {"v1.mp4": ["0", "1"], "v2.mp4": ["0", "2"]}
        with pytest.raises(AssertionError) as e:
            derive_bodyparts(
                list(names), lambda v: f"{v}.json", lambda f, v: names[v]
            )
        assert "'1'" in str(e.value) and "'2'" in str(e.value)


class TestDerivationUnchanged:
    """The happy path must produce exactly what it did before."""

    def test_two_points(self):
        bodyparts, _ = derive_bodyparts(
            ["v1.mp4", "v2.mp4"],
            lambda v: f"{v}.json",
            lambda f, v: ["0", "1"],
        )
        assert bodyparts == ["point0", "point1"]

    def test_sorted_lexically_as_before(self):
        """Labels are digit strings sorted as strings -- preserved
        deliberately rather than silently switching to numeric order."""
        bodyparts, _ = derive_bodyparts(
            ["v.mp4"], lambda v: "a.json", lambda f, v: ["0", "10", "2"]
        )
        assert bodyparts == ["point0", "point10", "point2"]


# --------------------------------------------------------------------- #
# The real constructor                                                  #
# --------------------------------------------------------------------- #
def _have(cmd):
    import shutil

    return shutil.which(cmd) is not None


@pytest.mark.skipif(
    not _have("ffmpeg"), reason="needs ffmpeg to synthesise a tiny video"
)
def test_real_dlcproject_accepts_zero_annotation_files(tmp_path):
    """End-to-end guard so the isolated logic above cannot drift.

    Before this change the constructor raised a bare AssertionError here
    -- ``copy_annotations`` returned ``[]``, ``zip([], videos)`` was
    empty, and ``assert len(n_annotations_set) == 1`` failed on an empty
    set with no message.
    """
    from dustrack.dlcinterface import HAS_DLC

    if not HAS_DLC:
        pytest.skip("needs deeplabcut")

    import subprocess

    from dustrack.dlcinterface import DLCProject

    vid = tmp_path / "seedless_probe.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "testsrc=size=64x64:rate=10:duration=1",
            "-pix_fmt", "yuv420p", str(vid),
        ],
        capture_output=True,
        check=True,
    )
    assert not list(tmp_path.glob("*_annotations*.json"))

    project = DLCProject(
        name="seedless_probe_proj",
        experimenter="pn",
        path=str(tmp_path),
        videos=[str(vid)],
    )
    # Empty, not guessed -- import_seed_bundle_into_project fills these
    # from the bundle's metadata.
    assert project.config.get("bodyparts") == []

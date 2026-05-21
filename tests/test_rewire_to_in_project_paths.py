"""Tests for :meth:`DUSTrack._rewire_to_in_project_paths`.

After ``create_dlc_project`` builds a DLC project, the live session
should switch all path references from the original (top-level) video +
annotation files to the in-project copies, so subsequent writes
(``apply_manual_corrections``, ``process_with_lk``,
``save_annotation_as``) land inside the project rather than next to the
original video.

The rewire helper is pure-ish (operates on ``self.fname`` and each
annotation layer's ``.fname`` / ``.fstem``) so it can be exercised
without launching the GUI by stubbing ``self`` with a small fake.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from dustrack.dlcinterface import DUSTrack


def _make_ann(fname, data=None):
    """A faux VideoAnnotation: just the attributes _rewire reads / writes."""
    a = SimpleNamespace()
    a.fname = str(fname)
    a.fstem = Path(fname).stem
    a.data = data or {}
    a.saved_to = None  # records the path that save() would have written to

    def save(path=None):
        target = path if path is not None else a.fname
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_text("{}", encoding="utf-8")
        a.saved_to = target

    a.save = save
    return a


def _make_project_layout(tmp_path):
    """Lay down a minimal project that satisfies ``_rewire``'s
    expectations: a ``videos/`` folder with a copied video file and the
    active-layer annotation already copied in.
    """
    top = tmp_path / "top"
    top.mkdir()
    orig_video = top / "ex.mp4"
    orig_video.write_bytes(b"\x00")

    project = top / "ex_manual-x-2026-05-18"
    project.mkdir()
    (project / "videos").mkdir()
    project_video = project / "videos" / "ex.mp4"
    project_video.write_bytes(b"\x00")
    return top, orig_video, project, project_video


class TestRewireToInProjectPaths:
    def test_self_fname_switches_to_in_project_video(self, tmp_path):
        top, orig_video, project, project_video = _make_project_layout(tmp_path)
        fake = SimpleNamespace()
        fake.fname = str(orig_video)
        fake.annotations = []
        fake._dlcproject = SimpleNamespace(
            video_list=[str(project_video)],
            path=str(project),
        )
        DUSTrack._rewire_to_in_project_paths(fake)
        assert fake.fname == str(project_video)

    def test_outside_project_json_migrates_to_videos_dir(self, tmp_path):
        top, orig_video, project, project_video = _make_project_layout(tmp_path)
        manual_path = top / "ex_annotations_manual.json"
        manual_path.write_text("{}", encoding="utf-8")
        ann = _make_ann(manual_path, data={"0": {0: [1.0, 2.0]}})
        fake = SimpleNamespace()
        fake.fname = str(orig_video)
        fake.annotations = [ann]
        fake._dlcproject = SimpleNamespace(
            video_list=[str(project_video)],
            path=str(project),
        )
        DUSTrack._rewire_to_in_project_paths(fake)
        expected = project / "videos" / "ex_annotations_manual.json"
        assert ann.fname == str(expected)
        assert ann.fstem == expected.stem
        # in-memory data was non-empty -> save() was triggered at the
        # new path so subsequent reloads have a destination on disk.
        assert ann.saved_to == str(expected)
        assert expected.exists()

    def test_inside_project_path_left_alone(self, tmp_path):
        top, orig_video, project, project_video = _make_project_layout(tmp_path)
        inside_path = project / "videos" / "iteration-0" / "exDLC_x_50.h5"
        inside_path.parent.mkdir(parents=True, exist_ok=True)
        inside_path.write_bytes(b"")
        ann = _make_ann(inside_path, data={"0": {0: [1.0, 2.0]}})
        fake = SimpleNamespace()
        fake.fname = str(orig_video)
        fake.annotations = [ann]
        fake._dlcproject = SimpleNamespace(
            video_list=[str(project_video)],
            path=str(project),
        )
        DUSTrack._rewire_to_in_project_paths(fake)
        assert ann.fname == str(inside_path)
        assert ann.saved_to is None  # no save() fired

    def test_non_json_outside_project_skipped(self, tmp_path):
        # DLC h5 files always live inside a project; if a phantom .h5
        # somehow exists outside, leave it alone rather than copying it.
        top, orig_video, project, project_video = _make_project_layout(tmp_path)
        h5_path = top / "stray.h5"
        h5_path.write_bytes(b"")
        ann = _make_ann(h5_path, data={"0": {0: [1.0, 2.0]}})
        fake = SimpleNamespace()
        fake.fname = str(orig_video)
        fake.annotations = [ann]
        fake._dlcproject = SimpleNamespace(
            video_list=[str(project_video)],
            path=str(project),
        )
        DUSTrack._rewire_to_in_project_paths(fake)
        assert ann.fname == str(h5_path)
        assert ann.saved_to is None

    def test_empty_layer_rewires_path_but_skips_save(self, tmp_path):
        # The buffer layer is typically empty. We still want its .fname
        # to migrate so a later add() + save() lands inside the project,
        # but we should NOT preemptively materialise an empty file.
        top, orig_video, project, project_video = _make_project_layout(tmp_path)
        buffer_path = top / "ex_annotations_buffer.json"
        buffer_path.write_text("{}", encoding="utf-8")
        ann = _make_ann(buffer_path, data={"0": {}})  # label present but empty
        fake = SimpleNamespace()
        fake.fname = str(orig_video)
        fake.annotations = [ann]
        fake._dlcproject = SimpleNamespace(
            video_list=[str(project_video)],
            path=str(project),
        )
        DUSTrack._rewire_to_in_project_paths(fake)
        expected = project / "videos" / "ex_annotations_buffer.json"
        assert ann.fname == str(expected)
        assert ann.saved_to is None
        assert not expected.exists()

    def test_outside_project_migrates_even_when_dlcproject_path_is_working_dir(self, tmp_path):
        """Regression for the train-pre-flight labeled-data bug.

        ``DLCProject.path`` is set to whatever was passed to the
        constructor -- for a freshly-created project that's the WORKING
        DIRECTORY (``deeplabcut.create_new_project`` creates the project
        at ``<working_dir>/<name>-<experimenter>-<date>/``). Pre-fix the
        rewire used ``self._dlcproject.path`` as ``project_root`` and the
        ``ann_path.relative_to(project_root)`` check returned a
        false-positive "already inside" for any annotation file sitting
        next to the original video at top-level (the working dir IS a
        prefix of that path). Layers stranded at their original
        locations; train pre-flight saved cleaned annotations there;
        ``extract_frames`` read the stale project copy and propagated
        the dropped point into ``labeled_data``.

        Fix: derive ``project_root`` from the in-project video path
        (``videos_dir.parent``), so it always points at the actual
        project directory regardless of what was passed to
        ``DLCProject(path=...)``.
        """
        top, orig_video, project, project_video = _make_project_layout(tmp_path)
        manual_path = top / "ex_annotations_manual.json"
        manual_path.write_text("{}", encoding="utf-8")
        ann = _make_ann(manual_path, data={"0": {0: [1.0, 2.0]}})
        fake = SimpleNamespace()
        fake.fname = str(orig_video)
        fake.annotations = [ann]
        # NOTE: ``path=str(top)`` -- mirrors the real production state
        # where ``DLCProject.__init__`` stored the working directory,
        # not the project directory. Pre-fix this would have caused
        # ``ex_annotations_manual.json`` to be treated as "already
        # inside the project tree" and skip migration.
        fake._dlcproject = SimpleNamespace(
            video_list=[str(project_video)],
            path=str(top),  # the bug-trigger
        )
        DUSTrack._rewire_to_in_project_paths(fake)
        expected = project / "videos" / "ex_annotations_manual.json"
        assert ann.fname == str(expected), (
            f"ann.fname should have migrated into the project's videos/ folder, "
            f"got {ann.fname!r}"
        )
        assert expected.exists()

    def test_layer_without_fname_skipped(self, tmp_path):
        top, orig_video, project, project_video = _make_project_layout(tmp_path)
        ann = SimpleNamespace(fname=None, fstem=None, data={}, save=lambda *a: None)
        fake = SimpleNamespace()
        fake.fname = str(orig_video)
        fake.annotations = [ann]
        fake._dlcproject = SimpleNamespace(
            video_list=[str(project_video)],
            path=str(project),
        )
        DUSTrack._rewire_to_in_project_paths(fake)
        # fname stays None, no crash
        assert ann.fname is None

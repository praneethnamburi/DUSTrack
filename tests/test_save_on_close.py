"""Tests for the rc2 save-on-close guard.

When the user closes the DUSTrack QMainWindow, an installed
``closeEvent`` patch scans every manual annotation layer for
in-memory-vs-disk diffs and (if any are found) prompts the user
with *Save all / Discard / Cancel*. The pure-data helpers
(:meth:`DUSTrack._scan_unsaved_layers`, :meth:`_format_unsaved_summary`)
are unit-tested here; the live Qt wiring (``_install_close_guard``,
``_prompt_save_on_close``) requires an interactive session.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from dustrack.dlcinterface import DUSTrack


# --- _format_unsaved_summary -------------------------------------------------

class TestFormatUnsavedSummary:
    def test_single_layer_all_three_kinds(self):
        unsaved = {
            "pn": {
                "added": [("0", 5), ("0", 6)],
                "removed": [("1", 3)],
                "modified": [("2", 7)],
            }
        }
        out = DUSTrack._format_unsaved_summary(unsaved)
        assert "'pn'" in out
        assert "+2 added" in out
        assert "-1 removed" in out
        assert "~1 modified" in out

    def test_layer_with_only_additions(self):
        unsaved = {"iter1": {"added": [("0", 1)], "removed": [], "modified": []}}
        out = DUSTrack._format_unsaved_summary(unsaved)
        assert "+1 added" in out
        assert "removed" not in out
        assert "modified" not in out

    def test_multiple_layers_each_on_own_line(self):
        unsaved = {
            "iter1": {"added": [("0", 1)], "removed": [], "modified": []},
            "iter2": {"added": [], "removed": [("0", 2)], "modified": []},
        }
        out = DUSTrack._format_unsaved_summary(unsaved)
        lines = out.splitlines()
        assert len(lines) == 2
        assert any("'iter1'" in l for l in lines)
        assert any("'iter2'" in l for l in lines)

    def test_empty_unsaved_returns_empty_string(self):
        assert DUSTrack._format_unsaved_summary({}) == ""


# --- _scan_unsaved_layers ----------------------------------------------------
# The scan walks self.annotations, filters by file pattern via
# _is_manual_annotation_layer, normalizes both sides, and diffs. We build
# a minimal stub object exposing only the attributes the method reads
# rather than spinning up a live DUSTrack instance.


def _make_ann_stub(name, fname, data):
    """Minimal ``VideoAnnotation``-shaped stub for the scan."""
    return SimpleNamespace(name=name, fname=str(fname), data=data)


def _scan_with_stub_self(video_fname, ann_stubs):
    """Invoke ``DUSTrack._scan_unsaved_layers`` against a stub ``self``.

    Bound-method invocation keeps the real method under test;
    the staticmethod helpers it relies on are bound onto the stub
    so the implementation does not have to know we're not a real
    :class:`DUSTrack`.
    """
    stub = SimpleNamespace(
        fname=str(video_fname),
        annotations=ann_stubs,
        _is_manual_annotation_layer=DUSTrack._is_manual_annotation_layer,
        _normalize_layer_data=DUSTrack._normalize_layer_data,
        _load_layer_disk_data=DUSTrack._load_layer_disk_data,
        _diff_ann_vs_disk=DUSTrack._diff_ann_vs_disk,
    )
    return DUSTrack._scan_unsaved_layers(stub)


class TestScanUnsavedLayers:
    def test_clean_layer_omitted(self, tmp_path):
        video = tmp_path / "v.mp4"
        ann_path = tmp_path / "v_annotations_pn.json"
        # Persist the same data to disk
        import json
        data = {"0": {5: [1.0, 1.0]}}
        with open(ann_path, "w") as f:
            json.dump({"0": {"5": [1.0, 1.0]}}, f)
        ann = _make_ann_stub("pn", ann_path, data)
        result = _scan_with_stub_self(video, [ann])
        assert result == {}

    def test_layer_with_added_frame_in_memory(self, tmp_path):
        video = tmp_path / "v.mp4"
        ann_path = tmp_path / "v_annotations_pn.json"
        import json
        with open(ann_path, "w") as f:
            json.dump({"0": {"5": [1.0, 1.0]}}, f)
        # In-memory has an extra frame on disk doesn't.
        data = {"0": {5: [1.0, 1.0], 6: [2.0, 2.0]}}
        ann = _make_ann_stub("pn", ann_path, data)
        result = _scan_with_stub_self(video, [ann])
        assert "pn" in result
        assert ("0", 6) in result["pn"]["added"]

    def test_layer_with_removed_frame_in_memory(self, tmp_path):
        video = tmp_path / "v.mp4"
        ann_path = tmp_path / "v_annotations_pn.json"
        import json
        with open(ann_path, "w") as f:
            json.dump({"0": {"5": [1.0, 1.0], "6": [2.0, 2.0]}}, f)
        data = {"0": {5: [1.0, 1.0]}}  # frame 6 deleted in memory
        ann = _make_ann_stub("pn", ann_path, data)
        result = _scan_with_stub_self(video, [ann])
        assert "pn" in result
        assert ("0", 6) in result["pn"]["removed"]

    def test_layer_with_modified_value(self, tmp_path):
        video = tmp_path / "v.mp4"
        ann_path = tmp_path / "v_annotations_pn.json"
        import json
        with open(ann_path, "w") as f:
            json.dump({"0": {"5": [1.0, 1.0]}}, f)
        data = {"0": {5: [9.0, 9.0]}}  # same frame, different coordinates
        ann = _make_ann_stub("pn", ann_path, data)
        result = _scan_with_stub_self(video, [ann])
        assert "pn" in result
        assert ("0", 5) in result["pn"]["modified"]

    def test_dlc_layer_excluded(self, tmp_path):
        # DLC trace / dlccorr layers are not manual annotation layers,
        # so the close-guard scan ignores them even when in-memory
        # differs from disk (they regenerate from the model).
        video = tmp_path / "v.mp4"
        ann_path = tmp_path / "v_annotations_dlccorr.json"
        import json
        with open(ann_path, "w") as f:
            json.dump({}, f)
        data = {"0": {5: [1.0, 1.0]}}
        ann = _make_ann_stub("dlccorr", ann_path, data)
        result = _scan_with_stub_self(video, [ann])
        assert result == {}

    def test_layer_with_no_disk_file_treated_as_fully_unsaved(self, tmp_path):
        video = tmp_path / "v.mp4"
        ann_path = tmp_path / "v_annotations_pn.json"  # does not exist
        data = {"0": {5: [1.0, 1.0]}}
        ann = _make_ann_stub("pn", ann_path, data)
        result = _scan_with_stub_self(video, [ann])
        assert "pn" in result
        # Both frames count as added relative to empty-on-disk.
        assert ("0", 5) in result["pn"]["added"]

    def test_multiple_layers_only_dirty_ones_returned(self, tmp_path):
        video = tmp_path / "v.mp4"
        clean_path = tmp_path / "v_annotations_pn.json"
        dirty_path = tmp_path / "v_annotations_iter1.json"
        import json
        with open(clean_path, "w") as f:
            json.dump({"0": {"5": [1.0, 1.0]}}, f)
        with open(dirty_path, "w") as f:
            json.dump({"0": {"5": [1.0, 1.0]}}, f)
        clean = _make_ann_stub("pn", clean_path, {"0": {5: [1.0, 1.0]}})
        dirty = _make_ann_stub("iter1", dirty_path, {"0": {5: [9.9, 9.9]}})
        result = _scan_with_stub_self(video, [clean, dirty])
        assert "pn" not in result
        assert "iter1" in result


# --- close-event dispatch ----------------------------------------------------
# Validate the cancel-default behaviour of the modal mapping and the
# unsaved-empty short-circuit. Live Qt is mocked.


class TestCloseEventDispatch:
    def test_no_unsaved_falls_through_immediately(self, monkeypatch):
        """When the scan returns empty, the original closeEvent fires
        without any prompt."""
        from dustrack.dlcinterface import DUSTrack as D

        # Build a minimal stub
        events_passed = []

        class StubWindow:
            _dustrack_close_guard_installed = False

            def closeEvent(self, event):
                events_passed.append(event)

        win = StubWindow()
        dustrack = SimpleNamespace(
            _find_qt_window=lambda: win,
            _scan_unsaved_layers=lambda: {},
            _prompt_save_on_close=lambda *a, **k: pytest.fail(
                "should not prompt when nothing unsaved"
            ),
            _save_unsaved_layers=lambda *a, **k: pytest.fail(
                "should not save when nothing unsaved"
            ),
        )
        D._install_close_guard(dustrack)
        # Trigger the patched closeEvent
        evt = SimpleNamespace(ignore=lambda: pytest.fail("should not ignore"))
        win.closeEvent(evt)
        assert events_passed == [evt]

    def test_cancel_prevents_close(self):
        """*Cancel* calls ``event.ignore()`` and skips the original."""
        from dustrack.dlcinterface import DUSTrack as D

        ignored = []
        original_called = []

        class StubWindow:
            _dustrack_close_guard_installed = False

            def closeEvent(self, event):
                original_called.append(event)

        win = StubWindow()
        dustrack = SimpleNamespace(
            _find_qt_window=lambda: win,
            _scan_unsaved_layers=lambda: {"pn": {"added": [("0", 1)]}},
            _prompt_save_on_close=lambda *a, **k: "cancel",
            _save_unsaved_layers=lambda *a, **k: pytest.fail("must not save on cancel"),
        )
        D._install_close_guard(dustrack)

        evt = SimpleNamespace(ignore=lambda: ignored.append(True))
        win.closeEvent(evt)
        assert ignored == [True]
        assert original_called == []  # original closeEvent did NOT fire

    def test_save_persists_then_closes(self):
        from dustrack.dlcinterface import DUSTrack as D

        saved = []
        original_called = []

        class StubWindow:
            _dustrack_close_guard_installed = False

            def closeEvent(self, event):
                original_called.append(event)

        win = StubWindow()
        dustrack = SimpleNamespace(
            _find_qt_window=lambda: win,
            _scan_unsaved_layers=lambda: {"pn": {"added": [("0", 1)]}},
            _prompt_save_on_close=lambda *a, **k: "save",
            _save_unsaved_layers=lambda u: saved.append(u),
        )
        D._install_close_guard(dustrack)

        evt = SimpleNamespace(ignore=lambda: pytest.fail("save path must close"))
        win.closeEvent(evt)
        assert saved == [{"pn": {"added": [("0", 1)]}}]
        assert original_called == [evt]

    def test_discard_skips_save_but_closes(self):
        from dustrack.dlcinterface import DUSTrack as D

        original_called = []

        class StubWindow:
            _dustrack_close_guard_installed = False

            def closeEvent(self, event):
                original_called.append(event)

        win = StubWindow()
        dustrack = SimpleNamespace(
            _find_qt_window=lambda: win,
            _scan_unsaved_layers=lambda: {"pn": {"added": [("0", 1)]}},
            _prompt_save_on_close=lambda *a, **k: "discard",
            _save_unsaved_layers=lambda *a, **k: pytest.fail(
                "discard must not save"
            ),
        )
        D._install_close_guard(dustrack)

        evt = SimpleNamespace(ignore=lambda: pytest.fail("discard path must close"))
        win.closeEvent(evt)
        assert original_called == [evt]

    def test_install_is_idempotent(self):
        """A second install pass (e.g. subclass __init__ re-entry) must
        not stack handlers."""
        from dustrack.dlcinterface import DUSTrack as D

        original_calls = []

        class StubWindow:
            _dustrack_close_guard_installed = False

            def closeEvent(self, event):
                original_calls.append(event)

        win = StubWindow()
        dustrack = SimpleNamespace(
            _find_qt_window=lambda: win,
            _scan_unsaved_layers=lambda: {},
            _prompt_save_on_close=lambda *a, **k: "cancel",
            _save_unsaved_layers=lambda *a, **k: None,
        )
        D._install_close_guard(dustrack)
        first_handler = win.closeEvent
        D._install_close_guard(dustrack)  # second call: no-op
        assert win.closeEvent is first_handler

    def test_scan_exception_does_not_block_close(self):
        """If the scan itself raises (e.g. annotations list torn down
        mid-shutdown), the guard must not strand the user with an
        un-closeable window."""
        from dustrack.dlcinterface import DUSTrack as D

        original_called = []

        class StubWindow:
            _dustrack_close_guard_installed = False

            def closeEvent(self, event):
                original_called.append(event)

        win = StubWindow()

        def boom():
            raise RuntimeError("annotations gone")

        dustrack = SimpleNamespace(
            _find_qt_window=lambda: win,
            _scan_unsaved_layers=boom,
            _prompt_save_on_close=lambda *a, **k: pytest.fail("no prompt expected"),
            _save_unsaved_layers=lambda *a, **k: None,
        )
        D._install_close_guard(dustrack)

        evt = SimpleNamespace(ignore=lambda: pytest.fail("must not ignore"))
        win.closeEvent(evt)
        assert original_called == [evt]

    def test_install_no_qt_window_is_noop(self):
        """mpl-fallback path: no Qt window means nothing to patch."""
        from dustrack.dlcinterface import DUSTrack as D

        dustrack = SimpleNamespace(_find_qt_window=lambda: None)
        # Should not raise.
        D._install_close_guard(dustrack)

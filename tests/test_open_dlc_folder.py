"""Tests for the dual-purpose Create-DLC-Project / Open-DLC-Folder button.

Inside a project, creating another one is meaningless -- but a
permanently greyed-out button is dead sidebar space. It repurposes to
"Open DLC Folder", which is what you actually want once a project
exists: predictions under ``videos/iteration-N/``, training frames under
``labeled-data/``, snapshots under ``dlc-models-pytorch/``.

The dispatch and the caption must agree on the same condition, so both
consult ``_session_inside_dlc_project``. These tests pin that agreement
-- a caption saying "Open DLC Folder" over an action that tries to
create a project would be worse than the greyed-out button it replaced.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from dustrack.dlcinterface import DUSTrack


def _fake(inside=None):
    """A stub exposing only what the dispatch reads."""
    return SimpleNamespace(
        _inside=inside,
        create_called=[],
        open_called=[],
    )


class TestDispatch:
    def test_outside_project_creates(self):
        fake = _fake(inside=None)
        fake.create_dlc_project = lambda *a, **k: fake.create_called.append(1)
        fake.open_dlc_folder = lambda: fake.open_called.append(1)
        with patch(
            "dustrack._dlc_paths._session_inside_dlc_project", return_value=None
        ):
            DUSTrack._create_or_open_dlc_project(fake)
        assert fake.create_called and not fake.open_called

    def test_inside_project_opens_folder(self, tmp_path):
        fake = _fake(inside=tmp_path)
        fake.create_dlc_project = lambda *a, **k: fake.create_called.append(1)
        fake.open_dlc_folder = lambda: fake.open_called.append(1)
        with patch(
            "dustrack._dlc_paths._session_inside_dlc_project", return_value=tmp_path
        ):
            DUSTrack._create_or_open_dlc_project(fake)
        assert fake.open_called and not fake.create_called

    def test_dispatch_forwards_args_when_creating(self):
        seen = {}
        fake = _fake()
        fake.create_dlc_project = lambda *a, **k: seen.update(args=a, kwargs=k)
        with patch(
            "dustrack._dlc_paths._session_inside_dlc_project", return_value=None
        ):
            DUSTrack._create_or_open_dlc_project(fake, "x", name="proj")
        assert seen["args"] == ("x",)
        assert seen["kwargs"] == {"name": "proj"}


class TestOpenFolder:
    def test_opens_the_project_root(self, tmp_path):
        fake = SimpleNamespace()
        with patch(
            "dustrack._dlc_paths._session_inside_dlc_project", return_value=tmp_path
        ), patch("dustrack.gui.sys.platform", "win32"), patch(
            "dustrack.gui.os.startfile", create=True
        ) as start:
            out = DUSTrack.open_dlc_folder(fake)
        start.assert_called_once_with(str(tmp_path))
        assert out == tmp_path

    def test_no_project_is_a_message_not_a_crash(self, capsys):
        fake = SimpleNamespace()
        with patch(
            "dustrack._dlc_paths._session_inside_dlc_project", return_value=None
        ):
            out = DUSTrack.open_dlc_folder(fake)
        assert out is None
        assert "Not inside a DLC project" in capsys.readouterr().out

    def test_failure_to_open_does_not_crash_the_gui(self, tmp_path, capsys):
        """A file-browser failure must not take the session down."""
        fake = SimpleNamespace()
        with patch(
            "dustrack._dlc_paths._session_inside_dlc_project", return_value=tmp_path
        ), patch("dustrack.gui.sys.platform", "win32"), patch(
            "dustrack.gui.os.startfile", create=True, side_effect=OSError("nope")
        ):
            out = DUSTrack.open_dlc_folder(fake)
        assert out is None
        assert "Could not open" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "platform,expected", [("darwin", "open"), ("linux", "xdg-open")]
    )
    def test_non_windows_uses_the_platform_opener(self, tmp_path, platform, expected):
        fake = SimpleNamespace()
        with patch(
            "dustrack._dlc_paths._session_inside_dlc_project", return_value=tmp_path
        ), patch("dustrack.gui.sys.platform", platform), patch(
            "dustrack.gui.subprocess.Popen"
        ) as popen:
            DUSTrack.open_dlc_folder(fake)
        popen.assert_called_once_with([expected, str(tmp_path)])

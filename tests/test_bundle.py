"""Tests for the :class:`_BundleState` dataclass that backs the
1.2.0a3 multi-video swap.

Pure-Python dataclass; no Qt / dnav / matplotlib involved. Verifies
the state-machine enum is enforced, defaults are sane, and the
``is_ready`` / ``is_terminal`` predicates report correctly.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from dustrack._bundle import (
    HYDRATION_FAILED,
    HYDRATION_HYDRATING,
    HYDRATION_PENDING,
    HYDRATION_READY,
    _BundleState,
)


class TestBundleConstruction:
    def test_minimum_fields(self):
        b = _BundleState(fname=Path("S:/p/v.mp4"), video_index=3)
        assert b.fname == Path("S:/p/v.mp4")
        assert b.video_index == 3
        # Defaults for the lifecycle / heavy state.
        assert b.hydration_state == HYDRATION_PENDING
        assert b.hydration_error is None
        assert b.reader is None
        assert b.annotations is None
        # Defaults for the UI snapshot.
        assert b.current_idx == 0
        assert b.selections == {}
        assert b.ax_lims == {
            "state": False,
            "x": [None, None],
            "y_trace_x": [None, None],
            "y_trace_y": [None, None],
        }
        assert b.image_view_state is None
        assert b.frames_of_interest == []

    def test_str_fname_promoted_to_path(self):
        b = _BundleState(fname="S:/p/v.mp4", video_index=0)
        assert isinstance(b.fname, Path)

    def test_unknown_hydration_state_rejected(self):
        with pytest.raises(ValueError, match="unknown hydration_state"):
            _BundleState(
                fname=Path("v.mp4"), video_index=0,
                hydration_state="bogus",
            )

    def test_ax_lims_default_isolation(self):
        """Two separately-defaulted bundles must not share the inner
        ax_lims list (the default_factory pattern is supposed to
        protect against shared-mutable-default leaks)."""
        a = _BundleState(fname=Path("a.mp4"), video_index=0)
        b = _BundleState(fname=Path("b.mp4"), video_index=1)
        a.ax_lims["x"][0] = 42
        assert b.ax_lims["x"] == [None, None]


class TestHydrationPredicates:
    @pytest.mark.parametrize("state,ready,terminal", [
        (HYDRATION_PENDING, False, False),
        (HYDRATION_HYDRATING, False, False),
        (HYDRATION_READY, True, True),
        (HYDRATION_FAILED, False, True),
    ])
    def test_predicates(self, state, ready, terminal):
        b = _BundleState(
            fname=Path("v.mp4"), video_index=0, hydration_state=state,
        )
        assert b.is_ready is ready
        assert b.is_terminal is terminal

"""Tests for the 1.2.0a3 bundle-list management surface
(``add_video`` / ``remove_video`` / ``replace_active_with``).

These tests cover the orchestration -- input validation, bundle-list
mutation, swap dispatch, renumbering -- without spinning up a full
DUSTrack. The hydration helpers (``_hydrate_phase1_bundle_data`` /
``_hydrate_bundle_data_only`` / ``_finalise_bundle_artists``) are
exercised by the existing integration / multi-video tests; here we
stub them out and assert the public methods route correctly.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from dustrack import dlcinterface
from dustrack._bundle import (
    HYDRATION_PENDING,
    HYDRATION_READY,
    _BundleState,
)


# ---------------------------------------------------------------------
# Stub-DUSTrack scaffolding
# ---------------------------------------------------------------------


class _StubBrowser:
    """Minimal stand-in that exposes just enough of the
    :class:`DUSTrack` surface for the bundle-API methods to operate.
    The actual hydration is replaced with a no-op that flips bundles
    to READY immediately so we can assert against the post-add list
    without real I/O.
    """

    def __init__(self):
        self._bundles = []
        self._active_index = 0
        self._video_queue = []
        self._hydration_worker = None
        self.fname = None
        self.data = None
        self.annotations = None
        self._dlcproject = None
        self._current_idx = 0
        self._ax_lims = {"state": False}
        self.frames_of_interest = []
        # Track swap_to calls for assertions.
        self.swap_to_calls = []
        self.refresh_calls = 0

    def _hydrate_bundle_sync(self, bundle, project=None):
        # Pretend hydration succeeded: heavy fields populated with
        # sentinels.
        bundle.reader = SimpleNamespace(__len__=lambda self=None: 100)
        bundle.annotations = SimpleNamespace(
            names=["iteration-0", "buffer"],
        )
        bundle.hydration_state = HYDRATION_READY
        bundle.hydration_error = None
        bundle.selections = {}

    def swap_to(self, index):
        self.swap_to_calls.append(index)
        if not (0 <= index < len(self._bundles)):
            return False
        self._active_index = index
        return True

    def _park_bundle_artists(self, bundle):
        pass

    def _refresh_nav_buttons(self):
        self.refresh_calls += 1


# Bind the real methods onto the stub class so we test the actual
# implementation logic.
def _bind_methods():
    from dustrack.dlcinterface import DUSTrack
    for name in (
        "add_video", "remove_video", "replace_active_with",
        "_validate_bundle_paths",
    ):
        setattr(_StubBrowser, name, getattr(DUSTrack, name))


_bind_methods()


@pytest.fixture
def stub_tracker(monkeypatch):
    """Fresh stub with a single seeded Phase 1 bundle (the seed-mode
    starting state). Tests add to / remove from this list."""
    tracker = _StubBrowser()
    # Pre-populate one bundle (the "seed" bundle in the seed-modal
    # analogue; for non-seed tests, just bundle 0 of a real tracker).
    seed = _BundleState(
        fname=Path("/seed.mp4"),
        video_index=0,
        project=None,
        hydration_state=HYDRATION_READY,
    )
    seed.annotations = SimpleNamespace(names=["_seed", "buffer"])
    seed.reader = SimpleNamespace(__len__=lambda self=None: 8)
    tracker._bundles = [seed]
    tracker.fname = str(seed.fname)
    tracker.annotations = seed.annotations
    tracker.data = seed.reader
    return tracker


# ---------------------------------------------------------------------
# add_video
# ---------------------------------------------------------------------


class TestAddVideo:
    def test_phase1_bare_video_appends_bundle(self, stub_tracker, tmp_path):
        v = tmp_path / "real.mp4"
        v.write_bytes(b"")
        indices = stub_tracker.add_video(v)
        assert indices == [1]
        assert len(stub_tracker._bundles) == 2
        new_bundle = stub_tracker._bundles[1]
        assert new_bundle.fname == v
        assert new_bundle.project is None  # Phase 1
        assert new_bundle.is_ready
        # _video_queue back-compat tracked.
        assert stub_tracker._video_queue == [v]
        # Did NOT auto-swap (set_active=False is the public default).
        assert stub_tracker._active_index == 0

    def test_set_active_swaps_after_append(self, stub_tracker, tmp_path):
        v = tmp_path / "real.mp4"
        v.write_bytes(b"")
        indices = stub_tracker.add_video(v, set_active=True)
        assert indices == [1]
        # swap_to(1) was called.
        assert stub_tracker.swap_to_calls == [1]
        assert stub_tracker._active_index == 1

    def test_empty_list_raises(self, stub_tracker):
        with pytest.raises(ValueError, match="empty path sequence"):
            stub_tracker.add_video([])

    def test_missing_file_raises_filenotfound(self, stub_tracker, tmp_path):
        with pytest.raises(FileNotFoundError):
            stub_tracker.add_video(tmp_path / "nope.mp4")

    def test_single_element_list_dispatch_equiv_to_scalar(
        self, stub_tracker, tmp_path,
    ):
        v = tmp_path / "real.mp4"
        v.write_bytes(b"")
        stub_tracker.add_video([v])
        assert len(stub_tracker._bundles) == 2
        assert stub_tracker._bundles[1].fname == v
        assert stub_tracker._bundles[1].project is None


# ---------------------------------------------------------------------
# remove_video
# ---------------------------------------------------------------------


class TestRemoveVideo:
    def test_remove_non_active_drops_bundle(self, stub_tracker, tmp_path):
        v = tmp_path / "extra.mp4"
        v.write_bytes(b"")
        stub_tracker.add_video(v)
        # _active_index is still 0, the seed. Remove bundle 1.
        assert stub_tracker.remove_video(1) is True
        assert len(stub_tracker._bundles) == 1
        assert stub_tracker._bundles[0].fname == Path("/seed.mp4")
        assert stub_tracker._active_index == 0

    def test_remove_active_swaps_to_next_first(self, stub_tracker, tmp_path):
        v = tmp_path / "extra.mp4"
        v.write_bytes(b"")
        stub_tracker.add_video(v)
        # Drop the active bundle (index 0). Should swap to index 1
        # FIRST, then drop the now-non-active old bundle.
        assert stub_tracker.remove_video(0) is True
        # swap_to(1) was called BEFORE the drop.
        assert 1 in stub_tracker.swap_to_calls
        # Post-drop: only one bundle remains.
        assert len(stub_tracker._bundles) == 1
        # Surviving bundle is the one we just added.
        assert stub_tracker._bundles[0].fname == v
        # Renumber: surviving bundle is index 0.
        assert stub_tracker._bundles[0].video_index == 0
        assert stub_tracker._active_index == 0

    def test_remove_active_at_tail_swaps_to_prev(self, stub_tracker, tmp_path):
        # Add two bundles. Swap to the tail. Remove tail.
        v0 = tmp_path / "a.mp4"
        v1 = tmp_path / "b.mp4"
        v0.write_bytes(b"")
        v1.write_bytes(b"")
        stub_tracker.add_video(v0)
        stub_tracker.add_video(v1)
        stub_tracker._active_index = 2
        # Remove index 2 (active, at tail). Should swap to index 1
        # (prev), then drop.
        assert stub_tracker.remove_video(2) is True
        assert 1 in stub_tracker.swap_to_calls
        assert len(stub_tracker._bundles) == 2

    def test_remove_refuses_to_empty_list(self, stub_tracker):
        # Only the seed bundle present.
        assert stub_tracker.remove_video(0) is False
        assert len(stub_tracker._bundles) == 1

    def test_remove_out_of_bounds_returns_false(self, stub_tracker):
        assert stub_tracker.remove_video(99) is False
        assert stub_tracker.remove_video(-1) is False

    def test_remove_renumbers_video_index(self, stub_tracker, tmp_path):
        # Add two bundles. Remove the middle one.
        v0 = tmp_path / "a.mp4"
        v1 = tmp_path / "b.mp4"
        v0.write_bytes(b"")
        v1.write_bytes(b"")
        stub_tracker.add_video(v0)  # index 1
        stub_tracker.add_video(v1)  # index 2
        # bundles: [seed(0), v0(1), v1(2)]
        stub_tracker.remove_video(1)
        # bundles: [seed(0), v1(1)]
        assert [b.video_index for b in stub_tracker._bundles] == [0, 1]
        assert [b.fname for b in stub_tracker._bundles] == [Path("/seed.mp4"), v1]

    def test_remove_below_active_shifts_active_down(
        self, stub_tracker, tmp_path,
    ):
        # Setup: active is index 2. Remove index 1.
        v0 = tmp_path / "a.mp4"
        v1 = tmp_path / "b.mp4"
        v0.write_bytes(b"")
        v1.write_bytes(b"")
        stub_tracker.add_video(v0)
        stub_tracker.add_video(v1)
        stub_tracker._active_index = 2
        stub_tracker.remove_video(1)
        # The removed index was below active, so active shifts 2 -> 1.
        assert stub_tracker._active_index == 1


# ---------------------------------------------------------------------
# replace_active_with
# ---------------------------------------------------------------------


class TestReplaceActiveWith:
    def test_seed_to_real_phase1_swap(self, stub_tracker, tmp_path):
        # Initial state: 1 bundle (the seed). Active = 0.
        v = tmp_path / "real.mp4"
        v.write_bytes(b"")

        final_indices = stub_tracker.replace_active_with(v)

        # After the operation:
        # - Old seed bundle is dropped.
        # - New bundle is at index 0.
        assert len(stub_tracker._bundles) == 1
        assert stub_tracker._bundles[0].fname == v
        # Active is the new bundle.
        assert stub_tracker._active_index == 0
        # final_indices points at the new bundle's post-removal index.
        assert final_indices == [0]

    def test_multi_video_replace(self, stub_tracker, tmp_path, force_has_dlc):
        # Replace seed with two videos -- requires Phase 2 (DLC project),
        # so the validation path will reject bare videos with a clear
        # message. Verify that. force_has_dlc bypasses the pre-validation
        # HAS_DLC guard so the intended ValueError (not ImportError) fires.
        v0 = tmp_path / "a.mp4"
        v1 = tmp_path / "b.mp4"
        v0.write_bytes(b"")
        v1.write_bytes(b"")
        with pytest.raises(ValueError, match="not inside a DLC project"):
            stub_tracker.replace_active_with([v0, v1])
        # The tracker is unchanged on validation failure (no partial
        # state).
        assert len(stub_tracker._bundles) == 1


# ---------------------------------------------------------------------
# _validate_bundle_paths
# ---------------------------------------------------------------------


class TestValidateBundlePaths:
    def test_scalar_bare_video(self, stub_tracker, tmp_path):
        v = tmp_path / "v.mp4"
        v.write_bytes(b"")
        project, paths = stub_tracker._validate_bundle_paths(v)
        assert project is None
        assert paths == [v]

    def test_scalar_string_works(self, stub_tracker, tmp_path):
        v = tmp_path / "v.mp4"
        v.write_bytes(b"")
        project, paths = stub_tracker._validate_bundle_paths(str(v))
        assert project is None
        assert paths == [v]

    def test_missing_file_raises(self, stub_tracker, tmp_path):
        with pytest.raises(FileNotFoundError):
            stub_tracker._validate_bundle_paths(tmp_path / "nope.mp4")

    def test_empty_list_raises(self, stub_tracker):
        with pytest.raises(ValueError, match="empty path sequence"):
            stub_tracker._validate_bundle_paths([])

    def test_single_element_list_normalises_to_scalar(
        self, stub_tracker, tmp_path,
    ):
        v = tmp_path / "v.mp4"
        v.write_bytes(b"")
        project, paths = stub_tracker._validate_bundle_paths([v])
        assert project is None
        assert paths == [v]

    def test_directory_not_dlc_project_raises(self, stub_tracker, tmp_path):
        d = tmp_path / "not_a_project"
        d.mkdir()
        with pytest.raises(ValueError, match="doesn't look like a DLC project"):
            stub_tracker._validate_bundle_paths(d)


# ---------------------------------------------------------------------
# _BundleState.project field round-trip
# ---------------------------------------------------------------------


class TestBundleStateProject:
    def test_default_project_is_none(self):
        b = _BundleState(fname=Path("/v.mp4"), video_index=0)
        assert b.project is None

    def test_project_round_trip(self):
        sentinel = SimpleNamespace(name="fake-project")
        b = _BundleState(
            fname=Path("/v.mp4"), video_index=0, project=sentinel,
        )
        assert b.project is sentinel

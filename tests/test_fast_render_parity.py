"""
Tier-2 (fast_render) parity + pick-event regression tests.

The two load-bearing claims about Tier 2 are:

1.  **Sub-pixel positional parity** -- a marker placed at a known data
    coordinate must render at the same coordinate in Tier 2 as it does
    in Tier 1. Annotators rely on this: the whole point of a point
    tracker is "the dot is where you said it was."
2.  **Pick-event equivalence** -- a mouse press at the display pixel
    of a known marker must select the same label whether the pick
    comes from matplotlib's ``pick_event`` (Tier 1) or from
    :class:`datanavigator._qt._QtPickAdapter` (Tier 2).

Tier 1 verification runs on Agg (matches conftest.py); Tier 2
verification needs a Qt canvas. The Qt-side asserts are skipped if no
Qt binding can be imported (CI without PyQt5/PySide6).
"""

import os

import numpy as np
import pytest

# Force offscreen Qt early so importing qtpy never opens a window.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_pointtracking import video_fname  # noqa: F401 (fixture)


def _qt_available():
    try:
        import qtpy  # noqa: F401
        from qtpy.QtWidgets import QApplication  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Tier 1 -- already exercised by tests/test_pointtracking.py. We only
# need a tiny direct positional check here so the parity comparison
# below has a known-good reference number to assert against.
# ---------------------------------------------------------------------------


def test_tier1_scatter_offsets_match_data_coords(video_fname, tmp_path):  # noqa: F811
    """A VideoAnnotation scatter records the same (x, y) it was given."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    import datanavigator
    from datanavigator.pointtracking import VideoAnnotation

    fig, ax = plt.subplots()
    ann = VideoAnnotation(vname=video_fname, name="parity")
    # add two labels with known positions
    coords = [(10.5, 20.5), (100.0, 200.0)]
    for label, (x, y) in enumerate(coords):
        ann.add(location=[x, y], label=str(label), frame_number=0)

    ann.plot_handles["ax_list_scatter"] = [ax]
    ann.setup_display_scatter([ax])
    ann.update_display_scatter(frame_number=0)

    handle = ann.plot_handles["labels_in_ax0"]
    offsets = np.asarray(handle.get_offsets())
    # Per-label palette length may exceed coords; only the first two
    # entries are meaningful here.
    np.testing.assert_allclose(offsets[0], coords[0], atol=1e-6)
    np.testing.assert_allclose(offsets[1], coords[1], atol=1e-6)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tier 2 -- Qt path, exercised directly through _QtScatterArtist /
# _QtPickAdapter without spinning up a full VideoPointAnnotator. Lets
# the test stay fast and avoid hitting the DLC dependency chain.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _qt_available(), reason="No Qt binding available")
def test_tier2_scatter_positional_parity():
    """Marker setPos preserves data coords within sub-pixel tolerance."""
    from qtpy.QtWidgets import QApplication

    from datanavigator._qt import (
        _make_qt_image_pane_class,
        _make_qt_scatter_artist_class,
    )

    app = QApplication.instance() or QApplication([])  # noqa: F841

    pane_cls = _make_qt_image_pane_class()
    scatter_cls = _make_qt_scatter_artist_class()
    pane = pane_cls()
    # Push a dummy 256x256 image so the scene rect is established.
    pane.set_image(np.zeros((256, 256, 3), dtype=np.uint8))

    group = pane.add_marker_group()
    palette = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    artist = scatter_cls(group, palette, picker_radius=5.0, image_pane=pane)

    coords = np.array(
        [[10.5, 20.5], [100.0, 200.0], [255.0, 255.0]], dtype=float
    )
    artist.set_offsets(coords)
    pane.show()
    # Drain the event loop so the scene actually realizes items.
    app.processEvents()

    # The artist's own offsets must round-trip.
    np.testing.assert_allclose(artist.get_offsets(), coords, atol=1e-6)
    # Each marker's position-in-scene must match the requested coord
    # within 0.5 px (the plan's load-bearing parity criterion).
    for i in range(len(coords)):
        pos = artist._items[i].pos()
        assert abs(pos.x() - coords[i, 0]) < 0.5, (
            f"marker {i} x drifted: {pos.x()} vs {coords[i, 0]}"
        )
        assert abs(pos.y() - coords[i, 1]) < 0.5, (
            f"marker {i} y drifted: {pos.y()} vs {coords[i, 1]}"
        )

    pane.hide()


@pytest.mark.skipif(not _qt_available(), reason="No Qt binding available")
def test_tier2_pick_adapter_hit_test_selects_nearest_marker():
    """A synthetic mouse press at a marker pixel fires pick for that label."""
    from qtpy.QtCore import QEvent, QPoint, Qt
    from qtpy.QtGui import QMouseEvent
    from qtpy.QtWidgets import QApplication

    from datanavigator._qt import (
        _make_qt_image_pane_class,
        _make_qt_scatter_artist_class,
    )

    app = QApplication.instance() or QApplication([])

    pane_cls = _make_qt_image_pane_class()
    scatter_cls = _make_qt_scatter_artist_class()
    pane = pane_cls()
    # 256x256 image -> scene rect is (0, 0, 256, 256).
    pane.set_image(np.zeros((256, 256, 3), dtype=np.uint8))

    group = pane.add_marker_group()
    palette = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    artist = scatter_cls(group, palette, picker_radius=5.0, image_pane=pane)

    coords = np.array(
        [[30.0, 30.0], [128.0, 128.0], [225.0, 225.0]], dtype=float
    )
    artist.set_offsets(coords)

    # Resize so the view has a meaningful viewport, then realize.
    pane.resize(400, 400)
    pane.show()
    app.processEvents()

    adapter = pane.install_pick_adapter()

    received = []
    adapter.connect_pick(lambda ev: received.append(("pick", ev.ind[0])))
    adapter.connect_button_press(
        lambda ev: received.append(("press", ev.button.name, ev.xdata, ev.ydata))
    )

    # For each known marker: map its scene coord to viewport pixel,
    # synthesize a left-button press there, deliver to the viewport's
    # event filter, and assert the pick fires with the right index.
    view = pane._view
    viewport = view.viewport()
    for i, (cx, cy) in enumerate(coords):
        viewport_pt = view.mapFromScene(cx, cy)
        # Re-map viewport_pt -> scene to validate the round-trip; if
        # the view fit is bad enough that the cursor pixel doesn't
        # land within picker_radius of the marker, the test is the
        # broken party, not the adapter.
        scene_check = view.mapToScene(viewport_pt)
        assert abs(scene_check.x() - cx) < 5.0, (
            f"viewport<->scene round-trip drift on marker {i}: "
            f"{scene_check.x()} vs {cx}"
        )
        received.clear()
        event = QMouseEvent(
            QEvent.MouseButtonPress,
            QPoint(viewport_pt),
            Qt.LeftButton,
            Qt.LeftButton,
            Qt.NoModifier,
        )
        QApplication.sendEvent(viewport, event)
        # The adapter should have fired exactly one pick (the marker
        # at coords[i]) plus one button-press.
        picks = [r for r in received if r[0] == "pick"]
        presses = [r for r in received if r[0] == "press"]
        assert len(presses) == 1, (
            f"marker {i}: expected 1 press, got {presses}"
        )
        assert len(picks) == 1, (
            f"marker {i}: expected 1 pick, got picks={picks} "
            f"all={received}"
        )
        assert picks[0][1] == i, (
            f"marker {i}: pick fired wrong index, got {picks[0][1]}"
        )

    pane.hide()


@pytest.mark.skipif(not _qt_available(), reason="No Qt binding available")
def test_tier2_pick_adapter_misses_outside_radius():
    """A click far from any marker fires button_press but not pick."""
    from qtpy.QtCore import QEvent, QPoint, Qt
    from qtpy.QtGui import QMouseEvent
    from qtpy.QtWidgets import QApplication

    from datanavigator._qt import (
        _make_qt_image_pane_class,
        _make_qt_scatter_artist_class,
    )

    app = QApplication.instance() or QApplication([])

    pane = _make_qt_image_pane_class()()
    pane.set_image(np.zeros((256, 256, 3), dtype=np.uint8))
    group = pane.add_marker_group()
    artist = _make_qt_scatter_artist_class()(
        group, [(1.0, 0.0, 0.0)], picker_radius=5.0, image_pane=pane,
    )
    artist.set_offsets(np.array([[100.0, 100.0]]))

    pane.resize(400, 400)
    pane.show()
    app.processEvents()
    adapter = pane.install_pick_adapter()

    pick_hits = []
    press_hits = []
    adapter.connect_pick(lambda ev: pick_hits.append(ev.ind[0]))
    adapter.connect_button_press(lambda ev: press_hits.append((ev.xdata, ev.ydata)))

    view = pane._view
    viewport = view.viewport()
    # Click on scene coord (0, 0) -- nowhere near (100, 100).
    viewport_pt = view.mapFromScene(0.0, 0.0)
    event = QMouseEvent(
        QEvent.MouseButtonPress,
        QPoint(viewport_pt),
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    QApplication.sendEvent(viewport, event)

    assert len(press_hits) == 1
    assert len(pick_hits) == 0

    pane.hide()

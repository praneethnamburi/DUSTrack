"""
Tier-1 (matplotlib path) positional parity check for VideoAnnotation.

This file used to also carry the Tier-2 fast_render Qt image-pane parity
+ pick-event regression tests that exercise
:class:`datanavigator._qt._QtScatterArtist` /
:class:`datanavigator._qt._QtPickAdapter` directly. Those tests stay in
datanavigator (see ``datanavigator/tests/test_fast_render_parity.py``)
where the Qt artists live -- they don't touch VideoAnnotation and don't
need the dustrack-side absorption to run.

`git log --follow tests/test_fast_render_parity.py` traces the full
pre-relocation history including both tiers.
"""

import os

import numpy as np
import pytest

# Force offscreen Qt early so importing qtpy never opens a window.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_pointtracking import video_fname  # noqa: F401 (fixture)


def test_tier1_scatter_offsets_match_data_coords(video_fname, tmp_path):  # noqa: F811
    """A VideoAnnotation scatter records the same (x, y) it was given."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from dustrack.pointtracking import VideoAnnotation

    fig, ax = plt.subplots()
    ann = VideoAnnotation(vname=video_fname, name="parity", n_labels=2)
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

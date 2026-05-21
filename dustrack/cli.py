"""Console-script entry point for DUSTrack.

Wired via ``[project.scripts]`` in ``pyproject.toml`` so users get the
no-command-line ergonomic shipped in 1.2.0a2:

    $ dustrack

pops the Qt video picker, opens whichever video(s) the user selects,
and blocks until the window closes. The first selected video becomes
the active session; any additional selections stash on
``tracker._video_queue`` for the future multi-video navigation work.

Intentionally argument-less for 1.2.0a2 -- proving out the
no-CLI ergonomic is the goal. A path-argument form (``dustrack
S:/path/to/video.mp4``) is a natural follow-up but not required by
the "no command line" success criterion.
"""
from __future__ import annotations

import sys

import matplotlib.pyplot as plt

from .dlcinterface import open as dustrack_open


def main() -> int:
    """Launch DUSTrack via the no-arg picker; block until the window closes.

    Returns the process exit code: ``0`` on a clean session or a
    cancelled picker, ``1`` if construction raised.
    """
    try:
        tracker = dustrack_open()
    except Exception:  # noqa: BLE001 -- surface a clean traceback at the CLI boundary.
        import traceback
        traceback.print_exc()
        return 1
    if tracker is None:
        # Picker cancelled -- nothing to show, exit cleanly.
        return 0
    # ``DUSTrack.__init__`` already called ``plt.show(block=False)``;
    # the CLI form needs a blocking show so the process doesn't exit
    # the moment construction returns.
    plt.show(block=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

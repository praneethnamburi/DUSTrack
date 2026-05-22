"""Per-video bundle state for the 1.2.0a3 multi-video swap.

A :class:`_BundleState` represents one video's worth of backend +
frontend state inside a multi-video DUSTrack session. The shell
(``DUSTrack``) holds a list of bundles (``self._bundles``) and rebinds
itself onto the active bundle on every swap. Bundles are populated
either synchronously inside :func:`dustrack.open` (the active bundle)
or off-thread by the background hydration worker (every other
queue entry).

The split between *heavy* fields (``reader``, ``annotations``, set
during hydration) and *snapshot* fields (always present, mutated on
every swap-out / swap-in) is the load-bearing distinction: a swap is
``shell snapshot -> bundle snapshot`` followed by
``bundle snapshot -> shell snapshot`` on the arriving side; nothing on
disk is read, nothing in memory is freed.

See ``specs/dustrack.md`` (Roadmap *Next 1.2.0* item 3) for the
end-to-end contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


def _default_ax_lims() -> dict:
    return {
        "state": False,
        "x": [None, None],
        "y_trace_x": [None, None],
        "y_trace_y": [None, None],
    }


# Hydration state machine. ``pending`` -> ``hydrating`` -> ``ready``
# is the happy path; ``failed`` is a terminal absorbing state with
# ``hydration_error`` populated. Bundles in ``pending``, ``hydrating``,
# or ``failed`` have ``reader is None`` and ``annotations is None``.
HYDRATION_PENDING = "pending"
HYDRATION_HYDRATING = "hydrating"
HYDRATION_READY = "ready"
HYDRATION_FAILED = "failed"

_HYDRATION_STATES = (
    HYDRATION_PENDING,
    HYDRATION_HYDRATING,
    HYDRATION_READY,
    HYDRATION_FAILED,
)


@dataclass
class _BundleState:
    """Per-video backend + frontend snapshot.

    Heavy fields (``reader``, ``annotations``) are populated during
    hydration; they stay ``None`` for ``pending`` / ``hydrating`` /
    ``failed`` bundles. Lightweight snapshot fields are always present
    and survive across swap-out / swap-in cycles, even before the
    first hydration -- their defaults define the bundle's "first
    visit" view (frame 0, no frozen axes, fit-to-frame image pane).

    The shell never reaches into a non-ready bundle's heavy fields;
    :meth:`DUSTrack.swap_to` blocks on hydration via a loading overlay
    before rebinding.
    """

    # Identity (always populated).
    fname: Path
    video_index: int  # 0-based position inside ``DUSTrack._bundles``

    # Project context (1.2.0a3 seed-modal cut). ``None`` = Phase 1
    # (bare video, no DLC project); a ``DLCProject`` = Phase 2. Stored
    # per-bundle so a tracker can hold a mix of Phase 1 + Phase 2
    # bundles (seed-modal flow: seed is Phase 1, picked may be either;
    # future ``add_video`` callers can mix freely). Rebound onto the
    # shell's ``_dlcproject`` in :meth:`DUSTrack._attach_bundle` on
    # every swap, so Workflow-button gates and project-aware code
    # paths see the active bundle's project.
    project: Any = None  # dustrack.dlcinterface.DLCProject | None

    # Heavy state, populated during hydration.
    reader: Any = None  # datanavigator.VideoReader
    annotations: Any = None  # dustrack.pointtracking.VideoAnnotations

    # Lightweight per-bundle UI snapshot.
    current_idx: int = 0
    selections: dict = field(default_factory=dict)
    ax_lims: dict = field(default_factory=_default_ax_lims)
    image_view_state: Any = None  # opaque blob, set/get via shell dispatch
    trace_view_state: Any = None  # {trace_x_xlim, trace_x_ylim, trace_y_ylim}
    enhance_state: Any = None  # {clahe_clip, gamma, brightness}
    frames_of_interest: list = field(default_factory=list)

    # Lifecycle.
    hydration_state: str = HYDRATION_PENDING
    hydration_error: Optional[str] = None

    def __post_init__(self) -> None:
        self.fname = Path(self.fname)
        if self.hydration_state not in _HYDRATION_STATES:
            raise ValueError(
                f"unknown hydration_state {self.hydration_state!r}; "
                f"expected one of {_HYDRATION_STATES!r}"
            )

    @property
    def is_ready(self) -> bool:
        """True iff this bundle's heavy state is populated and parked
        ready for a swap-in."""
        return self.hydration_state == HYDRATION_READY

    @property
    def is_terminal(self) -> bool:
        """True iff the bundle has reached an absorbing state
        (``ready`` or ``failed``) -- the hydration worker won't touch
        it again."""
        return self.hydration_state in (HYDRATION_READY, HYDRATION_FAILED)

"""Ultrasound image enhancement: CLAHE + gamma + brightness + slider plumbing.

Two layers:

* **Pure primitives** -- :func:`enhance_ultrasound_image` (the full
  pipeline), :func:`_apply_gamma_only` (per-channel gamma LUT without
  CLAHE), :func:`_enhance_is_passthrough` (bypass predicate),
  :func:`_auto_enhance_params` (one-shot heuristic from frame
  histogram), slider <-> parameter conversions
  (:func:`_slider_to_clahe_clip` / :func:`_slider_to_gamma` and their
  inverses).

* **EnhanceWidget factory** -- :func:`_make_enhance_widget_class`
  builds the two-slider Qt widget mounted below the statevars widget
  in :meth:`DUSTrack._add_enhance_widget`.

Extracted from ``dlcinterface.py`` in dustrack 1.2.0rc1. Public
``enhance_ultrasound_image`` was already an unofficial extension point
for downstream callers (no underscore prefix); kept callable as
``dustrack._image_enhance.enhance_ultrasound_image`` after the move.
"""

from __future__ import annotations

import cv2 as cv
import numpy as np


# ---------------------------------------------------------------------
# Full enhancement pipeline (public, called by DUSTrack.update on the
# active frame)
# ---------------------------------------------------------------------
def enhance_ultrasound_image(
    image, clahe_clip=2.0, clahe_grid=8, gamma=1.0, brightness=0
):
    """
    Enhance ultrasound image for better visibility.

    Args:
        image: Input image (RGB or grayscale)
        clahe_clip: CLAHE clip limit (higher = more contrast)
        clahe_grid: CLAHE tile grid size
        gamma: Gamma correction (>1 = brighter midtones, <1 = darker)
        brightness: Brightness offset (-255 to 255)

    Returns:
        Enhanced RGB image for matplotlib display.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    else:
        gray = image

    # Apply CLAHE
    clahe = cv.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    enhanced = clahe.apply(gray)

    # Apply gamma correction
    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(
            "uint8"
        )
        enhanced = cv.LUT(enhanced, table)

    # Apply brightness
    if brightness != 0:
        enhanced = np.clip(enhanced.astype(np.int16) + brightness, 0, 255).astype(
            np.uint8
        )

    # Convert back to RGB for matplotlib
    return cv.cvtColor(enhanced, cv.COLOR_GRAY2RGB)


# ---------------------------------------------------------------------
# Pure-function slider-to-param maps for the EnhanceWidget. Extracted
# at module scope so they're unit-testable without qtpy import.
# ---------------------------------------------------------------------
_CLAHE_CLIP_MIN = 1.0
_CLAHE_CLIP_MAX = 4.0
_GAMMA_MIN = 1.0
_GAMMA_MAX = 2.0  # extended 2026-05-19 (was 1.5) for darker ultrasound footage
_SLIDER_TICKS = 100  # integer slider range; sliders use 0..100


def _slider_to_clahe_clip(slider_value: int) -> float:
    """Map slider integer 0..100 to CLAHE clip limit in [1.0, 4.0]."""
    t = max(0, min(_SLIDER_TICKS, int(slider_value))) / _SLIDER_TICKS
    return _CLAHE_CLIP_MIN + t * (_CLAHE_CLIP_MAX - _CLAHE_CLIP_MIN)


def _slider_to_gamma(slider_value: int) -> float:
    """Map slider integer 0..100 to gamma in [1.0, 1.5]."""
    t = max(0, min(_SLIDER_TICKS, int(slider_value))) / _SLIDER_TICKS
    return _GAMMA_MIN + t * (_GAMMA_MAX - _GAMMA_MIN)


def _clahe_clip_to_slider(clip: float) -> int:
    """Inverse of :func:`_slider_to_clahe_clip` -- seed slider from current param."""
    span = _CLAHE_CLIP_MAX - _CLAHE_CLIP_MIN
    if span <= 0:
        return 0
    t = (float(clip) - _CLAHE_CLIP_MIN) / span
    return max(0, min(_SLIDER_TICKS, round(t * _SLIDER_TICKS)))


def _gamma_to_slider(gamma: float) -> int:
    """Inverse of :func:`_slider_to_gamma` -- seed slider from current param."""
    span = _GAMMA_MAX - _GAMMA_MIN
    if span <= 0:
        return 0
    t = (float(gamma) - _GAMMA_MIN) / span
    return max(0, min(_SLIDER_TICKS, round(t * _SLIDER_TICKS)))


def _apply_gamma_only(image, gamma: float):
    """Apply a gamma LUT to ``image`` without touching CLAHE or
    grayscale conversion.

    Per-slider bypass for the EnhanceWidget: when the user moves the
    gamma slider off zero with the clip slider still at zero, we
    don't want to spin up the full CLAHE pipeline (which forces an
    RGB->gray->RGB roundtrip + a CLAHE histogram pass at clip=1.0
    that *isn't* a true identity). This helper operates per-channel
    on the input directly via ``cv.LUT``, so moving gamma off 1.0
    transitions smoothly from raw -- the inverse-gamma LUT at
    gamma=1.0+epsilon differs from the identity LUT by less than 1
    unit in any bin, so the rendered frame is visually continuous
    with the bypassed state.

    Mirrors the gamma branch inside :func:`enhance_ultrasound_image`
    but operates on the original (possibly RGB) array rather than
    the grayscale intermediate.
    """
    inv_gamma = 1.0 / float(gamma)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(
        "uint8"
    )
    return cv.LUT(image, table)


def _enhance_is_passthrough(clahe_clip: float, gamma: float) -> bool:
    """True if the enhancement pipeline should be bypassed entirely.

    Returns ``True`` when both sliders sit at their minimum (the
    "no enhancement" position): ``clahe_clip <= 1.0`` AND
    ``gamma <= 1.0``. At that point :func:`enhance_ultrasound_image`
    would still run a CLAHE pass at clip=1.0 (minimal but non-zero
    effect) and would convert RGB->gray->RGB unconditionally, so the
    "bypass" semantic is implemented by short-circuiting at the
    image processor level rather than by tuning the parameter
    extremes. Replaces the pre-2026-05-19 ``_enhance_enabled``
    toggle that the deleted "Toggle enhance" button flipped.
    """
    return float(clahe_clip) <= _CLAHE_CLIP_MIN and float(gamma) <= _GAMMA_MIN


# Anchor points for the auto-enhance heuristic. Tuned 2026-05-19
# against real ultrasound footage in four passes; each pass was the
# user's "too aggressive" reaction to the previous one:
#
# - Pass 1 (LOW=40 / HIGH=180, DARK=50 / MID=130).
# - Pass 2 (LOW=0 / HIGH=120, DARK=20 / MID=90).
# - Pass 3 (LOW=0 / HIGH=100, DARK=20 / MID=75).
# - Pass 4 (current, LOW=0 / HIGH=75, DARK=0 / MID=25): bundled with
#   the gamma-max extension to 2.0. Calibrated against the S-corpus
#   DUSTrack clip where the user-target was clip~1.6, gamma~1.3 and
#   pass-3 was producing clip=2.17, gamma=1.5(capped). Inferred
#   stats: dyn-range ~61, p50 ~20. With pass-4 anchors and
#   gamma_max=2.0 those stats land at clip~1.56, gamma~1.20 --
#   below user target but close enough that the user dials up from
#   there. Anchors deliberately make "typical" ultrasound
#   (p50~60, dyn~80) a near-bypass (clip~1.0, gamma=1.0); Auto
#   only kicks in noticeably for dark + low-contrast frames.
# Adjust here, not in callers.
_AUTO_DYN_RANGE_LOW = 0.0  # at this dyn range, suggest clip=max
_AUTO_DYN_RANGE_HIGH = 75.0  # at this dyn range, suggest clip=min
_AUTO_MEDIAN_DARK = 0.0  # at this median, suggest gamma=max
_AUTO_MEDIAN_MID = 25.0  # at this median, suggest gamma=min


def _auto_enhance_params(image) -> tuple[float, float]:
    """Heuristic ``(clip, gamma)`` from the current frame.

    One-shot inference, called by :class:`EnhanceWidget`'s ``Auto``
    button. Reads the grayscale histogram of ``image`` and maps
    two robust statistics to slider-range parameters:

    - **CLAHE clip** is driven by the 5th-to-95th percentile dynamic
      range. Narrow dynamic range (a flat-looking frame) suggests
      pushing clip high; wide dynamic range (already-contrasty
      frame) suggests leaving clip low. Current anchors (pass 4):
      ``dyn=0 -> clip=_CLAHE_CLIP_MAX``,
      ``dyn=75 -> clip=_CLAHE_CLIP_MIN``.
    - **Gamma** is driven by the 50th percentile (median). A dark
      frame (low median) suggests pushing gamma high to lift the
      midtones; a balanced frame suggests gamma=1.0. Current
      anchors (pass 4):
      ``p50=0 -> gamma=_GAMMA_MAX``,
      ``p50=25 -> gamma=_GAMMA_MIN``.

    Both outputs are clamped to their slider ranges
    (``[_CLAHE_CLIP_MIN, _CLAHE_CLIP_MAX]`` and
    ``[_GAMMA_MIN, _GAMMA_MAX]``) so a degenerate frame (all-zero,
    all-white) can't drive the sliders past their ends.

    Pure function -- accepts a uint8 RGB or grayscale numpy array,
    returns ``(clip, gamma)`` floats. Lives at module scope so
    :file:`tests/test_enhance_widget_mapping.py` can unit-test the
    heuristic without standing up a Qt main loop.
    """
    if image.ndim == 3:
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    else:
        gray = image
    p5, p50, p95 = (float(x) for x in np.percentile(gray, [5, 50, 95]))

    dyn_range = p95 - p5
    t_clip = (_AUTO_DYN_RANGE_HIGH - dyn_range) / (
        _AUTO_DYN_RANGE_HIGH - _AUTO_DYN_RANGE_LOW
    )
    t_clip = max(0.0, min(1.0, t_clip))
    clip = _CLAHE_CLIP_MIN + t_clip * (_CLAHE_CLIP_MAX - _CLAHE_CLIP_MIN)

    t_gamma = (_AUTO_MEDIAN_MID - p50) / (_AUTO_MEDIAN_MID - _AUTO_MEDIAN_DARK)
    t_gamma = max(0.0, min(1.0, t_gamma))
    gamma = _GAMMA_MIN + t_gamma * (_GAMMA_MAX - _GAMMA_MIN)

    return float(clip), float(gamma)


def _make_enhance_widget_class():
    """Build :class:`EnhanceWidget` lazily, mirroring
    :func:`_make_progress_overlay_class`'s qtpy-import-on-demand pattern.

    Two-slider control mounted in the rc2 left-column dock below the
    statevars widget. CLAHE clip (1.0 -> 4.0) and gamma (1.0 -> 1.5).
    Brightness and CLAHE grid (8) stay at their constructor defaults --
    both ride below the "useful slider range" threshold for routine
    review. The widget itself owns no enable/disable state: both
    sliders at their minimum (clip=1.0 AND gamma=1.0) is the bypass
    -- :func:`_enhance_is_passthrough` short-circuits the image
    processor and returns the raw frame untouched.
    """
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSizePolicy,
        QSlider,
        QVBoxLayout,
        QWidget,
    )

    class EnhanceWidget(QWidget):
        """Two-slider widget for ultrasound image enhancement.

        Slider 1 -- CLAHE clip:  1.0 -> 4.0, default 2.0.
        Slider 2 -- Gamma:       1.0 -> 1.5, default 1.2.

        Sliders update ``self._owner._clahe_clip`` / ``_gamma`` directly
        on every value change and trigger ``self._owner.update()`` so
        the image redraws live.
        """

        def __init__(self, owner, parent=None):
            super().__init__(parent)
            self._owner = owner

            # Slightly darker bg than the parent dock so the enhance
            # section reads as a distinct group from the statevars
            # widget above it. Theme-adaptive via palette.darker().
            self.setAutoFillBackground(True)
            pal = self.palette()
            base = pal.color(self.backgroundRole())
            pal.setColor(self.backgroundRole(), base.darker(110))
            self.setPalette(pal)

            self.setFocusPolicy(Qt.NoFocus)
            self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

            outer = QVBoxLayout(self)
            outer.setContentsMargins(4, 4, 4, 4)
            outer.setSpacing(4)

            section_label = QLabel("Image enhance:", self)
            outer.addWidget(section_label)

            # CLAHE clip slider row.
            self._clip_label = QLabel(
                f"Clip: {float(owner._clahe_clip):.2f}",
                self,
            )
            outer.addWidget(self._clip_label)
            self._clip_slider = QSlider(Qt.Horizontal, self)
            self._clip_slider.setRange(0, _SLIDER_TICKS)
            self._clip_slider.setValue(_clahe_clip_to_slider(owner._clahe_clip))
            self._clip_slider.setFocusPolicy(Qt.NoFocus)
            self._clip_slider.valueChanged.connect(self._on_clip_changed)
            outer.addWidget(self._clip_slider)

            # Gamma slider row.
            self._gamma_label = QLabel(
                f"Gamma: {float(owner._gamma):.2f}",
                self,
            )
            outer.addWidget(self._gamma_label)
            self._gamma_slider = QSlider(Qt.Horizontal, self)
            self._gamma_slider.setRange(0, _SLIDER_TICKS)
            self._gamma_slider.setValue(_gamma_to_slider(owner._gamma))
            self._gamma_slider.setFocusPolicy(Qt.NoFocus)
            self._gamma_slider.valueChanged.connect(self._on_gamma_changed)
            outer.addWidget(self._gamma_slider)

            # One-shot trigger row: [None | Auto].
            # - None: snap both sliders to leftmost (passthrough).
            # - Auto: infer clip + gamma from the current frame's
            #   grayscale histogram, set the sliders once, redraw
            #   once. Subsequent frame navigations don't re-trigger
            #   Auto -- slider values stay put until the user
            #   (or another button click) moves them.
            button_row = QHBoxLayout()
            button_row.setContentsMargins(0, 0, 0, 0)
            button_row.setSpacing(4)
            self._none_button = QPushButton("None", self)
            self._none_button.setFocusPolicy(Qt.NoFocus)
            self._none_button.clicked.connect(self._on_none_clicked)
            button_row.addWidget(self._none_button)
            self._auto_button = QPushButton("Auto", self)
            self._auto_button.setFocusPolicy(Qt.NoFocus)
            self._auto_button.clicked.connect(self._on_auto_clicked)
            button_row.addWidget(self._auto_button)
            outer.addLayout(button_row)

        def _on_clip_changed(self, value: int) -> None:
            clip = _slider_to_clahe_clip(value)
            self._owner._clahe_clip = clip
            self._clip_label.setText(f"Clip: {clip:.2f}")
            self._owner.update()

        def _on_gamma_changed(self, value: int) -> None:
            gamma = _slider_to_gamma(value)
            self._owner._gamma = gamma
            self._gamma_label.setText(f"Gamma: {gamma:.2f}")
            self._owner.update()

        def _apply_param_pair(self, clip: float, gamma: float) -> None:
            """Set both sliders to (clip, gamma); one redraw at the end.

            Shared tail for the ``None`` and ``Auto`` button handlers.
            Slider signals are blocked during the dual ``setValue`` so
            the two ``valueChanged`` callbacks don't each fire
            ``owner.update()``; the redraw happens once. Slider
            positions are integer-quantized, so the actually-applied
            values are read back off the sliders to keep the labels
            + the owner's enhancement params in sync with the slider
            UI state of truth.
            """
            self._clip_slider.blockSignals(True)
            self._gamma_slider.blockSignals(True)
            self._clip_slider.setValue(_clahe_clip_to_slider(clip))
            self._gamma_slider.setValue(_gamma_to_slider(gamma))
            self._clip_slider.blockSignals(False)
            self._gamma_slider.blockSignals(False)

            actual_clip = _slider_to_clahe_clip(self._clip_slider.value())
            actual_gamma = _slider_to_gamma(self._gamma_slider.value())
            self._owner._clahe_clip = actual_clip
            self._owner._gamma = actual_gamma
            self._clip_label.setText(f"Clip: {actual_clip:.2f}")
            self._gamma_label.setText(f"Gamma: {actual_gamma:.2f}")
            self._owner.update()

        def _on_none_clicked(self) -> None:
            """Snap both sliders to leftmost (= passthrough).

            Convenience: undoes any Auto/manual enhancement in one
            click. The image processor's
            :func:`_enhance_is_passthrough` short-circuit fires after
            the redraw, so the next frame renders raw.
            """
            self._apply_param_pair(_CLAHE_CLIP_MIN, _GAMMA_MIN)

        def _on_auto_clicked(self) -> None:
            """One-shot auto-enhance from the current raw frame.

            Reads ``owner.data[owner._current_idx]`` (the same raw
            frame the image processor sees), runs
            :func:`_auto_enhance_params`, and applies the result.
            """
            owner = self._owner
            try:
                raw = owner.data[owner._current_idx].asnumpy()
            except Exception:
                # No frame available (e.g. video reader torn down):
                # surface nothing, leave sliders alone. Auto is best-
                # effort UI; a failure here shouldn't crash the
                # session.
                return
            clip, gamma = _auto_enhance_params(raw)
            self._apply_param_pair(clip, gamma)

        def sync_from_shell(self) -> None:
            """Move the slider knobs to match the owner's current
            ``_clahe_clip`` / ``_gamma`` without triggering the
            valueChanged cascade (and without calling
            ``owner.update()`` -- caller is responsible for that).

            Used by :meth:`DUSTrack._set_enhance_state` after a
            multi-video swap so the sliders visibly reflect the
            arriving bundle's saved enhance settings. Differs from
            :meth:`_apply_param_pair` in two ways: (1) it reads from
            the owner (not from caller-supplied args), and (2) it
            does not call ``owner.update()`` so the caller can batch
            the redraw with the rest of the swap.
            """
            self._clip_slider.blockSignals(True)
            self._gamma_slider.blockSignals(True)
            self._clip_slider.setValue(_clahe_clip_to_slider(self._owner._clahe_clip))
            self._gamma_slider.setValue(_gamma_to_slider(self._owner._gamma))
            self._clip_slider.blockSignals(False)
            self._gamma_slider.blockSignals(False)
            self._clip_label.setText(f"Clip: {self._owner._clahe_clip:.2f}")
            self._gamma_label.setText(f"Gamma: {self._owner._gamma:.2f}")

    return EnhanceWidget

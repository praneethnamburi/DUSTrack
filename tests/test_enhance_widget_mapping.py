"""Pure-function tests for the EnhanceWidget slider->param mapping.

The EnhanceWidget itself is Qt-only and painful to exercise headlessly
(synchronous-modal Qt code through a slider valueChanged callback that
calls back into DUSTrack.update()). The slider <-> parameter mapping
is the only branchy part of the widget worth pinning -- once these
are correct, the widget's job reduces to "drive update() on every
valueChanged" which is verified by manual smoke.
"""
import math

import pytest

from dustrack.dlcinterface import (
    _CLAHE_CLIP_MAX,
    _CLAHE_CLIP_MIN,
    _GAMMA_MAX,
    _GAMMA_MIN,
    _SLIDER_TICKS,
    _clahe_clip_to_slider,
    _enhance_is_passthrough,
    _gamma_to_slider,
    _slider_to_clahe_clip,
    _slider_to_gamma,
)


class TestClipSliderMapping:
    """CLAHE clip slider: integer 0..100 maps to float [1.0, 4.0]."""

    def test_slider_min_maps_to_clip_min(self):
        assert _slider_to_clahe_clip(0) == _CLAHE_CLIP_MIN == 1.0

    def test_slider_max_maps_to_clip_max(self):
        assert _slider_to_clahe_clip(_SLIDER_TICKS) == _CLAHE_CLIP_MAX == 4.0

    def test_slider_midpoint_is_2_5(self):
        # (1.0 + 4.0) / 2 = 2.5
        assert math.isclose(_slider_to_clahe_clip(50), 2.5, abs_tol=1e-9)

    def test_default_clip_2_0_roundtrips(self):
        # Default clip=2.0 -> slider value 33 (33.333... rounded).
        s = _clahe_clip_to_slider(2.0)
        assert s == 33
        # And the slider value maps back to ~2.0 within one tick (3%).
        assert math.isclose(_slider_to_clahe_clip(s), 2.0, abs_tol=0.05)

    def test_clip_to_slider_clamps_low(self):
        # Values below the range clamp to slider 0.
        assert _clahe_clip_to_slider(-1.0) == 0
        assert _clahe_clip_to_slider(0.5) == 0

    def test_clip_to_slider_clamps_high(self):
        assert _clahe_clip_to_slider(10.0) == _SLIDER_TICKS
        assert _clahe_clip_to_slider(4.5) == _SLIDER_TICKS

    def test_slider_value_clamped_low(self):
        # Slider values below 0 clamp to 0.
        assert _slider_to_clahe_clip(-5) == _CLAHE_CLIP_MIN

    def test_slider_value_clamped_high(self):
        # Slider values above ticks clamp to ticks.
        assert _slider_to_clahe_clip(_SLIDER_TICKS + 5) == _CLAHE_CLIP_MAX


class TestGammaSliderMapping:
    """Gamma slider: integer 0..100 maps to float [1.0, 1.5]."""

    def test_slider_min_maps_to_gamma_min(self):
        assert _slider_to_gamma(0) == _GAMMA_MIN == 1.0

    def test_slider_max_maps_to_gamma_max(self):
        assert _slider_to_gamma(_SLIDER_TICKS) == _GAMMA_MAX == 1.5

    def test_slider_midpoint_is_1_25(self):
        # (1.0 + 1.5) / 2 = 1.25
        assert math.isclose(_slider_to_gamma(50), 1.25, abs_tol=1e-9)

    def test_default_gamma_1_2_roundtrips(self):
        # Default gamma=1.2 -> slider value 40.
        s = _gamma_to_slider(1.2)
        assert s == 40
        assert math.isclose(_slider_to_gamma(s), 1.2, abs_tol=0.01)

    def test_gamma_to_slider_clamps(self):
        assert _gamma_to_slider(0.5) == 0
        assert _gamma_to_slider(2.0) == _SLIDER_TICKS


class TestRoundTrip:
    """Every slider tick round-trips back to itself (within rounding)."""

    @pytest.mark.parametrize("value", [0, 10, 33, 50, 75, 100])
    def test_clip_roundtrip(self, value):
        assert _clahe_clip_to_slider(_slider_to_clahe_clip(value)) == value

    @pytest.mark.parametrize("value", [0, 25, 40, 50, 80, 100])
    def test_gamma_roundtrip(self, value):
        assert _gamma_to_slider(_slider_to_gamma(value)) == value


class TestEnhanceIsPassthrough:
    """The slider-driven bypass predicate. ``True`` -> image_processor
    short-circuits and returns the raw frame; ``False`` -> the full
    CLAHE+gamma pipeline runs."""

    def test_both_sliders_at_min_is_passthrough(self):
        # Slider value 0 maps to clip=1.0 + gamma=1.0 -- the explicit
        # "no enhancement" position the EnhanceWidget defaults to.
        assert _enhance_is_passthrough(1.0, 1.0)

    def test_clip_nudged_off_min_is_not_passthrough(self):
        # One tick off min on the clip slider runs the pipeline.
        clip_one_tick = _slider_to_clahe_clip(1)
        assert clip_one_tick > _CLAHE_CLIP_MIN
        assert not _enhance_is_passthrough(clip_one_tick, 1.0)

    def test_gamma_nudged_off_min_is_not_passthrough(self):
        # One tick off min on the gamma slider runs the pipeline.
        gamma_one_tick = _slider_to_gamma(1)
        assert gamma_one_tick > _GAMMA_MIN
        assert not _enhance_is_passthrough(1.0, gamma_one_tick)

    def test_both_at_max_is_not_passthrough(self):
        assert not _enhance_is_passthrough(_CLAHE_CLIP_MAX, _GAMMA_MAX)

    def test_default_kwargs_are_passthrough(self):
        # DUSTrack.__init__ defaults: clahe_clip=1.0, gamma=1.0 ->
        # passthrough, so opening DUSTrack shows the raw frame.
        # Pin so a future default tweak doesn't silently re-enable
        # enhancement on startup.
        assert _enhance_is_passthrough(1.0, 1.0)

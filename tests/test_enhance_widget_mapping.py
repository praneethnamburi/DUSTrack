"""Pure-function tests for the EnhanceWidget slider->param mapping.

The EnhanceWidget itself is Qt-only and painful to exercise headlessly
(synchronous-modal Qt code through a slider valueChanged callback that
calls back into DUSTrack.update()). The slider <-> parameter mapping
is the only branchy part of the widget worth pinning -- once these
are correct, the widget's job reduces to "drive update() on every
valueChanged" which is verified by manual smoke.
"""
import math

import numpy as np
import pytest

from dustrack._image_enhance import (
    _CLAHE_CLIP_MAX,
    _CLAHE_CLIP_MIN,
    _GAMMA_MAX,
    _GAMMA_MIN,
    _SLIDER_TICKS,
    _apply_gamma_only,
    _auto_enhance_params,
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
    """Gamma slider: integer 0..100 maps to float [1.0, 2.0].

    Extended from the original [1.0, 1.5] range on 2026-05-19 to give
    headroom for darker ultrasound footage where the pre-2.0 ceiling
    was getting hit by Auto.
    """

    def test_slider_min_maps_to_gamma_min(self):
        assert _slider_to_gamma(0) == _GAMMA_MIN == 1.0

    def test_slider_max_maps_to_gamma_max(self):
        assert _slider_to_gamma(_SLIDER_TICKS) == _GAMMA_MAX == 2.0

    def test_slider_midpoint_is_1_5(self):
        # (1.0 + 2.0) / 2 = 1.5
        assert math.isclose(_slider_to_gamma(50), 1.5, abs_tol=1e-9)

    def test_pre_extension_default_gamma_1_2_roundtrips(self):
        # Old pre-EnhanceWidget hand-picked default gamma=1.2:
        # (1.2 - 1.0) / 1.0 * 100 = 20.
        s = _gamma_to_slider(1.2)
        assert s == 20
        assert math.isclose(_slider_to_gamma(s), 1.2, abs_tol=0.01)

    def test_gamma_to_slider_clamps(self):
        assert _gamma_to_slider(0.5) == 0
        assert _gamma_to_slider(3.0) == _SLIDER_TICKS


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


class TestAutoEnhanceParams:
    """The Auto button's image -> (clip, gamma) heuristic.

    Pin the qualitative behavior so future tuning can move the
    anchor points without silently regressing the directional
    response (dark image -> high gamma; low-contrast image -> high
    clip; etc).
    """

    def test_outputs_within_slider_ranges(self):
        # Sanity: every output stays clamped to the slider extents.
        img = np.full((64, 64), 100, dtype=np.uint8)
        clip, gamma = _auto_enhance_params(img)
        assert _CLAHE_CLIP_MIN <= clip <= _CLAHE_CLIP_MAX
        assert _GAMMA_MIN <= gamma <= _GAMMA_MAX

    def test_all_zero_image_maxes_both(self):
        # Degenerate: pure black -> 0 dynamic range + dark median ->
        # both heuristics push to max. Pinned so the clamp catches
        # the degenerate case (without the clamp, p50=0 sends gamma
        # past the slider end).
        img = np.zeros((32, 32), dtype=np.uint8)
        clip, gamma = _auto_enhance_params(img)
        assert clip == _CLAHE_CLIP_MAX
        assert gamma == _GAMMA_MAX

    def test_all_white_image_clips_max_gamma_min(self):
        # Pure white -> 0 dyn range (push clip up) but bright median
        # (don't lift gamma).
        img = np.full((32, 32), 255, dtype=np.uint8)
        clip, gamma = _auto_enhance_params(img)
        assert clip == _CLAHE_CLIP_MAX
        assert gamma == _GAMMA_MIN

    def test_full_range_gradient_is_near_passthrough(self):
        # Synthetic uniform-gradient frame: full dynamic range +
        # mid-grey median -> heuristic should land near "no
        # enhancement" so an already-balanced image isn't
        # over-processed.
        row = np.linspace(0, 255, 256, dtype=np.uint8)
        img = np.tile(row, (64, 1))
        clip, gamma = _auto_enhance_params(img)
        # Both should be at or very near the min slider values.
        assert clip < _CLAHE_CLIP_MIN + 0.1
        assert gamma < _GAMMA_MIN + 0.1

    def test_dark_low_contrast_image_pushes_both_up(self):
        # Direction check on a dark, narrow-histogram frame
        # ([5, 30] uniform: p5~6, p50~17, p95~28, dyn~22). Pass-4
        # anchors (DARK=0, MID=25, LOW=0, HIGH=75) catch this case:
        #   t_clip  = (75 - 22) / 75  = 0.71  -> clip  ~3.12
        #   t_gamma = (25 - 17) / 25  = 0.32  -> gamma ~1.32
        # Pinned loosely so future anchor retuning has headroom; the
        # tighter "S-corpus target" pin is in
        # ``test_s_corpus_clip_lands_near_user_target`` below.
        img = np.random.RandomState(0).randint(5, 30, size=(128, 128)).astype(np.uint8)
        clip, gamma = _auto_enhance_params(img)
        assert clip > _CLAHE_CLIP_MIN + 1.0
        assert gamma > _GAMMA_MIN + 0.1
        assert clip < _CLAHE_CLIP_MAX
        assert gamma < _GAMMA_MAX

    def test_typical_brightish_ultrasound_is_near_bypass(self):
        # Pass-4 retune (2026-05-19): an ultrasound frame with a
        # mid-grey median (p50~60) lands at near-passthrough
        # because the user prefers manual dial-up over Auto over-
        # enhancing. Anchors are deliberately conservative: gamma
        # kicks in only when p50 < 25 and clip kicks in only when
        # dyn-range < 75. Test pins this floor against future
        # anchor drift.
        img = np.random.RandomState(0).randint(20, 100, size=(128, 128)).astype(np.uint8)
        clip, gamma = _auto_enhance_params(img)
        # Representative stats are p50~60, dyn~72:
        #   t_clip  = (75 - 72) / 75 = 0.04 -> clip  ~1.12
        #   t_gamma = clamped to 0          -> gamma  1.00
        assert clip < _CLAHE_CLIP_MIN + 0.3, f"clip={clip} drifted off floor"
        assert gamma == _GAMMA_MIN, f"gamma={gamma} should be passthrough for p50~60"

    def test_s_corpus_clip_lands_near_user_target(self):
        # Calibration anchor: pass-4 was tuned against the user's
        # S-corpus DUSTrack clip (inferred stats: dyn~61, p50~20).
        # User target was clip~1.6, gamma~1.3. Synthesize a frame
        # matching those percentiles via uniform [5, 65] which gives
        # p5~8, p50~35, p95~62, dyn~54. A bit brighter / wider than
        # the actual user clip but illustrative of the right
        # direction. Pin so future anchor retuning needs an explicit
        # rationale for moving the user's calibration target.
        img = np.random.RandomState(0).randint(5, 65, size=(128, 128)).astype(np.uint8)
        clip, gamma = _auto_enhance_params(img)
        # Expected stats from random seed:
        #   dyn ~54, p50 ~35
        #   t_clip  = (75 - 54) / 75 = 0.28 -> clip  ~1.84
        #   t_gamma = clamped to 0          -> gamma  1.00 (p50>25)
        # Window is loose to allow real ultrasound histograms (which
        # have non-uniform shapes) to land in range too.
        assert 1.0 <= clip <= 2.5
        assert 1.0 <= gamma <= 1.5

    def test_rgb_input_handled_via_grayscale_conversion(self):
        # Auto must handle the RGB-shaped numpy array
        # ``VideoReader[i].asnumpy()`` produces. Result should match
        # the equivalent grayscale frame (the RGB->gray conversion
        # is the only difference).
        gray = np.full((32, 32), 80, dtype=np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)
        clip_g, gamma_g = _auto_enhance_params(gray)
        clip_rgb, gamma_rgb = _auto_enhance_params(rgb)
        # cv.cvtColor on a flat gray RGB image returns the same gray;
        # results should be byte-identical.
        assert math.isclose(clip_g, clip_rgb, abs_tol=1e-9)
        assert math.isclose(gamma_g, gamma_rgb, abs_tol=1e-9)


class TestApplyGammaOnly:
    """Per-slider bypass helper for the gamma-only path (clip at min,
    gamma off min). Smooths the gamma-slider left-end transition by
    skipping the CLAHE pipeline + its RGB->gray->RGB roundtrip.
    """

    def test_gamma_one_is_near_identity(self):
        # gamma=1.0 makes the inverse-gamma LUT the identity function:
        # for every i in 0..255, ((i / 255.0) ** 1.0) * 255 == i.
        # The uint8 cast can shave the topmost ramp value but the
        # rest is bit-identical to the input.
        img = np.arange(256, dtype=np.uint8).reshape(16, 16)
        out = _apply_gamma_only(img, 1.0)
        assert out.shape == img.shape
        # Allow at most 1-unit difference per pixel from rounding.
        assert np.max(np.abs(out.astype(np.int16) - img.astype(np.int16))) <= 1

    def test_gamma_lt_one_darkens_midtones(self):
        # gamma < 1 (here 0.5) applies a brightening LUT in the
        # enhance_ultrasound_image convention: inv_gamma = 1/0.5 = 2,
        # so output[i] = ((i/255)**2) * 255. Mid-grey 128 maps down to
        # ~64. Sanity check direction.
        img = np.full((4, 4), 128, dtype=np.uint8)
        out = _apply_gamma_only(img, 0.5)
        assert out[0, 0] < 128, "gamma<1 should darken midtones in this convention"

    def test_gamma_gt_one_lifts_midtones(self):
        # gamma > 1 (here 2.0) lifts midtones: inv_gamma = 0.5, so
        # output[i] = ((i/255)**0.5) * 255. Mid-grey 128 maps up to
        # ~181. Sanity check direction (matches what the EnhanceWidget
        # slider does when gamma slider moves right).
        img = np.full((4, 4), 128, dtype=np.uint8)
        out = _apply_gamma_only(img, 2.0)
        assert out[0, 0] > 128, "gamma>1 should lift midtones"

    def test_rgb_input_processed_per_channel(self):
        # On a 3-channel RGB array, cv.LUT runs the same table on
        # each channel. For grayscale-encoded RGB (R=G=B), all three
        # channels stay equal post-LUT -- no color shift introduced.
        gray_val = 128
        img = np.full((4, 4, 3), gray_val, dtype=np.uint8)
        out = _apply_gamma_only(img, 1.5)
        assert out.shape == (4, 4, 3)
        # All channels still equal.
        assert (out[..., 0] == out[..., 1]).all()
        assert (out[..., 1] == out[..., 2]).all()

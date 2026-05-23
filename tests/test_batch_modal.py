"""Tests for the Qt-free pieces of ``dustrack._batch_modal``.

Covers :func:`run_batch_jobs` dispatch (which underlying batch op is
called for which checkbox combo, and how the cancel hook short-circuits
between phases). The Qt overlay itself is exercised by hand; the
dispatcher is the load-bearing logic and stays unit-testable without
qtpy.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from dustrack._batch_modal import BatchJobSpec, BatchRunResults, run_batch_jobs


def _sources(tmp_path, n=2) -> list[Path]:
    return [tmp_path / f"v{i}.mp4" for i in range(n)]


def test_dispatch_files_calls_both_phases(tmp_path):
    src = _sources(tmp_path, n=1)
    spec = BatchJobSpec(source=src, convert_to_mono=True, build_toc=True)

    with patch("dustrack._batch_modal._batch.convert_to_mono") as mock_conv, patch(
        "dustrack._batch_modal._batch.build_toc"
    ) as mock_toc:
        mock_conv.return_value = [tmp_path / "v0_mono.mp4"]
        mock_toc.return_value = {str(tmp_path / "v0_mono.mp4"): "built"}
        results = run_batch_jobs(spec)

    assert mock_conv.called
    assert mock_toc.called
    # source list is passed through verbatim.
    assert mock_conv.call_args.args[0] == src
    assert results.converted == [tmp_path / "v0_mono.mp4"]
    assert results.toc_results == {str(tmp_path / "v0_mono.mp4"): "built"}
    assert results.error is None
    assert not results.cancelled


def test_dispatch_skips_convert_when_unchecked(tmp_path):
    spec = BatchJobSpec(source=_sources(tmp_path), convert_to_mono=False, build_toc=True)

    with patch("dustrack._batch_modal._batch.convert_to_mono") as mock_conv, patch(
        "dustrack._batch_modal._batch.build_toc"
    ) as mock_toc:
        mock_toc.return_value = {}
        run_batch_jobs(spec)

    mock_conv.assert_not_called()
    mock_toc.assert_called_once()


def test_cancel_between_phases_skips_toc(tmp_path):
    """If cancel fires after the convert phase, build_toc must not run."""
    spec = BatchJobSpec(source=_sources(tmp_path), convert_to_mono=True, build_toc=True)
    state = {"cancelled": False}

    def cancel_check():
        return state["cancelled"]

    def fake_convert(*a, **kw):
        # Simulate convert completing; flip cancel before TOC starts.
        state["cancelled"] = True
        return []

    with patch(
        "dustrack._batch_modal._batch.convert_to_mono", side_effect=fake_convert
    ), patch("dustrack._batch_modal._batch.build_toc") as mock_toc:
        results = run_batch_jobs(spec, cancel_check=cancel_check)

    mock_toc.assert_not_called()
    assert results.cancelled is True


def test_phase_callback_routes_phase_tag(tmp_path):
    spec = BatchJobSpec(source=_sources(tmp_path), convert_to_mono=True, build_toc=True)
    seen: list[tuple] = []

    def cb(phase, idx, total, path, status):
        seen.append((phase, status))

    def fake_convert(sources, *, progress_callback=None, **kw):
        if progress_callback is not None:
            progress_callback(0, 1, Path("a.mp4"), "ok")
        return []

    def fake_toc(sources, *, progress_callback=None, **kw):
        if progress_callback is not None:
            progress_callback(0, 1, Path("a.mp4"), "built")
        return {}

    with patch(
        "dustrack._batch_modal._batch.convert_to_mono", side_effect=fake_convert
    ), patch("dustrack._batch_modal._batch.build_toc", side_effect=fake_toc):
        run_batch_jobs(spec, progress_callback=cb)

    assert ("convert", "ok") in seen
    assert ("toc", "built") in seen


def test_error_in_convert_does_not_block_toc(tmp_path):
    """The dispatcher captures per-phase errors but continues to the
    next phase so partial progress still happens."""
    spec = BatchJobSpec(source=_sources(tmp_path), convert_to_mono=True, build_toc=True)

    with patch(
        "dustrack._batch_modal._batch.convert_to_mono",
        side_effect=RuntimeError("ffmpeg not found"),
    ), patch("dustrack._batch_modal._batch.build_toc") as mock_toc:
        mock_toc.return_value = {}
        results = run_batch_jobs(spec)

    assert "convert_to_mono failed" in (results.error or "")
    mock_toc.assert_called_once()


def test_run_batch_returns_batch_run_results_type(tmp_path):
    spec = BatchJobSpec(source=_sources(tmp_path), convert_to_mono=False, build_toc=False)
    out = run_batch_jobs(spec)
    assert isinstance(out, BatchRunResults)

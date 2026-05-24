# API Reference

```{eval-rst}
.. automodule:: dustrack
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
```

## Entry point

The recommended way to start a DUSTrack session. Hand it a path and it
figures out whether you're starting fresh on a bare video or resuming
inside a DLC project, dispatching to :class:`~dustrack.gui.DUSTrack`
or :meth:`~dustrack.dlcinterface.DLCProject.annotate` accordingly. Direct
construction of :class:`~dustrack.gui.DUSTrack` still works for
advanced use, but ``dustrack.open(...)`` is the documented surface.

```{eval-rst}
.. autofunction:: dustrack._open.open
```

## DUSTrack GUI

```{eval-rst}
.. autoclass:: dustrack.gui.DUSTrack
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
```

## Deep learning

Interface class for using ResNets via DeepLabCut.

```{eval-rst}
.. autoclass:: dustrack.dlcinterface.DLCProject
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
```

## Optical flow

Module for optical flow based postprocessing using the LK-RSTC algorithm.

```{eval-rst}
.. automodule:: dustrack.lk_filter
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
```

## Batch processing

Helpers for warming a folder of ultrasound videos before an annotation
session. Pre-build the PyAV+TOC sidecars (`build_toc`) so
`dustrack.open(...)` returns instantly on every file, and re-encode
yuv420p captures as h265 monochrome (`convert_to_mono`) to fix the
chroma-noise penalty that yuv420p clips carry into DLC inference.

The batch surface is also wired into the GUI: `dustrack.open()` (with
no path) pops a welcome modal whose top-right **Batch process...**
button opens a click-driven interface for the same operations; the
Tools menu on the main window has the same entry for use mid-session.

### Mono encoding — CRF quality reference

Ultrasound capture pipelines typically store frames as h264 yuv420p,
which folds chroma-plane noise back into the gray channel during
RGB→gray conversion and costs ~0.7 px median DLC keypoint error.
`convert_to_mono` re-encodes as h265 4:0:0 monochrome (`-c:v libx265
-pix_fmt gray`) so the chroma planes are dropped entirely — both for
storage savings (~6% smaller at default settings) and for DLC accuracy.

The CRF default is 22 (chosen for parity with capture-side quality).
Higher CRF values trade pixel accuracy for storage savings. The
following table summarises the tradeoff measured on a 30 s pia02
ultrasound subclip (2010 frames × 2 keypoints, `interosseous_pn24`
ResNet-50 model, pixel error relative to a lossless mono reference):

| CRF | file size (relative) | median error (px) | p95 (px) |
|-----|----------------------|-------------------|----------|
| 22 (default) | 100% | 0.37 | 1.04 |
| 24 | 72% | 0.47 | 1.29 |
| 26 | 51% | 0.58 | 1.60 |
| 28 | 35% | 0.73 | 1.96 |

All values stay sub-pixel at the median; the p95 statistic is the
better gauge of how often DLC trace post-processing (LK refinement)
will need to clean up a wobble. CRF 28 sits at the edge where p95
starts to approach the LK refinement window, so 22-26 is the
recommended operating range. Storage-constrained workflows can pass
`crf=24` or `crf=26` to `convert_to_mono` (or to
`immersionlab.telemed.crop_video(mono=True, crf=...)` for the
one-pass crop+mono variant).

```{eval-rst}
.. automodule:: dustrack.batch
    :members:
    :undoc-members:
    :show-inheritance:
```

## Helpers
```{eval-rst}
.. autoclass:: dustrack.annotations.VideoAnnotation
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
    :exclude-members: postprocess

**postprocess**
    Post-processes the annotation using LK-RSTC filter.
    See :func:`~dustrack.lk_filter.lk_moving_average_filter` for details.

.. autoclass:: dustrack._file_management.VideoFileManager
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__

.. autofunction:: dustrack._file_management._extract_frames
.. autofunction:: dustrack._file_management._extract_frames_decord

.. autofunction:: dustrack._file_management.get_annotation_file_name
.. autofunction:: dustrack._file_management.make_annotation_file_name
.. autofunction:: dustrack._file_management.merge_annotations_in_folder
```

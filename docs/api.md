# API Reference

This page provides detailed documentation for all modules and functions in the DUSTrack package.

## Overview
```{eval-rst}
.. automodule:: dustrack
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
```

## DUSTrack GUI

```{eval-rst}
.. autoclass:: dustrack.dlcinterface.DUSTrack
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
```

## DeepLabCut Interface

```{eval-rst}
.. autoclass:: dustrack.dlcinterface.DLCProject
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
```

## LK-RSTC Post-Processing

```{eval-rst}
.. automodule:: dustrack.postprocess
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
```

## Helpers
```{eval-rst}
.. autoclass:: dustrack.dlcinterface.VideoAnnotation
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
    :exclude-members: postprocess

**postprocess**
    Post-processes the annotation using LK-RSTC filter. 
    See :func:`~dustrack.postprocess.lk_moving_average_filter` for details.

.. autoclass:: dustrack.dlcinterface.VideoFileManager
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__

.. autofunction:: dustrack.dlcinterface._extract_frames
.. autofunction:: dustrack.dlcinterface._extract_frames_decord

.. autofunction:: dustrack.dlcinterface.get_annotation_file_name
.. autofunction:: dustrack.dlcinterface.make_annotation_file_name
.. autofunction:: dustrack.dlcinterface.merge_annotations_in_folder
```
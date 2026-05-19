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
inside a DLC project, dispatching to :class:`~dustrack.dlcinterface.DUSTrack`
or :meth:`~dustrack.dlcinterface.DLCProject.annotate` accordingly. Direct
construction of :class:`~dustrack.dlcinterface.DUSTrack` still works for
advanced use, but ``dustrack.open(...)`` is the documented surface.

```{eval-rst}
.. autofunction:: dustrack.dlcinterface.open
```

## DUSTrack GUI

```{eval-rst}
.. autoclass:: dustrack.dlcinterface.DUSTrack
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
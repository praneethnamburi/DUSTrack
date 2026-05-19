"""
Programmatic probe for the "press t over image axes doesn't add annotation"
bug. Run in dlc env.

Steps:
  1. Open DLCProject + annotate() to get a real DUSTrack instance.
  2. SAFELY disconnect every action_func from every button so we
     can't trigger destructive callbacks.
  3. ALSO replace VideoPointAnnotator.add_annotation with a sniffer
     that records the event but doesn't actually add the annotation
     (no file writes).
  4. Synthesize a Qt mouse move event to the CENTER of self._ax_image's
     display bbox. This is the same coordinate path matplotlib uses to
     compute event.inaxes for keyPressEvents.
  5. Synthesize Qt keyClick('t').
  6. Print: what did matplotlib observe in the resulting KeyEvent?
     - event.key
     - event.xdata, event.ydata (data coordinates)
     - event.inaxes (which axes the cursor was over)
     - event.inaxes is ret._ax_image (does it match?)

If event.inaxes is ret._ax_image AND xdata/ydata are valid -> the
add_annotation check should pass. If not, the bug is at the
mpl-axes-lookup layer; the rest of the diagnostic narrows it.
"""

import os
import sys
import logging
import warnings

os.environ.setdefault("QT_API", "pyside6")
logging.getLogger("numexpr.utils").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module="shibokensupport.signature.parser")

import matplotlib
matplotlib.use("QtAgg")

from matplotlib import pyplot as plt
from qtpy.QtCore import QPoint, Qt
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))


CONFIG = r"M:\DLC_MODELS\general\interosseous_pn24-x-2025-10-24\config.yaml"


def main():
    from dustrack import DLCProject
    import datanavigator

    print(f"datanavigator: {datanavigator.__version__} ({datanavigator.__file__})")
    print(f"matplotlib   : {matplotlib.__version__}")
    print()

    print("opening DLCProject + annotate() (loads UI)...")
    g = DLCProject(CONFIG)
    ret = g.annotate(video_index=0)
    for _ in range(10):
        app.processEvents()
    print("UI loaded.\n")

    # SAFETY: disconnect every button slot. Cannot fire any real callback.
    for name in ret.buttons.names:
        b = ret.buttons[name]
        if hasattr(b, "_qt_btn"):
            try:
                b._qt_btn.clicked.disconnect()
            except (TypeError, RuntimeError):
                pass

    # Replace add_annotation with a sniffer (records the event, doesn't write).
    recorded_events = []
    original_add_annotation = type(ret).add_annotation

    def sniff_add_annotation(self, event):
        recorded_events.append({
            "key": event.key,
            "xdata": event.xdata,
            "ydata": event.ydata,
            "inaxes": event.inaxes,
            "inaxes_is_ax_image": event.inaxes is self._ax_image,
            "x_pixel": getattr(event, "x", None),
            "y_pixel": getattr(event, "y", None),
        })

    # Bind on the instance
    import types
    ret.add_annotation = types.MethodType(sniff_add_annotation, ret)
    # Rebind the keypress dispatch entry to the new method
    from datanavigator.core import KeyBinding
    prev = ret._keypressdict["t"]
    ret._keypressdict["t"] = KeyBinding(
        callback=ret.add_annotation, description=prev.description, group=prev.group
    )

    canvas = ret.figure.canvas
    print(f"figure size : {ret.figure.get_size_inches()} inches")
    print(f"canvas size : {canvas.size().width()}x{canvas.size().height()} pixels")
    print(f"_ax_image   : {ret._ax_image}")
    print(f"  position  : {ret._ax_image.get_position()} (figure coords)")
    print(f"  bbox      : {ret._ax_image.bbox.bounds} (display coords)")

    # Compute the center of _ax_image in canvas pixel coordinates.
    bbox = ret._ax_image.bbox
    cx_display = (bbox.x0 + bbox.x1) / 2
    cy_display = (bbox.y0 + bbox.y1) / 2
    # matplotlib's display coordinate y is bottom-up; Qt's is top-down.
    cy_qt = canvas.height() - cy_display
    target = QPoint(int(cx_display), int(cy_qt))
    print(f"  ax_image center (display)  : ({cx_display:.1f}, {cy_display:.1f})")
    print(f"  ax_image center (Qt coords): {target.x()}, {target.y()}")

    # Synthesize the mouse move so matplotlib's cursor tracking updates.
    print("\nsynthesizing mouse move to ax_image center...")
    QTest.mouseMove(canvas, target)
    for _ in range(5):
        app.processEvents()

    # Synthesize key press 't'
    print("synthesizing keyClick('t')...")
    QTest.keyClick(canvas, Qt.Key_T)
    for _ in range(5):
        app.processEvents()

    print(f"\nrecorded {len(recorded_events)} add_annotation invocations:")
    for i, ev in enumerate(recorded_events):
        print(f"  [{i}] {ev}")

    # Also report what's in the figure axes list, for context.
    print(f"\nall axes in the figure ({len(ret.figure.axes)} total):")
    for i, ax in enumerate(ret.figure.axes):
        marker = " <- _ax_image" if ax is ret._ax_image else ""
        print(f"  [{i}] bbox={ax.bbox.bounds}{marker}")

    plt.close(ret.figure)
    return 0


if __name__ == "__main__":
    sys.exit(main())

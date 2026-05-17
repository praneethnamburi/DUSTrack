"""
Targeted probe for the 'z' key binding (VideoPointAnnotator's event
add-marker via events.add(add_key='z', ...)).

This script:
  1. Constructs a synthetic browser with the 'z' binding registered
     directly via add_key_binding (mimicking what events.add does).
  2. Watches every keypress: Qt key code, character, modifier state,
     and the EXACT string mpl reports.
  3. Confirms whether the 'z' binding fires.

You'll see ONE printed line per keypress. The line shows:
  - 'mpl_key=' the literal string matplotlib reports
  - 'in_keymap=' whether that string is a key in self._keypressdict
  - 'fired=' whether the binding ran

Try:
  1. Press plain z (no shift, no caps lock). Expect mpl_key='z', fired=True.
  2. Press Shift+Z. Expect mpl_key='Z', fired=False (no Z binding).
  3. Press something else (e.g. 'a'). Expect mpl_key='a', fired=False.
  4. Close window.

If pressing plain z reports mpl_key='Z' even without explicit shift,
then either Caps Lock is on or your keyboard layout sends shift-by-default.
"""

import os
import sys
import logging
import time

os.environ.setdefault("QT_API", "pyside6")
logging.getLogger("numexpr.utils").setLevel(logging.WARNING)

import warnings
# Suppress the PySide6 shiboken QFileDialog parser warnings (cosmetic
# only, known PySide6 6.4.x issue).
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module="shibokensupport.signature.parser")

import matplotlib
matplotlib.use("QtAgg")

from matplotlib import pyplot as plt
from qtpy.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from datanavigator.core import GenericBrowser


def main():
    fired_count = {"z": 0, "other": 0}

    def on_z_fired():
        fired_count["z"] += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([1, 2, 3, 4, 5], [2, 4, 3, 5, 4])
    ax.set_title("Press 'z' (no shift, no caps lock) over the plot")

    b = GenericBrowser(figure_handle=fig)
    b.add_key_binding("z", on_z_fired, "test z binding")

    # Instrument: log every mpl key_press_event AND whether the binding fired.
    def log_key(event):
        t = time.strftime("%H:%M:%S")
        in_keymap = event.key in b._keypressdict
        before = fired_count["z"]
        # GenericBrowser.__call__ is what mpl_connect routed our handler to;
        # by the time this log_key runs (we connect it AFTER GenericBrowser's
        # __init__), the dispatcher has already run.
        # We sample fired_count after.
        time.sleep(0.01)  # let the prior dispatch complete
        after = fired_count["z"]
        fired = after > before
        print(f"[{t}] mpl_key={event.key!r:12s} "
              f"xdata={event.xdata} ydata={event.ydata} "
              f"in_keymap={in_keymap} fired_in_this_event={fired}",
              flush=True)

    fig.canvas.mpl_connect("key_press_event", log_key)

    print(f"binding registered: 'z' -> on_z_fired")
    print(f"_keypressdict keys: {sorted(b._keypressdict.keys())}")
    print()
    print("Press keys and watch the output. Plain 'z' should fire.")
    print("Close the window when done.\n")

    plt.show()
    print(f"\nfinal fired count for 'z' binding: {fired_count['z']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

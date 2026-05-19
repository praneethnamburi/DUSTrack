"""Smoke probe: open a VideoPointAnnotator-shaped browser, click the
"Keyboard shortcuts" button, and check the rc2 grouped dialog renders.

Run:
    QT_API=pyqt5 python tests/qt_learning/21_dustrack_keybindings_smoke.py
"""

import os
import sys

os.environ.setdefault("QT_API", "pyqt5")

import matplotlib
matplotlib.use("QtAgg")

from matplotlib import pyplot as plt
from qtpy.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from datanavigator.core import GenericBrowser


def main():
    fig = plt.figure()
    b = GenericBrowser(figure_handle=fig)
    b.set_default_keybindings()

    # Declare a button-attached binding to verify the hint wiring.
    def my_action(event=None):
        pass

    btn = b.buttons.add(text="My action", action_func=my_action)
    b.add_key_binding(
        "ctrl+m", my_action, "My demo action",
        group="Demo", on_button=True,
    )

    for _ in range(5):
        app.processEvents()

    assert btn.name == "My action  (ctrl+m)", (
        f"expected hint on button face, got {btn.name!r}"
    )
    print(f"button label: {btn.name!r}")

    b.show_key_bindings()
    for _ in range(5):
        app.processEvents()

    dialog = b._keybinding_dialog
    assert dialog is not None and dialog.isVisible()

    # Walk children to verify sections exist.
    from qtpy.QtWidgets import QGroupBox
    section_titles = [c.title() for c in dialog.findChildren(QGroupBox)]
    print(f"sections found: {section_titles}")
    assert "Navigation" in section_titles
    assert "View" in section_titles
    assert "Demo" in section_titles
    # "Other" should be last if present.

    print("dustrack-keybindings smoke OK")
    dialog.close()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())

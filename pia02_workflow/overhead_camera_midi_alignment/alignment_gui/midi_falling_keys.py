"""MIDI Falling Keys Visualization Tool.

A real-time MIDI visualizer with falling notes and an interactive piano keyboard.
Requires PyQt5: pip install PyQt5
"""

import os
import sys
import time
import dataclasses
from bisect import bisect_left

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout,
        QHBoxLayout, QPushButton, QLabel, QSlider, QSizePolicy,
    )
    from PyQt5.QtCore import Qt, QTimer, QRectF
    from PyQt5.QtGui import QPainter, QColor, QBrush, QPen, QFont
except ImportError:
    raise ImportError("PyQt5 is required. Install with: pip install PyQt5")

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from pia02_workflow.overhead_camera_midi_alignment.midi import Log, MIDI_TO_NOTE


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class NoteData:
    pitch: int
    start: float
    end: float
    velocity: int
    name: str
    is_black: bool
    color: object  # QColor


def extract_notes(midi_log):
    """Return (list[NoteData], total_duration) from a midi.Log."""
    raw = midi_log.notes
    if not raw:
        return [], 0.0

    vels = [n.velocity for n in raw]
    vmin, vmax = min(vels), max(vels)
    vrange = vmax - vmin if vmax > vmin else 1

    notes = []
    for n in raw:
        if n.pitch < 21 or n.pitch > 108:
            continue
        name = MIDI_TO_NOTE.get(n.pitch, f"M{n.pitch}")
        is_black = '#' in name
        norm = (n.velocity - vmin) / vrange
        hue = 0.6 * (1.0 - norm)  # blue (soft) → red (loud)
        color = QColor.fromHsvF(hue, 0.7, 0.9)
        notes.append(NoteData(n.pitch, n.start, n.end, n.velocity, name, is_black, color))

    notes.sort(key=lambda nd: nd.start)
    total = midi_log.duration
    return notes, total


# ---------------------------------------------------------------------------
# Piano keyboard (bottom strip)
# ---------------------------------------------------------------------------

class PianoKeyboardWidget(QWidget):
    NUM_WHITE = 52  # A0 – C8

    def __init__(self, parent=None):
        super().__init__(parent)
        self._active = set()
        self._rects = {}        # pitch → QRectF
        self._is_black = {}     # pitch → bool
        self.setFixedHeight(80)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    # geometry ---------------------------------------------------------------

    def _rebuild(self):
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return

        wkw = w / self.NUM_WHITE
        p2wi = {}   # pitch → white-key index
        wi = 0
        for p in range(21, 109):
            name = MIDI_TO_NOTE[p]
            blk = '#' in name
            self._is_black[p] = blk
            if not blk:
                p2wi[p] = wi
                wi += 1

        self._rects = {}
        for p in range(21, 109):
            if not self._is_black[p]:
                idx = p2wi[p]
                self._rects[p] = QRectF(idx * wkw, 0, wkw, h)
            else:
                cx = (p2wi[p - 1] + 1) * wkw
                bw = wkw * 0.6
                bh = h * 0.62
                self._rects[p] = QRectF(cx - bw / 2, 0, bw, bh)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rebuild()

    # public -----------------------------------------------------------------

    def set_active_pitches(self, pitches):
        if pitches != self._active:
            self._active = pitches
            self.update()

    def pitch_x_center(self, pitch):
        r = self._rects.get(pitch)
        return r.x() + r.width() / 2 if r else None

    def pitch_width(self, pitch):
        r = self._rects.get(pitch)
        return r.width() if r else 10

    # paint ------------------------------------------------------------------

    def paintEvent(self, event):
        if not self._rects:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)

        W_FILL = QColor(255, 255, 255)
        B_FILL = QColor(0, 0, 0)
        W_ACT  = QColor('#5DADE2')
        B_ACT  = QColor('#2E86C1')
        BORDER = QPen(QColor(80, 80, 80), 1)

        # white keys first
        for pitch in range(21, 109):
            if self._is_black.get(pitch):
                continue
            r = self._rects[pitch]
            p.setBrush(QBrush(W_ACT if pitch in self._active else W_FILL))
            p.setPen(BORDER)
            p.drawRect(r)

        # black keys on top
        for pitch in range(21, 109):
            if not self._is_black.get(pitch):
                continue
            r = self._rects[pitch]
            p.setBrush(QBrush(B_ACT if pitch in self._active else B_FILL))
            p.setPen(BORDER)
            p.drawRect(r)

        p.end()


# ---------------------------------------------------------------------------
# Falling notes area
# ---------------------------------------------------------------------------

class FallingNotesWidget(QWidget):
    VIS_MIN = 2.0
    VIS_MAX = 20.0

    def __init__(self, notes, keyboard, parent=None):
        super().__init__(parent)
        self._notes = notes
        self._starts = [n.start for n in notes]
        self._max_dur = max((n.end - n.start for n in notes), default=1.0)
        self._kb = keyboard
        self._t = 0.0
        self._vis = 5.0
        self.setMinimumHeight(200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_current_time(self, t):
        self._t = t
        self.update()

    def wheelEvent(self, event):
        d = event.angleDelta().y()
        step = 0.5
        if d > 0:
            self._vis = max(self.VIS_MIN, self._vis - step)
        elif d < 0:
            self._vis = min(self.VIS_MAX, self._vis + step)
        self.update()

    def paintEvent(self, event):
        if not self._kb._rects:
            return

        pr = QPainter(self)
        W, H = self.width(), self.height()
        pr.fillRect(0, 0, W, H, QColor(30, 30, 30))

        t0 = self._t                    # bottom = now
        t1 = self._t + self._vis        # top = future
        pps = H / self._vis             # pixels per second

        # grid every 1 s
        grid_pen = QPen(QColor(60, 60, 60), 1)
        txt_pen  = QPen(QColor(100, 100, 100))
        pr.setFont(QFont("Consolas", 8))
        for ts in range(int(t0) + 1, int(t1) + 1):
            y = H - (ts - t0) * pps
            if 0 <= y <= H:
                pr.setPen(grid_pen)
                pr.drawLine(0, int(y), W, int(y))
                pr.setPen(txt_pen)
                pr.drawText(5, int(y) - 3, f"{ts // 60}:{ts % 60:02d}")

        # notes
        lo = bisect_left(self._starts, t0 - self._max_dur)
        hi = bisect_left(self._starts, t1)
        pr.setPen(Qt.NoPen)
        for i in range(lo, hi):
            n = self._notes[i]
            if n.end <= t0:
                continue
            xc = self._kb.pitch_x_center(n.pitch)
            if xc is None:
                continue
            nw = self._kb.pitch_width(n.pitch)
            yt = H - (n.end   - t0) * pps
            yb = H - (n.start - t0) * pps
            pr.setBrush(QBrush(n.color))
            pr.drawRoundedRect(QRectF(xc - nw / 2 + 1, yt, nw - 2, yb - yt), 3, 3)

        # now-line
        pr.setPen(QPen(QColor(255, 255, 255, 180), 2))
        pr.drawLine(0, H - 1, W, H - 1)
        pr.end()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MidiFallingKeysWindow(QMainWindow):

    def __init__(self, midi_log):
        super().__init__()

        notes, dur = extract_notes(midi_log)
        self._notes = notes
        self._dur = dur
        self._t = 0.0
        self._playing = False
        self._wall0 = 0.0
        self._midi0 = 0.0

        fname = os.path.basename(midi_log.filename) if midi_log.filename else "Untitled"
        self.setWindowTitle(f"MIDI Falling Keys \u2014 {fname}")
        self.resize(1200, 700)

        # widgets
        self._kb = PianoKeyboardWidget()
        self._fall = FallingNotesWidget(notes, self._kb)

        self._play_btn = QPushButton("Play")
        self._stop_btn = QPushButton("Stop")
        self._lbl = QLabel()
        self._lbl.setMinimumWidth(160)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, max(1, int(dur * 10)))
        self._slider.valueChanged.connect(self._on_slider)
        self._play_btn.clicked.connect(self._on_play_pause)
        self._stop_btn.clicked.connect(self._on_stop)

        # layout
        row = QHBoxLayout()
        row.addWidget(self._play_btn)
        row.addWidget(self._stop_btn)
        row.addStretch()
        row.addWidget(self._lbl)

        tbar = QVBoxLayout()
        tbar.addLayout(row)
        tbar.addWidget(self._slider)
        tw = QWidget()
        tw.setLayout(tbar)

        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._fall)
        root.addWidget(self._kb)
        root.addWidget(tw)

        c = QWidget()
        c.setLayout(root)
        self.setCentralWidget(c)

        # timer ~60 fps
        self._timer = QTimer()
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        self._refresh()

    # -- transport -----------------------------------------------------------

    def _on_play_pause(self):
        if self._playing:
            self._playing = False
            self._play_btn.setText("Play")
        else:
            if self._t >= self._dur:
                self._t = 0.0
            self._playing = True
            self._play_btn.setText("Pause")
            self._wall0 = time.perf_counter()
            self._midi0 = self._t

    def _on_stop(self):
        self._playing = False
        self._play_btn.setText("Play")
        self._t = 0.0
        self._refresh()

    def _on_slider(self, val):
        self._t = val / 10.0
        if self._playing:
            self._wall0 = time.perf_counter()
            self._midi0 = self._t
        self._refresh()

    def _tick(self):
        if not self._playing:
            return
        t = self._midi0 + (time.perf_counter() - self._wall0)
        if t >= self._dur:
            t = self._dur
            self._playing = False
            self._play_btn.setText("Play")
        self._t = t
        self._refresh()

    def _refresh(self):
        t = self._t
        self._fall.set_current_time(t)

        # active pitches
        active = set()
        for n in self._notes:
            if n.start > t:
                break
            if n.end > t:
                active.add(n.pitch)
        self._kb.set_active_pitches(active)

        # time label
        self._lbl.setText(f"{_fmt(t)} / {_fmt(self._dur)}")

        # slider (no feedback loop)
        self._slider.blockSignals(True)
        self._slider.setValue(int(t * 10))
        self._slider.blockSignals(False)

    # -- keyboard shortcuts --------------------------------------------------

    def keyPressEvent(self, event):
        k = event.key()
        if k == Qt.Key_Space:
            self._on_play_pause()
        elif k == Qt.Key_Left:
            self._seek_rel(-1)
        elif k == Qt.Key_Right:
            self._seek_rel(1)
        elif k == Qt.Key_Home:
            self._seek_abs(0)
        elif k == Qt.Key_End:
            self._seek_abs(self._dur)
            self._playing = False
            self._play_btn.setText("Play")
        else:
            super().keyPressEvent(event)

    def _seek_rel(self, delta):
        self._t = max(0, min(self._dur, self._t + delta))
        if self._playing:
            self._wall0 = time.perf_counter()
            self._midi0 = self._t
        self._refresh()

    def _seek_abs(self, t):
        self._t = t
        if self._playing:
            self._wall0 = time.perf_counter()
            self._midi0 = t
        self._refresh()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(t):
    m = int(t) // 60
    s = t - m * 60
    return f"{m}:{s:04.1f}"


def main(midi_file_path):
    app = QApplication(sys.argv)
    midi_log = Log(midi_file_path)
    window = MidiFallingKeysWindow(midi_log)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = r"\\192.168.1.104\home\piano\data\045\disklavier\20250815_144525_pia02_s045_007_tempo_ramp_ap345.mid"
    main(path)

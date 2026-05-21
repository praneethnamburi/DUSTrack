# Force the non-interactive Agg backend before any pyplot import so the
# test suite runs cleanly headless (CI, local devs without an MPLBACKEND
# env var set). Mirrors datanavigator/tests/conftest.py -- shared
# helpers stayed in dnav until 1.5.0a1 / 1.2.0a1 relocated
# test_pointtracking.py here.
import matplotlib

matplotlib.use("Agg", force=True)

import pytest
import pysampled

import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent
from matplotlib.backend_bases import MouseEvent

import datanavigator


@pytest.fixture
def matplotlib_figure():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    yield fig, ax
    plt.close(fig)


@pytest.fixture
def mock_figure():
    fig = plt.figure()
    yield fig
    plt.close(fig)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def signal_list():
    return [
        pysampled.generate_signal("white_noise"),
        pysampled.generate_signal("sine_wave"),
        pysampled.generate_signal("three_sine_waves"),
    ]


@pytest.fixture(scope="session", autouse=True)
def setup_folders(tmp_path_factory):
    """Point datanavigator's global clip/cache folders at a session-scoped
    temp dir, restore on session end.

    Was function-scope + autouse, which fired around every one of the
    ~350 tests; no test in this suite reads the globals during
    execution (the video fixtures pass ``dest_folder=`` explicitly),
    so the per-test churn was pure overhead. If a future test ever
    relies on a clean clip/cache state, that test should request a
    function-scoped helper explicitly rather than re-broadening this.
    """
    curr_clip_dir = datanavigator.get_clip_folder()
    curr_cache_dir = datanavigator.get_cache_folder()
    clip_dir = tmp_path_factory.mktemp("clips")
    cache_dir = tmp_path_factory.mktemp("cache")
    datanavigator.set_clip_folder(str(clip_dir))
    datanavigator.set_cache_folder(str(cache_dir))
    yield str(clip_dir), str(cache_dir)
    datanavigator.set_clip_folder(curr_clip_dir)
    datanavigator.set_cache_folder(curr_cache_dir)


def simulate_key_press(figure, key="a", **kwargs):
    """Simulate a key press event on the given figure."""
    event = KeyEvent(
        name="key_press_event",
        canvas=figure.canvas,
        key=key,
        guiEvent=None,
    )
    for k, v in kwargs.items():
        setattr(event, k, v)
    return event


def simulate_key_press_at_xy(fax, key="1", xdata=0.5, ydata=0.5):
    """Simulate a key press event with cursor positioned at (xdata, ydata)."""
    fig, ax = fax
    x_pixel, y_pixel = ax.transData.transform((xdata, ydata))
    event = KeyEvent(
        name="key_press_event",
        canvas=fig.canvas,
        key=key,
        x=x_pixel,
        y=y_pixel,
        guiEvent=None,
    )
    event.xdata = xdata
    event.ydata = ydata
    event.inaxes = ax
    return event


def simulate_mouse_click(fax, xdata=0.5, ydata=0.5, button=1):
    """Simulate a mouse click event at (xdata, ydata) on the given axis."""
    fig, ax = fax
    event = MouseEvent(
        name="button_press_event",
        canvas=ax.figure.canvas,
        x=ax.transData.transform((xdata, ydata))[0],
        y=ax.transData.transform((xdata, ydata))[1],
        button=button,
        key=None,
        step=0,
        dblclick=False,
        guiEvent=None,
    )
    return event


def press_browser_button(button: datanavigator.Button):
    """Simulate a full mouse click (press + release) on a matplotlib Button."""
    if not hasattr(button, "ax"):
        raise AttributeError(
            "The button object must have an 'ax' attribute representing its Matplotlib Axes."
        )

    button_ax = button.ax
    if button_ax is None or button_ax.figure is None or button_ax.figure.canvas is None:
        raise ValueError(
            "The button's Axes or its associated canvas is not properly configured."
        )

    bbox = button_ax.get_position()
    x = bbox.x0 + 0.5 * bbox.width
    y = bbox.y0 + 0.5 * bbox.height
    canvas = button_ax.figure.canvas
    canvas_width, canvas_height = canvas.get_width_height()

    canvas_x = canvas_width * x
    canvas_y = canvas_height * y

    press_event = MouseEvent(
        name="button_press_event",
        canvas=canvas,
        x=canvas_x,
        y=canvas_y,
        button=1,
        key=None,
    )
    canvas.callbacks.process("button_press_event", press_event)

    release_event = MouseEvent(
        name="button_release_event",
        canvas=canvas,
        x=canvas_x,
        y=canvas_y,
        button=1,
        key=None,
    )
    canvas.callbacks.process("button_release_event", release_event)

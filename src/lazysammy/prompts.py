"""Interactive prompt helpers for visually picking points and boxes.

These utilities open a matplotlib figure where the user can click to
place prompts directly on a video frame.  They are designed for use
inside Jupyter notebooks.

**Backend requirements:**

- ``preview_frame()`` works with any backend (including ``%matplotlib inline``).
- ``pick_points_on_image()`` and ``pick_box_on_image()`` require an
  **interactive** backend to receive mouse clicks.  Use one of:

  - ``%matplotlib widget`` (recommended - needs ``ipympl``)
  - ``%matplotlib notebook`` (classic Jupyter only)

The interactive figures **close automatically** when you finish:

- **Points:** press **Enter**, **q**, or **Esc** to finish (or click *n* points
  if a fixed count was requested).
- **Box:** the figure closes after the second corner click.

Typical usage::

    %matplotlib widget          # <-- required for interactive clicking

    from lazysammy.prompts import PromptPicker

    picker = PromptPicker(session)

    # Click to place points
    points, labels = picker.pick_points(frame_idx=0)

    # Draw a box by clicking two corners
    box = picker.pick_box(frame_idx=0)

    # Full interactive flow: pick, add, and preview in one call
    picker.add_points_interactive(frame_idx=0, obj_id=1)
    picker.add_box_interactive(frame_idx=0, obj_id=2)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from lazysammy.utils import list_frame_files, load_image

if TYPE_CHECKING:
    from lazysammy.video import VideoSession

logger = logging.getLogger(__name__)


def _check_interactive_backend() -> None:
    """Warn if the current matplotlib backend doesn't support interaction."""
    import matplotlib

    backend = matplotlib.get_backend().lower()
    non_interactive = {"agg", "module://matplotlib_inline.backend_inline", "svg", "pdf", "ps"}
    if backend in non_interactive or "inline" in backend:
        logger.warning(
            "Current matplotlib backend (%s) does not support interactive "
            "clicking.  Use '%%matplotlib widget' (needs ipympl) or "
            "'%%matplotlib notebook' for click-based prompts.  "
            "preview_frame() still works for static display.",
            backend,
        )


def is_interactive_backend() -> bool:
    """Return ``True`` when the active matplotlib backend accepts clicks.

    The default inline backend used by most notebooks cannot deliver mouse
    events, so ``pick_points_on_image`` / ``pick_box_on_image`` would return
    empty results silently.  Call this to detect that up-front.

    Returns:
        ``True`` for GUI/interactive backends, ``False`` for inline/static ones.

    Example::

        if not is_interactive_backend():
            print("Enable %matplotlib widget for click prompts")
    """
    import matplotlib

    backend = matplotlib.get_backend().lower()
    if "inline" in backend:
        return False
    return backend not in {"agg", "svg", "pdf", "ps", "cairo", "template"}


def _backend_name() -> str:
    """Return the current matplotlib backend name (for messages)."""
    import matplotlib

    return str(matplotlib.get_backend())


def _set_coord_readout(ax: Any) -> None:
    """Attach a live ``x=..., y=...`` readout to the axes' status bar.

    Matplotlib types ``Axes.format_coord`` as a method, so assigning a
    callable to it is flagged by type checkers. This wrapper centralises the
    single, deliberate assignment.

    Args:
        ax: The matplotlib axes to annotate.
    """
    setattr(ax, "format_coord", lambda x, y: f"x={x:.0f}, y={y:.0f}")  # noqa: B010


# ---------------------------------------------------------------------------
# Standalone helpers (no session required)
# ---------------------------------------------------------------------------


def preview_frame(
    video_dir: str | Path,
    frame_idx: int = 0,
    *,
    figsize: tuple[int, int] = (12, 8),
    show_grid: bool = True,
    grid_step: int = 50,
) -> Any:
    """Display a frame with a coordinate grid and mouse-position readout.

    Useful for visually identifying coordinates before adding prompts.

    Args:
        video_dir: Directory of frames (or video file path - frames must
            already be extracted).
        frame_idx: Zero-based index of the frame to show.
        figsize: Matplotlib figure size.
        show_grid: Overlay a faint coordinate grid.
        grid_step: Spacing (pixels) between grid lines.

    Returns:
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt

    frames = list_frame_files(video_dir)
    if frame_idx >= len(frames):
        msg = f"frame_idx {frame_idx} out of range (0-{len(frames) - 1})"
        raise IndexError(msg)

    img = load_image(frames[frame_idx])
    h, w = img.shape[:2]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img)
    ax.set_title(
        f"Frame {frame_idx}  |  {w}x{h}  |  hover to read (x, y)",
        fontsize=11,
    )

    if show_grid:
        for x in range(0, w, grid_step):
            ax.axvline(x, color="white", linewidth=0.3, alpha=0.4)
        for y in range(0, h, grid_step):
            ax.axhline(y, color="white", linewidth=0.3, alpha=0.4)
        # Label a few ticks
        ax.set_xticks(range(0, w, grid_step))
        ax.set_yticks(range(0, h, grid_step))
        ax.tick_params(labelsize=7, colors="gray")
    else:
        ax.axis("off")

    # Live (x, y) in the status bar (interactive backends only)
    _set_coord_readout(ax)
    fig.tight_layout()
    fig.canvas.draw_idle()
    plt.show()
    return fig


def pick_points_on_image(
    image: str | Path | npt.NDArray[np.uint8],
    n: int = -1,
    *,
    figsize: tuple[int, int] = (12, 8),
    fg_color: str = "lime",
    bg_color: str = "red",
    default_label: int = 1,
    title: str | None = None,
) -> tuple[list[list[float]], list[int]]:
    """Interactively click points on an image.

    **Left-click** adds a *foreground* point, **right-click** adds a
    *background* point.  Press **Enter**, **q**, or **Esc** to finish.
    The figure closes automatically when done.

    Args:
        image: Image to annotate (path or array).
        n: Number of points to collect (``-1`` = unlimited, finish with Enter).
        figsize: Figure size.
        fg_color: Colour for foreground markers.
        bg_color: Colour for background markers.
        default_label: Label for left-click (``1`` fg, ``0`` bg).
        title: Custom window title.

    Returns:
        ``(points, labels)`` - lists ready to pass to ``add_points()``.
    """
    import matplotlib.pyplot as plt

    img = load_image(image) if not isinstance(image, np.ndarray) else image

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img)
    ax.set_title(title or "Left=fg  Right=bg  Enter/q=done")
    _set_coord_readout(ax)

    points: list[list[float]] = []
    labels: list[int] = []
    cids: list[int] = []  # connection ids to disconnect later

    def _finish() -> None:
        """Disconnect events and close the figure."""
        for c in cids:
            fig.canvas.mpl_disconnect(c)
        cids.clear()
        plt.close(fig)

    def _onclick(event: Any) -> None:
        if event.inaxes != ax:
            return
        x, y = event.xdata, event.ydata
        if event.button == 1:  # left
            label = default_label
        elif event.button == 3:  # right
            label = 1 - default_label
        else:
            return
        points.append([x, y])
        labels.append(label)
        color = fg_color if label == 1 else bg_color
        ax.plot(x, y, "o", color=color, markersize=8, markeredgecolor="white", markeredgewidth=1.2)
        ax.annotate(
            f"({x:.0f},{y:.0f})",
            (x, y),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=7,
            color=color,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.15", "fc": "black", "alpha": 0.6},
        )
        fig.canvas.draw_idle()
        if n > 0 and len(points) >= n:
            _finish()

    def _onkey(event: Any) -> None:
        if event.key in ("enter", "escape", "q"):
            _finish()

    _check_interactive_backend()
    cids.append(fig.canvas.mpl_connect("button_press_event", _onclick))
    cids.append(fig.canvas.mpl_connect("key_press_event", _onkey))
    fig.tight_layout()
    fig.canvas.draw_idle()
    plt.show()

    if not points and not is_interactive_backend():
        logger.warning(
            "No points were collected because the %r backend cannot receive "
            "clicks. Switch to '%%matplotlib widget' (ipympl) and re-run this "
            "cell.",
            _backend_name(),
        )
    return points, labels


def pick_box_on_image(
    image: str | Path | npt.NDArray[np.uint8],
    *,
    figsize: tuple[int, int] = (12, 8),
    box_color: str = "lime",
    title: str | None = None,
) -> list[float]:
    """Interactively draw a bounding box on an image.

    Click two corners - the figure closes automatically after the second
    click.  Press **q** or **Esc** to cancel.

    Args:
        image: Image to annotate.
        figsize: Figure size.
        box_color: Rectangle colour.
        title: Custom window title.

    Returns:
        ``[x1, y1, x2, y2]`` ready to pass to ``add_box()``.
    """
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    img = load_image(image) if not isinstance(image, np.ndarray) else image

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img)
    ax.set_title(title or "Click two corners to draw a box (q=cancel)")
    _set_coord_readout(ax)

    corners: list[list[float]] = []
    cids: list[int] = []

    def _finish() -> None:
        for c in cids:
            fig.canvas.mpl_disconnect(c)
        cids.clear()
        plt.close(fig)

    def _onclick(event: Any) -> None:
        if event.inaxes != ax or event.button != 1:
            return
        x, y = event.xdata, event.ydata
        corners.append([x, y])
        ax.plot(x, y, "+", color=box_color, markersize=12, markeredgewidth=2)
        ax.annotate(
            f"({x:.0f},{y:.0f})",
            (x, y),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=7,
            color=box_color,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.15", "fc": "black", "alpha": 0.6},
        )
        if len(corners) == 2:
            x1, y1 = corners[0]
            x2, y2 = corners[1]
            bx1, by1 = min(x1, x2), min(y1, y2)
            bx2, by2 = max(x1, x2), max(y1, y2)
            rect = patches.Rectangle(
                (bx1, by1),
                bx2 - bx1,
                by2 - by1,
                linewidth=2,
                edgecolor=box_color,
                facecolor=box_color,
                alpha=0.15,
            )
            ax.add_patch(rect)
            ax.set_title(f"Box: [{bx1:.0f}, {by1:.0f}, {bx2:.0f}, {by2:.0f}]")
            fig.canvas.draw_idle()
            _finish()
            return
        fig.canvas.draw_idle()

    def _onkey(event: Any) -> None:
        if event.key in ("escape", "q"):
            _finish()

    _check_interactive_backend()
    cids.append(fig.canvas.mpl_connect("button_press_event", _onclick))
    cids.append(fig.canvas.mpl_connect("key_press_event", _onkey))
    fig.tight_layout()
    fig.canvas.draw_idle()
    plt.show()

    if len(corners) < 2:
        if not is_interactive_backend():
            logger.warning(
                "No box was drawn because the %r backend cannot receive "
                "clicks. Switch to '%%matplotlib widget' (ipympl) and re-run "
                "this cell.",
                _backend_name(),
            )
        return []
    x1, y1 = corners[0]
    x2, y2 = corners[1]
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


# ---------------------------------------------------------------------------
# Session-aware prompt picker
# ---------------------------------------------------------------------------


class PromptPicker:
    """Interactive prompt picker bound to a :class:`VideoSession`.

    Provides one-step methods that let you visually click on a frame,
    automatically send the prompt to the session, and preview the result.

    Example::

        picker = PromptPicker(session)
        picker.add_points_interactive(frame_idx=0, obj_id=1)
        picker.add_box_interactive(frame_idx=0, obj_id=2)
    """

    def __init__(self, session: VideoSession) -> None:
        self._session = session
        self._frame_files = list_frame_files(session.video_dir)

    # ------- convenience --------

    def get_frame(self, frame_idx: int) -> npt.NDArray[np.uint8]:
        """Load frame *frame_idx* as an RGB numpy array."""
        if frame_idx < 0 or frame_idx >= len(self._frame_files):
            msg = f"frame_idx {frame_idx} out of range (0-{len(self._frame_files) - 1})"
            raise IndexError(msg)
        return load_image(self._frame_files[frame_idx])

    def preview(self, frame_idx: int = 0, **kwargs: Any) -> Any:
        """Show a frame with a coordinate grid (see :func:`preview_frame`)."""
        return preview_frame(self._session.video_dir, frame_idx, **kwargs)

    # ------- interactive prompts --------

    def pick_points(
        self,
        frame_idx: int = 0,
        n: int = -1,
        **kwargs: Any,
    ) -> tuple[list[list[float]], list[int]]:
        """Interactively pick points on a frame.

        Returns ``(points, labels)`` without adding them to the session.
        """
        img = self.get_frame(frame_idx)
        return pick_points_on_image(
            img,
            n=n,
            title=f"Frame {frame_idx} - pick points",
            **kwargs,
        )

    def pick_box(
        self,
        frame_idx: int = 0,
        **kwargs: Any,
    ) -> list[float]:
        """Interactively draw a box on a frame.

        Returns ``[x1, y1, x2, y2]`` without adding it to the session.
        """
        img = self.get_frame(frame_idx)
        return pick_box_on_image(
            img,
            title=f"Frame {frame_idx} - draw box",
            **kwargs,
        )

    def add_points_interactive(
        self,
        frame_idx: int,
        obj_id: int,
        n: int = -1,
        *,
        clear_old: bool = True,
        show_result: bool = True,
        **kwargs: Any,
    ) -> tuple[list[list[float]], list[int]]:
        """Pick points, add them to the session, and optionally preview.

        Args:
            frame_idx: Frame to annotate.
            obj_id: Object identifier.
            n: Number of points (``-1`` = click until Enter).
            clear_old: Replace existing prompts for this object.
            show_result: Show the resulting mask after adding.

        Returns:
            ``(points, labels)`` that were added.
        """
        from lazysammy.visualization import show_video_frame

        points, labels = self.pick_points(frame_idx, n=n, **kwargs)
        if not points:
            logger.warning("No points selected - nothing added.")
            return points, labels

        fm = self._session.add_points(
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
            clear_old=clear_old,
        )
        if show_result:
            import matplotlib.pyplot as plt

            show_video_frame(self.get_frame(frame_idx), fm, alpha=0.5)
            plt.show()
        return points, labels

    def add_box_interactive(
        self,
        frame_idx: int,
        obj_id: int,
        *,
        show_result: bool = True,
        **kwargs: Any,
    ) -> list[float]:
        """Draw a box, add it to the session, and optionally preview.

        Args:
            frame_idx: Frame to annotate.
            obj_id: Object identifier.
            show_result: Show the resulting mask after adding.

        Returns:
            ``[x1, y1, x2, y2]`` that was added.
        """
        from lazysammy.visualization import show_video_frame

        box = self.pick_box(frame_idx, **kwargs)
        if not box:
            logger.warning("No box drawn - nothing added.")
            return box

        fm = self._session.add_box(
            frame_idx=frame_idx,
            obj_id=obj_id,
            box=box,
        )
        if show_result:
            import matplotlib.pyplot as plt

            show_video_frame(self.get_frame(frame_idx), fm, alpha=0.5)
            plt.show()
        return box

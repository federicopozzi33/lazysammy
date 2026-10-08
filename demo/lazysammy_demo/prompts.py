"""Typed prompt model for the demo's image and video tabs.

Prompts are typed objects rather than ``dict``s with a ``"type"`` string, so
drawing, session application, and summary rendering are polymorphic instead of
re-branched on the type at every call site.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from lazysammy_demo.drawing import draw_label


@dataclass
class PointPrompt:
    """One object's point prompts on a single frame."""

    frame_idx: int
    obj_id: int
    points: list[list[float]]
    labels: list[int]

    def draw(self, img: np.ndarray, color: tuple[int, int, int]) -> None:
        """Draw the points onto *img* in place."""
        for pt, lab in zip(self.points, self.labels, strict=False):
            x, y = int(pt[0]), int(pt[1])
            point_color = color if lab == 1 else (128, 128, 128)
            cv2.circle(img, (x, y), 7, point_color, -1)
            cv2.circle(img, (x, y), 7, (255, 255, 255), 2)
            tag = f"obj{self.obj_id}+" if lab == 1 else f"obj{self.obj_id}-"
            draw_label(img, tag, x + 10, y - 6, color)

    def apply(self, session: Any) -> None:
        """Register this prompt on a video tracking session."""
        session.add_points(
            frame_idx=self.frame_idx,
            obj_id=self.obj_id,
            points=self.points,
            labels=self.labels,
        )

    def summary_html(self) -> str:
        """Render the prompt body for the prompt-list summary."""
        fg_tag = '<span style="color:green">fg</span>'
        bg_tag = '<span style="color:red">bg</span>'
        pts = ", ".join(
            f"({pt[0]:.0f},{pt[1]:.0f}) {fg_tag if lb == 1 else bg_tag}"
            for pt, lb in zip(self.points, self.labels, strict=False)
        )
        return f"Obj {self.obj_id} | Frame {self.frame_idx} | {pts}"


@dataclass
class BoxPrompt:
    """One object's bounding-box prompt on a single frame."""

    frame_idx: int
    obj_id: int
    box: list[float]

    def draw(self, img: np.ndarray, color: tuple[int, int, int]) -> None:
        """Draw the box onto *img* in place."""
        bx = self.box
        cv2.rectangle(img, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), color, 2)
        draw_label(img, f"obj{self.obj_id}", int(bx[0]) + 4, int(bx[1]) - 6, color, scale=0.5)

    def apply(self, session: Any) -> None:
        """Register this prompt on a video tracking session."""
        session.add_box(frame_idx=self.frame_idx, obj_id=self.obj_id, box=self.box)

    def summary_html(self) -> str:
        """Render the prompt body for the prompt-list summary."""
        b = self.box
        return (
            f"Obj {self.obj_id} | Frame {self.frame_idx} | "
            f"box ({b[0]:.0f},{b[1]:.0f}) to ({b[2]:.0f},{b[3]:.0f})"
        )


@dataclass
class TextPrompt:
    """A SAM 3 text concept prompt for a video frame.

    Unlike point/box prompts, a text prompt is not tied to one object id: SAM 3
    detects every matching instance and assigns each its own id.
    """

    frame_idx: int
    text: str

    def draw(self, img: np.ndarray, color: tuple[int, int, int]) -> None:
        """Draw a label for the concept in the top-left corner."""
        draw_label(img, f'"{self.text}"', 10, 24, color, scale=0.6)

    def apply(self, session: Any) -> None:
        """Register this concept prompt on a SAM 3 video session."""
        session.add_text(frame_idx=self.frame_idx, text=self.text)

    def summary_html(self) -> str:
        """Render the prompt body for the prompt-list summary."""
        return f'Concept "{self.text}" | Frame {self.frame_idx}'


VideoPrompt = PointPrompt | BoxPrompt | TextPrompt


@dataclass
class ImagePrompts:
    """Accumulated image-tab prompts for a single object on frame 0.

    ``point`` holds the committed point clicks; ``box_corners`` holds the
    pending box corners (0, 1, or 2) before they become a :class:`BoxPrompt`.
    """

    point: PointPrompt | None = None
    box_corners: list[list[float]] = field(default_factory=list)

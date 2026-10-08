"""Drawing helpers for the Gradio demo.

Pure OpenCV/numpy helpers with no Gradio or model dependencies, so they can be
unit-tested and reused by every tab.
"""

from __future__ import annotations

import cv2
import numpy as np

from lazysammy import ConceptPrediction

_COLORS = [
    (255, 0, 0),
    (0, 200, 0),
    (0, 80, 255),
    (255, 200, 0),
    (255, 0, 200),
    (0, 200, 200),
    (180, 0, 0),
    (0, 160, 0),
]


def color_for_obj(obj_id: int) -> tuple[int, int, int]:
    """Return a stable BGR colour for *obj_id*."""
    return _COLORS[obj_id % len(_COLORS)]


def draw_label(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    *,
    scale: float = 0.45,
) -> None:
    """Draw *text* with a white outline so it stays legible on any frame."""
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_points_on_image(
    img: np.ndarray,
    points: list[list[float]],
    *,
    radius: int = 8,
) -> np.ndarray:
    """Draw point prompts with fg/bg colouring and +/- labels."""
    out = img.copy()
    for p in points:
        x, y, label = int(p[0]), int(p[1]), int(p[2])
        color = (0, 220, 0) if label == 1 else (220, 50, 50)
        cv2.circle(out, (x, y), radius, color, -1)
        cv2.circle(out, (x, y), radius, (255, 255, 255), 2)
        draw_label(out, "+" if label == 1 else "-", x + radius + 2, y + 4, color, scale=0.55)
    return out


def draw_box_corners_on_image(
    img: np.ndarray,
    corners: list[list[float]],
) -> np.ndarray:
    """Draw partial / complete box prompt."""
    out = img.copy()
    if len(corners) >= 1:
        x1, y1 = int(corners[0][0]), int(corners[0][1])
        cv2.drawMarker(out, (x1, y1), (0, 200, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(
            out,
            "corner 1",
            (x1 + 10, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )
    if len(corners) >= 2:
        x2, y2 = int(corners[1][0]), int(corners[1][1])
        bx1, by1 = min(x1, x2), min(y1, y2)
        bx2, by2 = max(x1, x2), max(y1, y2)
        cv2.rectangle(out, (bx1, by1), (bx2, by2), (0, 200, 255), 2)
    return out


def draw_concept_boxes(image: np.ndarray, prediction: ConceptPrediction) -> np.ndarray:
    """Draw each detected instance's bounding box and score.

    Boxes are aligned with masks by index; a prediction with masks but no boxes
    (the interactive path) simply draws nothing.
    """
    out = image.copy()
    for i, mask in enumerate(prediction.masks):
        if i >= len(prediction.boxes):
            break
        color = color_for_obj(i)
        x1, y1, x2, y2 = (int(v) for v in prediction.boxes[i])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        draw_label(out, f"{mask.score:.2f}", x1 + 4, y1 - 6, color, scale=0.5)
    return out

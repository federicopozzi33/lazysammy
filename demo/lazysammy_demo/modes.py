"""Click-mode constants and helpers for the demo's image and video tabs."""

from __future__ import annotations

MODE_FOREGROUND = "Add foreground point"
MODE_BACKGROUND = "Add background point"
MODE_BOX = "Draw a box (2 clicks)"

MODE_CHOICES = [MODE_FOREGROUND, MODE_BACKGROUND, MODE_BOX]


def is_box_mode(mode: str) -> bool:
    """Return True when *mode* is the two-click box mode."""
    return mode == MODE_BOX


def label_for_mode(mode: str) -> int:
    """Return the point label (1 foreground, 0 background) for *mode*."""
    return 0 if mode == MODE_BACKGROUND else 1


def render_mode_instruction(mode: str) -> str:
    """Return the HTML instruction banner for *mode*."""
    if mode == MODE_BACKGROUND:
        return (
            '<div class="click-instruction click-bg">'
            "Click on the image to add <b>background</b> points "
            "(areas to exclude)</div>"
        )
    if mode == MODE_BOX:
        return (
            '<div class="click-instruction click-box">'
            "Click <b>two corners</b> on the image to define a "
            "bounding box</div>"
        )
    return (
        '<div class="click-instruction click-fg">'
        "Click on the image to add <b>foreground</b> points "
        "(objects to include)</div>"
    )

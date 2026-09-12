"""Guard tests for authoring hygiene.

These keep a few easy-to-regress, purely textual conventions enforced:
ASCII-only sources (no emoji or typographic dashes in code, docs, or UI
strings) and a README whose opening actually leads with the motivation.

They are deliberately cheap and dependency-free so they run on every test
invocation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Characters that render as "typography" rather than plain typing, plus the
# emoji that tend to creep into UI copy.
FORBIDDEN_CHARS = {
    "\u2014": "em dash",
    "\u2013": "en dash",
    "\u2026": "ellipsis glyph",
    "\u00b7": "middle dot",
    "\u2022": "bullet glyph",
    "\u2192": "rightwards arrow",
    "\u2264": "less-than-or-equal glyph",
    "\u2265": "greater-than-or-equal glyph",
    "\u201c": "curly double quote (open)",
    "\u201d": "curly double quote (close)",
    "\u2018": "curly single quote (open)",
    "\u2019": "curly single quote (close)",
    "\u2795": "heavy plus emoji",
    "\u2796": "heavy minus emoji",
    "\u274c": "cross mark emoji",
    "\u2714": "check mark glyph",
}

# Text files that should stay plain ASCII. The notebook is loaded as JSON and
# checked separately, and always uses explicit escapes for non-ASCII.
TEXT_FILES = [
    "README.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "pyproject.toml",
]


def _python_files() -> list[Path]:
    paths: list[Path] = []
    for folder in ("src", "tests", "demo", "scripts"):
        paths.extend(sorted((REPO_ROOT / folder).rglob("*.py")))
    return paths


def _offending(text: str) -> list[str]:
    found = []
    for char, name in FORBIDDEN_CHARS.items():
        if char in text:
            found.append(f"{name} (U+{ord(char):04X})")
    return found


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_python_sources_avoid_typography_tells(path: Path) -> None:
    """Python sources should use plain ASCII punctuation, not typography."""
    text = path.read_text(encoding="utf-8")
    problems = _offending(text)
    assert not problems, f"{path.relative_to(REPO_ROOT)} contains {problems}"


@pytest.mark.parametrize("relative", TEXT_FILES)
def test_text_files_avoid_typography_tells(relative: str) -> None:
    """Markdown and config files should use plain ASCII punctuation."""
    path = REPO_ROOT / relative
    problems = _offending(path.read_text(encoding="utf-8"))
    assert not problems, f"{relative} contains {problems}"


def test_notebook_sources_avoid_typography_tells() -> None:
    """The notebook's cell text should be plain ASCII too."""
    nb_path = REPO_ROOT / "examples" / "video_tracking.ipynb"
    if not nb_path.exists():
        pytest.skip("Example notebook is not present.")

    notebook = json.loads(nb_path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"], start=1):
        source = "".join(cell.get("source", []))
        problems = _offending(source)
        assert not problems, f"notebook cell {index} contains {problems}"


def test_readme_leads_with_motivation() -> None:
    """The README opening should still lead with the motivation section."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    motivation_at = readme.find("## Motivation")

    assert motivation_at != -1, "README has no Motivation section"
    # It should appear before the installation and quick start sections.
    for later in ("## Installation", "## Quick start"):
        position = readme.find(later)
        assert position != -1, f"README is missing {later}"
        assert motivation_at < position, f"Motivation should precede {later}"

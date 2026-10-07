# Contributing to lazysammy

Thanks for your interest in improving `lazysammy`. This project is a thin,
typed convenience layer over [SAM 2](https://github.com/facebookresearch/sam2);
contributions that keep it thin are especially welcome.

## Development setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/federicopozzi33/easier-sam2.git
cd easier-sam2
uv sync
```

`uv sync` installs the `dev` dependency group (pytest, ruff, mypy) by default.
For working on the notebook or demo, include those extras too:

```bash
uv sync --extra notebook --extra demo
```

## Before you open a pull request

Run these before opening a pull request:

```bash
uv run ruff check src/ tests/ demo/
uv run ruff format --check src/ tests/ demo/
uv run mypy src/
uv run pytest
```

All four must pass. If you changed behavior, please also run the end-to-end
suite, which loads real weights and is deselected by default:

```bash
uv run pytest -m integration
```

Tips for that run:

- It defaults to the `small` model. Use a smaller/faster one with
  `LAZYSAMMY_TEST_MODEL=tiny`.
- `HF_HUB_OFFLINE=1` makes it fail fast instead of downloading weights, so use
  it to confirm results come from already-cached assets.

## Guidelines

- **Keep the abstraction thin.** Prefer delegating to SAM 2 over
  reimplementing model behavior.
- **Preserve type coverage.** `src/` is checked with `mypy --strict`; new public
  APIs need annotations and Google-style docstrings.
- **Validate at the boundary.** Public entry points should reject malformed
  prompts with a clear `ValueError`/`IndexError` rather than failing deep inside
  the model. Helpers live in `src/lazysammy/validation.py`.
- **Add tests with the fix.** Bug fixes should come with a regression test;
  see `tests/test_regressions.py` for the pattern.
- **Don't add heavyweight test dependencies.** Unit tests must not download
  weights or require a GPU: mark anything that does with
  `@pytest.mark.integration`.
- **Public API changes belong in the README and `CHANGELOG.md`.**

## Reporting bugs

Please include:

- `lazysammy` version, Python version, and `torch` version
- your device (CUDA / MPS / CPU)
- the model size in use
- a minimal reproduction, and the full traceback

## License

By contributing, you agree that your contributions are licensed under the
Apache 2.0 License, the same license that covers this project.

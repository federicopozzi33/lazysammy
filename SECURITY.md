# Security Policy

## Scope

`lazysammy` is a thin Python wrapper that loads and runs SAM 2 model weights
locally. It does not run a network service by default and does not process
untrusted input on any server you did not start yourself.

The most relevant risks are therefore:

- **Model weights**: downloaded from the HuggingFace Hub on first use, or loaded
  from a local checkpoint path you supply. Only load checkpoints you trust.
- **The demo app**: `demo/app.py` binds to `127.0.0.1` by default. If you
  deliberately expose it (`LAZYSAMMY_DEMO_HOST=0.0.0.0`, or a tunnel/`--share`),
  treat it as an untrusted-input surface and put authentication in front of it.
- **Untrusted media**: decoding malformed images or videos is delegated to
  OpenCV and Pillow; keep those dependencies up to date.

## Supported versions

This project is at `0.1.x`. Security fixes are applied to the latest release
only.

## Reporting a vulnerability

Please **do not** open a public issue for a security problem.

Instead, use GitHub's private vulnerability reporting on the repository
("Security" → "Report a vulnerability"), or contact the maintainer directly
through their GitHub profile. Include:

- a description of the issue and its impact
- steps to reproduce
- affected versions
- any suggested fix

You can expect an acknowledgement within a few days. Please allow time for a
fix and release before disclosing publicly.

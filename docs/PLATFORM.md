# Platform validation

QuotientGuard's wheel is pure Python (`py3-none-any`) and the core has no runtime dependencies.

## Local release validation

The v1.0.0 wheel was built and clean-installed under WSL/Linux with Python 3.14.4. The installed wheel passed:

- `quotientguard --version`;
- the paper demo;
- the full 69-test suite copied outside the source tree;
- the 5-case public certification suite;
- exact compatible fixture smoke.

The development PC did not expose a native Windows `python` or `py` command in PowerShell, so a local native-Windows venv smoke could not be run.

## Public release gate

Repository CI covers:

- `ubuntu-latest`;
- `windows-latest`;
- `macos-latest`;

across Python 3.10, 3.11, 3.12, 3.13, and 3.14.

The `v1.0.0` tag/release must not be created until that matrix is green on the public repository.

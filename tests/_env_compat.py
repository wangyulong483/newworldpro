"""沙箱环境兼容垫片：把 ``tempfile.mkdtemp`` 换成普通 mkdir，
避免 Windows ACL 受限沙箱中临时目录不可写导致的测试假错误（见 README §9）。

Test-environment compatibility shim (Windows + DSH file sandbox).

Why this exists
---------------
On Windows, :func:`tempfile.mkdtemp` creates the directory with a security
descriptor that the DSH sandboxed process **cannot write into afterwards**:
every test that does ``with tempfile.TemporaryDirectory() as tmp: (tmp/"x").write(...)``
fails with ``PermissionError: [Errno 13]`` and even the cleanup step fails with
``[WinError 5]``.  That produced 6 false test errors in the P0 baseline even
though the test bodies themselves are correct.

``patch_tempdir()`` replaces ``tempfile.mkdtemp`` with a plain
``Path.mkdir``-based implementation (inheriting the normal workspace ACL) and
only does it on Windows.  It changes **no test semantics**: the directory is
still unique, temporary and removed afterwards.

Usage (top of a test module, before the tempfile-using tests run)::

    from tests._env_compat import patch_tempdir
    patch_tempdir()"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

_APPLIED = False


def patch_tempdir() -> None:
    """Make ``tempfile.mkdtemp`` writable under the Windows file sandbox."""
    global _APPLIED
    if _APPLIED or os.name != "nt":
        return
    _APPLIED = True

    def _mkdtemp(suffix: str | None = None, prefix: str | None = None, dir: str | None = None) -> str:
        base = Path(dir) if dir is not None else Path(tempfile.gettempdir())
        base.mkdir(parents=True, exist_ok=True)
        name = f"{prefix or 'tmp'}{uuid.uuid4().hex[:10]}{suffix or ''}"
        target = base / name
        target.mkdir()
        return str(target)

    tempfile.mkdtemp = _mkdtemp  # type: ignore[assignment]


def claim_port() -> int:  # pragma: no cover - convenience for future tests
    """Return an unused localhost TCP port (helper used by GUI smoke tests)."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])

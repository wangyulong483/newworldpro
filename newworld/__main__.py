"""支持 ``python -m newworld`` 的入口（直接转到 CLI）。"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())

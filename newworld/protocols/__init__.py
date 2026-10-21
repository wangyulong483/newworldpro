"""实验协议包（P5 registry 注册）。

导入本包会触发所有协议的 @register_protocol 注册；各协议模块由
experiments/run.py 或 newworld.experiment.run_experiment 通过名字调用。

P5 起协议是"可插拔文件"：新增协议 = 新模块 + @register_protocol，
本文件只在最后做一次 import 汇总。集成阶段会改为显式 import；当前
用容错 loader 让各协议可以独立开发/测试（缺失模块被跳过，不掩盖
模块内部的 ImportError）。
"""

from __future__ import annotations

import importlib

from . import common  # noqa: F401  (共用工具)

_PROTOCOL_MODULES = (
    "p1_learning",
    "p3_social",
    "p4_evolution",
    "p5_scaling",
    "p6_communication",
    "p6_memory",
)

for _module_name in _PROTOCOL_MODULES:
    try:
        importlib.import_module(f"{__name__}.{_module_name}")
    except ModuleNotFoundError as _exc:  # 仅容忍"该协议文件尚未落地"
        if _exc.name != f"{__name__}.{_module_name}":
            raise

__all__ = ["common"]

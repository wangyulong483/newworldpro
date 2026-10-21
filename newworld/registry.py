"""组件注册表（P5 工程化基石）。

YOLO 式"注册制"的最小实现：策略、实验协议、社会机制等可插拔组件
不再由调用方写死 if/else，而是各自用装饰器登记到命名注册表，再由
YAML 配置里的名字查表构造。这样新增一个策略/协议 = 新文件 + 一个
装饰器，不需要改 CLI、实验脚本或实验室后端。

设计约定
--------
* 注册表按"种类"分开（policy / protocol / social），避免同名冲突；
* 默认禁止覆盖注册（拼写错误或重复定义应在 import 时立刻炸掉），
  测试可用 overwrite=True；
* 工厂可以是类、函数或任何 callable；create() 原样转发参数；
* 注册顺序稳定（dict 保序），names() 输出可直接写进文档；
* 本模块只依赖标准库，不 import 项目内模块（避免循环依赖）。
"""

from __future__ import annotations

from typing import Any, Callable, Iterable


class RegistryError(LookupError):
    """注册表查询/注册错误（缺名、重名、类型不符）。"""


class Registry:
    """命名工厂注册表。

    典型用法::

        POLICIES = Registry("policy")

        @POLICIES.register("neural")
        class NeuralPolicy:
            ...

        policy = POLICIES.create("neural", obs_dim=9)
    """

    def __init__(self, kind: str) -> None:
        if not kind:
            raise ValueError("注册表种类不能为空")
        self.kind = str(kind)
        self._factories: dict[str, Callable[..., Any]] = {}
        self._notes: dict[str, str] = {}

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------
    def register(
        self,
        name: str,
        factory: Callable[..., Any] | None = None,
        *,
        overwrite: bool = False,
        note: str = "",
    ):
        """注册（可作装饰器，也可直接调用）。"""
        name = self._check_name(name)

        def _install(target: Callable[..., Any]):
            if name in self._factories and not overwrite:
                raise RegistryError(
                    f"{self.kind} 注册表里已经有 {name!r}，"
                    "如确需替换请传 overwrite=True"
                )
            if not callable(target):
                raise RegistryError(f"{self.kind}:{name} 的工厂不可调用")
            self._factories[name] = target
            self._notes[name] = note or (
                getattr(target, "__doc__", "") or ""
            ).strip().split("\n")[0]
            return target

        if factory is None:
            return _install
        return _install(factory)

    # ------------------------------------------------------------------
    # 查询/构造
    # ------------------------------------------------------------------
    def get(self, name: str) -> Callable[..., Any]:
        """取工厂；未注册时抛 RegistryError（附可用名字）。"""
        try:
            return self._factories[str(name)]
        except KeyError:
            raise RegistryError(
                f"未知 {self.kind}: {name!r}；"
                f"可用: {', '.join(self.names()) or '（空）'}"
            ) from None

    def create(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """查表并构造：factory(*args, **kwargs)。"""
        return self.get(name)(*args, **kwargs)

    def factory(self, name: str) -> Callable[..., Any]:
        """get 的别名（语义更清楚时用）。"""
        return self.get(name)

    # ------------------------------------------------------------------
    # 枚举/诊断
    # ------------------------------------------------------------------
    def names(self) -> list[str]:
        """全部已注册名字（注册顺序）。"""
        return list(self._factories)

    def notes(self) -> dict[str, str]:
        """名字 -> 一行说明（文档/状态栏用）。"""
        return dict(self._notes)

    def describe(self) -> dict[str, dict[str, str]]:
        return {name: {"note": self._notes.get(name, "")} for name in self.names()}

    def __contains__(self, name: object) -> bool:
        return str(name) in self._factories

    def __len__(self) -> int:
        return len(self._factories)

    def __iter__(self) -> Iterable[str]:
        return iter(self.names())

    # ------------------------------------------------------------------
    def clear(self) -> None:
        """清空（仅测试/临时注册用；业务代码不要调用）。"""
        self._factories.clear()
        self._notes.clear()

    def _check_name(self, name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            raise RegistryError("注册名必须是非空字符串")
        return name.strip()


# ---------------------------------------------------------------------------
# 全局注册表（各模块 import 后自行登记）
# ---------------------------------------------------------------------------
#: 群伙策略：rule / random / neural / neural_comm / ...
POLICIES = Registry("policy")
#: 实验协议：learning_compare / social_ablation / scaling_sweep / ...
PROTOCOLS = Registry("protocol")
#: 社会/认知机制：social / communication / memory / ...
MECHANISMS = Registry("mechanism")


def register_policy(name: str, *, note: str = "") -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """策略注册装饰器。"""
    return POLICIES.register(name, note=note)


def register_protocol(name: str, *, note: str = "") -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """实验协议注册装饰器。"""
    return PROTOCOLS.register(name, note=note)


def register_mechanism(name: str, *, note: str = "") -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """机制注册装饰器。"""
    return MECHANISMS.register(name, note=note)


__all__ = [
    "Registry",
    "RegistryError",
    "POLICIES",
    "PROTOCOLS",
    "MECHANISMS",
    "register_policy",
    "register_protocol",
    "register_mechanism",
]

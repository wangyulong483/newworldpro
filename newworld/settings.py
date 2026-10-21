"""YAML 配置层（P5 工程化基石）：实验的一切旋钮都进配置文件。

设计目标（YOLO / HF Trainer 的工程模式）
----------------------------------------
1. **单一事实来源**：世界尺寸、群伙参数、策略种类、社会/认知机制、
   运行预算、输出目录全部由 YAML 描述；代码里不再散落魔法数字。
2. **严格模式**：未知字段直接报错（拼写错误不会静默生效），
   数值范围在构造时校验；配置可 dump 回 YAML 复现。
3. **继承 + 覆盖**：配置文件可用 "_base: base.yaml" 继承（可多层），
   命令行用 "--set world.width=128 run.seeds=[0,1,2]" 做点号覆盖。
4. **可做实验指纹**：config_hash 是规范化 JSON 的 sha256，模型
   zoo / 实验 summary 都记录它，防止"跑的是哪份配置"说不清。

配置文件结构::

    name: p1_scarce
    protocol: learning_compare        # 注册表里的协议名
    world:   {width: 256, height: 160, seed: 0, food_regrow_per_s: 0.0005}
    swarm:   {initial_population: 120, capacity_hard: 3000}
    policy:  {kind: neural, hidden: 32}
    social:  {enable_barter: true, ...}
    comm:    {enabled: false}
    memory:  {enabled: false}
    run:     {seconds: 60, dt: 0.05, seeds: [0, 1, 2]}
    protocol_params: {...}            # 协议私有参数

本模块不 import 仿真模块（避免循环依赖）；需要把配置转成
MapConfig/SwarmConfig 时在方法里延迟 import。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ConfigError(ValueError):
    """配置结构/取值错误（带字段路径，便于定位）。"""


# ---------------------------------------------------------------------------
# 严格反序列化
# ---------------------------------------------------------------------------
def _is_union(annotation: Any) -> bool:
    return get_origin(annotation) is not None and str(get_origin(annotation)) in (
        "typing.Union", "<class 'types.UnionType'>", "types.UnionType"
    )


def _coerce(annotation: Any, value: Any, path: str) -> Any:
    """按类型注解做保守转换：不猜、不丢信息，类型明显不符就报错。"""
    if annotation in (Any, object):
        return value
    # Optional[X] / X | None
    if _is_union(annotation):
        options = [arg for arg in get_args(annotation) if arg is not type(None)]
        if value is None and type(None) in get_args(annotation):
            return None
        if len(options) == 1:
            return _coerce(options[0], value, path)
        for option in options:
            try:
                return _coerce(option, value, path)
            except ConfigError:
                continue
        raise ConfigError(f"{path}: 无法匹配联合类型 {annotation}")
    origin = get_origin(annotation)
    if origin in (list, tuple):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: 需要列表，得到 {type(value).__name__}")
        args = get_args(annotation)
        item_type = args[0] if args else Any
        converted = [_coerce(item_type, item, f"{path}[{i}]") for i, item in enumerate(value)]
        return tuple(converted) if origin is tuple else converted
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: 需要映射，得到 {type(value).__name__}")
        args = get_args(annotation)
        key_type, val_type = (args + (Any, Any))[:2] if len(args) < 2 else args[:2]
        return {
            _coerce(key_type, key, f"{path}.<key>"): _coerce(val_type, val, f"{path}.{key}")
            for key, val in value.items()
        }
    if dataclasses.is_dataclass(annotation):
        return _from_dict(annotation, value, path)
    if annotation is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{path}: 需要布尔值，得到 {value!r}")
    if annotation is int:
        if isinstance(value, bool):
            raise ConfigError(f"{path}: 需要整数，得到布尔值")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise ConfigError(f"{path}: 需要整数，得到 {value!r}")
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: 需要数值，得到 {value!r}")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: 需要字符串，得到 {value!r}")
        return value
    return value


def _from_dict(cls: type, data: Any, path: str = "") -> Any:
    """严格构造 dataclass：未知字段报错，缺失字段用默认值。"""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path or cls.__name__}: 需要映射，得到 {type(data).__name__}")
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(
            f"{path or cls.__name__}: 未知字段 {unknown}；"
            f"可用: {sorted(known)}"
        )
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        key = f"{path}.{f.name}" if path else f.name
        kwargs[f.name] = _coerce(hints.get(f.name, Any), data[f.name], key)
    try:
        return cls(**kwargs)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{path or cls.__name__}: {exc}") from exc


def _to_plain(value: Any) -> Any:
    """dataclass/tuple -> 纯 JSON/YAML 类型。"""
    if dataclasses.is_dataclass(value):
        return {f.name: _to_plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, tuple):
        return [_to_plain(item) for item in value]
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_plain(val) for key, val in value.items()}
    if isinstance(value, Path):
        return str(value)
    return value


# ---------------------------------------------------------------------------
# 配置数据类
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WorldSettings:
    """地图生成/生态参数（映射到 MapConfig）。"""

    width: int = 256
    height: int = 160
    seed: int = 0
    ocean_margin: int = 6
    noise_scale: int = 64
    sea_level: float = 0.42
    mountain_level: float = 0.74
    snow_level: float = 0.86
    food_regrow_per_s: float = 0.012
    forest_spread_per_s: float = 0.0004

    def __post_init__(self) -> None:
        if self.width < 32 or self.height < 32:
            raise ValueError("world 尺寸至少 32x32")
        if not 0 <= self.ocean_margin < min(self.width, self.height) // 2:
            raise ValueError("ocean_margin 越界")
        if not 0.0 <= self.sea_level < self.mountain_level < self.snow_level <= 1.0:
            raise ValueError("阈值必须满足 0 <= sea < mountain < snow <= 1")
        if self.food_regrow_per_s < 0 or self.forest_spread_per_s < 0:
            raise ValueError("再生/蔓延速率不能为负")

    def to_map_config(self):
        from .map_engine import MapConfig

        return MapConfig(
            width=self.width,
            height=self.height,
            seed=self.seed,
            ocean_margin=self.ocean_margin,
            noise_scale=self.noise_scale,
            sea_level=self.sea_level,
            mountain_level=self.mountain_level,
            snow_level=self.snow_level,
            food_regrow_per_s=self.food_regrow_per_s,
            forest_spread_per_s=self.forest_spread_per_s,
        )


@dataclass(frozen=True)
class SwarmSettings:
    """群伙人口/代谢/移动参数（映射到 SwarmConfig）。"""

    initial_population: int = 120
    capacity_hard: int = 3000
    metabolism_per_s: float = 0.010
    bite_size: float = 0.35
    mature_age_s: float = 12.0
    breed_cooldown_s: float = 14.0
    breed_food_threshold: float = 0.12
    breed_hunger_threshold: float = 0.55
    breed_probability_per_s: float = 0.10
    old_age_s: float = 90.0
    old_age_hazard_per_s: float = 0.05
    move_speed: float = 3.0
    wander: float = 0.55
    forage_hunger: float = 0.95

    def __post_init__(self) -> None:
        if self.initial_population < 1:
            raise ValueError("initial_population >= 1")
        if self.capacity_hard < self.initial_population:
            raise ValueError("capacity_hard 不能小于 initial_population")
        for name in ("metabolism_per_s", "bite_size", "move_speed"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} 不能为负")
        if not 0.0 <= self.wander <= 1.0:
            raise ValueError("wander 必须在 [0,1]")

    def to_swarm_config(self):
        from .swarm import SwarmConfig

        return SwarmConfig(
            initial_population=self.initial_population,
            capacity_hard=self.capacity_hard,
            metabolism_per_s=self.metabolism_per_s,
            bite_size=self.bite_size,
            mature_age_s=self.mature_age_s,
            breed_cooldown_s=self.breed_cooldown_s,
            breed_food_threshold=self.breed_food_threshold,
            breed_hunger_threshold=self.breed_hunger_threshold,
            breed_probability_per_s=self.breed_probability_per_s,
            old_age_s=self.old_age_s,
            old_age_hazard_per_s=self.old_age_hazard_per_s,
            move_speed=self.move_speed,
            wander=self.wander,
            forage_hunger=self.forage_hunger,
        )


@dataclass(frozen=True)
class PolicySettings:
    """策略选择与超参；kind 必须在注册表的 POLICIES 中注册。"""

    kind: str = "rule"
    seed: int = 0
    hidden: int = 32
    checkpoint: str | None = None
    lr: float = 0.02
    gamma: float = 0.96
    batch_size: int = 512
    epochs_per_episode: int = 2
    normalize_advantage: bool = True
    grad_clip: float = 0.5

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("policy.kind 不能为空")
        if self.hidden < 1:
            raise ValueError("policy.hidden >= 1")
        if self.lr <= 0:
            raise ValueError("policy.lr > 0")


@dataclass(frozen=True)
class SocialSettings:
    """P3 社会机制开关（映射到 SocialConfig）。"""

    enabled: bool = True
    enable_barter: bool = True
    barter_gap: float = 0.30
    barter_transfer: float = 0.15
    barter_radius: float = 2.5
    barter_interval_s: float = 1.0
    enable_teaching: bool = True
    teach_radius: float = 4.0
    teach_chance_per_s: float = 0.06
    teach_interval_s: float = 2.0
    enable_inheritance: bool = True
    inherit_radius: float = 12.0
    initial_skill_rate: float = 0.08

    def to_social_config(self):
        from .social_mech import SocialConfig

        return SocialConfig(
            enable_barter=self.enable_barter,
            barter_gap=self.barter_gap,
            barter_transfer=self.barter_transfer,
            barter_radius=self.barter_radius,
            barter_interval_s=self.barter_interval_s,
            enable_teaching=self.enable_teaching,
            teach_radius=self.teach_radius,
            teach_chance_per_s=self.teach_chance_per_s,
            teach_interval_s=self.teach_interval_s,
            enable_inheritance=self.enable_inheritance,
            inherit_radius=self.inherit_radius,
            initial_skill_rate=self.initial_skill_rate,
        )


@dataclass(frozen=True)
class CommSettings:
    """P6 通信通道参数（语义不硬编码：符号数/半径/成本）。"""

    enabled: bool = False
    n_symbols: int = 4
    radius: float = 6.0
    message_cost: float = 0.0
    seed: int = 0
    train: bool = True
    lr: float = 0.05
    batch_size: int = 256
    epochs: int = 1

    def __post_init__(self) -> None:
        if self.enabled and self.n_symbols < 2:
            raise ValueError("comm.n_symbols >= 2")
        if self.radius <= 0:
            raise ValueError("comm.radius > 0")


@dataclass(frozen=True)
class MemorySettings:
    """P6 生成式记忆/反思参数（去 LLM：新近×重要×相关 检索 + 周期反思）。"""

    enabled: bool = False
    capacity: int = 24
    recency_halflife_s: float = 30.0
    weight_recency: float = 1.0
    weight_importance: float = 1.0
    weight_relevance: float = 1.0
    top_k: int = 3
    reflection_every_s: float = 20.0
    reflection_min_events: int = 6
    seed: int = 0

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("memory.capacity >= 1")
        if self.top_k < 1 or self.top_k > self.capacity:
            raise ValueError("memory.top_k 必须在 [1, capacity]")
        if self.recency_halflife_s <= 0:
            raise ValueError("memory.recency_halflife_s > 0")
        if self.reflection_every_s < 0:
            raise ValueError("memory.reflection_every_s >= 0")


@dataclass(frozen=True)
class RunSettings:
    """运行预算与产物路径。"""

    seconds: float = 120.0
    dt: float = 0.05
    daylight: float = 0.8
    seeds: tuple[int, ...] = (0,)
    eval_seconds: float = 120.0
    output_dir: str = "experiments/results"
    save_models: bool = True

    def __post_init__(self) -> None:
        if self.seconds <= 0 or self.dt <= 0:
            raise ValueError("run.seconds/dt 必须 > 0")
        if not 0.0 <= self.daylight <= 1.0:
            raise ValueError("run.daylight 必须在 [0,1]")
        if not self.seeds:
            raise ValueError("run.seeds 不能为空")
        if min(self.seeds) < 0:
            raise ValueError("run.seeds 不能为负")

    @property
    def steps(self) -> int:
        return max(1, int(round(self.seconds / self.dt)))


@dataclass(frozen=True)
class ExperimentSettings:
    """一份完整实验配置（YAML 的根对象）。"""

    name: str = "experiment"
    protocol: str = "learning_compare"
    description: str = ""
    world: WorldSettings = field(default_factory=WorldSettings)
    swarm: SwarmSettings = field(default_factory=SwarmSettings)
    policy: PolicySettings = field(default_factory=PolicySettings)
    social: SocialSettings = field(default_factory=SocialSettings)
    comm: CommSettings = field(default_factory=CommSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    run: RunSettings = field(default_factory=RunSettings)
    protocol_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name 不能为空")
        if not self.protocol:
            raise ValueError("protocol 不能为空")

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)

    def hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# YAML 读取 / 继承 / 覆盖
# ---------------------------------------------------------------------------
def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ConfigError(f"配置继承出现环: {chain}")
    if not path.exists():
        raise ConfigError(f"配置文件不存在: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: YAML 解析失败: {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: 顶层必须是映射")
    data = dict(data)
    bases = data.pop("_base", None)
    if bases is None:
        return data
    if isinstance(bases, str):
        bases = [bases]
    if not isinstance(bases, (list, tuple)):
        raise ConfigError(f"{path}: _base 必须是字符串或列表")
    merged: dict[str, Any] = {}
    for base_name in bases:
        base_path = (path.parent / str(base_name)).resolve()
        merged = _deep_merge(merged, _load_raw(base_path, (*stack, path)))
    return _deep_merge(merged, data)


def load_config(path: str | Path) -> ExperimentSettings:
    """从 YAML 载入配置（支持 _base 继承 + 严格字段校验）。"""
    raw = _load_raw(Path(path))
    return _from_dict(ExperimentSettings, raw)


def apply_overrides(
    settings: ExperimentSettings, overrides: list[str] | tuple[str, ...]
) -> ExperimentSettings:
    """应用点号覆盖，例如 ["world.width=128", "run.seeds=[0,1]"]。

    值用 YAML 解析（因此 true/false、数字、列表都能写），键必须
    已存在于配置 schema 中（否则报 ConfigError，防拼写错误静默忽略）。
    """
    if not overrides:
        return settings
    raw = settings.to_dict()
    for item in overrides:
        text = str(item).strip()
        if not text:
            continue
        if "=" not in text:
            raise ConfigError(f"覆盖项需要 key=value 形式: {text!r}")
        key, _, value_text = text.partition("=")
        key = key.strip()
        if not key:
            raise ConfigError(f"覆盖项缺少键: {text!r}")
        try:
            value = yaml.safe_load(value_text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"覆盖项 {key} 的值无法解析: {exc}") from exc
        _set_dotted(raw, key.split("."), value)
    return _from_dict(ExperimentSettings, raw)


def _set_dotted(tree: dict[str, Any], parts: list[str], value: Any) -> None:
    node: Any = tree
    for index, part in enumerate(parts):
        last = index == len(parts) - 1
        if not isinstance(node, dict):
            raise ConfigError(f"覆盖路径 {'.'.join(parts)} 经过非映射节点")
        if last:
            node[part] = value
            return
        if part not in node:
            raise ConfigError(f"覆盖路径不存在: {'.'.join(parts)}（{part} 无此字段）")
        node = node[part]


def save_config(settings: ExperimentSettings, path: str | Path) -> Path:
    """把配置写回 YAML（实验产物随 summary 一起归档）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(settings.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return target


def config_hash(settings: ExperimentSettings) -> str:
    """配置指纹（模型 zoo / summary 记录用）。"""
    return settings.hash()


__all__ = [
    "ConfigError",
    "WorldSettings",
    "SwarmSettings",
    "PolicySettings",
    "SocialSettings",
    "CommSettings",
    "MemorySettings",
    "RunSettings",
    "ExperimentSettings",
    "load_config",
    "apply_overrides",
    "save_config",
    "config_hash",
    "PROJECT_ROOT",
]

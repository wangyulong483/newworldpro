"""模型 zoo（P5 工程化）：统一 checkpoint 产出格式与元数据。

YOLO/Ultralytics 的 model zoo 思路：训练产物不是一个裸 npz，而是一个
自描述目录::

    assets/zoo/<name>/
        policy.npz      # 权重（NeuralSwarmPolicy.save 格式）
        card.json       # 架构/维度/seed/创建时间/git commit/config hash
                               # 权重 sha256（可校验未被手改）
        metrics.json    # 该 checkpoint 的评估指标（协议写入）
        config.yaml     # 训练/评估所用配置快照（可选）

这样"这个模型是什么、怎么来的、在什么配置下训的、指标多少"都能在
一个目录里回答（回应用户"不要连训练的模型是什么都不知道"）。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ZOO_ROOT = PROJECT_ROOT / "assets" / "zoo"
FORMAT_VERSION = 1
_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")


class ZooError(ValueError):
    """模型 zoo 读写/校验错误。"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_commit() -> str:
    """当前 git commit（无 git / 非仓库时返回空串，不阻塞保存）。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelCard:
    """一个 checkpoint 的元数据（card.json 的 schema）。"""

    name: str
    kind: str = "neural"
    obs_dim: int = 9
    n_actions: int = 5
    hidden: int = 32
    seed: int = 0
    format_version: int = FORMAT_VERSION
    created_at: str = ""
    git_commit: str = ""
    config_hash: str = ""
    weights_sha256: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "obs_dim": int(self.obs_dim),
            "n_actions": int(self.n_actions),
            "hidden": int(self.hidden),
            "seed": int(self.seed),
            "format_version": int(self.format_version),
            "created_at": self.created_at,
            "git_commit": self.git_commit,
            "config_hash": self.config_hash,
            "weights_sha256": self.weights_sha256,
            "description": self.description,
            "tags": list(self.tags),
            "metrics": dict(self.metrics),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelCard":
        if not isinstance(data, dict):
            raise ZooError("card 必须是映射")
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ZooError(f"card 未知字段: {sorted(unknown)}")
        values = dict(data)
        values["tags"] = list(values.get("tags") or [])
        values["metrics"] = dict(values.get("metrics") or {})
        values["extra"] = dict(values.get("extra") or {})
        return cls(**values)


class ModelZoo:
    """assets/zoo 目录的读写门面（路径全部限制在 root 下）。"""

    def __init__(self, root: str | Path = DEFAULT_ZOO_ROOT) -> None:
        self.root = Path(root)

    # ------------------------------------------------------------------
    def _safe_name(self, name: str) -> str:
        if not isinstance(name, str) or not _NAME_RE.match(name):
            raise ZooError(
                f"非法模型名 {name!r}；只允许字母数字开头，含 [A-Za-z0-9_.-]"
            )
        return name

    def model_dir(self, name: str) -> Path:
        return self.root / self._safe_name(name)

    def exists(self, name: str) -> bool:
        return (self.model_dir(name) / "card.json").exists()

    # ------------------------------------------------------------------
    def save(
        self,
        name: str,
        policy,
        *,
        metrics: dict[str, Any] | None = None,
        config: Any | None = None,
        tags: list[str] | tuple[str, ...] = (),
        description: str = "",
        extra: dict[str, Any] | None = None,
    ) -> ModelCard:
        """保存策略 + 元数据，返回 ModelCard。

        policy 必须提供 save(path)（目前是 NeuralSwarmPolicy）。
        config 若提供 ExperimentSettings，会写 config.yaml 并记录 hash。
        """
        self._safe_name(name)
        if not callable(getattr(policy, "save", None)):
            raise ZooError(f"策略 {type(policy).__name__} 不支持 save，不能入 zoo")
        directory = self.model_dir(name)
        directory.mkdir(parents=True, exist_ok=True)
        weights = policy.save(directory / "policy.npz")
        config_hash = ""
        if config is not None:
            if hasattr(config, "hash") and hasattr(config, "to_dict"):
                config_hash = str(config.hash())
                try:
                    import yaml

                    (directory / "config.yaml").write_text(
                        yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True),
                        encoding="utf-8",
                    )
                except Exception:  # pragma: no cover - 配置快照失败不应毁掉模型
                    pass
            else:
                config_hash = hashlib.sha256(
                    json.dumps(config, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest()
        card = ModelCard(
            name=name,
            kind=str(getattr(policy, "kind", type(policy).__name__)),
            obs_dim=int(getattr(policy, "obs_dim", 0)),
            n_actions=int(getattr(policy, "n_actions", 0)),
            hidden=int(getattr(policy, "hidden", 0)),
            seed=int(getattr(policy, "seed", 0)),
            created_at=_utc_now(),
            git_commit=_git_commit(),
            config_hash=config_hash,
            weights_sha256=_sha256_file(Path(weights)),
            description=description,
            tags=[str(tag) for tag in tags],
            metrics=dict(metrics or {}),
            extra=dict(extra or {}),
        )
        (directory / "card.json").write_text(
            json.dumps(card.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (directory / "metrics.json").write_text(
            json.dumps(card.metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return card

    # ------------------------------------------------------------------
    def load_card(self, name: str) -> ModelCard:
        path = self.model_dir(name) / "card.json"
        if not path.exists():
            raise ZooError(f"模型不存在: {name}（{path}）")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ZooError(f"{path}: card.json 解析失败: {exc}") from exc
        return ModelCard.from_dict(data)

    def load(self, name: str, *, seed: int | None = None):
        """载入策略；seed 缺省用 card 里记录的 seed。"""
        from .swarm_policy import NeuralSwarmPolicy

        card = self.load_card(name)
        if card.kind != "neural":
            raise ZooError(f"模型 {name} 的 kind={card.kind} 不是可载入的神经策略")
        policy = NeuralSwarmPolicy.load(
            self.model_dir(name) / "policy.npz",
            seed=int(card.seed if seed is None else seed),
        )
        if int(policy.obs_dim) != int(card.obs_dim) or int(policy.n_actions) != int(card.n_actions):
            raise ZooError(
                f"模型 {name} 权重维度与 card 不一致: "
                f"{policy.obs_dim}x{policy.n_actions} vs {card.obs_dim}x{card.n_actions}"
            )
        return policy

    # ------------------------------------------------------------------
    def names(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(
            child.name for child in self.root.iterdir()
            if child.is_dir() and (child / "card.json").exists()
        )

    def cards(self) -> list[ModelCard]:
        return [self.load_card(name) for name in self.names()]

    def get(self, name: str) -> ModelCard:
        return self.load_card(name)

    def verify(self, name: str) -> bool:
        """校验权重文件 sha256 与 card 一致（防手改/损坏）。"""
        card = self.load_card(name)
        weights = self.model_dir(name) / "policy.npz"
        if not weights.exists():
            return False
        return _sha256_file(weights) == card.weights_sha256

    def manifest(self, path: str | Path | None = None) -> dict[str, Any]:
        """汇总所有 card（可写 manifest.json 给前端/文档读取）。"""
        cards = self.cards()
        payload = {
            "format_version": FORMAT_VERSION,
            "generated_at": _utc_now(),
            "git_commit": _git_commit(),
            "models": [card.to_dict() for card in cards],
        }
        if path is not None:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return payload


__all__ = [
    "DEFAULT_ZOO_ROOT",
    "FORMAT_VERSION",
    "ModelCard",
    "ModelZoo",
    "ZooError",
]

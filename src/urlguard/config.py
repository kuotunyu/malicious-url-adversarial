"""型別化設定樹 + 從 YAML 載入。

把散落在 notebook 各 cell 的 magic number (vocab_size=5000, max_length=100,
epsilon=0.05, seed=42 ...) 收斂到單一來源,讓每個實驗都可由一份 YAML 完整描述、
可版本控管、可復現。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class DataConfig:
    dataset: str = "sid321axn/malicious-urls-dataset"
    csv_name: str = "malicious_phish.csv"
    raw_dir: str = "data/raw"
    downsample: bool = True


@dataclass
class FeaturesConfig:
    vocab_size: int = 5000
    max_length: int = 100
    oov_token: str = "<OOV>"
    char_level: bool = True
    lowercase: bool = True
    padding: str = "post"  # "post" | "pre"
    truncating: str = "post"  # "post" | "pre"


@dataclass
class ModelConfig:
    type: str = "dense_pool"  # "dense_pool" | "lstm"
    embedding_dim: int = 16
    dense_units: int = 24
    lstm_units: int = 64
    dropout: float = 0.0


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 32
    test_size: float = 0.2
    stratify: bool = True
    threshold: float = 0.5
    learning_rate: float = 1e-3


@dataclass
class AttackConfig:
    method: str = "fgsm"  # "fgsm" | "pgd"
    epsilon: float = 0.05
    sweep: list[float] = field(default_factory=lambda: [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3])
    pgd_steps: int = 10
    pgd_alpha: float = 0.01
    confidence_threshold: float = 0.9


@dataclass
class DefenseConfig:
    method: str = "fgsm"  # 生成對抗樣本用的攻擊
    epsilon: float = 0.1
    adv_weight: float = 0.5
    epochs: int = 5


@dataclass
class Config:
    seed: int = 42
    artifacts_dir: str = "artifacts"
    images_dir: str = "docs/images"
    data: DataConfig = field(default_factory=DataConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)

    # ---- 便利屬性 ----
    @property
    def artifacts_path(self) -> Path:
        return Path(self.artifacts_dir)

    @property
    def images_path(self) -> Path:
        return Path(self.images_dir)


def _build(cls: type, data: dict[str, Any] | None) -> Any:
    """遞迴地把 dict 建成(可能巢狀的)dataclass,未提供的欄位用預設值。

    使用 get_type_hints 解析型別:因為本模組啟用了 ``from __future__ import
    annotations``,dataclass field.type 會是字串(如 "DataConfig"),需先解析成
    真正的型別物件才能判斷是否為巢狀 dataclass。
    """
    data = data or {}
    if not dataclasses.is_dataclass(cls):
        return data
    hints = get_type_hints(cls)
    field_names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - field_names
    if unknown:
        raise ValueError(f"{cls.__name__} 收到未知的設定欄位: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name in field_names:
        if name not in data:
            continue
        ftype = hints.get(name)
        if dataclasses.is_dataclass(ftype):
            kwargs[name] = _build(ftype, data[name])  # type: ignore[arg-type]
        else:
            kwargs[name] = data[name]
    return cls(**kwargs)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _coerce(value: str) -> Any:
    """把 CLI 的字串值粗略轉成 bool/int/float/list。"""
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    if "," in value:
        return [_coerce(v.strip()) for v in value.split(",") if v.strip() != ""]
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            continue
    return value


def _apply_dotted(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    """把 'attack.epsilon' -> value 寫進巢狀 dict。"""
    parts = dotted_key.split(".")
    node = data
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def load_config(
    path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> Config:
    """從 YAML 載入設定;overrides 為 ['attack.epsilon=0.1', ...] 形式的點路徑覆蓋。"""
    data: dict[str, Any] = {}
    if path is not None:
        text = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
    if overrides:
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"--set 需為 key=value 形式,收到: {item!r}")
            key, raw = item.split("=", 1)
            _apply_dotted(data, key.strip(), _coerce(raw.strip()))
    return _build(Config, data)


def to_dict(cfg: Config) -> dict[str, Any]:
    """序列化回純 dict(存 artifact / 記錄實驗用)。"""
    return dataclasses.asdict(cfg)

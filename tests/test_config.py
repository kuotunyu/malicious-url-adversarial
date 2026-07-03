"""設定載入 / 覆蓋 / 驗證。"""

from __future__ import annotations

import pytest

from urlguard.config import load_config


def test_default_config_values():
    cfg = load_config("configs/default.yaml")
    assert cfg.seed == 42
    assert cfg.features.vocab_size == 5000
    assert cfg.features.max_length == 100
    assert cfg.train.stratify is True
    assert cfg.model.type == "dense_pool"


def test_fast_ci_config_is_tiny():
    cfg = load_config("configs/fast_ci.yaml")
    assert cfg.features.vocab_size == 200
    assert cfg.train.epochs == 1


def test_dotted_overrides_apply_and_coerce_types():
    cfg = load_config(
        "configs/fast_ci.yaml",
        overrides=["attack.epsilon=0.25", "model.type=lstm", "train.stratify=false"],
    )
    assert cfg.attack.epsilon == 0.25
    assert isinstance(cfg.attack.epsilon, float)
    assert cfg.model.type == "lstm"
    assert cfg.train.stratify is False


def test_list_override_parses():
    cfg = load_config("configs/fast_ci.yaml", overrides=["attack.sweep=0.0,0.1,0.2"])
    assert cfg.attack.sweep == [0.0, 0.1, 0.2]


def test_unknown_field_raises():
    with pytest.raises(ValueError):
        load_config("configs/fast_ci.yaml", overrides=["attack.not_a_field=1"])


def test_bad_override_format_raises():
    with pytest.raises(ValueError):
        load_config("configs/fast_ci.yaml", overrides=["missing_equals"])

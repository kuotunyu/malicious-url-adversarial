"""共用 fixtures。所有測試都以 repo root 為工作目錄,讓 configs/ 與 tests/fixtures 相對路徑可解析。

需要 TensorFlow 的 fixture(built_model)會在 TF 未安裝時自動 skip,
因此純資料/特徵/指標測試在無 TF 環境也能完整跑。"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FAST_CI_CONFIG = str(REPO_ROOT / "configs" / "fast_ci.yaml")


@pytest.fixture(autouse=True)
def _chdir_repo_root(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)


@pytest.fixture
def tiny_config():
    from urlguard.config import load_config

    return load_config(FAST_CI_CONFIG)


@pytest.fixture
def tiny_df(tiny_config):
    from urlguard.data import load_balanced

    return load_balanced(tiny_config)


@pytest.fixture
def fitted_tokenizer(tiny_config, tiny_df):
    from urlguard.features import build_tokenizer

    return build_tokenizer(tiny_df["url"].tolist(), tiny_config.features)


@pytest.fixture
def encoded(tiny_df, fitted_tokenizer):
    x = fitted_tokenizer.encode(tiny_df["url"].tolist())
    y = tiny_df["label"].to_numpy()
    return x, y


@pytest.fixture
def built_model(tiny_config, fitted_tokenizer):
    pytest.importorskip("tensorflow")
    from urlguard.model.build import build_model, compile_model

    model = build_model(
        tiny_config.model,
        num_tokens=fitted_tokenizer.num_tokens,
        max_length=tiny_config.features.max_length,
    )
    compile_model(model, tiny_config.train.learning_rate)
    return model

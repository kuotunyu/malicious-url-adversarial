"""模型建立(需要 TensorFlow;未安裝時 skip)。"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorflow")


def test_output_shape_and_range(built_model, encoded):
    x, _ = encoded
    prob = built_model.predict(x, verbose=0)
    assert prob.shape == (len(x), 1)
    assert np.all((prob >= 0.0) & (prob <= 1.0))
    assert not np.isnan(prob).any()


def test_embedding_layer_named(built_model):
    from urlguard.model.build import EMBEDDING_LAYER_NAME

    # 攻擊模組依賴此命名層存在
    assert built_model.get_layer(EMBEDDING_LAYER_NAME) is not None
    assert built_model.layers[0].name == EMBEDDING_LAYER_NAME


def test_lstm_variant_builds(tiny_config, fitted_tokenizer):
    from urlguard.model.build import build_model

    tiny_config.model.type = "lstm"
    m = build_model(
        tiny_config.model,
        num_tokens=fitted_tokenizer.num_tokens,
        max_length=tiny_config.features.max_length,
    )
    assert m.output_shape == (None, 1)

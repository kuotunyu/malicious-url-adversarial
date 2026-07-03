"""對抗訓練(需要 TensorFlow;未安裝時 skip)。

註:在數十列的 toy fixture 上訓練 1 個 epoch 無法保證統計上的 robustness 提升,
因此這裡驗證『機制正確、輸出合法、不產生 NaN』,而非斷言一定變 robust
(真正的防禦效果請跑 configs/default.yaml 於完整資料集觀察)。"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorflow")


def test_adversarial_train_produces_valid_model(tiny_config, tiny_df, fitted_tokenizer):
    from urlguard.defenses.adv_training import adversarial_train

    x = fitted_tokenizer.encode(tiny_df["url"].tolist())
    y = tiny_df["label"].to_numpy()
    model = adversarial_train(
        tiny_config,
        x,
        y,
        num_tokens=fitted_tokenizer.num_tokens,
        max_length=tiny_config.features.max_length,
    )
    prob = model.predict(x, verbose=0)
    assert prob.shape == (len(x), 1)
    assert np.all((prob >= 0) & (prob <= 1))
    assert not np.isnan(prob).any()


def test_robustness_report_structure(tiny_config, tiny_df, fitted_tokenizer, built_model):
    from urlguard.defenses.adv_training import robustness_report

    x = fitted_tokenizer.encode(tiny_df["url"].tolist())
    y = tiny_df["label"].to_numpy()
    rep = robustness_report(built_model, x, y, tiny_config, label="vanilla")
    for k in ["clean_accuracy", "robust_acc_fgsm", "robust_acc_pgd", "sweep"]:
        assert k in rep
    assert 0.0 <= rep["robust_acc_fgsm"] <= 1.0
    assert 0.0 <= rep["robust_acc_pgd"] <= 1.0

"""FGSM 攻擊(需要 TensorFlow;未安裝時 skip)。"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorflow")


def _bce(y, p, eps=1e-7):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y, dtype=float)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def test_classifier_from_embeddings_matches_full_model(built_model, encoded):
    from urlguard.attacks.base import build_classifier_from_embeddings, embed

    x, _ = encoded
    clf = build_classifier_from_embeddings(built_model)
    emb = embed(built_model, x)
    from_emb = clf.predict(emb, verbose=0).ravel()
    full = built_model.predict(x, verbose=0).ravel()
    # 兩條路徑共享權重,輸出應一致
    np.testing.assert_allclose(from_emb, full, atol=1e-5)


def test_fgsm_increases_loss_and_no_nan(built_model, encoded):
    from urlguard.attacks.fgsm import fgsm_attack

    x, _ = encoded
    y = np.ones(len(x), dtype=int)  # 視為惡意:攻擊應壓低惡意機率、增大 loss
    res = fgsm_attack(built_model, x, y, epsilon=0.05)
    assert not np.isnan(res["adv_prob"]).any()
    # FGSM 沿梯度上升方向,對抗樣本的 BCE loss 應 >= 乾淨 loss(一階保證)
    assert _bce(y, res["adv_prob"]) >= _bce(y, res["orig_prob"]) - 1e-6


def test_epsilon_sweep_structure(built_model, encoded):
    from urlguard.attacks.fgsm import epsilon_sweep

    x, _ = encoded
    y = np.ones(len(x), dtype=int)
    out = epsilon_sweep(built_model, x, y, [0.0, 0.1, 0.3], threshold=0.5, method="fgsm")
    assert len(out["results"]) == 3
    first = out["results"][0]
    assert first["epsilon"] == 0.0
    # eps=0 時對抗=原始,平均機率應等於原始平均
    assert abs(first["mean_adv_prob"] - out["mean_orig_prob"]) < 1e-6
    for r in out["results"]:
        assert {"epsilon", "success_rate", "mean_adv_prob", "robust_accuracy"} <= set(r)

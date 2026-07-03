"""PGD 攻擊(需要 TensorFlow;未安裝時 skip)。"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorflow")


def test_pgd_stays_within_linf_ball(built_model, encoded):
    import tensorflow as tf

    from urlguard.attacks.base import build_classifier_from_embeddings, embed
    from urlguard.attacks.pgd import pgd_perturb

    x, _ = encoded
    y = np.ones(len(x), dtype=int)
    clf = build_classifier_from_embeddings(built_model)
    emb = embed(built_model, x)
    epsilon = 0.1
    adv = pgd_perturb(clf, emb, y, epsilon=epsilon, steps=5, alpha=0.03)
    linf = float(tf.reduce_max(tf.abs(adv - emb)))
    assert linf <= epsilon + 1e-5


def test_pgd_reduces_malicious_prob(built_model, encoded):
    from urlguard.attacks.pgd import pgd_attack

    x, _ = encoded
    y = np.ones(len(x), dtype=int)
    res = pgd_attack(built_model, x, y, epsilon=0.2, steps=10, alpha=0.05)
    assert not np.isnan(res["adv_prob"]).any()
    # 對惡意樣本,PGD 應平均壓低惡意機率
    assert res["adv_prob"].mean() <= res["orig_prob"].mean() + 1e-6

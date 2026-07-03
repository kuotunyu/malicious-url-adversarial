"""評估指標。"""

from __future__ import annotations

import numpy as np

from urlguard.evaluation.metrics import classification_metrics, recall_at_fpr, robust_accuracy


def test_perfect_separation_gives_auc_one():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.01, 0.2, 0.8, 0.99])
    m = classification_metrics(y, p)
    assert m["roc_auc"] == 1.0
    assert m["pr_auc"] == 1.0
    assert m["accuracy"] == 1.0


def test_metric_keys_and_ranges():
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([0.3, 0.6, 0.2, 0.4, 0.9])
    m = classification_metrics(y, p)
    for k in ["accuracy", "precision_malicious", "recall_malicious", "f1_malicious", "roc_auc"]:
        assert 0.0 <= m[k] <= 1.0
    assert np.array(m["confusion_matrix"]).shape == (2, 2)


def test_recall_at_fpr_returns_valid_threshold():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    r = recall_at_fpr(y, p, target_fpr=0.0)
    assert r["recall"] == 1.0
    assert 0.0 <= r["threshold"] <= 1.0


def test_robust_accuracy():
    y = np.array([1, 1, 0, 0])
    adv = np.array([0.2, 0.9, 0.1, 0.8])  # 第 1、4 個被翻
    assert robust_accuracy(y, adv) == 0.5


def test_single_class_auc_is_nan():
    y = np.array([1, 1, 1])
    p = np.array([0.6, 0.7, 0.8])
    m = classification_metrics(y, p)
    assert np.isnan(m["roc_auc"])

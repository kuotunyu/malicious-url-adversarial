"""分類與 robustness 指標。

資安場景重點:漏報(False Negative,把惡意判為良性)成本最高,因此除了
accuracy 之外,必須看 recall、PR-AUC,以及固定低誤報率(FPR)下的 recall。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def _as_labels(y_prob: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(y_prob).ravel() >= threshold).astype(int)


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """回傳 accuracy / precision / recall / f1 / roc_auc / pr_auc / 混淆矩陣等。"""
    y_true = np.asarray(y_true).ravel().astype(int)
    y_prob = np.asarray(y_prob).ravel().astype(float)
    y_pred = _as_labels(y_prob, threshold)

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],  # 明確指定,避免單一類別時與 target_names 長度不符而報錯
        target_names=["benign", "malicious"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # AUC 需要兩個類別都存在
    both_classes = len(np.unique(y_true)) == 2
    roc_auc = float(roc_auc_score(y_true, y_prob)) if both_classes else float("nan")
    pr_auc = float(average_precision_score(y_true, y_prob)) if both_classes else float("nan")

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_malicious": float(report["malicious"]["precision"]),
        "recall_malicious": float(report["malicious"]["recall"]),
        "f1_malicious": float(report["malicious"]["f1-score"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
        "report": report,
    }


def recall_at_fpr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_fpr: float = 0.01,
) -> dict[str, float]:
    """在指定最大 FPR 下可達到的最高 recall(TPR)與對應門檻。

    回答『在真實流量(良性遠多於惡意)只能容忍 1% 誤報時,還能抓到多少惡意?』
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_prob = np.asarray(y_prob).ravel().astype(float)
    if len(np.unique(y_true)) < 2:
        return {"target_fpr": float(target_fpr), "recall": float("nan"), "threshold": float("nan")}
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    mask = fpr <= target_fpr
    if not mask.any():
        return {"target_fpr": float(target_fpr), "recall": 0.0, "threshold": 1.0}
    idx = int(np.argmax(tpr[mask]))
    chosen = np.where(mask)[0][idx]
    # roc_curve 的 thresholds[0] 可能是 +inf,取實際門檻時夾到 [0,1]
    threshold = float(min(max(thr[chosen], 0.0), 1.0))
    return {
        "target_fpr": float(target_fpr),
        "recall": float(tpr[chosen]),
        "threshold": threshold,
    }


def robust_accuracy(
    y_true: np.ndarray,
    adv_prob: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """對抗樣本下的 accuracy(越高代表模型越 robust)。"""
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = _as_labels(adv_prob, threshold)
    return float((y_pred == y_true).mean())

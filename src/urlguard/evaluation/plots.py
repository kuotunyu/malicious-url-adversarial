"""繪圖:把結果輸出成 PNG(供 README 內嵌)。使用 Agg 後端,無需顯示器。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion_matrix(cm, path: str | Path) -> Path:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Pred benign", "Pred malicious"])
    ax.set_yticks([0, 1], ["Actual benign", "Actual malicious"])
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:d}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=13,
            )
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, path)


def plot_history(history: dict, path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    if "loss" in history:
        ax.plot(history["loss"], label="train loss", marker="o")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="val loss", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training / Validation Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_roc_pr(y_true, y_prob, path: str | Path) -> Path:
    y_true = np.asarray(y_true).ravel().astype(int)
    y_prob = np.asarray(y_prob).ravel().astype(float)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ax1.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})", color="C0")
    ax1.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate (Recall)")
    ax1.set_title("ROC Curve")
    ax1.legend()
    ax1.grid(alpha=0.3)

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    ax2.plot(rec, prec, label=f"PR (AP={ap:.3f})", color="C1")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend()
    ax2.grid(alpha=0.3)
    return _save(fig, path)


def plot_attack_bar(orig_prob: float, adv_prob: float, epsilon: float, path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    probs = [float(orig_prob), float(adv_prob)]
    colors = ["#d62728" if p > 0.5 else "#2ca02c" for p in probs]
    bars = ax.bar(["Original", "Adversarial"], probs, color=colors, edgecolor="black", alpha=0.85)
    ax.axhline(0.5, ls="--", color="gray", label="Decision boundary (0.5)")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Malicious probability")
    ax.set_title(f"FGSM Attack (epsilon={epsilon})")
    for b, p in zip(bars, probs, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, p + 0.02, f"{p:.1%}", ha="center", fontweight="bold")
    ax.legend()
    return _save(fig, path)


def plot_epsilon_sweep(sweeps: dict[str, dict], path: str | Path) -> Path:
    """一或多條攻擊成功率曲線。sweeps: {label: sweep_result_dict}。"""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for label, sweep in sweeps.items():
        rows = sweep["results"]
        xs = [r["epsilon"] for r in rows]
        ys = [100.0 * r["success_rate"] for r in rows]
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel("Epsilon (perturbation strength)")
    ax.set_ylabel("Attack success rate (%)")
    ax.set_title("Attack Success Rate vs Epsilon")
    ax.set_ylim(-2, 102)
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, path)


def plot_robustness_comparison(sweep_vanilla: dict, sweep_robust: dict, path: str | Path) -> Path:
    """防禦前後對照:vanilla vs adversarially-trained 的攻擊成功率曲線。"""
    return plot_epsilon_sweep(
        {"vanilla": sweep_vanilla, "adversarially-trained": sweep_robust}, path
    )

"""FGSM(Fast Gradient Sign Method)單步攻擊 + epsilon 掃描。

公式: adv = emb + epsilon * sign(∇_emb J(θ, emb, y))
"""

from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf

from urlguard.attacks.base import build_classifier_from_embeddings, embed, signed_gradient


def fgsm_attack(model, x, y, epsilon: float, classifier=None) -> dict[str, Any]:
    """對一批樣本做單步 FGSM,回傳原始與對抗機率。"""
    classifier = classifier or build_classifier_from_embeddings(model)
    emb = tf.convert_to_tensor(embed(model, x))
    signed = signed_gradient(classifier, emb, y)
    adv = emb + epsilon * signed
    orig_prob = classifier(emb, training=False).numpy().ravel()
    adv_prob = classifier(adv, training=False).numpy().ravel()
    return {
        "epsilon": float(epsilon),
        "orig_prob": orig_prob,
        "adv_prob": adv_prob,
    }


def attack_success_rate(
    y: np.ndarray,
    orig_prob: np.ndarray,
    adv_prob: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """攻擊成功率:在『原本就被正確偵測的惡意樣本』中,被翻成良性的比例。

    只計算 label=1 且原始機率>=threshold 的樣本,避免把本來就分錯的樣本灌水。
    """
    y = np.asarray(y).ravel().astype(int)
    orig_prob = np.asarray(orig_prob).ravel()
    adv_prob = np.asarray(adv_prob).ravel()
    target = (y == 1) & (orig_prob >= threshold)
    denom = int(target.sum())
    if denom == 0:
        return float("nan")
    flipped = int(((adv_prob < threshold) & target).sum())
    return flipped / denom


def epsilon_sweep(
    model,
    x,
    y,
    epsilons: list[float],
    threshold: float = 0.5,
    method: str = "fgsm",
    pgd_steps: int = 10,
    pgd_alpha: float = 0.01,
) -> dict[str, Any]:
    """掃描多個 epsilon,回傳每個強度的攻擊成功率與 robust accuracy。

    把原 notebook 的單點『攻擊失敗』改寫為量化的攻擊強度分析。
    """
    classifier = build_classifier_from_embeddings(model)
    emb = tf.convert_to_tensor(embed(model, x))
    y_arr = np.asarray(y).ravel().astype(int)
    orig_prob = classifier(emb, training=False).numpy().ravel()

    results = []
    for eps in epsilons:
        if method == "fgsm":
            signed = signed_gradient(classifier, emb, y_arr)
            adv = emb + float(eps) * signed
        elif method == "pgd":
            from urlguard.attacks.pgd import pgd_perturb

            adv = pgd_perturb(classifier, emb, y_arr, float(eps), pgd_steps, pgd_alpha)
        else:
            raise ValueError(f"未知 method: {method!r}(fgsm / pgd)")
        adv_prob = classifier(adv, training=False).numpy().ravel()
        adv_pred = (adv_prob >= threshold).astype(int)
        results.append(
            {
                "epsilon": float(eps),
                "success_rate": attack_success_rate(y_arr, orig_prob, adv_prob, threshold),
                "mean_adv_prob": float(adv_prob.mean()),
                "robust_accuracy": float((adv_pred == y_arr).mean()),
            }
        )
    return {
        "method": method,
        "threshold": float(threshold),
        "n_samples": int(len(y_arr)),
        "mean_orig_prob": float(orig_prob.mean()),
        "results": results,
    }

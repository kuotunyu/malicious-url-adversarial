"""PGD(Projected Gradient Descent)迭代攻擊。

比單步 FGSM 更強:多步小幅前進,每步後把擾動投影回以原點為中心、半徑 epsilon 的
L-infinity 球內。通常能把惡意機率壓破 0.5,示範一個真正成功的攻擊。
"""

from __future__ import annotations

from typing import Any

import tensorflow as tf

from urlguard.attacks.base import build_classifier_from_embeddings, embed, loss_object


def pgd_perturb(
    classifier: tf.keras.Model,
    embeddings: tf.Tensor,
    labels,
    epsilon: float,
    steps: int = 10,
    alpha: float = 0.01,
) -> tf.Tensor:
    """在 embedding 空間對一批樣本做 PGD,回傳對抗 embedding。"""
    labels_t = tf.reshape(tf.cast(labels, tf.float32), (-1, 1))
    loss_obj = loss_object()
    emb0 = tf.identity(tf.convert_to_tensor(embeddings))
    adv = tf.identity(emb0)
    for _ in range(int(steps)):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            pred = classifier(adv, training=False)
            loss = loss_obj(labels_t, pred)
        grad = tape.gradient(loss, adv)
        adv = adv + alpha * tf.sign(grad)  # 梯度上升:增大 loss
        # 投影回 L-inf(epsilon)球
        adv = emb0 + tf.clip_by_value(adv - emb0, -epsilon, epsilon)
    return adv


def pgd_attack(
    model,
    x,
    y,
    epsilon: float,
    steps: int = 10,
    alpha: float = 0.01,
    classifier=None,
) -> dict[str, Any]:
    """對一批樣本做 PGD,回傳原始與對抗機率。"""
    classifier = classifier or build_classifier_from_embeddings(model)
    emb = tf.convert_to_tensor(embed(model, x))
    adv = pgd_perturb(classifier, emb, y, epsilon, steps, alpha)
    orig_prob = classifier(emb, training=False).numpy().ravel()
    adv_prob = classifier(adv, training=False).numpy().ravel()
    return {
        "epsilon": float(epsilon),
        "steps": int(steps),
        "alpha": float(alpha),
        "orig_prob": orig_prob,
        "adv_prob": adv_prob,
    }

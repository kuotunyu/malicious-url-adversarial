"""對抗訓練(Adversarial Training)。

每個 batch 在 embedding 空間即時生成 FGSM/PGD 對抗樣本,以『乾淨 + 對抗』的
混合損失更新**所有**權重(含 embedding)。攻擊與防禦都在同一 embedding 空間,
因此 robustness 前後可直接比較(Goodfellow 2015 / Madry 2018 的簡化版)。

擾動方向以 ``tf.stop_gradient`` 視為常數,這是對抗訓練的標準做法:我們要模型對
「當前這個擾動」變 robust,而不是去對擾動本身求導。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf

from urlguard.attacks.base import build_classifier_from_embeddings, get_embedding_layer, loss_object
from urlguard.attacks.fgsm import epsilon_sweep
from urlguard.evaluation.metrics import classification_metrics, robust_accuracy
from urlguard.logging_utils import get_logger
from urlguard.model.build import build_model, compile_model
from urlguard.seed import set_seed

logger = get_logger(__name__)


def _make_perturbation(classifier, emb, y, method, epsilon, steps, alpha):
    """生成對抗擾動(常數,不帶梯度)。"""
    if method == "pgd":
        from urlguard.attacks.pgd import pgd_perturb

        adv = pgd_perturb(classifier, emb, y, epsilon, steps, alpha)
        return tf.stop_gradient(adv - emb)
    # FGSM
    loss_obj = loss_object()
    y_t = tf.reshape(tf.cast(y, tf.float32), (-1, 1))
    with tf.GradientTape() as tape:
        tape.watch(emb)
        pred = classifier(emb, training=False)
        loss = loss_obj(y_t, pred)
    signed = tf.sign(tape.gradient(loss, emb))
    return tf.stop_gradient(epsilon * signed)


def adversarial_train(
    cfg,
    x_train: np.ndarray,
    y_train: np.ndarray,
    num_tokens: int,
    max_length: int,
) -> tf.keras.Model:
    """從頭訓練一個對抗強化模型(架構同 cfg.model),回傳硬化後的模型。"""
    set_seed(cfg.seed)
    d = cfg.defense
    model = build_model(cfg.model, num_tokens=num_tokens, max_length=max_length)
    compile_model(model, cfg.train.learning_rate)  # 建 optimizer/metrics(也讓 model 完整 build)

    emb_layer = get_embedding_layer(model)
    classifier = build_classifier_from_embeddings(model)  # 與 model body 共享權重
    optimizer = model.optimizer
    loss_obj = loss_object()
    w = float(d.adv_weight)

    x_t = tf.convert_to_tensor(x_train)
    y_t = tf.convert_to_tensor(np.asarray(y_train).reshape(-1, 1).astype("float32"))
    ds = (
        tf.data.Dataset.from_tensor_slices((x_t, y_t))
        .shuffle(buffer_size=min(len(x_train), 10000), seed=cfg.seed)
        .batch(cfg.train.batch_size)
    )

    def train_step(xb, yb):
        emb0 = emb_layer(xb)  # 用於生成擾動
        perturb = _make_perturbation(
            classifier, emb0, yb, d.method, d.epsilon, cfg.attack.pgd_steps, cfg.attack.pgd_alpha
        )
        with tf.GradientTape() as tape:
            emb = emb_layer(xb)  # 重算,連上 embedding 權重供求導
            clean_pred = classifier(emb, training=True)
            adv_pred = classifier(emb + perturb, training=True)
            loss = (1.0 - w) * loss_obj(yb, clean_pred) + w * loss_obj(yb, adv_pred)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables, strict=True))
        return loss

    for epoch in range(d.epochs):
        losses = []
        for xb, yb in ds:
            losses.append(float(train_step(xb, yb)))
        logger.info("[adv-train] epoch %d/%d  mean_loss=%.4f", epoch + 1, d.epochs, np.mean(losses))

    return model


def robustness_report(
    model: tf.keras.Model,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    cfg,
    label: str = "model",
) -> dict[str, Any]:
    """回傳一個模型的乾淨指標 + 在 defense.epsilon 下 FGSM/PGD 的 robust accuracy + epsilon 掃描。"""
    clean_prob = model.predict(x_eval, verbose=0).ravel()
    clean = classification_metrics(y_eval, clean_prob, threshold=cfg.train.threshold)

    from urlguard.attacks.fgsm import fgsm_attack
    from urlguard.attacks.pgd import pgd_attack

    eps = cfg.defense.epsilon
    fgsm_res = fgsm_attack(model, x_eval, y_eval, eps)
    pgd_res = pgd_attack(
        model, x_eval, y_eval, eps, steps=cfg.attack.pgd_steps, alpha=cfg.attack.pgd_alpha
    )
    sweep = epsilon_sweep(
        model, x_eval, y_eval, cfg.attack.sweep, threshold=cfg.train.threshold, method="fgsm"
    )
    return {
        "label": label,
        "clean_accuracy": clean["accuracy"],
        "clean_recall_malicious": clean["recall_malicious"],
        "clean_roc_auc": clean["roc_auc"],
        "clean_pr_auc": clean["pr_auc"],
        "robust_acc_fgsm": robust_accuracy(y_eval, fgsm_res["adv_prob"], cfg.train.threshold),
        "robust_acc_pgd": robust_accuracy(y_eval, pgd_res["adv_prob"], cfg.train.threshold),
        "epsilon": float(eps),
        "sweep": sweep,
    }

"""攻擊共用工具:拆出 embedding→輸出 子模型、取得 embedding、計算梯度方向。"""

from __future__ import annotations

import tensorflow as tf

from urlguard.model.build import EMBEDDING_LAYER_NAME


def get_embedding_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    return model.get_layer(EMBEDDING_LAYER_NAME)


def build_classifier_from_embeddings(model: tf.keras.Model) -> tf.keras.Model:
    """建立「embedding 向量 → 惡意機率」的子模型,重用已訓練層(共享權重)。

    取代原 notebook 對 ``model.layers[1:]`` 的即興迴圈:此處以顯式 Input 決定性地
    重建,並可被單元測試。輸入形狀 (max_length, embedding_dim)。
    """
    emb_layer = get_embedding_layer(model)
    max_length = model.input_shape[1]
    emb_dim = emb_layer.output_dim
    inp = tf.keras.Input(shape=(max_length, emb_dim), name="embeddings")
    x = inp
    for layer in model.layers[1:]:  # 跳過 Embedding
        x = layer(x)
    return tf.keras.Model(inp, x, name="classifier_from_embeddings")


def embed(model: tf.keras.Model, x) -> tf.Tensor:
    """把 token id 序列轉成 embedding 向量 (n, max_length, emb_dim)。"""
    emb_layer = get_embedding_layer(model)
    return emb_layer(tf.convert_to_tensor(x))


def loss_object() -> tf.keras.losses.Loss:
    return tf.keras.losses.BinaryCrossentropy()


def signed_gradient(
    classifier: tf.keras.Model,
    embeddings: tf.Tensor,
    labels,
) -> tf.Tensor:
    """FGSM 的核心:loss 對 embedding 的梯度方向 sign(∇J)。

    對惡意樣本(label=1)沿此方向加擾動會『增大 loss』→ 壓低惡意機率 → 欺騙模型。
    """
    labels_t = tf.reshape(tf.cast(labels, tf.float32), (-1, 1))
    embeddings = tf.convert_to_tensor(embeddings)
    loss_obj = loss_object()
    with tf.GradientTape() as tape:
        tape.watch(embeddings)
        pred = classifier(embeddings, training=False)
        loss = loss_obj(labels_t, pred)
    grad = tape.gradient(loss, embeddings)
    return tf.sign(grad)

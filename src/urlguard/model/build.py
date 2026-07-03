"""建立 Keras 模型。

與原 notebook 的差異:
  * 移除 ``Embedding(..., input_length=...)`` —— 該參數在 Keras 3 / TF 2.20 已被
    忽略/棄用;改用顯式的 ``Input(shape=(max_length,))``。
  * Embedding 層命名為 ``embedding``,方便攻擊模組定位並拆出「embedding→輸出」子模型。
  * 以 config 選擇 dense_pool(輕量基線)或 lstm 架構(接上原 notebook 註解掉的 LSTM)。
"""

from __future__ import annotations

import tensorflow as tf

EMBEDDING_LAYER_NAME = "embedding"


def build_model(model_cfg, num_tokens: int, max_length: int) -> tf.keras.Model:
    """依 ModelConfig 建立未編譯的模型。

    Args:
        model_cfg: ModelConfig(type / embedding_dim / dense_units / lstm_units / dropout)。
        num_tokens: tokenizer 的索引總數(= Embedding 的 input_dim,須 > 最大 token id)。
        max_length: 輸入序列長度。
    """
    model = tf.keras.Sequential(name=f"urlguard_{model_cfg.type}")
    model.add(tf.keras.Input(shape=(max_length,), dtype="int32", name="tokens"))
    model.add(
        tf.keras.layers.Embedding(
            input_dim=num_tokens,
            output_dim=model_cfg.embedding_dim,
            name=EMBEDDING_LAYER_NAME,
        )
    )

    if model_cfg.type == "dense_pool":
        model.add(tf.keras.layers.GlobalAveragePooling1D(name="gap"))
        model.add(tf.keras.layers.Dense(model_cfg.dense_units, activation="relu", name="dense"))
    elif model_cfg.type == "lstm":
        model.add(tf.keras.layers.LSTM(model_cfg.lstm_units, name="lstm"))
    else:
        raise ValueError(f"未知的 model.type: {model_cfg.type!r}(可用: dense_pool / lstm)")

    if model_cfg.dropout and model_cfg.dropout > 0:
        model.add(tf.keras.layers.Dropout(model_cfg.dropout, name="dropout"))

    model.add(tf.keras.layers.Dense(1, activation="sigmoid", name="output"))
    return model


def compile_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    """以 Adam + binary_crossentropy 編譯。"""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

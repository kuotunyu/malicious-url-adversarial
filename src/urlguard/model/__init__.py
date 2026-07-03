"""模型層:建立與訓練 Keras 模型(匯入本子套件會連帶匯入 TensorFlow)。"""

from __future__ import annotations

from urlguard.model.build import EMBEDDING_LAYER_NAME, build_model

__all__ = ["build_model", "EMBEDDING_LAYER_NAME"]

"""集中化隨機種子設定,讓實驗可復現。"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """設定 python / numpy / tensorflow 的隨機種子。

    TensorFlow 以延遲方式匯入:未安裝 TF 的環境(例如只跑資料/特徵測試)
    仍可呼叫本函式而不報錯。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:  # pragma: no cover - TF 未安裝時略過
        pass

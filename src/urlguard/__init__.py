"""urlguard — 惡意網址偵測 + 對抗式攻防 (FGSM / PGD / 對抗訓練)。

模組邊界:
    data      讀取/下載/降採樣資料 (僅 pandas,不碰 TF)
    features  CharTokenizer 字元級編碼 (純 Python/numpy,可測試)
    model     建立/訓練 Keras 模型 (延遲 import tensorflow)
    attacks   FGSM / PGD 對抗攻擊 (embedding 特徵空間)
    defenses  對抗訓練
    evaluation 指標與繪圖 (sklearn / matplotlib)
    app       Streamlit demo
    cli       Typer 指令入口 (唯一解析參數處)
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]

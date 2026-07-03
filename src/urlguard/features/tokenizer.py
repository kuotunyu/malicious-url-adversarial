"""字元級 Tokenizer(純 Python / numpy)。

為什麼自己寫而不用 keras.preprocessing.text.Tokenizer:
  1. 該類別在 Keras 3 已被標記為 deprecated。
  2. 純 Python 實作讓特徵層不依賴 TensorFlow,可獨立單元測試。
  3. 完全掌控 OOV、padding、詞彙表上限與 save/load 格式。

索引配置:
  0 = <PAD>(補零),1 = <OOV>(未見字元),2.. = 依訓練集頻率排序的字元。
Embedding 層的 input_dim 應使用 ``tokenizer.num_tokens``(= 已配置索引數)。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

PAD_ID = 0
OOV_ID = 1
_RESERVED = 2  # PAD + OOV


class CharTokenizer:
    """字元(或簡易單字)層級的 tokenizer。

    重要:``fit`` 只能在**訓練集**上呼叫,避免測試集字元分佈洩漏進前處理。
    """

    def __init__(
        self,
        vocab_size: int = 5000,
        max_length: int = 100,
        oov_token: str = "<OOV>",
        char_level: bool = True,
        lowercase: bool = True,
        padding: str = "post",
        truncating: str = "post",
    ) -> None:
        if padding not in {"post", "pre"}:
            raise ValueError("padding 必須是 'post' 或 'pre'")
        if truncating not in {"post", "pre"}:
            raise ValueError("truncating 必須是 'post' 或 'pre'")
        self.vocab_size = int(vocab_size)
        self.max_length = int(max_length)
        self.oov_token = oov_token
        self.char_level = char_level
        self.lowercase = lowercase
        self.padding = padding
        self.truncating = truncating
        self.char2id: dict[str, int] = {}

    # ---- 內部 ----
    def _normalize(self, text: str) -> str:
        return text.lower() if self.lowercase else text

    def _split(self, text: str) -> list[str]:
        text = self._normalize(text)
        if self.char_level:
            return list(text)
        return [t for t in _WORD_SPLIT(text) if t]

    # ---- API ----
    @property
    def num_tokens(self) -> int:
        """已配置的索引總數(含 PAD/OOV),即 Embedding 的 input_dim。"""
        return _RESERVED + len(self.char2id)

    def fit(self, texts: list[str]) -> CharTokenizer:
        """在(僅)訓練集文字上建立詞彙表。"""
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(self._split(str(text)))
        # 依頻率(高→低)、同頻以字元排序,確保決定性
        keep = self.vocab_size - _RESERVED
        ordered = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        self.char2id = {ch: i + _RESERVED for i, (ch, _) in enumerate(ordered[:keep])}
        return self

    def texts_to_sequences(self, texts: list[str]) -> list[list[int]]:
        """轉成整數序列(未 padding);未見字元 → OOV_ID。"""
        if not self.char2id:
            raise RuntimeError("Tokenizer 尚未 fit,請先在訓練集上呼叫 fit()")
        seqs: list[list[int]] = []
        for text in texts:
            seqs.append([self.char2id.get(tok, OOV_ID) for tok in self._split(str(text))])
        return seqs

    def encode(self, texts: list[str], max_length: int | None = None) -> np.ndarray:
        """轉成定長 int32 陣列 (n, max_length),含截斷與補零。"""
        max_length = self.max_length if max_length is None else int(max_length)
        seqs = self.texts_to_sequences(texts)
        out = np.full((len(seqs), max_length), PAD_ID, dtype=np.int32)
        for i, seq in enumerate(seqs):
            if len(seq) > max_length:
                seq = seq[-max_length:] if self.truncating == "pre" else seq[:max_length]
            if not seq:
                continue
            if self.padding == "pre":
                out[i, max_length - len(seq) :] = seq
            else:
                out[i, : len(seq)] = seq
        return out

    # ---- 持久化 ----
    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "max_length": self.max_length,
            "oov_token": self.oov_token,
            "char_level": self.char_level,
            "lowercase": self.lowercase,
            "padding": self.padding,
            "truncating": self.truncating,
            "char2id": self.char2id,
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def from_dict(cls, data: dict) -> CharTokenizer:
        tok = cls(
            vocab_size=data["vocab_size"],
            max_length=data["max_length"],
            oov_token=data.get("oov_token", "<OOV>"),
            char_level=data.get("char_level", True),
            lowercase=data.get("lowercase", True),
            padding=data.get("padding", "post"),
            truncating=data.get("truncating", "post"),
        )
        tok.char2id = {k: int(v) for k, v in data["char2id"].items()}
        return tok

    @classmethod
    def load(cls, path: str | Path) -> CharTokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


def _WORD_SPLIT(text: str) -> list[str]:
    """word-level 的簡易切分:以非英數字元為分隔(URL 少有空白)。"""
    import re

    return re.split(r"[^a-z0-9]+", text) if text else []


def build_tokenizer(train_texts: list[str], features_cfg) -> CharTokenizer:
    """依 FeaturesConfig 在**訓練集**上建立並 fit 一個 CharTokenizer。"""
    tok = CharTokenizer(
        vocab_size=features_cfg.vocab_size,
        max_length=features_cfg.max_length,
        oov_token=features_cfg.oov_token,
        char_level=features_cfg.char_level,
        lowercase=features_cfg.lowercase,
        padding=features_cfg.padding,
        truncating=features_cfg.truncating,
    )
    return tok.fit(list(train_texts))

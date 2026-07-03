"""特徵工程:把 URL 字串轉成定長整數序列。"""

from __future__ import annotations

from urlguard.features.tokenizer import CharTokenizer, build_tokenizer

__all__ = ["CharTokenizer", "build_tokenizer"]

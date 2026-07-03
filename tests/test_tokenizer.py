"""CharTokenizer:編碼形狀、padding、截斷、OOV、僅訓練集 fit、save/load。"""

from __future__ import annotations

import numpy as np

from urlguard.config import FeaturesConfig
from urlguard.features import CharTokenizer, build_tokenizer
from urlguard.features.tokenizer import OOV_ID, PAD_ID


def _cfg(**kw):
    base = dict(vocab_size=50, max_length=10, char_level=True, lowercase=True)
    base.update(kw)
    return FeaturesConfig(**base)


def test_encode_shape_and_pad():
    tok = build_tokenizer(["abc", "abcdef"], _cfg(max_length=8))
    enc = tok.encode(["abc"])
    assert enc.shape == (1, 8)
    # 'abc' 只有 3 個字元,其餘應為 PAD(0)
    assert (enc[0, 3:] == PAD_ID).all()
    assert enc.dtype == np.int32


def test_truncation_post():
    tok = build_tokenizer(["abcdefghijklmnop"], _cfg(max_length=5, truncating="post"))
    enc = tok.encode(["abcdefghijklmnop"])
    assert enc.shape == (1, 5)
    assert (enc[0] != PAD_ID).all()  # 全被填滿(截斷,無 padding)


def test_oov_for_unseen_char():
    tok = build_tokenizer(["abc"], _cfg())
    seq = tok.texts_to_sequences(["z"])[0]
    assert seq == [OOV_ID]


def test_fit_only_sees_given_texts():
    tok = build_tokenizer(["aaa"], _cfg())
    # 'b' 未出現在訓練文字 → 不在詞彙表
    assert "b" not in tok.char2id
    assert "a" in tok.char2id


def test_num_tokens_and_max_id_bound():
    tok = build_tokenizer(["abcabc", "abcd"], _cfg())
    enc = tok.encode(["abcd", "zzz"])  # 含 OOV
    assert tok.num_tokens == 2 + len(tok.char2id)
    assert int(enc.max()) < tok.num_tokens  # Embedding input_dim 安全


def test_vocab_size_cap_respected():
    tok = build_tokenizer(["abcdefg"], _cfg(vocab_size=4))  # 2 保留 + 最多 2 字元
    assert len(tok.char2id) <= 2
    assert tok.num_tokens <= 4


def test_save_load_roundtrip(tmp_path):
    tok = build_tokenizer(["hello.com", "phish.ru"], _cfg())
    p = tmp_path / "tok.json"
    tok.save(p)
    tok2 = CharTokenizer.load(p)
    assert tok2.to_dict() == tok.to_dict()
    np.testing.assert_array_equal(tok.encode(["hello"]), tok2.encode(["hello"]))

"""資料層:標籤編碼、降採樣、載入。"""

from __future__ import annotations

import pandas as pd

from urlguard.data import add_label, downsample, load_balanced


def test_add_label_maps_benign_to_zero():
    df = pd.DataFrame(
        {"url": ["a", "b", "c", "d"], "type": ["benign", "phishing", "malware", "defacement"]}
    )
    out = add_label(df)
    assert out["label"].tolist() == [0, 1, 1, 1]


def test_add_label_is_case_insensitive():
    df = pd.DataFrame({"url": ["a", "b"], "type": ["Benign", "BENIGN"]})
    assert add_label(df)["label"].tolist() == [0, 0]


def test_downsample_balances_classes():
    df = pd.DataFrame(
        {"url": [str(i) for i in range(30)], "type": ["benign"] * 20 + ["phishing"] * 10}
    )
    df = add_label(df)
    balanced = downsample(df, seed=0)
    counts = balanced["label"].value_counts().to_dict()
    assert counts[0] == counts[1] == 10


def test_load_balanced_from_fixture(tiny_config):
    df = load_balanced(tiny_config)
    assert set(df["label"].unique()) == {0, 1}
    assert {"url", "type", "label"}.issubset(df.columns)


def test_load_balanced_is_deterministic(tiny_config):
    a = load_balanced(tiny_config)
    b = load_balanced(tiny_config)
    assert a["url"].tolist() == b["url"].tolist()

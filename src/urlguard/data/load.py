"""讀取資料集、二元標籤編碼、處理類別不平衡。

原始資料集欄位: url, type(benign / phishing / defacement / malware)。
標籤: benign -> 0,其餘一律 -> 1(惡意)。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BENIGN = "benign"
URL_COL = "url"
TYPE_COL = "type"
LABEL_COL = "label"


def read_raw(cfg) -> pd.DataFrame:
    """從 raw_dir/csv_name 讀取原始 CSV。"""
    csv_path = Path(cfg.data.raw_dir) / cfg.data.csv_name
    if not csv_path.exists():
        raise FileNotFoundError(
            f"找不到資料檔 {csv_path}。請先執行 `urlguard download-data`,"
            f"或確認 config 的 data.raw_dir / data.csv_name 設定正確。"
        )
    df = pd.read_csv(csv_path)
    missing = {URL_COL, TYPE_COL} - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要欄位: {sorted(missing)}(現有欄位: {list(df.columns)})")
    return df


def add_label(df: pd.DataFrame) -> pd.DataFrame:
    """新增 label 欄:benign -> 0,其餘 -> 1。"""
    out = df.copy()
    out[LABEL_COL] = (out[TYPE_COL].astype(str).str.lower() != BENIGN).astype(int)
    return out


def downsample(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """將多數類降採樣至與少數類同量,並打亂順序。

    比原 notebook『把 benign 砍到 malicious 數量』更穩健:不假設哪一類是多數,
    自動取兩類的最小數量。
    """
    if LABEL_COL not in df.columns:
        raise ValueError("downsample 前需先呼叫 add_label")
    groups = [g for _, g in df.groupby(LABEL_COL)]
    if len(groups) < 2:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = min(len(g) for g in groups)
    balanced = pd.concat([g.sample(n=n, random_state=seed) for g in groups])
    return balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_balanced(cfg) -> pd.DataFrame:
    """完整資料流程:讀檔 -> 標籤 -> (可選)降採樣 -> 洗牌。回傳含 url/type/label 的 DataFrame。"""
    df = add_label(read_raw(cfg))
    if cfg.data.downsample:
        df = downsample(df, seed=cfg.seed)
    else:
        df = df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    return df

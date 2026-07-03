"""端到端訓練 + artifact 存取。

修正原 notebook 的兩個問題:
  1. 資料洩漏:原 cell 6 在 train_test_split 之前對『全資料』fit tokenizer,
     測試集字元分佈洩漏進前處理。此處改為 **先切分、只用訓練集 url fit**。
  2. 未分層:train_test_split 補上 ``stratify``。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from urlguard.config import to_dict
from urlguard.data import load_balanced
from urlguard.evaluation.metrics import classification_metrics, recall_at_fpr
from urlguard.features import CharTokenizer, build_tokenizer
from urlguard.logging_utils import get_logger
from urlguard.model.build import build_model, compile_model
from urlguard.seed import set_seed

logger = get_logger(__name__)


def artifact_paths(cfg) -> dict[str, Path]:
    d = Path(cfg.artifacts_dir)
    return {
        "dir": d,
        "model": d / "model.keras",
        "tokenizer": d / "tokenizer.json",
        "metrics": d / "metrics.json",
        "history": d / "history.json",
        "config": d / "config.json",
    }


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(data), ensure_ascii=False, indent=2), encoding="utf-8")


def save_artifacts(cfg, model, tokenizer, metrics, history) -> dict[str, Path]:
    p = artifact_paths(cfg)
    p["dir"].mkdir(parents=True, exist_ok=True)
    model.save(p["model"])
    tokenizer.save(p["tokenizer"])
    _write_json(p["metrics"], metrics)
    _write_json(p["history"], history)
    _write_json(p["config"], to_dict(cfg))
    logger.info("已儲存 artifacts 至 %s", p["dir"])
    return p


def load_trained(artifacts_dir: str | Path) -> tuple[tf.keras.Model, CharTokenizer]:
    """載入已訓練的模型與 tokenizer。"""
    d = Path(artifacts_dir)
    model_path = d / "model.keras"
    tok_path = d / "tokenizer.json"
    if not model_path.exists() or not tok_path.exists():
        raise FileNotFoundError(
            f"在 {d} 找不到 model.keras / tokenizer.json,請先執行 `urlguard train`。"
        )
    model = tf.keras.models.load_model(model_path)
    tokenizer = CharTokenizer.load(tok_path)
    return model, tokenizer


def predict_proba(model, tokenizer: CharTokenizer, urls: list[str]) -> np.ndarray:
    """回傳每個 URL 的惡意機率(0~1)。"""
    encoded = tokenizer.encode([str(u) for u in urls])
    return model.predict(encoded, verbose=0).ravel()


def prepare_splits(cfg, df=None):
    """決定性地切出 train/test(url 字串層級)。train 與 evaluate/attack 共用,確保同一切分。"""
    if df is None:
        df = load_balanced(cfg)
    urls = df["url"].astype(str).tolist()
    labels = df["label"].astype(int).to_numpy()
    stratify = labels if cfg.train.stratify else None
    return train_test_split(
        urls,
        labels,
        test_size=cfg.train.test_size,
        random_state=cfg.seed,
        stratify=stratify,
    )


def reproduce_test_set(cfg, tokenizer, df=None):
    """用已存 tokenizer 重現訓練當時的測試集(編碼後)。回傳 (x_test, y_test, x_te_urls)。"""
    _, x_te_urls, _, y_test = prepare_splits(cfg, df)
    return tokenizer.encode(x_te_urls), y_test, x_te_urls


def train(cfg, df=None, save: bool = True) -> dict[str, Any]:
    """訓練並(可選)儲存 artifacts。回傳含 model/tokenizer/metrics/history 的 dict。"""
    set_seed(cfg.seed)
    x_tr_urls, x_te_urls, y_train, y_test = prepare_splits(cfg, df)
    logger.info("訓練集 %d 筆 / 測試集 %d 筆", len(x_tr_urls), len(x_te_urls))

    # 關鍵:tokenizer 只在訓練集 url 上 fit(修正洩漏)
    tokenizer = build_tokenizer(x_tr_urls, cfg.features)
    x_train = tokenizer.encode(x_tr_urls)
    x_test = tokenizer.encode(x_te_urls)
    logger.info("詞彙表大小(num_tokens)= %d", tokenizer.num_tokens)

    model = build_model(
        cfg.model, num_tokens=tokenizer.num_tokens, max_length=cfg.features.max_length
    )
    compile_model(model, cfg.train.learning_rate)

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=cfg.train.epochs,
        batch_size=cfg.train.batch_size,
        verbose=2,
    )

    y_prob = model.predict(x_test, verbose=0).ravel()
    metrics = classification_metrics(y_test, y_prob, threshold=cfg.train.threshold)
    metrics["recall_at_fpr_1pct"] = recall_at_fpr(y_test, y_prob, target_fpr=0.01)
    logger.info(
        "val: acc=%.4f  recall(malicious)=%.4f  ROC-AUC=%.4f  PR-AUC=%.4f",
        metrics["accuracy"],
        metrics["recall_malicious"],
        metrics["roc_auc"],
        metrics["pr_auc"],
    )

    result: dict[str, Any] = {
        "model": model,
        "tokenizer": tokenizer,
        "x_test": x_test,
        "y_test": y_test,
        "y_prob": y_prob,
        "metrics": metrics,
        "history": history.history,
    }
    if save:
        result["paths"] = save_artifacts(cfg, model, tokenizer, metrics, history.history)
    return result

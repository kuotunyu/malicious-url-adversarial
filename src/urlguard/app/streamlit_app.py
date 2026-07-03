"""Streamlit demo:輸入 URL → 即時惡意機率;並可用 epsilon slider 觀察對抗擾動。

以 `urlguard serve` 啟動(會設定環境變數 URLGUARD_ARTIFACTS)。此 app 只『載入』
已訓練的 artifact,絕不訓練;找不到模型時給明確提示。
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

EXAMPLES = [
    "google.com/search?q=weather",
    "github.com/pytorch/pytorch",
    "paypal-verify.g00gle-account.ru/login",
    "free-giftcard-amaz0n.com/claim?id=999",
    "download-free-movies.stream/setup.exe",
]


@st.cache_resource
def _load(artifacts_dir: str):
    from urlguard.model.train import load_trained

    return load_trained(artifacts_dir)


def _verdict(prob: float, threshold: float = 0.5) -> None:
    if prob >= threshold:
        st.error(f"🔴 惡意 (malicious) — 機率 {prob:.1%}")
    else:
        st.success(f"🟢 良性 (benign) — 惡意機率 {prob:.1%}")


def main() -> None:
    st.set_page_config(page_title="URLGuard — 惡意網址偵測 + 對抗攻防", page_icon="🛡️")
    st.title("🛡️ URLGuard")
    st.caption("惡意網址偵測 + FGSM/PGD 對抗式攻防 demo")

    artifacts_dir = os.environ.get("URLGUARD_ARTIFACTS", "artifacts")
    if not (Path(artifacts_dir) / "model.keras").exists():
        st.warning(
            f"找不到模型 artifact({artifacts_dir}/model.keras)。\n\n"
            "請先執行 `urlguard train --config configs/default.yaml`。"
        )
        st.stop()

    model, tokenizer = _load(artifacts_dir)

    st.subheader("1️⃣ 偵測")
    example = st.selectbox("範例 URL(或於下方自行輸入)", [""] + EXAMPLES)
    url = st.text_input("輸入網址", value=example or "")

    if url:
        prob = float(model.predict(tokenizer.encode([url]), verbose=0).ravel()[0])
        _verdict(prob)
        st.progress(min(max(prob, 0.0), 1.0))

        st.subheader("2️⃣ 對抗擾動(白箱 embedding 空間攻擊)")
        st.caption(
            "在 embedding 特徵空間對此 URL 施加擾動,觀察模型判斷如何被改變。"
            "注意:擾動後的 embedding 不對應可輸入的真實 URL,僅示範決策邊界脆弱性。"
        )
        col1, col2 = st.columns(2)
        method = col1.selectbox("攻擊方法", ["fgsm", "pgd"])
        epsilon = col2.slider("Epsilon(擾動強度)", 0.0, 0.5, 0.1, 0.01)

        if method == "pgd":
            from urlguard.attacks.pgd import pgd_attack

            res = pgd_attack(model, tokenizer.encode([url]), [1], epsilon, steps=10, alpha=0.02)
        else:
            from urlguard.attacks.fgsm import fgsm_attack

            res = fgsm_attack(model, tokenizer.encode([url]), [1], epsilon)

        adv_prob = float(res["adv_prob"][0])
        c1, c2 = st.columns(2)
        c1.metric("原始惡意機率", f"{prob:.1%}")
        c2.metric("對抗後惡意機率", f"{adv_prob:.1%}", delta=f"{(adv_prob - prob):.1%}")
        if prob >= 0.5 and adv_prob < 0.5:
            st.error("⚠️ 攻擊成功:模型被騙,把惡意 URL 判成良性(False Negative)")
        else:
            st.info("此強度尚未翻轉判斷,試著調大 epsilon 或改用 PGD。")


if __name__ == "__main__":
    main()

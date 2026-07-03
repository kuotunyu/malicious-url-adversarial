"""產生 tests/fixtures/sample_urls.csv —— 供 CI/單元測試使用的極小手標資料集。

刻意只有數十列且不含真實資料集,讓測試與 CI 不需下載 Kaggle、不需憑證,
且 repo 保持輕量。欄位與原始資料集一致: url, type。
"""

from __future__ import annotations

import csv
from pathlib import Path

BENIGN = [
    "google.com/search?q=weather",
    "github.com/pytorch/pytorch",
    "en.wikipedia.org/wiki/Machine_learning",
    "stackoverflow.com/questions/12345/how-to-parse-json",
    "amazon.com/dp/B0000000",
    "youtube.com/watch?v=dQw4w9WgXcQ",
    "news.ycombinator.com/item?id=1",
    "python.org/downloads/release/python-3120",
    "developer.mozilla.org/en-US/docs/Web/HTTP",
    "reddit.com/r/MachineLearning",
    "netflix.com/browse",
    "linkedin.com/in/some-user",
    "apple.com/tw/iphone",
    "microsoft.com/zh-tw/microsoft-365",
    "tensorflow.org/tutorials/keras/classification",
    "kaggle.com/datasets",
    "medium.com/@author/a-great-post",
    "nytimes.com/section/technology",
    "bbc.co.uk/news/world",
    "gov.tw/news/index.html",
    "scu.edu.tw/admission",
    "dropbox.com/home",
    "spotify.com/tw/premium",
    "booking.com/city/tw/taipei.html",
    "cloudflare.com/learning/ddos/what-is-a-ddos-attack",
]

MALICIOUS = [
    ("paypal-verify.g00gle-account.ru/login", "phishing"),
    ("secure-appleid.apple.com.verify-login.tk/signin", "phishing"),
    ("update-your-bank.hsbc.confirm-info.cn/index", "phishing"),
    ("free-giftcard-amaz0n.com/claim?id=999", "phishing"),
    ("microsoft-support-alert.win32-fix.info/call", "phishing"),
    ("192.168.99.13/wp-admin/hacked/shell.php", "defacement"),
    ("example-blog.com/wp-content/uploads/2018/x.php", "defacement"),
    ("victim-site.org/index.php?page=../../etc/passwd", "defacement"),
    ("cheapmeds-online.top/buy-now/viagra", "defacement"),
    ("download-free-movies.stream/setup.exe", "malware"),
    ("cracked-software-hub.ru/keygen/adobe.exe", "malware"),
    ("invoice-2024-pdf.attachment-open.zip.exe", "malware"),
    ("track-your-package-dhl.delivery-update.xyz/confirm", "phishing"),
    ("faceb00k-login-security.check-account.ml/home", "phishing"),
    ("netfl1x-billing-update.account-verify.ga/pay", "phishing"),
    ("bit.ly/3xF4kePhish", "phishing"),
    ("0nlinebanking-chase.secure-session.cc/auth", "phishing"),
    ("win-a-free-iphone15.prize-claim.click/now", "phishing"),
    ("adobe-flashplayer-update.download-now.pw/flash.exe", "malware"),
    ("crypto-wallet-recovery.metamask-support.io/seed", "phishing"),
    ("router-firmware-update.mikrotik-fix.su/upgrade.bin", "malware"),
    ("myuniversity-portal.login-secure.online/sso", "phishing"),
    ("steamcommunity-tradeoffer.gift-cs2.ru/claim", "phishing"),
    ("dhl-express.tracking-id-88213.top/label.php", "defacement"),
    ("unsubscribe-now.click-here-to-claim.buzz/reward", "phishing"),
]


def build_rows() -> list[dict[str, str]]:
    rows = [{"url": u, "type": "benign"} for u in BENIGN]
    rows += [{"url": u, "type": t} for u, t in MALICIOUS]
    return rows


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "sample_urls.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "type"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"寫入 {len(rows)} 列 -> {out}")


if __name__ == "__main__":
    main()

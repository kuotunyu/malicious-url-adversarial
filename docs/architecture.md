# 架構與設計

## 從 notebook 到套件

原始講座 notebook(保留於 [`notebooks/01_workshop_original.ipynb`](../notebooks/01_workshop_original.ipynb))
把資料處理、訓練、攻擊全塞在單一檔案。本專案將其重構為可安裝、可測試、可由 CLI 驅動的套件。

## 模組邊界

| 模組 | 職責 | 依賴 |
|---|---|---|
| `config` | dataclass 設定樹 + YAML 載入 + 點路徑覆蓋 | PyYAML |
| `data` | 下載 / 讀取 / 標籤編碼 / 降採樣 | pandas(**不依賴 TF**) |
| `features` | `CharTokenizer` 字元級編碼 | 純 Python / numpy(**不依賴 TF**) |
| `model` | 建立 / 訓練 / artifact 存取 | TensorFlow |
| `attacks` | FGSM / PGD / epsilon sweep | TensorFlow |
| `defenses` | 對抗訓練 | TensorFlow |
| `evaluation` | 指標(sklearn)/ 繪圖(matplotlib) | sklearn / matplotlib(**不依賴 TF**) |
| `app` | Streamlit demo | streamlit |
| `cli` | Typer 指令入口(唯一解析參數處) | typer |

**設計原則**:資料與特徵層刻意不依賴 TensorFlow,好處是這些層可在沒有 TF 的環境
獨立單元測試,CI 也能快速跑純模組測試。TF 相關測試以 `pytest.importorskip("tensorflow")`
在缺 TF 時自動 skip,在完整環境(CI)則跑端到端 smoke test。

## 我在原始 workshop 上修正的問題

1. **資料洩漏(最關鍵)**:原 notebook `tokenizer.fit_on_texts(df_balanced['url'])` 在
   `train_test_split` **之前**對全資料擬合,測試集的字元分佈洩漏進前處理,使驗證分數偏樂觀。
   → `model/train.py` 改為**先切分、只用訓練集 url fit**(見 `prepare_splits` / `build_tokenizer`)。
2. **未分層抽樣**:`train_test_split` 補上 `stratify`,確保 train/test 類別比例一致。
3. **已棄用參數**:`Embedding(input_length=...)` 在 Keras 3 / TF 2.20 被忽略 → 改用顯式 `Input(shape=(max_length,))`。
4. **脆弱的攻擊子模型**:原 notebook 以即興迴圈重建 `model.layers[1:]`;改為
   `attacks/base.build_classifier_from_embeddings` 顯式、決定性地重建,並可單元測試。
5. **magic number**:vocab_size / max_length / epsilon / seed 等全部收斂到 `configs/*.yaml`。

## Artifact 佈局

```
artifacts/
├── model.keras          # 訓練好的模型
├── tokenizer.json       # CharTokenizer(含詞彙表)
├── metrics.json         # 訓練時的評估指標
├── history.json         # 訓練 loss 曲線
├── config.json          # 該次實驗的完整設定(可復現)
├── eval_metrics.json    # urlguard evaluate 產生
├── attack_results.json  # urlguard attack 產生(含 epsilon sweep)
├── defense_results.json # urlguard adv-train 產生(防禦前後對照)
└── robust/              # 對抗訓練後的模型
```

資料集與 artifacts 皆不入 git(見 `.gitignore`);測試改用 `tests/fixtures/sample_urls.csv`(約 50 列)。

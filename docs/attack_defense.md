# 對抗式攻防:FGSM / PGD 與對抗訓練

## 為什麼在 embedding 空間攻擊

URL 是**離散字元序列**,無法直接對輸入做梯度下降(字元沒有連續的「中間值」)。
因此我們對模型內部第一層 `Embedding` 的**輸出向量**施加擾動。這在文獻上稱為
feature-space / latent-space 攻擊,足以證明分類器的**決策邊界是脆弱的**。

`attacks/base.build_classifier_from_embeddings` 會把訓練好的模型拆成
「embedding 向量 → 惡意機率」的子模型(重用已訓練層、共享權重),攻擊即對此子模型的輸入求梯度。

## FGSM(Fast Gradient Sign Method)

單步攻擊:

$$adv = emb + \epsilon \cdot \text{sign}\big(\nabla_{emb} J(\theta, emb, y)\big)$$

- 訓練是**降低** loss;攻擊是沿 `sign(∇J)` 方向**增大** loss。
- 對惡意樣本(label=1),增大 BCE loss 等於把預測機率往 0(良性)推 → 欺騙模型。
- `epsilon` 控制擾動強度:越大越容易成功,但失真也越大。

### 從「攻擊失敗」到「攻擊強度分析」

原講座 notebook 在 `epsilon=0.05` 下惡意機率從 0.976 掉到 0.615,因為仍 > 0.5 而判定
「攻擊失敗」。但這其實是**巨大的信心崩塌**,只是還沒越界。正確的呈現方式是
**epsilon 掃描**:對整批高信心惡意樣本掃 `epsilon ∈ {0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3}`,
計算**攻擊成功率**(原本被正確偵測、現在被翻成良性的比例),畫成曲線。

> 攻擊成功率只計算「label=1 且原始機率 ≥ 門檻」的樣本,避免把本來就分錯的樣本灌水。

## PGD(Projected Gradient Descent)

FGSM 的多步強化版:

1. 每步前進一小步 `alpha`:`adv ← adv + alpha · sign(∇J)`
2. 投影回以原點為中心、半徑 `epsilon` 的 **L∞ 球**:`adv ← emb₀ + clip(adv − emb₀, −ε, +ε)`
3. 重複 `k` 步。

PGD 通常能把惡意機率壓破 0.5,示範一個**真正成功**的攻擊,也建立攻擊強度的階梯(FGSM < PGD)。

## 對抗訓練(Adversarial Training)

防禦方法:在訓練時就讓模型「見過」對抗樣本。

- 每個 batch 於 embedding 空間即時生成 FGSM/PGD 擾動,以混合損失更新**所有**權重:

$$L = (1-w)\cdot \text{BCE}(y, f(emb)) + w \cdot \text{BCE}(y, f(emb + \delta))$$

- 擾動方向 `δ` 以 `tf.stop_gradient` 視為**常數**——這是對抗訓練的標準做法:我們要模型對
  「當前這個擾動」變 robust,而不是去對擾動本身求導。
- 攻擊與防禦都在同一 embedding 空間,因此可用**同一條 epsilon 掃描**直接比較 vanilla vs 對抗訓練模型。

### 權衡代價

對抗訓練通常以**乾淨準確率略降**換取 robustness 提升。`urlguard adv-train` 會同時報告
兩個模型的乾淨 accuracy 與 robust accuracy,把這個 trade-off 攤開來看。

## 限制與威脅模型(誠實聲明)

- 本專案的攻擊是**白箱**、需要梯度存取。
- 擾動發生在**連續 embedding 空間**,擾動後的向量**不一定對應任何可輸入的真實 URL 字串**,
  因此屬於「證明脆弱性」的示範,而非可部署的端到端攻擊。
- 真實世界的離散字元級黑箱攻擊(HotFlip、TextFooler、homoglyph/typosquatting 改寫)是未來工作。

## 面試常見問題對照

| 問題 | 對應證據 |
|---|---|
| 白箱還黑箱?攻在哪一層? | 本文件威脅模型 + `attacks/base.py` |
| eps=0.05 沒成功,怎麼證明模型脆弱? | epsilon 掃描曲線(`urlguard attack --sweep`) |
| 離散文字怎麼做梯度攻擊? | embedding 空間攻擊 + 上述限制聲明 |
| 對抗訓練的代價? | `defense_results.json` 的乾淨 vs robust accuracy |
| val_acc 有沒有資料洩漏? | 已修正 tokenizer fit 順序(見 architecture.md) |
| accuracy 0.94 在資安夠好嗎? | PR-AUC + 固定低 FPR 下的 recall |

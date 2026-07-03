"""對抗式攻擊(embedding 特徵空間白箱)。

因為 URL 是離散字元、無法直接對輸入做梯度下降,我們對模型內部的 Embedding
向量施加擾動,證明決策邊界的脆弱性。這屬於白箱、需梯度存取的攻擊示範,
擾動後的 embedding 不一定對應任何可輸入的真實 URL —— 此限制在 docs 中明確標註。
"""

from __future__ import annotations

from urlguard.attacks.base import build_classifier_from_embeddings, embed
from urlguard.attacks.fgsm import attack_success_rate, epsilon_sweep, fgsm_attack
from urlguard.attacks.pgd import pgd_attack, pgd_perturb

__all__ = [
    "build_classifier_from_embeddings",
    "embed",
    "fgsm_attack",
    "epsilon_sweep",
    "attack_success_rate",
    "pgd_attack",
    "pgd_perturb",
]

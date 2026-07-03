"""防禦:對抗訓練(在同一 embedding 空間硬化模型)。"""

from __future__ import annotations

from urlguard.defenses.adv_training import adversarial_train, robustness_report

__all__ = ["adversarial_train", "robustness_report"]

"""從 Kaggle 下載資料集(冪等:已存在則跳過)。"""

from __future__ import annotations

import shutil
from pathlib import Path

from urlguard.logging_utils import get_logger

logger = get_logger(__name__)


def download_dataset(cfg) -> Path:
    """下載資料集並把目標 CSV 複製到 cfg.data.raw_dir。回傳 CSV 路徑。

    需要 Kaggle 憑證(~/.kaggle/kaggle.json 或 KAGGLE_USERNAME/KAGGLE_KEY)。
    kagglehub 以延遲方式匯入,讓不需下載的流程不必安裝它。
    """
    raw_dir = Path(cfg.data.raw_dir)
    dest = raw_dir / cfg.data.csv_name
    if dest.exists():
        logger.info("資料已存在,跳過下載: %s", dest)
        return dest

    import kagglehub  # 延遲匯入

    logger.info("正在從 Kaggle 下載資料集: %s", cfg.data.dataset)
    path = Path(kagglehub.dataset_download(cfg.data.dataset))
    logger.info("下載完成: %s", path)

    src = path / cfg.data.csv_name
    if not src.exists():
        candidates = list(path.rglob(cfg.data.csv_name))
        if not candidates:
            raise FileNotFoundError(
                f"下載目錄 {path} 中找不到 {cfg.data.csv_name}。"
                f"目錄內容: {[p.name for p in path.iterdir()]}"
            )
        src = candidates[0]

    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    logger.info("已複製到: %s", dest)
    return dest

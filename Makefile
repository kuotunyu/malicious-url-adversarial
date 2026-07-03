# 開發常用指令入口。Windows 使用者若無 make,可直接執行每個 target 對應的指令。
.PHONY: setup data lint format type test train evaluate attack adv-train serve plots ci clean

PYTHON ?= python
EPS ?= 0.1

setup:            ## 建立虛擬環境並安裝(含 dev/app)+ pre-commit
	$(PYTHON) -m pip install -e ".[dev,app]"
	pre-commit install || true

data:             ## 從 Kaggle 下載資料集(需 ~/.kaggle/kaggle.json)
	urlguard download-data

lint:             ## 靜態檢查(不修改)
	ruff check .
	black --check .

format:           ## 自動修正格式
	ruff check --fix .
	black .

type:             ## 型別檢查
	mypy src/urlguard

test:             ## 跑測試 + 覆蓋率(使用 tiny fixture,不下載資料)
	pytest --cov=urlguard --cov-report=term-missing

train:            ## 以完整設定訓練
	urlguard train --config configs/default.yaml

evaluate:         ## 評估並輸出圖表
	urlguard evaluate --config configs/default.yaml

attack:           ## FGSM epsilon sweep(可用 EPS 覆蓋單點)
	urlguard attack --config configs/default.yaml --method fgsm --sweep

adv-train:        ## 對抗訓練(PGD)並重新評估 robustness
	urlguard adv-train --config configs/default.yaml --epsilon $(EPS)

serve:            ## 啟動 Streamlit demo
	urlguard serve

ci:               ## 在本機重現 CI(lint + type + test)
	$(MAKE) lint
	$(MAKE) type
	$(MAKE) test

clean:            ## 清除快取與產物
	rm -rf artifacts .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +

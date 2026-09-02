"""urlguard 命令列介面(Typer)。

子命令:
    download-data  從 Kaggle 下載資料集
    train          訓練模型並存 artifacts
    evaluate       載入模型、輸出混淆矩陣 / ROC / PR 圖與指標
    attack         FGSM/PGD 攻擊 + epsilon 掃描
    adv-train      對抗訓練並比較防禦前後 robustness
    serve          啟動 Streamlit demo

所有指令都吃 --config,並可用可重複的 --set key=value 覆蓋(如 --set attack.epsilon=0.1)。
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from urlguard import __version__
from urlguard.config import load_config
from urlguard.logging_utils import get_logger

app = typer.Typer(add_completion=False, help="惡意網址偵測 + 對抗式攻防工具")
logger = get_logger("urlguard.cli")

ConfigOpt = typer.Option("configs/default.yaml", "--config", "-c", help="設定檔路徑")
SetOpt = typer.Option(None, "--set", help="覆蓋設定,可重複。例: --set attack.epsilon=0.1")


def _load(config: str, set_: list[str] | None):
    cfg = load_config(config, overrides=set_)
    return cfg


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


@app.command()
def version() -> None:
    """顯示版本。"""
    typer.echo(f"urlguard {__version__}")


@app.command("download-data")
def download_data(config: str = ConfigOpt, set_: list[str] | None = SetOpt) -> None:
    """從 Kaggle 下載資料集到 data.raw_dir(冪等)。"""
    from urlguard.data.download import download_dataset

    cfg = _load(config, set_)
    dest = download_dataset(cfg)
    typer.echo(f"資料就緒: {dest}")


@app.command()
def train(config: str = ConfigOpt, set_: list[str] | None = SetOpt) -> None:
    """訓練模型並儲存 artifacts。"""
    from urlguard.model.train import train as run_train

    cfg = _load(config, set_)
    result = run_train(cfg, save=True)
    m = result["metrics"]
    typer.echo(
        f"完成。val accuracy={m['accuracy']:.4f}  recall(malicious)={m['recall_malicious']:.4f}  "
        f"ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}"
    )
    typer.echo(f"artifacts: {result['paths']['dir']}")


@app.command()
def evaluate(config: str = ConfigOpt, set_: list[str] | None = SetOpt) -> None:
    """載入已訓練模型,輸出混淆矩陣 / ROC / PR 圖與指標。"""
    from urlguard.evaluation import plots
    from urlguard.evaluation.metrics import classification_metrics, recall_at_fpr
    from urlguard.model.train import load_trained, reproduce_test_set

    cfg = _load(config, set_)
    model, tokenizer = load_trained(cfg.artifacts_dir)
    x_test, y_test, _ = reproduce_test_set(cfg, tokenizer)
    y_prob = model.predict(x_test, verbose=0).ravel()

    metrics = classification_metrics(y_test, y_prob, threshold=cfg.train.threshold)
    metrics["recall_at_fpr_1pct"] = recall_at_fpr(y_test, y_prob, 0.01)

    img = cfg.images_path
    plots.plot_confusion_matrix(metrics["confusion_matrix"], img / "confusion_matrix.png")
    plots.plot_roc_pr(y_test, y_prob, img / "roc_pr.png")
    _write_json(cfg.artifacts_path / "eval_metrics.json", metrics)

    typer.echo(
        f"accuracy={metrics['accuracy']:.4f}  recall(malicious)={metrics['recall_malicious']:.4f}  "
        f"ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}"
    )
    typer.echo(f"recall@FPR=1%: {metrics['recall_at_fpr_1pct']}")
    typer.echo(f"圖已輸出至 {img}")


@app.command()
def attack(
    config: str = ConfigOpt,
    set_: list[str] | None = SetOpt,
    method: str = typer.Option("fgsm", help="fgsm 或 pgd"),
    sweep: bool = typer.Option(True, help="是否做 epsilon 掃描"),
) -> None:
    """對高信心惡意樣本發動攻擊,輸出攻擊結果與圖。"""
    import numpy as np

    from urlguard.attacks.fgsm import epsilon_sweep, fgsm_attack
    from urlguard.attacks.pgd import pgd_attack
    from urlguard.evaluation import plots
    from urlguard.model.train import load_trained, reproduce_test_set

    cfg = _load(config, set_)
    model, tokenizer = load_trained(cfg.artifacts_dir)
    x_test, y_test, _ = reproduce_test_set(cfg, tokenizer)
    y_prob = model.predict(x_test, verbose=0).ravel()

    # 挑「模型高信心且正確」的惡意樣本
    target_mask = (y_test == 1) & (y_prob >= cfg.attack.confidence_threshold)
    if not target_mask.any():
        target_mask = y_test == 1
        logger.warning("無高信心惡意樣本,改用全部惡意樣本")
    idx = np.where(target_mask)[0]
    x_target, y_target = x_test[idx], y_test[idx]
    typer.echo(f"攻擊目標樣本數: {len(idx)}(method={method})")

    out: dict = {"method": method, "n_targets": int(len(idx))}

    # 單點攻擊(第一個樣本)供 before/after bar
    single_x = x_target[:1]
    single_y = y_target[:1]
    if method == "pgd":
        single = pgd_attack(
            model,
            single_x,
            single_y,
            cfg.attack.epsilon,
            steps=cfg.attack.pgd_steps,
            alpha=cfg.attack.pgd_alpha,
        )
    else:
        single = fgsm_attack(model, single_x, single_y, cfg.attack.epsilon)
    orig_p = float(single["orig_prob"][0])
    adv_p = float(single["adv_prob"][0])
    out["single_point"] = {"epsilon": cfg.attack.epsilon, "orig_prob": orig_p, "adv_prob": adv_p}
    plots.plot_attack_bar(orig_p, adv_p, cfg.attack.epsilon, cfg.images_path / "attack_bar.png")
    typer.echo(f"單點 epsilon={cfg.attack.epsilon}: {orig_p:.4f} -> {adv_p:.4f}")

    if sweep:
        sweep_res = epsilon_sweep(
            model,
            x_target,
            y_target,
            cfg.attack.sweep,
            threshold=cfg.train.threshold,
            method=method,
            pgd_steps=cfg.attack.pgd_steps,
            pgd_alpha=cfg.attack.pgd_alpha,
        )
        out["sweep"] = sweep_res
        plots.plot_epsilon_sweep({method: sweep_res}, cfg.images_path / "epsilon_sweep.png")
        for r in sweep_res["results"]:
            typer.echo(
                f"  eps={r['epsilon']:<5} success={r['success_rate']:.2%}  "
                f"mean_adv_prob={r['mean_adv_prob']:.3f}"
            )

    _write_json(cfg.artifacts_path / "attack_results.json", out)
    typer.echo(f"攻擊結果已存至 {cfg.artifacts_path / 'attack_results.json'}")


@app.command("adv-train")
def adv_train(
    config: str = ConfigOpt,
    set_: list[str] | None = SetOpt,
    epsilon: float | None = typer.Option(None, help="覆蓋 defense.epsilon"),
) -> None:
    """對抗訓練,並比較防禦前後 robustness。"""
    from urlguard.defenses.adv_training import adversarial_train, robustness_report
    from urlguard.evaluation import plots
    from urlguard.model.train import load_trained, prepare_splits, reproduce_test_set

    overrides = list(set_ or [])
    if epsilon is not None:
        overrides.append(f"defense.epsilon={epsilon}")
    cfg = _load(config, overrides)

    baseline, tokenizer = load_trained(cfg.artifacts_dir)
    x_tr_urls, _, y_train, _ = prepare_splits(cfg)
    x_train = tokenizer.encode(x_tr_urls)
    x_test, y_test, _ = reproduce_test_set(cfg, tokenizer)

    typer.echo("開始對抗訓練...")
    robust = adversarial_train(
        cfg, x_train, y_train, num_tokens=tokenizer.num_tokens, max_length=cfg.features.max_length
    )

    # 存 robust 模型
    robust_dir = cfg.artifacts_path / "robust"
    robust_dir.mkdir(parents=True, exist_ok=True)
    robust.save(robust_dir / "model.keras")
    tokenizer.save(robust_dir / "tokenizer.json")

    rep_base = robustness_report(baseline, x_test, y_test, cfg, label="vanilla")
    rep_robust = robustness_report(robust, x_test, y_test, cfg, label="adversarially-trained")
    comparison = {"epsilon": cfg.defense.epsilon, "vanilla": rep_base, "robust": rep_robust}
    _write_json(cfg.artifacts_path / "defense_results.json", comparison)

    plots.plot_robustness_comparison(
        rep_base["sweep"], rep_robust["sweep"], cfg.images_path / "robustness_before_after.png"
    )

    typer.echo(f"--- 防禦前後對照(eps={cfg.defense.epsilon:.3f})---")
    for rep in (rep_base, rep_robust):
        typer.echo(
            f"  {rep['label']:<24} clean_acc={rep['clean_accuracy']:.4f}  "
            f"robust_acc(FGSM)={rep['robust_acc_fgsm']:.4f}  robust_acc(PGD)={rep['robust_acc_pgd']:.4f}"
        )
    typer.echo(f"結果已存至 {cfg.artifacts_path / 'defense_results.json'}")


@app.command()
def serve(
    config: str = ConfigOpt,
    set_: list[str] | None = SetOpt,
    port: int = typer.Option(8501, help="Streamlit 埠號"),
) -> None:
    """啟動 Streamlit demo(需要 pip install .[app])。"""
    import os
    import subprocess
    import sys

    cfg = _load(config, set_)
    app_path = Path(__file__).parent / "app" / "streamlit_app.py"
    env = dict(os.environ)
    env["URLGUARD_ARTIFACTS"] = str(cfg.artifacts_dir)
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)]
    typer.echo(f"啟動 Streamlit: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=False)


if __name__ == "__main__":
    app()

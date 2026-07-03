"""CLI smoke test。--help 只需 typer;train/attack 端到端需要 TensorFlow。"""

from __future__ import annotations

import pytest

typer = pytest.importorskip("typer")
from typer.testing import CliRunner  # noqa: E402

runner = CliRunner()


def test_help_lists_subcommands():
    from urlguard.cli import app

    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ["download-data", "train", "evaluate", "attack", "adv-train", "serve"]:
        assert cmd in result.output


def test_version():
    from urlguard.cli import app

    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "urlguard" in result.output


def test_train_then_attack_end_to_end(tmp_path):
    pytest.importorskip("tensorflow")
    from urlguard.cli import app

    art = str(tmp_path / "artifacts")
    img = str(tmp_path / "images")
    common = [
        "--config",
        "configs/fast_ci.yaml",
        "--set",
        f"artifacts_dir={art}",
        "--set",
        f"images_dir={img}",
    ]

    r_train = runner.invoke(app, ["train", *common])
    assert r_train.exit_code == 0, r_train.output
    assert (tmp_path / "artifacts" / "model.keras").exists()
    assert (tmp_path / "artifacts" / "tokenizer.json").exists()

    r_attack = runner.invoke(app, ["attack", *common, "--method", "fgsm"])
    assert r_attack.exit_code == 0, r_attack.output
    assert (tmp_path / "artifacts" / "attack_results.json").exists()

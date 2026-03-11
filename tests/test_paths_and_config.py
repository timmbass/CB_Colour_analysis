from pathlib import Path

import pytest

from src.config import load_analysis_config, load_calibration_config, load_dynamic_color_config
from src.copy_renderer import load_copy_rules
from src.paths import COPY_RULES_PATH, DYNAMIC_COLORS_CONFIG_PATH, ROOT_DIR, resolve_repo_path


def test_resolve_repo_path_from_relative() -> None:
    assert resolve_repo_path("dynamic_colors_config.json") == DYNAMIC_COLORS_CONFIG_PATH


def test_copy_rules_load_from_repo_relative_path_after_cwd_change(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(Path("/tmp"))
    rules = load_copy_rules("assets/copy_rules.v1.json")
    assert "templates" in rules


def test_dynamic_color_config_loads_from_repo_relative_path_after_cwd_change(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(Path("/tmp"))
    cfg = load_dynamic_color_config("dynamic_colors_config.json")
    assert "families" in cfg


def test_load_analysis_config_validates_structure(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad.json"
    cfg_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        load_analysis_config(cfg_path)


def test_load_calibration_config_uses_defaults_for_missing_file(tmp_path: Path) -> None:
    cfg = load_calibration_config(tmp_path / "missing.json")
    assert cfg.source == "default"
    assert cfg.gamma == 3.0


def test_repo_root_contains_expected_assets() -> None:
    assert ROOT_DIR.joinpath(COPY_RULES_PATH.relative_to(ROOT_DIR)).exists()

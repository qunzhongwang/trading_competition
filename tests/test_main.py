"""Tests for main.py — config loading and component initialization."""

from __future__ import annotations

import pytest

from main import load_config


class TestLoadConfig:
    def test_loads_default_config(self):
        config = load_config("config/default.yaml")
        assert config["mode"] == "paper"
        assert "symbols" in config
        assert "alpha" in config

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_config_has_required_sections(self):
        config = load_config("config/default.yaml")
        required = [
            "mode",
            "symbols",
            "exchange",
            "data",
            "features",
            "alpha",
            "strategy",
            "risk",
            "execution",
            "paper",
        ]
        for key in required:
            assert key in config, f"Missing config section: {key}"

    def test_model_path_points_to_artifacts(self):
        config = load_config("config/default.yaml")
        model_path = config["alpha"]["model_path"]
        assert "artifacts/" in model_path

    def test_symbols_are_list(self):
        config = load_config("config/default.yaml")
        assert isinstance(config["symbols"], list)
        assert len(config["symbols"]) > 0

    def test_risk_limits_present(self):
        config = load_config("config/default.yaml")
        risk = config["risk"]
        assert "max_portfolio_exposure" in risk
        assert "trailing_stop_pct" in risk
        assert "daily_drawdown_limit" in risk

from pathlib import Path

import pytest

from tests.shared import model_getter


def _configure_missing_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point model lookup at an empty LM Studio home."""
    pointer = tmp_path / ".lmstudio-home-pointer"
    pointer.write_text(str(tmp_path / "lmstudio"))
    original_expanduser = Path.expanduser

    def expanduser(path: Path) -> Path:
        if str(path) == "~/.lmstudio-home-pointer":
            return pointer
        return original_expanduser(path)

    monkeypatch.setattr(Path, "expanduser", expanduser)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test")
    monkeypatch.delenv("MLX_ENGINE_TEST_DOWNLOAD_MODELS", raising=False)


def test_model_getter_skips_missing_fixture_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Automated tests skip missing model fixtures without reading stdin."""
    _configure_missing_model(monkeypatch, tmp_path)
    monkeypatch.delenv("MLX_ENGINE_TEST_REQUIRE_MODELS", raising=False)

    with pytest.raises(pytest.skip.Exception, match="model fixture is not installed"):
        model_getter("example/model")


def test_model_getter_fails_when_models_are_required(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Required-model runs fail clearly when a fixture is unavailable."""
    _configure_missing_model(monkeypatch, tmp_path)
    monkeypatch.setenv("MLX_ENGINE_TEST_REQUIRE_MODELS", "1")

    with pytest.raises(
        pytest.fail.Exception, match="required model fixture is missing"
    ):
        model_getter("example/model")

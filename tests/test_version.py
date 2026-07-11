from pathlib import Path

from mlx_engine import version as version_module


def test_runtime_version_reads_revision_from_installed_runtime(
    monkeypatch, tmp_path: Path
) -> None:
    """Version reporting reads REVISION beside the installed virtualenv."""
    runtime_dir = tmp_path / "runtime"
    venv_dir = runtime_dir / ".venv"
    venv_dir.mkdir(parents=True)
    (runtime_dir / "REVISION").write_text("abc123\n")
    monkeypatch.setattr(version_module.sys, "prefix", str(venv_dir))
    monkeypatch.setattr(version_module, "version", lambda _: "1.2.3")

    assert version_module.runtime_version() == "1.2.3+abc123"

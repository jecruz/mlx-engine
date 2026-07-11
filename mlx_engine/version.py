"""Installed runtime version and source revision reporting."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys


def runtime_version() -> str:
    """Return the installed package version and captured source revision."""
    try:
        package_version = version("mlx-engine-internal")
    except PackageNotFoundError:
        package_version = "development"
    revision_candidates = (
        Path(sys.prefix).resolve().parent / "REVISION",
        Path(__file__).resolve().parent.parent / "REVISION",
    )
    revision_path = next((path for path in revision_candidates if path.exists()), None)
    revision = revision_path.read_text().strip() if revision_path else "unknown"
    return f"{package_version}+{revision}"


def main() -> None:
    """Print the installed runtime version for deployment verification."""
    print(runtime_version())

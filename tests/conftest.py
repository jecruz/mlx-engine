import os

import pytest


def pytest_addoption(parser):
    """Add command-line options for heavy and model-backed tests."""
    parser.addoption(
        "--heavy",
        action="store_true",
        default=False,
        help="run heavy tests (e.g., tests that require large models or long execution time)",
    )
    parser.addoption(
        "--require-models",
        action="store_true",
        default=False,
        help="fail instead of skip when a required local model fixture is missing",
    )
    parser.addoption(
        "--download-models",
        action="store_true",
        default=False,
        help="explicitly allow lms to download missing model fixtures",
    )


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "heavy: mark test as heavy (requires --heavy option to run)"
    )
    config.addinivalue_line(
        "markers", "model: test uses an installed local model fixture"
    )
    if config.getoption("--require-models"):
        os.environ["MLX_ENGINE_TEST_REQUIRE_MODELS"] = "1"
    if config.getoption("--download-models"):
        os.environ["MLX_ENGINE_TEST_DOWNLOAD_MODELS"] = "1"


def pytest_collection_modifyitems(config, items):
    """Skip heavy tests unless --heavy option is provided."""
    if config.getoption("--heavy"):
        # --heavy given in cli: do not skip heavy tests
        return

    skip_heavy = pytest.mark.skip(reason="need --heavy option to run")
    for item in items:
        if "heavy" in item.keywords:
            item.add_marker(skip_heavy)

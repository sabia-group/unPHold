from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--data-dir",
        default=None,
        help="Root of test data (default: tests/data/ inside the repo)",
    )


@pytest.fixture
def data_dir(request):
    d = request.config.getoption("--data-dir")
    return Path(d) if d is not None else Path(__file__).parent / "data"

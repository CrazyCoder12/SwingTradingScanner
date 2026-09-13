import os
import subprocess
import sys

import pytest

from swing_trading_scanner.cli import COMMANDS
from swing_trading_scanner.config import require_credentials


def test_missing_credentials_are_actionable():
    with pytest.raises(ValueError, match="ALPACA_API_KEY"):
        require_credentials()


def test_exported_credentials_are_used(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
    assert require_credentials() == ("test-key", "test-secret")


@pytest.mark.parametrize("command", [None, *COMMANDS])
def test_help_works_without_credentials(command, tmp_path):
    args = [sys.executable, "-m", "swing_trading_scanner"]
    if command:
        args.append(command)
    result = subprocess.run(
        args + ["--help"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()

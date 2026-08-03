from typer.testing import CliRunner

from rekka_ai.cli import app

runner = CliRunner()


def test_detect_prints_greeting() -> None:
    result = runner.invoke(app, ["detect"])
    assert result.exit_code == 0
    assert "Hello from rekka-ai" in result.stdout

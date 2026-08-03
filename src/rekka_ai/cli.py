import typer

app = typer.Typer(help="Truck detection from aerial images.")


@app.callback()
def main() -> None:
    """Keep subcommand dispatch even with a single command registered."""


@app.command()
def detect() -> None:
    """Print a greeting."""
    typer.echo("Hello from rekka-ai!")

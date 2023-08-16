import typer
from loguru import logger
from rich.console import Console

from velocount import __version__
from velocount.commands import dropest_bc_correct, run, run10x, run_dropest, run_smartseq2

# install(show_locals=True, width=300, extra_lines=6, word_wrap=True)

logger.remove()

console = Console()


def version_callback(value: bool) -> None:
    """Prints the version of the package."""
    if value:
        console.print(f"[yellow]fcsjanitor[/] version: [bold blue]{__version__}[/]")
        raise typer.Exit()


velocount = typer.Typer(
    name="velocount",
    help=("Process scRNA-seq data for RNA velocity analysis"),
    add_completion=False,
    rich_markup_mode="markdown",
    no_args_is_help=True,
)

velocount.add_typer(run10x.app, name="run10x")
velocount.add_typer(run.app, name="run")
velocount.add_typer(run_smartseq2.app, name="run_smartseq2")
velocount.add_typer(dropest_bc_correct.app, name="dropest_bc_correct")

# velocount.add_typer(run_smartseq2.app, name="runsmartseq2")
velocount.add_typer(run_dropest.app, name="rundropest")

# @velocount.command(no_args_is_help=True)
# def velocount_help():
# pass

if __name__ == "__main__":
    velocount()

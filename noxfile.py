"""Nox sessions."""
import shutil
import sys
from pathlib import Path
from textwrap import dedent

import nox

package = "velocyto"
python_versions = ["3.10"]
nox.needs_version = ">= 2021.6.6"
nox.options.sessions = (
    "pre-commit",
    "mypy",
    "tests",
    "typeguard",
    "xdoctest",
    "docs-build",
)

locations = (
    "src",
    "tests",
)


@nox.session(python=["3.10"])
def lint(session: nox.Session) -> None:
    """Run ruff code formatter."""
    check_files = ["src", "tests", "doc", "noxfile.py"]
    session.install("ruff >=0.5.1")
    session.run("ruff", "check", "--fix", *check_files)
    session.run("ruff", "format", "--diff", *check_files)


def activate_virtualenv_in_precommit_hooks(session: nox.Session) -> None:
    """Activate virtualenv in hooks installed by pre-commit.
    This function patches git hooks installed by pre-commit to activate the
    session's virtual environment. This allows pre-commit to locate hooks in
    that environment when invoked from git.
    Args:
        session: The Session object.
    """
    assert session.bin is not None  # noqa: S101

    virtualenv = session.env.get("VIRTUAL_ENV")
    if virtualenv is None:
        return

    hookdir = Path(".git") / "hooks"
    if not hookdir.is_dir():
        return

    for hook in hookdir.iterdir():
        if hook.name.endswith(".sample") or not hook.is_file():
            continue

        text = hook.read_text()
        bindir = repr(session.bin)[1:-1]  # strip quotes
        if not (
            Path("A") == Path("a") and bindir.lower() in text.lower() or bindir in text
        ):
            continue

        lines = text.splitlines()
        if not (lines[0].startswith("#!") and "python" in lines[0].lower()):
            continue

        header = dedent(
            f"""\
            import os
            os.environ["VIRTUAL_ENV"] = {virtualenv!r}
            os.environ["PATH"] = os.pathsep.join((
                {session.bin!r},
                os.environ.get("PATH", ""),
            ))
            """
        )

        lines.insert(1, header)
        hook.write_text("\n".join(lines))


@nox.session(name="pre-commit", python="3.10")
def precommit(session: nox.Session) -> None:
    """Lint using pre-commit."""
    args = session.posargs or ["check", "--fix"]
    session.install(
        "ruff",
    )
    session.run("pre-commit", *args)
    if args and args[0] == "install":
        activate_virtualenv_in_precommit_hooks(session)


@nox.session(python=python_versions)
def mypy(session: nox.Session) -> None:
    """Type-check using mypy."""
    args = session.posargs or locations
    session.run_always("pdm", "install", "-G", ".", external=True)
    session.install("mypy", "pytest")
    session.run("mypy", "--install-types", "--non-interactive", "out", *args)
    if not session.posargs:
        session.run(
            "mypy",
            "--install-types",
            "--non-interactive",
            f"--python-executable={sys.executable}",
            "noxfile.py",
        )


@nox.session(python=python_versions)
def tests(session: nox.Session) -> None:
    """Run the test suite."""
    session.run_always("pdm", "install", "-G", "test", external=True)
    # session.install("coverage[toml]", "pytest", "pygments", "hypothesis")
    session.run("pytest")
    # try:
    #     session.run("coverage", "run", "--parallel", "-m", "pytest", *session.posargs)
    # finally:
    #     if session.interactive:
    #         session.notify("coverage", posargs=[])


@nox.session
def coverage(session: nox.Session) -> None:
    """Produce the coverage report."""
    # args = session.posargs or ["report"]

    session.install("coverage[toml]", "codecov")

    if not session.posargs and any(Path().glob(".coverage.*")):
        session.run("coverage", "combine")

    session.run("coverage", "xml", "--fail-under=0")
    session.run("codecov", *session.posargs)
    # session.run("coverage", *args)


@nox.session(python=python_versions)
def typeguard(session: nox.Session) -> None:
    """Runtime type checking using Typeguard."""
    session.run_always("pdm", "install", "-G", "test", external=True)
    session.install("pytest", "typeguard", "pygments", "hypothesis")
    session.run("pytest", f"--typeguard-packages={package}", *session.posargs)


@nox.session(python=python_versions)
def xdoctest(session: nox.Session) -> None:
    """Run examples with xdoctest."""
    args = session.posargs or ["all"]
    session.run_always("pdm", "install", "-G", "test", external=True)
    session.install("xdoctest[colors]")
    session.run("python", "-m", "xdoctest", package, *args)


@nox.session(name="docs-build", python=python_versions)
def docs_build(session: nox.Session) -> None:
    """Build the documentation."""
    args = session.posargs or ["docs", "docs/_build"]
    session.run_always("pdm", "install", "-G", "test", external=True)
    session.install("sphinx", "sphinx-click", "sphinx-rtd-theme")

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-build", *args)


@nox.session(python=python_versions)
def docs(session: nox.Session) -> None:
    """Build and serve the documentation with live reloading on file changes."""
    args = session.posargs or ["--open-browser", "docs", "docs/_build"]
    session.run_always("pdm", "install", "-G", "test", external=True)
    session.install("sphinx", "sphinx-autobuild", "sphinx-click", "sphinx-rtd-theme")

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-autobuild", *args)

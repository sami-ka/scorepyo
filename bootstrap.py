import argparse
import os
import subprocess
import sys
from typing import Iterable, List

# from utils.call import call, python_call


def call(cmd: List[str], **kwargs):  # pragma: no cover
    """Run a subprocess command and raise if it fails.

    Args:
        cmd: List of command parts.
        **kwargs: Optional keyword arguments passed to `subprocess.run`.

    Raises:
        click.exceptions.Exit: If `subprocess.run` returns non-zero code.
    """
    # pylint: disable=subprocess-run-check
    code = subprocess.run(cmd, **kwargs).returncode
    return code


def python_call(module: str, arguments: Iterable[str], **kwargs):  # pragma: no cover
    """Run a subprocess command that invokes a Python module."""
    call([sys.executable, "-m", module] + list(arguments), **kwargs)


def create_env(**kwargs):
    envname = kwargs["env_name"]
    python_call(
        module="pip",
        arguments=["install", "-e", "core_module/.[dev,ipy,doc]", "--no-cache-dir"],
    )

    python_call(
        module="ipykernel", arguments=["install", "--user", f"--name={envname}"]
    )

    python_call(module="pip", arguments=["install", "artifacts-keyring"])

    if os.name == "nt":
        python_call(module="pip", arguments=["install", "pypiwin32"])

    call(["conda", "install", "-y", "pandoc"])


def format(**_kwargs):  # prefix by underscore to avoid pylint to say that it is unused
    python_call(
        module="black",
        arguments=["--diff", ".", "--config", "./core_module/pyproject.toml"],
    )
    python_call(
        module="isort",
        arguments=["--diff", ".", "--settings-path", "./core_module/pyproject.toml"],
    )


def format_CI(
    **_kwargs,
):  # prefix by underscore to avoid pylint to say that it is unused
    python_call(
        module="black",
        arguments=[
            "--check",
            "--diff",
            ".",
            "--config",
            "./core_module/pyproject.toml",
        ],
    )
    python_call(
        module="isort",
        arguments=[
            "--check",
            "--diff",
            ".",
            "--settings-path",
            "./core_module/pyproject.toml",
        ],
    )


def test(**_kwargs):  # prefix by underscore to avoid pylint to say that it is unused
    python_call(
        module="pytest",
        arguments=["-rfs", "--cov=scorepyo", "--cov-report", "term-missing"],
    )


def linting(**_kwargs):  # prefix by underscore to avoid pylint to say that it is unused
    python_call(
        module="pylint",
        arguments=["--rcfile=./core_module/pyproject.toml", "."],
    )


def mypy(**_kwargs):  # prefix by underscore to avoid pylint to say that it is unused
    python_call(
        module="mypy",
        arguments=["--config-file", "./core_module/pyproject.toml", "core_module/"],
    )


def bandit(**_kwargs):  # prefix by underscore to avoid pylint to say that it is unused
    python_call(
        module="bandit",
        arguments=["-c", "./core_module/pyproject.toml", "-r", "."],
    )


def CI(**_kwargs):
    format_CI()
    test()
    linting()
    mypy()
    bandit()

def build_doc(**_kwargs):
    call(
        cmd=["sphinx-build","-b", "html", "-a", "-E", "./docs/source", "./docs/build"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("function")
    parser.add_argument("-e", "--env-name")
    args = vars(parser.parse_args())
    function = args["function"]
    remaining_arguments = {k: v for k, v in args.items() if k != "function"}
    globals()[function](**remaining_arguments)
    
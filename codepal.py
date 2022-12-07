"""Devops utils"""
import argparse
import os
import subprocess
import sys
from typing import Iterable, List


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
    """Install required package from setup.cfg in the current conda environment and register it"""
    assert (
        "env_name" in kwargs.keys()
    ), "Please fill the current environment name with -e option"
    envname = kwargs["env_name"]
    python_call(
        module="pip",
        arguments=["install", "-e", "core_module/.[dev,ipy,doc]", "--no-cache-dir"],
    )

    python_call(
        module="ipykernel", arguments=["install", "--user", f"--name={envname}"]
    )

    # python_call(module="pip", arguments=["install", "artifacts-keyring"])

    if os.name == "nt":
        python_call(module="pip", arguments=["install", "pypiwin32"])

    # call(["conda", "install", "-y", "pandoc"])


def format(**_kwargs):  # prefix by underscore to avoid pylint to say that it is unused
    """Call black and isort with parameters from pyproject.toml"""
    python_call(
        module="black",
        arguments=[".", "--config", "./core_module/pyproject.toml"],
    )
    python_call(
        module="isort",
        arguments=[".", "--settings-path", "./core_module/pyproject.toml"],
    )


def format_CI(
    **_kwargs,
):  # prefix by underscore to avoid pylint to say that it is unused
    """Call black and isort as check-only with parameters from pyproject.toml.
    This function is the one used in the CI pipeline."""
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
    """Call pytest"""
    python_call(
        module="pytest",
        arguments=["-rfs", "--cov=scorepyo", "--cov-report", "term-missing"],
    )


def lint(**_kwargs):  # prefix by underscore to avoid pylint to say that it is unused
    """Call pylint with parameters in pyproject.toml"""
    python_call(
        module="pylint",
        arguments=["--rcfile=./core_module/pyproject.toml", "."],
    )


def mypy(**_kwargs):  # prefix by underscore to avoid pylint to say that it is unused
    """Call mypy with parameters in pyproject.toml"""
    python_call(
        module="mypy",
        arguments=["--config-file", "./core_module/pyproject.toml", "core_module/"],
    )


def bandit(**_kwargs):  # prefix by underscore to avoid pylint to say that it is unused
    """Call bandit with parameters in pyproject.toml"""
    python_call(
        module="bandit",
        arguments=["-c", "./core_module/pyproject.toml", "-r", "."],
    )


def CI(**_kwargs):
    """Launch all CI steps locally"""
    # Black+isort
    format_CI()

    # Pytest
    test()

    # Pylint
    lint()

    # Mypy
    mypy()

    # Bandit
    bandit()


def build_doc(**_kwargs):
    "Call jupyter-book to build-doc."
    call(cmd=["jupyter-book", "build", "docs"])


def build_wheel(
    **_kwargs,
):  # prefix by underscore to avoid pylint to say that it is unused
    """Build wheel"""
    python_call(
        module="build",
        arguments=["core_module"],
    )


def check_wheel(
    **_kwargs,
):  # prefix by underscore to avoid pylint to say that it is unused
    """Build wheel"""
    call(cmd=["twine", "check", "core_module/dist/*"])


def upload_wheel_test(
    **_kwargs,
):  # prefix by underscore to avoid pylint to say that it is unused
    """Build wheel"""
    call(cmd=["twine", "upload", "-r", "testpypi", "core_module/dist/*"])


def upload_wheel_real(
    **_kwargs,
):  # prefix by underscore to avoid pylint to say that it is unused
    """Build wheel"""
    call(cmd=["twine", "upload", "core_module/dist/*"])


def publish_doc(
    **_kwargs,
):  # prefix by underscore to avoid pylint to say that it is unused
    """Build wheel"""
    call(cmd=["ghp-import", "-n", "-p", "-f", "docs/_build/html"])


if __name__ == "__main__":
    """Parse the function name to call as an argument+potential additional parameters for create_env function
    Call the specified function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("function")
    parser.add_argument("-e", "--env-name")
    # parser.add_argument("-t", "--test")
    args = vars(parser.parse_args())
    function = args["function"]
    remaining_arguments = {k: v for k, v in args.items() if k != "function"}
    globals()[function](**remaining_arguments)

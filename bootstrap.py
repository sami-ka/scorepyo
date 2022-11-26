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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("envname")
    args = parser.parse_args()
    python_call(
        module="pip",
        arguments=["install", "-e", "core_module/.[dev,ipy,doc]"],
    )

    python_call(
        module="ipykernel", arguments=["install", "--user", f"--name={args.envname}"]
    )

    python_call(module="pip", arguments=["install", "artifacts-keyring"])

    if os.name == "nt":
        python_call(module="pip", arguments=["install", "pypiwin32"])

    call(["conda", "install", "-y", "pandoc"])

"""Build helper for installing MLatom from this directory.

The package metadata lives in pyproject.toml. This file adds the one build step
pyproject.toml cannot express: setuptools copies .py files into a wheel as 0644,
which would leave mlatom/shell_cmd.py - run directly by `$mlatom input.inp` -
without its executable bit, and `$mlatom` would fail with "Permission denied".
"""
import os
from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPyExecutableShellCmd(build_py):
    """Keep mlatom/shell_cmd.py executable in the built wheel."""

    def run(self):
        build_py.run(self)
        target = os.path.join(self.build_lib, 'mlatom', 'shell_cmd.py')
        if os.path.exists(target):
            os.chmod(target, 0o755)


setup(cmdclass={'build_py': BuildPyExecutableShellCmd})

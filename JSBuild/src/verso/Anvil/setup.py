import os
import subprocess

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py


class BuildNative(build_py):
    def run(self):
        repo_root = os.path.dirname(__file__)
        script_paths = [
            os.path.join(repo_root, "Saguaro", "build_secure.sh"),
            os.path.join(repo_root, "core", "native", "build_native.sh"),
        ]
        for script_path in script_paths:
            if os.path.exists(script_path):
                print(f"Running native build: {script_path}")
                subprocess.check_call(["bash", script_path])
            else:
                print(f"Native build script not found, skipping: {script_path}")

        super().run()


ROOT_PACKAGES = find_packages(exclude=["saguaro", "saguaro.*"])
SAGUARO_PACKAGES = find_packages(where="Saguaro", include=["saguaro", "saguaro.*"])


setup(
    name="anvil",
    version="0.1.0",
    description="Anvil Agent with HighNoon Native Acceleration",
    packages=ROOT_PACKAGES + SAGUARO_PACKAGES,
    package_dir={"saguaro": "Saguaro/saguaro"},
    install_requires=[
        "numpy",
        "jsonschema",
        "requests",
        "pyyaml",
    ],
    cmdclass={
        "build_py": BuildNative,
    },
    entry_points={
        "console_scripts": [
            "anvil=cli.repl:main",
            "saguaro=saguaro.bootstrap:main",
            "anvil-aes-lint=core.aes.lint:main",
        ],
    },
    author="Anvil Team",
)

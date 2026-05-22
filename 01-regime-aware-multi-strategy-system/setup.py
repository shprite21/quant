from __future__ import annotations

from pathlib import Path

from setuptools import find_namespace_packages, setup


ROOT = Path(__file__).resolve().parent


def read_requirements() -> list[str]:
    requirements_path = ROOT / "requirements.txt"
    with requirements_path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip() and not line.startswith("#")]


setup(
    name="project-a-regime-aware-backtesting",
    version="0.1.0",
    description="Regime-aware multi-strategy backtesting system powered by Gaussian HMMs.",
    author="Arnaav Raj",
    python_requires=">=3.11",
    packages=find_namespace_packages(include=["src", "src.*"]),
    install_requires=read_requirements(),
    include_package_data=True,
)

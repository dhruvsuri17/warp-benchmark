from pathlib import Path

from setuptools import setup, find_packages

_req = Path(__file__).resolve().parent / "requirements.txt"
_install_requires = []
if _req.is_file():
    _install_requires = [
        line.strip()
        for line in _req.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="warp-benchmark",
    version="1.0.0",
    description="WARP benchmark: primal-dual warm-starting for interior-point AC-OPF solvers",
    python_requires=">=3.10",
    packages=find_packages(),
    py_modules=["normalizer"],
    install_requires=_install_requires,
)

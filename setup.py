# TODO: setup.py packaging
from setuptools import find_packages, setup

HYPHEN_E_DOT = "-e ."


def get_requirements(file_path: str):
    with open(file_path) as f:
        reqs = [line.strip() for line in f.readlines()]
        if HYPHEN_E_DOT in reqs:
            reqs.remove(HYPHEN_E_DOT)
    return reqs


setup(
    name="footvision",
    version="0.1.0",
    author="your-name",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
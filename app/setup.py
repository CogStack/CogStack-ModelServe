from setuptools import setup, find_packages

INSTALL_REQUIRES = [requirement.strip() for requirement in open("requirements.txt").readlines()]

setup(
    name="cogstack-model-serve",
    version="0.0.1",
    description="A model serving system for CogStack NLP solutions",
    packages=find_packages(),
    setup_requires=[
        "wheel",
    ],
    install_requires=INSTALL_REQUIRES
)

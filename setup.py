from setuptools import find_packages, setup

setup(
    name="deathsaurus",
    author="Jason Nance",
    version="0.0.2",
    packages=find_packages(),
    license="LICENSE.txt",
    description="Discord bot for interfacing with Transformer models.",
    install_requires=[
        "torch >= 1.12.1",
        "click >= 8.1",
        "transformers >= 4.21",
        "tqdm >= 4.31.1",
        "discord.py >= 1.7",
    ],
    python_requires=">=3.10",
    entry_points={"console_scripts": ["deathsaurus = deathsaurus.main:main"]},
)

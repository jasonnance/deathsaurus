from setuptools import find_packages, setup

setup(
    name="deathsaurus",
    author="Jason Nance",
    version="0.0.1",
    packages=find_packages(),
    license="LICENSE.txt",
    description="Discord bot for interfacing with Transformer models.",
    install_requires=[
        "torch >= 1.3.1",
        "click >= 7.0",
        "transformers >= 2.2.1",
        "tqdm >= 4.31.1",
        "discord.py >= 1.2.5",
    ],
    python_requires=">=3.6",
    entry_points={"console_scripts": ["deathsaurus = deathsaurus.main:main"]},
)

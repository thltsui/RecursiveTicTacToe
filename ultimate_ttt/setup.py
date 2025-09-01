#!/usr/bin/env python3
"""
Setup script for Ultimate Tic Tac Toe.
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Ultimate Tic Tac Toe - An advanced variant of classic Tic Tac Toe"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return ['numpy>=1.21.0', 'torch>=1.9.0']

setup(
    name="ultimate-ttt",
    version="1.0.0",
    description="Ultimate Tic Tac Toe - An advanced variant of classic Tic Tac Toe",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Ultimate Tic Tac Toe Team",
    author_email="",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Games/Entertainment :: Board Games",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
    entry_points={
        "console_scripts": [
            "ultimate-ttt=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="tic-tac-toe, ultimate, game, board-game, reinforcement-learning, ai",
    project_urls={
        "Bug Reports": "",
        "Source": "",
        "Documentation": "",
    },
)

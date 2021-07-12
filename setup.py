#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

# with open("requirements.txt") as f:
#   requirements = f.read().splitlines()

requirements = [
    "numpy",
    "scipy",
    "astropy",
    "matplotlib",
    "powerbox",
    "cached-property",
]

test_requirements = []

setup(
    author="Tyler Cox",
    author_email="tyler.a.cox@berkeley.edu",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Python package for simulating and cross-corcorrelating intensity mapping observations",
    entry_points={
        "console_scripts": [
            "xcorr=xcorr.cli:main",
        ],
    },
    install_requires=requirements,
    license="BSD license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="xcorr",
    name="xcorr",
    packages=find_packages(include=["xcorr", "xcorr.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/tyler-a-cox/xcorr",
    version="0.1.0",
    zip_safe=False,
)

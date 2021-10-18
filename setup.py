from distutils.core import setup

from setuptools import find_packages

setup(
    name="cdse_covid",
    packages=find_packages(),
    # 3.6 and up, but not Python 4
    python_requires=">=3.7",
    install_requires=["attrs>=19.2.0", "vistautils>=0.21.0"],
    scripts=[],
    classifiers=[],
)

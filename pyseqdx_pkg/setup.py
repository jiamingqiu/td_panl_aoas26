# setup.py
from setuptools import setup, find_packages

setup(
    name="pyseqdx",
    version="0.0.1",
    packages=find_packages(),
    python_requires='==3.12.3',
    install_requires=[
        # "torch>=2.3.0", # maybe pytorch?
        "numpy>=1.26.4",
        "pandas>=2.2.1",
        "matplotlib>=3.8.4"
    ],
    include_package_data=True
)
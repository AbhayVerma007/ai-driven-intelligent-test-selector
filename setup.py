from setuptools import find_packages, setup

setup(
    name="testwise-ai",
    version="0.1.0",
    description="AI-powered intelligent test selection for CI pipelines",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=[
        "scikit-learn>=1.3.0",
        "pandas>=2.1.0",
        "numpy>=1.24.0",
        "joblib>=1.3.0",
        "pytest>=7.4.0",
        "gitpython>=3.1.0",
        "pyyaml>=6.0",
        "matplotlib>=3.8.0",
    ],
    entry_points={
        "console_scripts": [
            "testwise=src.cli:main",
        ],
    },
)

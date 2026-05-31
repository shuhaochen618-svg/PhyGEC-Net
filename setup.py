from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="phygec_net",
    version="1.0.0",
    author="PhyGEC-Net Authors",
    author_email="",
    description="Physics-Guided Error Correction Network for Day-Ahead Electricity Load Forecasting",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/phygec_net",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=parse_requirements("requirements.txt"),
)

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tajik_text_segmentation",
    version="0.1.7",
    author="Sobir Bobiev",
    author_email="sobir.bobiev@gmail.com",
    description="A package for Tajik text segmentation using a heuristic algorithm and neural network.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sobir-git/tajik-text-segmentation",
    packages=find_packages(exclude=['training', 'tests']),
    package_data={
        "tajik_text_segmentation": ["checkpoints/*", "config.json"],
    },
    install_requires=open("requirements.txt").readlines(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires='>=3.6',
)

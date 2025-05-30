from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cliven",
    version="0.1.0",
    author="Kreyon aka vikas",
    author_email="vikaskumar783588@gmail.com",
    description="Chat with your PDFs using local AI models",
    long_description="Chat with your PDFs using local AI models",
    long_description_content_type="text/markdown",
    url="https://github.com/krey-yon/cliven",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "typer>=0.9.0",
        "rich>=13.0.0",
        "pdfplumber>=0.7.0",
        "sentence-transformers>=2.2.0",
        "chromadb>=0.4.0",
        "langchain>=0.0.300",
        "requests>=2.28.0",
        "pathlib",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cliven=main.cliven:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

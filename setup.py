from setuptools import find_packages, setup

setup(
    name="mcqgenerator",
    version="0.0.1",
    author="yomna waleed",
    author_email="yomnawaleed2022@gmail.com",
    install_requires=["openai", "langchain", "streamlit", "python-dotenv", "PyPDF2"],
    packages=find_packages(),
)

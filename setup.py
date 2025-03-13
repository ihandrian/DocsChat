from setuptools import setup, find_packages

setup(
    name="document-analysis-assistant",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "python-dotenv",
        "PyPDF2",
        "langchain",
        "faiss-cpu",
        "sentence-transformers",
        "huggingface_hub",
        "matplotlib",
        "seaborn",
        "pandas",
        "docx2txt",
        "openpyxl",
        "ebooklib",
        "beautifulsoup4",
    ],
    author="Irfan Handrian",
    author_email="handrian.irfan@gmail.com",
    description="A document analysis assistant that can process multiple document types and provide chat, summarization, and visualization capabilities",
    keywords="document, analysis, chat, AI, visualization, epub, djvu",
    python_requires=">=3.8",
)


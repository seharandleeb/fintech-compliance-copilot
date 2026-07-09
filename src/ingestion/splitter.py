"""
Text Splitter for ObliQA Dataset.

This module splits LangChain Documents into smaller chunks
using RecursiveCharacterTextSplitter.

Output:
    List[Document]
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.logging_config import logger
from config.settings import get_settings
from src.ingestion.loader import load_obliqa_documents

settings = get_settings()


def split_documents(
    documents: list[Document],
) -> list[Document]:
    """
    Split documents into smaller chunks.

    Parameters
    ----------
    documents : list[Document]
        List of LangChain Documents.

    Returns
    -------
    list[Document]
        List of the first 300 chunked documents.
    """

    logger.info("Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        add_start_index=True,
    )

    # Split all documents
    chunks = text_splitter.split_documents(documents)

    logger.info(
        f"Total generated chunks: {len(chunks)}"
    )

    # Keep only the first 300 chunks
    chunks = chunks[:300]

    logger.info(
        f"Keeping first {len(chunks)} chunks."
    )

    return chunks


if __name__ == "__main__":

    docs = load_obliqa_documents()

    chunks = split_documents(docs)

    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    print(f"Total Documents : {len(docs)}")
    print(f"Total Chunks    : {len(chunks)}")

    print("\n" + "=" * 60)
    print("FIRST CHUNK")
    print("=" * 60)

    print(chunks[0].page_content)

    print("\nMetadata:\n")
    print(chunks[0].metadata)

    print("\n" + "=" * 60)
    print("LAST CHUNK")
    print("=" * 60)

    print(chunks[-1].page_content)

    print("\nMetadata:\n")
    print(chunks[-1].metadata)
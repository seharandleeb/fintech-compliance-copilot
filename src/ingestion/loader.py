"""
Data Loader for ObliQA Dataset.

This module loads the ObliQA JSON dataset using LangChain JSONLoader
and converts each record into LangChain Documents.
"""

import json

from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document

from config.logging_config import logger
from config.settings import get_settings

settings = get_settings()


def metadata_func(record: dict, metadata: dict) -> dict:
    """
    Extract metadata from each JSON record.
    """

    metadata["question_id"] = record.get("QuestionID")

    metadata["document_ids"] = [
        p.get("DocumentID")
        for p in record.get("Passages", [])
    ]

    metadata["passage_ids"] = [
        p.get("PassageID")
        for p in record.get("Passages", [])
    ]

    metadata["num_passages"] = len(
        record.get("Passages", [])
    )

    return metadata


def load_obliqa_documents() -> list[Document]:
    """
    Load ObliQA dataset using LangChain JSONLoader.
    """

    dataset_path = settings.dataset_path

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    logger.info("Loading dataset using JSONLoader...")

    loader = JSONLoader(
        file_path=str(dataset_path),
        jq_schema=".[]",
        text_content=False,
        metadata_func=metadata_func,
    )

    raw_documents = loader.load()

    documents = []

    for doc in raw_documents:

        # JSONLoader returns page_content as JSON string
        record = json.loads(doc.page_content)

        question = record["Question"]

        answer = "\n\n".join(
            passage["Passage"]
            for passage in record["Passages"]
        )

        page_content = (
            f"Question:\n{question}\n\n"
            f"Answer:\n{answer}"
        )

        documents.append(
            Document(
                page_content=page_content,
                metadata=doc.metadata,
            )
        )

    logger.info(
        f"Successfully loaded {len(documents)} documents."
    )

    return documents


if __name__ == "__main__":

    docs = load_obliqa_documents()

    print(f"\nTotal Documents: {len(docs)}")

    print("\nFirst Document\n")

    print(docs[0].page_content[:700])

    print("\nMetadata\n")

    print(docs[0].metadata)
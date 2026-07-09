"""
Data Loader for ObliQA Dataset.

This module loads the raw ObliQA JSON dataset and converts each
record into a LangChain Document object.

Output:
    List[Document]
"""

from pathlib import Path

from langchain_core.documents import Document

from config.logging_config import logger
from config.settings import get_settings


settings = get_settings()


def load_obliqa_documents() -> list[Document]:
    """
    Load the ObliQA dataset and convert it into LangChain Documents.

    Returns
    -------
    list[Document]
        List of LangChain Document objects.
    """

    dataset_path = settings.dataset_path

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    logger.info("Loading dataset...")

    import json

    with open(dataset_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    documents = []

    for record in data:

        question = record["Question"]

        passages = record["Passages"]

        answer = "\n\n".join(
            passage["Passage"]
            for passage in passages
        )

        page_content = (
            f"Question:\n{question}\n\n"
            f"Answer:\n{answer}"
        )

        metadata = {
            "question_id": record["QuestionID"],
            "document_ids": [
                passage["DocumentID"]
                for passage in passages
            ],
            "passage_ids": [
                passage["PassageID"]
                for passage in passages
            ],
            "num_passages": len(passages),
        }

        documents.append(
            Document(
                page_content=page_content,
                metadata=metadata,
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

    print("\nMetadata:\n")

    print(docs[0].metadata)
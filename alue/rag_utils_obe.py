import argparse
import os
import re
import uuid
from re import Pattern
from typing import Any

import chromadb
from cleantext import clean
from config import EMBEDDING_MODELS
from haystack import Document, Pipeline, component
from haystack.components.converters import (
    PyPDFToDocument,
    TextFileToDocument,
)
from haystack.components.embedders import (
    OpenAIDocumentEmbedder,
    SentenceTransformersDocumentEmbedder,
)
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.routers import FileTypeRouter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import ComponentDevice, Secret
from dotenv import load_dotenv


load_dotenv()

def _secret_from_env(*env_keys: str) -> Secret:
    
    for key in env_keys:
        value = os.getenv(key)
        if value:
            return Secret.from_token(value)
    
    raise ValueError(
        f"Misssing required API key env variable. Tried: {', '.join(env_keys)}"
    )


@component
class CustomDocumentCleaner:
    r"""
    A custom components to clean documents that will be loaded to our OpenSearchDocumentStore

    Attributes
    ----------
    fix_unicode : bool
        If True, fix 'broken' unicode such as mojibake and garbled HTML entities to proper unicode.
    to_ascii : bool
        If True, transliterate to closest ASCII representation.
    no_line_breaks : bool
        If True, remove line breaks by converting "\n" or "\r" to a single space.
    no_urls : bool
        If True, replace all URLs with a special token.
    no_emails : bool
        If True, replace all email addresses with a special token.
    no_phone_numbers : bool
        If True, replace all phone numbers with a special token.
    no_punct : bool
        If True, remove punctuations.
    no_numbers : bool
        If True, remove numbers.
    no_digits : bool
        If True, replace all digits with a special token.
    no_currency_symbols : bool
        If True, replace all currency symbols with a special token.
    lower : bool
        If True, convert all characters to lowercase.
    custom_regex_pattern : Pattern[str]
        Custom regex pattern to be applied on the text.

    Methods
    -------
    run(documents: List[Document]) -> List[Document]
        Clean the given list of documents and return the cleaned documents.
    """

    def __init__(
        self,
        fix_unicode: bool = True,
        to_ascii: bool = True,
        no_line_breaks: bool = False,
        no_urls: bool = True,
        no_emails: bool = True,
        no_phone_numbers: bool = False,
        no_punct: bool = False,
        no_numbers: bool = False,
        no_digits: bool = False,
        no_currency_symbols: bool = True,
        lower: bool = False,
        custom_regex_pattern: Pattern[
            str
        ] = r"[^\x00-\x7F]+",  # remove all non-ascii characters
    ):
        """
        Constructs all the necessary attributes for the CustomDocumentCleaner object.

        Parameters
        ----------
        fix_unicode : bool, optional
            If True, fix 'broken' unicode such as mojibake and garbled HTML entities to proper unicode.
        to_ascii : bool, optional
            If True, transliterate to closest ASCII representation.
        no_line_breaks : bool, optional
            If True, remove line breaks
        no_urls : bool, optional
            If True, replace all URLs with a special token.
        no_emails : bool, optional
            If True, replace all email addresses with a special token.
        no_phone_numbers : bool, optional
            If True, replace all phone numbers with a special token.
        no_punct : bool, optional
            If True, remove punctuations.
        no_numbers : bool, optional
            If True, remove numbers.
        no_digits : bool, optional
            If True, replace all digits with a special token.
        no_currency_symbols : bool, optional
            If True, replace all currency symbols with a special token.
        lower : bool, optional
            If True, convert all characters to lowercase.
        custom_regex_pattern : Pattern[str], optional
            Custom regex pattern to be applied on the text.
        """
        self.fix_unicode = fix_unicode
        self.to_ascii = to_ascii
        self.no_line_breaks = no_line_breaks
        self.no_urls = no_urls
        self.no_emails = no_emails
        self.no_phone_numbers = no_phone_numbers
        self.no_punct = no_punct
        self.no_numbers = no_numbers
        self.no_digits = no_digits
        self.no_currency_symbols = no_currency_symbols
        self.lower = lower
        self.custom_regex_pattern = custom_regex_pattern

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> list[Document]:
        """
        Clean Documents

        Parameters
        ----------
        documents : List[Document]
            list of unprocessed documents

        Returns
        -------
        List[Document]
            list of cleaned documents
        """
        for i, document in enumerate(documents):
            if self.custom_regex_pattern:
                documents[i].content = re.sub(
                    self.custom_regex_pattern, " ", document.content
                )
            documents[i].content = clean(
                document.content,
                fix_unicode=self.fix_unicode,
                to_ascii=self.to_ascii,
                no_line_breaks=self.no_line_breaks,
                no_urls=self.no_urls,
                no_emails=self.no_emails,
                no_phone_numbers=self.no_phone_numbers,
                no_punct=self.no_punct,
                no_numbers=self.no_numbers,
                no_digits=self.no_digits,
                no_currency_symbols=self.no_currency_symbols,
                lower=self.lower,
            )

        return {"documents": documents}


class ProcessDocuments:
    def __init__(
        self,
        document_store: Any,
        use_local: bool = False,
        embedding_endpoint: str = EMBEDDING_MODELS["BAAI/bge-m3"]["aip_endpoint"],
        embedding_path: str = EMBEDDING_MODELS["BAAI/bge-m3"]["local_path"],
        use_splitter: bool = False,
        split_by: str = "word",
        split_length: int = 128,
        split_overlap: int = 0,
        split_threshold: int = 0,
    ):
        """
        Initialize the ProcessDocuments class. This class initializes the
        preprocessing pipeline to read in documents and store in Chroma.
        Args:
            document_store (Any): Chroma Document Store where docs will be stored
            embedding_endpoint (str): Embedding model to use, default is AIP Embedding endpoint
            embedding_path (str): Embedding model to use locally
            use_splitter (bool): Whether to chunk documents or not
            split_by (str): Method to split documents (word, sentence, passage, page)
            split_length (int): Length of each split chunk
            split_overlap (int): Overlap between chunks
            split_threshold (int): Minimum length of a document fragment

        """
        self.embedding_model = embedding_endpoint if not use_local else embedding_path
        self.document_store = document_store
        file_type_router = FileTypeRouter(mime_types=["text/plain", "application/pdf"])
        text_file_converter = TextFileToDocument()
        pdf_converter = PyPDFToDocument()
        # pdf_converter = PDFMinerToDocument()
        document_joiner = DocumentJoiner()
        document_cleaner = CustomDocumentCleaner()

        if not use_local:
            document_embedder = OpenAIDocumentEmbedder(
                api_key=_secret_from_env("EMBEDDINGS_API_KEY"),
                api_base_url=self.embedding_model,
                model="tei",
            )
        else:
            device = ComponentDevice.from_str("cuda:2")
            document_embedder = SentenceTransformersDocumentEmbedder(
                model=self.embedding_model, device=device
            )
        document_writer = DocumentWriter(
            self.document_store, policy=DuplicatePolicy.SKIP
        )
        # create pipeline to process documents
        self.chunk_creation_pipeline = Pipeline()
        self.write_chunks_to_index_pipeline = Pipeline()

        self.chunk_creation_pipeline.add_component(
            instance=file_type_router, name="file_type_router"
        )
        self.chunk_creation_pipeline.add_component(
            instance=text_file_converter, name="text_file_converter"
        )
        self.chunk_creation_pipeline.add_component(
            instance=pdf_converter, name="pypdf_converter"
        )
        self.chunk_creation_pipeline.add_component(
            instance=document_joiner, name="document_joiner"
        )
        self.chunk_creation_pipeline.add_component(
            instance=document_cleaner, name="document_cleaner"
        )
        if use_splitter:
            document_splitter = DocumentSplitter(
                split_by=split_by,
                split_length=split_length,
                split_overlap=split_overlap,
                split_threshold=split_threshold,
            )
            self.chunk_creation_pipeline.add_component(
                instance=document_splitter, name="document_splitter"
            )

        self.chunk_creation_pipeline.connect(
            "file_type_router.text/plain", "text_file_converter.sources"
        )
        self.chunk_creation_pipeline.connect(
            "file_type_router.application/pdf", "pypdf_converter.sources"
        )
        self.chunk_creation_pipeline.connect("text_file_converter", "document_joiner")
        self.chunk_creation_pipeline.connect("pypdf_converter", "document_joiner")
        self.chunk_creation_pipeline.connect("document_joiner", "document_cleaner")

        if use_splitter:
            self.chunk_creation_pipeline.connect(
                "document_cleaner", "document_splitter"
            )

        self.write_chunks_to_index_pipeline.add_component(
            instance=document_embedder, name="document_embedder"
        )
        self.write_chunks_to_index_pipeline.add_component(
            instance=document_writer, name="document_writer"
        )

        self.write_chunks_to_index_pipeline.connect(
            "document_embedder", "document_writer"
        )

    def run_document_process(
        self, directory: str = "", doc_id_field: str = "ACN"
    ) -> None:
        """
        Run the document processing pipeline.
        Args:
            directory (str): The directory where the documents are stored.
        """
        file_paths = [
            os.path.join(directory, file_path) for file_path in os.listdir(directory)
        ]

        output = self.chunk_creation_pipeline.run(
            {"file_type_router": {"sources": file_paths}},
        )

        if "document_splitter" in output:
            for doc in output["document_splitter"]["documents"]:
                doc.meta.pop("split_id", None)
                doc.meta.pop("split_idx_start", None)
                doc.meta.pop("_split_overlap", None)
                doc.meta[doc_id_field] = doc.meta["source_id"]
        elif "document_cleaner" in output:
            for doc in output["document_cleaner"]["documents"]:
                doc.meta[doc_id_field] = str(uuid.uuid4())

        documents = output[
            "document_cleaner" if "document_cleaner" in output else "document_splitter"
        ]["documents"]
        self.write_chunks_to_index_pipeline.run(
            {"document_embedder": {"documents": documents}}
        )


def load_or_create_db(collection_name: str, persist_path: str):
    client = chromadb.PersistentClient(path=persist_path)
    document_store = client.get_or_create_collection(name=collection_name)

    return document_store


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and store documents for RAG.")
    parser.add_argument(
        "--use_local",
        action="store_true",
        help="Whether to use a local embedding model",
    )
    parser.add_argument(
        "--use_splitter",
        action="store_true",
        help="Whether to use the DocumentSplitter",
    )
    parser.add_argument(
        "--split_by",
        type=str,
        default="word",
        choices=["word", "sentence", "passage", "page"],
        help="Method to split documents",
    )
    parser.add_argument(
        "--split_length", type=int, default=128, help="Length of each split chunk"
    )
    parser.add_argument(
        "--split_overlap", type=int, default=0, help="Overlap between chunks"
    )
    parser.add_argument(
        "--split_threshold",
        type=int,
        default=0,
        help="Minimum length of a document fragment",
    )
    parser.add_argument(
        "--directory",
        type=str,
        default="docs",
        help="Directory where the documents are stored",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="rag_docs",
        help="Name of Chroma Collection that holds all the documents for RAG. Chroma is a vector database used to store and retrieve document embeddings.",
    )
    parser.add_argument(
        "--persist-path",
        type=str,
        default="./rag_docs",
        help="Path to persist the Chroma Collection",
    )
    args = parser.parse_args()

    # document_store = ChromaDocumentStore(
    #     collection_name="testing_doc",
    #     persist_path="./testing_doc",
    # )

    document_store = load_or_create_db(
        collection_name=args.collection_name, persist_path=args.persist_path
    )

    doc_upload = ProcessDocuments(
        document_store=document_store,
        use_local=args.use_local,
        use_splitter=args.use_splitter,
        split_by=args.split_by,
        split_length=args.split_length,
        split_overlap=args.split_overlap,
        split_threshold=args.split_threshold,
    )
    doc_upload.run_document_process(directory=args.directory)

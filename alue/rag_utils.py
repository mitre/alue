import json
import logging
import os
from pprint import pformat
from typing import Optional, List, Dict, Any, Tuple
import argparse

from chromadb import (
    Collection,
    EmbeddingFunction,
    Metadata,
    PersistentClient,
)
from chromadb.api import ClientAPI
from chromadb.api.types import Embeddable
from chromadb.utils import embedding_functions
from unstructured.chunking.basic import chunk_elements
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import (
    Element,
    ElementMetadata,
    CompositeElement,
)
from unstructured.partition.pdf import partition_pdf
from settings import get_settings

def setup_logger(name: str) -> logging.Logger:
    """Set up and configure a logger with console output.

    Args:
        name: Name to be used for the logger instance.

    Returns:
        Configured logging.Logger instance with console handler and formatter.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    return logger

logger = setup_logger(__name__)

def get_embedding_function(provider: str, model: str = None, url: str = None):
    settings = get_settings()
    if provider == "openai":
        api_key = settings.openai_api_key_str
        if not api_key:
            raise ValueError("Embedding API key not found. Set EMBEDDING_API_KEY environment variable.")
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=model or "text-embedding-3-small"
        )

    elif provider == "ollama":
        embedding_functions.OllamaEmbeddingFunction(
            model_name=model or "nomic-embed-text",
            url=f"{url or 'http://localhost:11434'}/api/embeddings"
        )

    elif provider == "hf":
        api_key = settings.hf_token_str
        if not api_key:
            raise ValueError("HuggingFace token not found. Set HF_TOKEN environment variable.")
        return embedding_functions.HuggingFaceEmbeddingFunction(
            api_key=api_key,
            model_name=model or "sentence-transformers/all-MiniLM-L6-v2"
        )

    elif provider == "local":
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model or "all-MiniLM-L6-v2"
        )

    elif provider == "openai-compatible":
        api_key = settings.embedding_api_key_str
        if not api_key:
            raise ValueError("Embedding API key not found. Set EMBEDDING_API_KEY environment variable.")
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=model or "default",
            api_base=url
        )
    
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


class DocumentProcessor:

    # Define block types to keep
    EXTRACT_IMAGE_BLOCK_TYPES: List[str] = ["Image", "Table"]
    # Constants that control the number of characters per text chunk.
    # "HARD": No chunk will exceed this length. A single element that exceeds this length
    # will be divided into two or more chunks using text-splitting.
    # "SOFT": Chunk length >= this value is not extended to include the next element, even
    # if that element would fit without exceeding max_characters. A "soft max" length that
    # can be used in conjunction with max_characters to limit most chunks to a preferred
    # length while still allowing larger elements to be included in a single chunk without
    # resorting to text-splitting.
    CHUNK_HARD_MAX_CHARS = 1200
    CHUNK_SOFT_MAX_CHARS = 700
    OVERLAP_SIZE = 50  # ie. no overlap
    OVERLAP_ALL = False
    # Specific to the `by_title` strategy
    # Combines elements (ex: a series of titles) until a section reaches a length of N chars.
    # Defaults to `max_characters`, which combines chunks whenever space allows. Note this
    # value is "capped" at the `new_after_n_chars` value ("SOFT MAX") since a value higher
    # than that would not change this parameter's effect.
    COMBINE_TEXT_UNDER_N_CHARS = 500
    MULTIPAGE_SECTIONS = True

    document_directory_path: Optional[str]
    output_path: str
    temp_document_path: Optional[str]
    partition_strategy: str 
    hi_res_model_name: str 
    extracted_image_block_types: Optional[List[str]] 
    write_image_block_types_metadata: Optional[bool] 
    elements_to_keep: List[str] 
    chunk_hard_max_chars: int
    chunk_soft_max_chars: int
    overlap_size: int 
    combine_text_under_n_chars: int
    multipage_sections: bool 
    chunk_type: str 
    write_to_file: bool


    def __init__(
        self,
        document_directory_path: Optional[str],
        output_path: str,
        temp_document_path: Optional[str] = None,
        partition_strategy: str = "hi_res",
        hi_res_model_name: str = "yolox",
        extracted_image_block_types: Optional[List[str]] = EXTRACT_IMAGE_BLOCK_TYPES,
        write_image_block_types_metadata: Optional[bool] = False,
        elements_to_keep: List[str] = ["NarrativeText", "ListItem", "Title"],
        chunk_hard_max_chars: int = CHUNK_HARD_MAX_CHARS,
        chunk_soft_max_chars: int = CHUNK_SOFT_MAX_CHARS,
        overlap_size: int = OVERLAP_SIZE,
        combine_text_under_n_chars: int = COMBINE_TEXT_UNDER_N_CHARS,
        multipage_sections: bool = MULTIPAGE_SECTIONS,
        chunk_type: str = "title",
        write_to_file: bool = True,
    ):
        self.document_directory_path = document_directory_path
        self.output_path = output_path
        self.temp_document_path = temp_document_path
        self.partition_strategy = partition_strategy
        self.hi_res_model_name = hi_res_model_name
        self.extracted_image_block_types = extracted_image_block_types 
        self.write_image_block_types_metadata = write_image_block_types_metadata
        self.elements_to_keep = elements_to_keep 
        self.chunk_hard_max_chars = chunk_hard_max_chars
        self.chunk_soft_max_chars = chunk_soft_max_chars
        self.overlap_size = overlap_size
        self.combine_text_under_n_chars = combine_text_under_n_chars
        self.multipage_sections = multipage_sections
        self.chunk_type = chunk_type
        self.write_to_file = write_to_file


    def process_document_directory(
        self
    ) -> List[Dict[str,Any]]:
        
        chunks = []
        if self.document_directory_path:
            for dirpath, _, filenames in os.walk(self.document_directory_path):
                logger.info(f"Processing documents in {dirpath}")

                for filename in filenames:
                    if filename.lower().endswith("pdf"):
                        document_path = os.path.join(dirpath, filename)
                        logger.info(f"Processing document {document_path}")
                        doc_chunks = self.process_single_document(document_path)
                        logger.info(f"Retrieved {len(doc_chunks)} chunks from {document_path}")
                        chunks.extend(doc_chunks)

            logger.info(f"Returning {len(chunks)} chunks from documents in {self.document_directory_path}")
        else: 
            logger.info(f"Paramter 'document_directory_path' must be set to generate document chunks. Returning empty chunks list.")

        return chunks
        

    def write_image_block_metadata(
        self,
        elements: List["Element"],
        metadata_fpath: str,
        document_path: str,
        extracted_image_block_types: Optional[List[str]] = EXTRACT_IMAGE_BLOCK_TYPES,
    ) -> None:
        """Write image metadata to metadata file."""
        logger.info(f"Writing image block metadata to: {metadata_fpath}")
        with open(metadata_fpath, "w", encoding="utf-8") as f:
            f.write(f"Metadata for extracted image blocks from {document_path}\n")
            f.write(f"Extracted image block types: {extracted_image_block_types}\n")
            f.write(f"Number of extracted image blocks: {len(elements)}\n")
            f.write("Metadata for each extracted image block:\n")
            for element in elements:
                if extracted_image_block_types and element.category in extracted_image_block_types:
                    element_metadata = element.metadata
                    element_metadata_str = pformat(element_metadata.to_dict())
                    f.write(f"{element_metadata_str}\n")


    def process_single_document(
        self,
        document_path: str
    ) -> List[Dict[str,Any]]:
        
        document_name = os.path.splitext(os.path.basename(document_path))[0]
        artifacts_dir = os.path.join(self.output_path, document_name, "artifacts")
        extracted_images_dir = os.path.join(artifacts_dir, "extracted_images")
        os.makedirs(extracted_images_dir, exist_ok=True)
        chunks_dir = os.path.join(artifacts_dir, "chunks")
        os.makedirs(chunks_dir, exist_ok=True)
        jsonl_dir = os.path.join(artifacts_dir, "json")
        os.makedirs(jsonl_dir, exist_ok=True)

        # extract elements from PDF
        logger.info(f"Partitioning pdf for {document_name}")
        elements = partition_pdf(
            filename=document_path,
            strategy=self.partition_strategy,
            languages=["eng"],
            infer_table_structure=False,
            extract_image_block_types=self.extracted_image_block_types,
            extract_image_block_output_dir=extracted_images_dir,
            hi_res_model_name=self.hi_res_model_name,
        )

        # write image metadata to document metadata file
        if self.write_image_block_types_metadata:
            metadata_fpath = os.path.join(
                artifacts_dir, "metadata_for_extracted_images.txt"
            )
            self.write_image_block_metadata(
                elements, metadata_fpath, document_path, self.extracted_image_block_types
            )

        # filter file elements before chunking, only keeping specific element types
        clean_elements = []
        for element in elements:
            if element.category in self.elements_to_keep:
                clean_elements.append(element)

        # specify arguments for chunking then pivot on chunking type
        basic_kwargs = {
            "max_characters": self.chunk_hard_max_chars,
            "new_after_n_chars": self.chunk_soft_max_chars,
            "overlap": self.overlap_size,
        }
        title_kwargs = {
            "combine_text_under_n_chars": self.combine_text_under_n_chars,
            "multipage_sections": self.multipage_sections,
            **basic_kwargs,
        }
        chunks_basic_template = "type_{chunk_type}_soft_lim_{soft_max}_hard_lim_{hard_max}_overlap_{overlap_size}.txt"
        chunks_title_template = (
            os.path.splitext(chunks_basic_template)[0]
            + "_element_soft_lim_{combine_text_under_n}_multipage_{multipage_ok}.txt"
        )
        logger.info(f"Chunking pdf elements for {document_name}")
        
        # create chunks based on type of chunking specified
        if self.chunk_type == "title":
            # chunk by title
            chunks = chunk_by_title(clean_elements, **title_kwargs)
            if self.write_to_file:
                # create the filenames and export chunks
                chunks_title_fname = chunks_title_template.format(
                    chunk_type=self.chunk_type,
                    soft_max=self.chunk_soft_max_chars,
                    hard_max=self.chunk_hard_max_chars,
                    overlap_size=self.overlap_size,
                    combine_text_under_n=self.combine_text_under_n_chars,
                    multipage_ok=self.multipage_sections,
                )
                # write chunks to text
                self.save_chunks_to_txt(
                    chunks, 
                    os.path.join(chunks_dir, chunks_title_fname), 
                    document_name
                )
                self.save_chunks_to_json(
                    chunks, 
                    os.path.join(jsonl_dir, f"{document_name}-title.jsonl"), 
                    document_name,
                    doc_path=document_path,
                )

        elif self.chunk_type == "basic":
            # create chunks
            chunks = chunk_elements(clean_elements, **basic_kwargs)
            if self.write_to_file:
                # create the filenames and export chunks
                chunks_basic_fname = chunks_basic_template.format(
                    chunk_type=self.chunk_type,
                    soft_max=self.chunk_soft_max_chars,
                    hard_max=self.chunk_hard_max_chars,
                    overlap_size=self.overlap_size,
                )
                # write chunks to text
                self.save_chunks_to_txt(
                    chunks, 
                    os.path.join(chunks_dir, chunks_basic_fname), 
                    document_name
                )
                self.save_chunks_to_json(
                    chunks,
                    os.path.join(jsonl_dir, f"{document_name}-basic.jsonl"),
                    document_name,
                    doc_path=document_path,
                )

        else:
            chunks = []
        
        output_chunks = self.to_json(chunks, document_name, document_path)

        return output_chunks


    def save_chunks_to_txt(
        self,
        chunks: List["Element"], 
        fname: str, 
        identifier: str
    ) -> None:
        logger.info(f"Saving chunks to {fname}")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(f"Identifier: {identifier}\n\n")
            for chunk in chunks:
                f.write(f"{str(chunk)}\n\n")
                f.write(
                    "----------------------------------------------------------------------------------------------------\n"
                )
    

    def prepare_chunk_metadata(
        self,
        chunk_metadata: "ElementMetadata",
        identifier: str,
        file_dir: Optional[str] = None,
        file_name: Optional[str] = None,
    ) -> Dict[str, str]:
        metadata = {
            "file_directory": (file_dir if file_dir else chunk_metadata.file_directory),
            "filename": file_name if file_name else chunk_metadata.filename,
            "page_number": chunk_metadata.page_number,
            "chunk_id": identifier,
        }
        return metadata

    
    def to_json(
        self,
        chunks: List["Element|CompositeElement"],
        identifier: str,
        doc_path: Optional[str] = None,
    ) -> List[Dict[str,Any]]:
        
        file_info = {}
        if doc_path is not None:
            file_dir, file_name = os.path.split(doc_path)
            file_info.update({"file_dir": file_dir, "file_name": file_name})
        
        output = []
        for i,chunk in enumerate(chunks):
            metadata_inputs = {
                "chunk_metadata": chunk.metadata,
                "identifier": f"{identifier}-{i}",
            }
            metadata_inputs.update(file_info)
            chunk_metadata = self.prepare_chunk_metadata(**metadata_inputs)
            chunk_dict = {
                "text": chunk.text, 
                "metadata": chunk_metadata
            }
            output.append(chunk_dict)
        
        return output


    def save_chunks_to_json(
        self,
        chunks: List["Element"],
        fname: str,
        identifier: str,
        doc_path: Optional[str] = None,
    ) -> None:
        output_chunks = self.to_json(chunks, identifier, doc_path)
        logger.info(f"Saving chunks to {fname}")
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(output_chunks, f, indent=4)


    @staticmethod
    def load_chunks_from_json(
        file_path: str,
    ) -> List[Dict[str,Any]]:
        chunks = []
        with open(file_path, "r", encoding="utf-8") as f:
            chunks.extend(json.load(f))
        return chunks


class ChromaInterface:

    client: Optional[ClientAPI]  
    database_path: str


    def __init__(
        self,
        database_path: str,
    ):
        self.client = None
        self.database_path = database_path


    def load_or_create_db(
        self,
    ) -> None:
        if self.client is None and self.database_path:
            self.client = PersistentClient(path=self.database_path)


    def get_or_create_collection(
        self,
        collection_name: str,
        embedding_function: Optional[EmbeddingFunction[Embeddable]] = None,
    ) -> Collection:
        

        if embedding_function is None:
            embedding_function = get_embedding_function("local")
        
        if self.client is None:
            # create client if it does not exist
            self.load_or_create_db()

        collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function)
                
        return collection


    def organize_document_chunks(
        self,
        document_chunks: List[Dict[str,Any]]
    ) -> Tuple[List[str], List[str], List[Metadata]]:
        ids = []
        documents = []
        metadatas = []
        for chunk in document_chunks:
            ids.append(chunk["metadata"]["chunk_id"])
            documents.append(chunk["text"])
            metadatas.append(chunk["metadata"])
        return (ids, documents, metadatas)


    def add_document_chunks(
        self,
        collection_name: str,
        embedding_function: EmbeddingFunction[Embeddable],
        document_chunks: List[Dict[str,Any]]
    ) -> Collection:
        # get collection
        collection = self.get_or_create_collection(collection_name, embedding_function)
        # map chunks from DocumentProcessor to required sublists
        (ids, documents, metadatas) = self.organize_document_chunks(document_chunks)
        # load into database collection
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        return collection
    

    def query_collection(
        self,
        query: str,
        collection_name: str,
        embedding_function,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Query ChromaDB collection and return formatted results."""
        collection = self.get_or_create_collection(collection_name, embedding_function)
        
        # Build query parameters
        query_params = {
            "query_texts": [query],
            "n_results": n_results
        }
        if where:
            query_params["where"] = where
        
        results = collection.query(**query_params)
        
        # Format results
        formatted_results = []
        for i in range(len(results["documents"][0])):
            formatted_results.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results["distances"] else None
            })
        
        return formatted_results


def main():
    parser = argparse.ArgumentParser(description="Process documents and load into ChromaDB")
    
    # Basic paths and names
    parser.add_argument("--document-directory", type=str, required=True,
                       help="Directory containing PDF documents to process")
    parser.add_argument("--database-path", type=str, default="./chroma_db",
                       help="Path to store ChromaDB database")
    parser.add_argument("--collection-name", type=str, default="documents",
                       help="Name for the ChromaDB collection")
    parser.add_argument("--output-path", type=str, default="./output",
                       help="Path to store processing artifacts")
    
    # Document processing options
    parser.add_argument("--partition-strategy", type=str, default="hi_res",
                       choices=["hi_res", "fast", "ocr_only", "auto"],
                       help="PDF partitioning strategy: 'fast' for lightweight processing, 'hi_res' for detailed extraction")
    parser.add_argument("--chunk-hard-max", type=int, default=1200,
                       help="Maximum characters per chunk (hard limit)")
    parser.add_argument("--chunk-soft-max", type=int, default=700,
                       help="Preferred characters per chunk (soft limit)")
    parser.add_argument("--overlap-size", type=int, default=50,
                       help="Character overlap between chunks")
    
    # Embedding provider selection
    parser.add_argument("--embedding-provider", type=str, default="local",
                       choices=["openai", "ollama", "hf", "local", "openai-compatible"],
                       help="Embedding provider to use")
    parser.add_argument("--embedding-model", type=str, default=None,
                       help="Specific model name (optional)")
    parser.add_argument("--embedding-url", type=str, default="http://localhost:11434",
                       help="URL for Ollama or OpenAI-compatible endpoints")
    
    args = parser.parse_args()
    
    # Create document processor with chunking parameters
    doc_processor = DocumentProcessor(
        document_directory_path=args.document_directory,
        output_path=args.output_path,
        partition_strategy=args.partition_strategy,
        chunk_hard_max_chars=args.chunk_hard_max,
        chunk_soft_max_chars=args.chunk_soft_max,
        overlap_size=args.overlap_size
    )
    
    # Process documents
    print(f"Processing documents from {args.document_directory}...")
    chunks = doc_processor.process_document_directory()
    print(f"Generated {len(chunks)} chunks")
    
    # Create ChromaDB interface
    chroma = ChromaInterface(database_path=args.database_path)
    
    embedding_function = get_embedding_function(
        args.embedding_provider,
        args.embedding_model,
        args.embedding_url
    )
    
    # Load into ChromaDB
    print(f"Loading chunks into ChromaDB collection '{args.collection_name}'...")
    collection = chroma.add_document_chunks(
        collection_name=args.collection_name,
        embedding_function=embedding_function,
        document_chunks=chunks
    )
    
    print(f"Successfully loaded {len(chunks)} chunks into ChromaDB")
    print(f"Collection '{args.collection_name}' now has {collection.count()} total documents")

if __name__ == "__main__":
    main()
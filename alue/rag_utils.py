import json
import logging
import os
from pprint import pformat
from typing import Optional, List, Dict, Any

from unstructured.chunking.basic import chunk_elements
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import (
    Element,
    ElementMetadata,
    CompositeElement,
)
from unstructured.partition.pdf import partition_pdf

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

    document_directory_path: str
    output_path: str


    def __init__(
        self,
        document_directory_path: str,
        output_path: str
    ):
        self.document_directory_path = document_directory_path
        self.output_path = output_path


    def process_document_directory(
        self,
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
    ) -> List[Dict[str,Any]]:
        
        chunks = []
        for dirpath, _, filenames in os.walk(self.document_directory_path):
            logger.info(f"Processing documents in {dirpath}")

            for filename in filenames:
                document_path = os.path.join(dirpath, filename)
                logger.info(f"Processing document {document_path}")
                doc_chunks = self.process_single_document(
                    document_path,
                    temp_document_path,
                    partition_strategy,
                    hi_res_model_name,
                    extracted_image_block_types,
                    write_image_block_types_metadata,
                    elements_to_keep,
                    chunk_hard_max_chars,
                    chunk_soft_max_chars,
                    overlap_size,
                    combine_text_under_n_chars,
                    multipage_sections,
                    chunk_type,
                    write_to_file,
                )
                logger.info(f"Retrieved {len(doc_chunks)} chunks from {document_path}")
                chunks.extend(doc_chunks)

        logger.info(f"Returning {len(chunks)} chunks from documents in {self.document_directory_path}")
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
        document_path: str,
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
            strategy=partition_strategy,
            languages=["eng"],
            infer_table_structure=False,
            extract_image_block_types=extracted_image_block_types,
            extract_image_block_output_dir=extracted_images_dir,
            hi_res_model_name=hi_res_model_name,
        )

        # write image metadata to document metadata file
        if write_image_block_types_metadata:
            metadata_fpath = os.path.join(
                artifacts_dir, "metadata_for_extracted_images.txt"
            )
            self.write_image_block_metadata(
                elements, metadata_fpath, document_path, extracted_image_block_types
            )

        # filter file elements before chunking, only keeping specific element types
        clean_elements = []
        for element in elements:
            if element.category in elements_to_keep:
                clean_elements.append(element)

        # specify arguments for chunking then pivot on chunking type
        basic_kwargs = {
            "max_characters": chunk_hard_max_chars,
            "new_after_n_chars": chunk_soft_max_chars,
            "overlap": overlap_size,
        }
        title_kwargs = {
            "combine_text_under_n_chars": combine_text_under_n_chars,
            "multipage_sections": multipage_sections,
            **basic_kwargs,
        }
        chunks_basic_template = "type_{chunk_type}_soft_lim_{soft_max}_hard_lim_{hard_max}_overlap_{overlap_size}.txt"
        chunks_title_template = (
            os.path.splitext(chunks_basic_template)[0]
            + "_element_soft_lim_{combine_text_under_n}_multipage_{multipage_ok}.txt"
        )
        logger.info(f"Chunking pdf elements for {document_name}")
        
        # create chunks based on type of chunking specified
        if chunk_type == "title":
            # chunk by title
            chunks = chunk_by_title(clean_elements, **title_kwargs)
            if write_to_file:
                # create the filenames and export chunks
                chunks_title_fname = chunks_title_template.format(
                    chunk_type=chunk_type,
                    soft_max=chunk_soft_max_chars,
                    hard_max=chunk_hard_max_chars,
                    overlap_size=overlap_size,
                    combine_text_under_n=combine_text_under_n_chars,
                    multipage_ok=multipage_sections,
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

        elif chunk_type == "basic":
            # create chunks
            chunks = chunk_elements(clean_elements, **basic_kwargs)
            if write_to_file:
                # create the filenames and export chunks
                chunks_basic_fname = chunks_basic_template.format(
                    chunk_type=chunk_type,
                    soft_max=chunk_soft_max_chars,
                    hard_max=chunk_hard_max_chars,
                    overlap_size=overlap_size,
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
            if file_info:
                metadata_inputs.update(file_info)
            chunk_metadata = self.prepare_chunk_metadata(**metadata_inputs)
            chunk_dict = {"text": chunk.text, "metadata": chunk_metadata}
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
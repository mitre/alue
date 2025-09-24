import pytest
import os

from rag_utils import (
    DocumentProcessor, 
    ChromaInterface,
)

    
def test_single_document():
    current_dir = os.path.dirname(__file__)
    test_document = os.path.join(current_dir, "resources", "FAA_Order_8040.1C.pdf")
    output_dir = os.path.join(current_dir, "output")
    
    # create document processor and process test document
    processor = DocumentProcessor(
        document_directory_path=None, 
        output_path=output_dir
    )
    chunks = processor.process_single_document(
        document_path=test_document
    )
    print(f"Length of chunks {len(chunks)}")

    assert(len(chunks) == 9)
    assert(chunks[8]["text"] == "FAA Form 1320-19 (8-89)(Representation)")
    assert(chunks[8]["metadata"]["file_directory"] == "/hawes/projects/AVIATION_NLP/fy25/users/chen/repos/gitlab-alue/alue/tests/resources")
    assert(chunks[8]["metadata"]["filename"] == "FAA_Order_8040.1C.pdf")
    assert(chunks[8]["metadata"]["page_number"] == 4)
    assert(chunks[8]["metadata"]["chunk_id"] == "FAA_Order_8040.1C-8")


def test_database_creation():
    current_dir = os.path.dirname(__file__)
    test_document = os.path.join(current_dir, "resources", "FAA_Order_8040.1C.pdf")
    output_dir = os.path.join(current_dir, "output")
    
    # create document chunks
    processor = DocumentProcessor(
        document_directory_path=None, 
        output_path=output_dir
    )
    chunks = processor.process_single_document(
        document_path=test_document
    )

    # write to database
    chroma_interface = ChromaInterface(
        database_path=os.path.join(output_dir, "chromadb")
    )
    chroma_interface.add_document_chunks(
        collection_name="test_collection", 
        embedding_function=ChromaInterface.get_local_embedding_function(),
        document_chunks=chunks
    )

    

    # read from database
    collection = chroma_interface.get_collection(collection_name="test_collection")
    getResult = collection.get(ids=["FAA_Order_8040.1C-8"])
    queryResult = collection.query(query_texts=["disposition of records"], n_results=1)
    # assert presence of collection and contents
    assert(collection.count() == 9)
    assert(getResult["documents"][0] == "FAA Form 1320-19 (8-89)(Representation)")
    assert(queryResult["ids"][0][0] == "FAA_Order_8040.1C-6")







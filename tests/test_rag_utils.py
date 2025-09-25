import pytest
import os

from alue.rag_utils import (
    DocumentProcessor, 
    ChromaInterface,
    get_embedding_function
)


@pytest.fixture(scope="module")
def current_dir():
    return os.path.dirname(__file__)


@pytest.fixture(scope="module")
def output_dir(current_dir):
    return os.path.join(current_dir, "output")


@pytest.fixture(scope="module")
def auto_doc_processor(output_dir):
    return DocumentProcessor(
        document_directory_path=None, 
        output_path=output_dir,
        partition_strategy="auto",
    )


@pytest.fixture(scope="module")
def hi_res_doc_processor(output_dir):
    return DocumentProcessor(
        document_directory_path=None, 
        output_path=output_dir,
        partition_strategy="hi_res",
    )


@pytest.fixture(scope="module")
def chroma_interface(output_dir):
    return ChromaInterface(
        database_path=os.path.join(output_dir, "chromadb")
    )
    

def test_single_document_auto_chunking(current_dir, auto_doc_processor):
    test_document = os.path.join(current_dir, "resources", "JO_1030.1D_ATO_Safety_Guidance.pdf")

    # create document processor and process test document
    processor = auto_doc_processor
    chunks = processor.process_single_document(
        document_path=test_document
    )

    assert(len(chunks) == 13)
    assert(chunks[8]["text"] == (
        "(1) Coordination of ATO-SGs.\n\n(a) ATO Service Units. The OPR must provide the AJI-311 "
        "POC with a list of all ATO Service Units to which the ATO-SG should be distributed for "
        "coordination. The period for ATO grid coordination is 10 working days. The AJI-311 POC "
        "will submit all comments received to the OPR for adjudication and editing if required.\n\n"
        "(b) Acquisition Executive Board (AEB) and Acquisition System Advisory Group (ASAG) Approval "
        "of ATO-SG Information. If a proposed ATO-SG contains guidance related to the FAA Acquisition "
        "Management System, the AJI-311 POC must also submit the draft ATO-SG to the ASAG for review "
        "prior to AEB approval. The AJI-311 POC will submit all comments received to the OPR for "
        "adjudication and editing if required."
    ))
    assert(chunks[8]["metadata"]["filename"] == "JO_1030.1D_ATO_Safety_Guidance.pdf")
    assert(chunks[8]["metadata"]["page_number"] == 3)
    assert(chunks[8]["metadata"]["chunk_id"] == "JO_1030.1D_ATO_Safety_Guidance-8")

    
def test_single_document_hires_chunking(current_dir, hi_res_doc_processor):
    test_document = os.path.join(current_dir, "resources", "FAA_Order_8040.1C.pdf")
    
    # create document processor and process test document
    processor = hi_res_doc_processor
    chunks = processor.process_single_document(
        document_path=test_document
    )

    assert(len(chunks) == 9)
    assert(chunks[8]["text"] == "FAA Form 1320-19 (8-89)(Representation)")
    assert(chunks[8]["metadata"]["file_directory"] == "/hawes/projects/AVIATION_NLP/fy25/users/chen/repos/gitlab-alue/alue/tests/resources")
    assert(chunks[8]["metadata"]["filename"] == "FAA_Order_8040.1C.pdf")
    assert(chunks[8]["metadata"]["page_number"] == 4)
    assert(chunks[8]["metadata"]["chunk_id"] == "FAA_Order_8040.1C-8")


def test_database_creation(current_dir, hi_res_doc_processor, chroma_interface):
    test_document = os.path.join(current_dir, "resources", "FAA_Order_8040.1C.pdf")
    
    # create document chunks
    processor = hi_res_doc_processor
    chunks = processor.process_single_document(
        document_path=test_document
    )

    # write to database
    chroma_client = chroma_interface
    chroma_client.add_document_chunks(
        collection_name="test_collection", 
        embedding_function=get_embedding_function(),
        document_chunks=chunks
    )

        # read from database
    collection = chroma_client.get_or_create_collection(collection_name="test_collection")
    getResult = collection.get(ids=["FAA_Order_8040.1C-8"])
    queryResult = collection.query(query_texts=["disposition of records"], n_results=1)
    # assert presence of collection and contents
    assert(collection.count() == 9)
    assert(getResult["documents"][0] == "FAA Form 1320-19 (8-89)(Representation)")
    assert(queryResult["ids"][0][0] == "FAA_Order_8040.1C-6")


def test_query_collection(current_dir, hi_res_doc_processor, chroma_interface):
    test_document = os.path.join(current_dir, "resources", "FAA_Order_8040.1C.pdf")
    
    # create document chunks
    processor = hi_res_doc_processor
    chunks = processor.process_single_document(
        document_path=test_document
    )

    # write to database
    chroma_client = chroma_interface
    chroma_client.add_document_chunks(
        collection_name="test_collection", 
        embedding_function=get_embedding_function("local"),
        document_chunks=chunks
    )

    # query database
    formatted_results = chroma_client.query_collection(
        query="disposition of records", 
        collection_name="test_collection",
        embedding_function=get_embedding_function(),
        n_results=1
    )
    
    assert(formatted_results[0]["text"] == (
        "10. Quest for Information. You can get more information or ask questions "
        "about this order by contacting the Aircraft Certification Service, Aircraft "
        "Engineering Division, Delegation and Airworthiness Programs Branch (AIR-140), "
        "telephone (405) 954-4103.\n\n11. Records Management. Refer to FAA Orders 0000.1, "
        "FAA Standard Subject Classification System, 1350.14, Records Management; and "
        "1350.15, Records, Organization, Transfer, and\n\nDestruction Standards, or see "
        "your office Records Management Officer/Directives Management Officer for "
        "guidance regarding retention or disposition of records."
    ))
    assert(formatted_results[0]["metadata"]["filename"] == "FAA_Order_8040.1C.pdf")
    assert(formatted_results[0]["metadata"]["chunk_id"] == "FAA_Order_8040.1C-6")
    assert(formatted_results[0]["distance"] == 0.9738633632659912)



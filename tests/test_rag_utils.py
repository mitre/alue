import pytest
import os

from rag_utils import DocumentProcessor
    
def test_single_document():
    current_dir = os.path.dirname(__file__)
    test_document = os.path.join(current_dir, "resources", "FAA_Order_8040.1C.pdf")
    output_dir = os.path.join(current_dir, "output")
    
    processor = DocumentProcessor(None, output_dir)
    
    chunks = processor.process_single_document(test_document)
    assert(len(chunks) > 0)

import json
import random
from typing import (
    Any, 
    Dict, 
    List,
    Tuple
    )
from pathlib import Path


class DataLoader:
    """Unified data loader for all ALUE tasks.
    
    This class provides a consistent interface for loading and processing
    various ALUE task data formats including JSONL, JSON, and nested formats
    like SQuAD.
    
    Attributes:
        file_path: Path object pointing to the data file.
        raw_data: The raw loaded data from the file.
    """

    def __init__(self, 
                 file_path: str) -> None:
        """Initialize the DataLoader and load data from file.
        
        Args:
            file_path: Path to the JSON or JSONL file containing task data.
            
        Raises:
            FileNotFoundError: If the specified file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        self.file_path = Path(file_path)

        if self.file_path.suffix == '.jsonl':
            # Load JSONL file - each line is a separate JSON object
            with open(self.file_path) as f:
                self.raw_data = [json.loads(line.strip()) for line in f if line.strip()]
        else:
            with open(self.file_path) as f:
                self.raw_data = json.load(f)

    def get_examples(self, 
                     num_examples: int = 5, 
                     randomize: bool = False, 
                     seed: int = 42) -> List[Dict[str, Any]]:
        """Extract few-shot examples from the dataset.
        
        Args:
            num_examples: Number of examples to retrieve. Defaults to 5.
            randomize: Whether to randomly sample examples. Defaults to False.
            seed: Random seed for reproducibility when randomize is True. 
                Defaults to 42.
                
        Returns:
            A list of example dictionaries in normalized format.
        """
        if num_examples <= 0:
            return []
            
        examples = self._extract_examples()
        return self._sample_data(examples, num_examples, randomize, seed)

    def get_test_data(self) -> List[Dict[str, Any]]:
        """Get all test data for inference or evaluation.
        
        Returns:
            A list of test data items in normalized format.
        """
        return self._extract_test_data()

    def get_task_info(self) -> Dict[str, str]:
        """Extract task metadata and information.
        
        Returns:
            A dictionary containing task information such as name, description,
            and other metadata. Returns empty dict if no task info exists.
        """
        return self.raw_data.get("task_info", {})

    def _extract_examples(self) -> List[Dict[str, Any]]:
        """Extract the examples section from raw data.
        
        This method handles multiple data formats including JSONL with split
        fields and nested JSON structures. Examples are never confused with
        test data.
        
        Returns:
            A list of normalized example dictionaries.
        """
        # Handle JSONL format with split field
        if isinstance(self.raw_data, list) and any(item.get("split") for item in self.raw_data):
            examples = [item for item in self.raw_data if item.get("split") == "example"]
            return self._normalize_items(examples)
        
        if "examples" in self.raw_data:
            return self._normalize_items(self.raw_data["examples"])
        
        if "data" in self.raw_data and isinstance(self.raw_data["data"], list):
            for item in self.raw_data["data"]:
                if "examples" in item:
                    return self._normalize_items(item["examples"])
        return []

    def _extract_test_data(self) -> List[Dict[str, Any]]:
        """Extract the test data section from raw data.
        
        This method handles multiple formats including JSONL with split fields,
        nested SQuAD format with paragraphs, and standard list formats.
        
        Returns:
            A list of normalized test data dictionaries.
        """
        # Handle JSONL format with split field
        if isinstance(self.raw_data, list) and any(item.get("split") for item in self.raw_data):
            test_data = [item for item in self.raw_data if item.get("split") == "test"]
            return self._normalize_items(test_data)
        
        # Handle nested SQuAD format with paragraphs
        if "data" in self.raw_data and isinstance(self.raw_data["data"], list):
            test_items = []
            for item in self.raw_data["data"]:
                if "paragraphs" in item:
                    # Flatten paragraph QA pairs into individual items
                    for para in item["paragraphs"]:
                        context = para.get("context", "")
                        for qa in para.get("qas", []):
                            test_items.append({
                                "input": qa.get("question", ""),
                                "output": qa.get("answers", [{}])[0].get("text", ""),
                                "context": context,
                                "metadata": {"format": "squad", "id": qa.get("id", "")}
                            })
                    return test_items
        
        # Original logic for other formats
        if "data" in self.raw_data and isinstance(self.raw_data["data"], list):
            return self._normalize_items(self.raw_data["data"])
        elif isinstance(self.raw_data, list):
            return self._normalize_items(self.raw_data)
        
        return []

    def _normalize_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert items to a standard normalized format.
        
        This method handles different input formats including RAG format with
        queries, classification format with text and labels, and items already
        in standard format.
        
        Args:
            items: A list of dictionaries in various formats.
            
        Returns:
            A list of dictionaries in standardized format with 'input', 
            'output', and optional 'context' and 'metadata' fields.
        """
        normalized = []
        for item in items:
            # Handle different input formats
            if "query" in item:
                # RAG format
                normalized.append({
                    "input": item["query"],
                    "output": item.get("answer", ""),
                    "context": item.get("context", ""),
                    "metadata": {"format": "rag"}
                })
            elif "text" in item:
                # Simple classification format
                normalized.append({
                    "input": item["text"],
                    "output": item.get("label", ""),
                    "metadata": {"format": "classification"}
                })
            else:
                # Already in standard format or close to it
                normalized.append(item)
        return normalized

    def _sample_data(self, data: List[Dict], num_examples: int, randomize: bool, seed: int) -> List[Dict]:
        """Sample data with optional randomization.
        
        Args:
            data: The list of data items to sample from.
            num_examples: Number of examples to sample.
            randomize: Whether to randomly sample or take first N items.
            seed: Random seed for reproducibility when randomize is True.
            
        Returns:
            A sampled list of data items, either randomized or sequential.
        """
        if not data:
            return []
            
        if randomize:
            random.seed(seed)
            data = random.sample(data, min(len(data), num_examples))
        else:
            data = data[:num_examples]
            
        return data


def load_data(file_path: str) -> DataLoader:
    """Load any ALUE dataset and return a DataLoader instance.
    
    Args:
        file_path: Path to the dataset file (JSON or JSONL).
        
    Returns:
        A DataLoader instance initialized with the specified file.
    """
    return DataLoader(file_path)


def load_task_data(file_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load task data with both examples and test data.
    
    This is a convenience function that loads a dataset and returns both
    the few-shot examples and the test data in a single call.
    
    Args:
        file_path: Path to the dataset file (JSON or JSONL).
        
    Returns:
        A tuple containing (examples, test_data) where both are lists of
        normalized data dictionaries.
    """
    loader = DataLoader(file_path)
    return loader.get_examples(), loader.get_test_data()
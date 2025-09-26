import json
import random
from typing import Any, Dict, List
from pathlib import Path


class DataLoader:
    """Unified data loader for all ALUE tasks."""

    def __init__(self, file_path: str):
        """Load and normalize data from JSON file."""
        self.file_path = Path(file_path)

        if self.file_path.suffix == '.jsonl':
            # Load JSONL file - each line is a separate JSON object
            with open(self.file_path) as f:
                self.raw_data = [json.loads(line.strip()) for line in f if line.strip()]
        else:
            with open(self.file_path) as f:
                self.raw_data = json.load(f)

    def get_examples(self, num_examples: int = 5, randomize: bool = False, seed: int = 42) -> List[Dict[str, Any]]:
        """Extract few-shot examples."""
        if num_examples <= 0:
            return []
            
        examples = self._extract_examples()
        return self._sample_data(examples, num_examples, randomize, seed)

    def get_test_data(self) -> List[Dict[str, Any]]:
        """Get all test data for inference/evaluation."""
        return self._extract_test_data()

    def get_task_info(self) -> Dict[str, str]:
        """Extract task information."""
        return self.raw_data.get("task_info", {})

    def _extract_examples(self) -> List[Dict[str, Any]]:
        """Extract examples section (never test data)."""
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
        """Extract test data section.""" 

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
        """Convert items to standard format."""
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
        """Sample data with optional randomization."""
        if not data:
            return []
            
        if randomize:
            random.seed(seed)
            data = random.sample(data, min(len(data), num_examples))
        else:
            data = data[:num_examples]
            
        return data


# Simplified interface functions
def load_data(file_path: str) -> DataLoader:
    """Load any ALUE dataset."""
    return DataLoader(file_path)


def load_task_data(file_path: str) -> tuple[List[Dict], List[Dict]]:
    """Generic function to load any task data. Returns (examples, test_data)."""
    loader = DataLoader(file_path)
    return loader.get_examples(), loader.get_test_data()
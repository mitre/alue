import json
import random
from typing import Any, Dict, List
from pathlib import Path


class DataLoader:
    """Unified data loader for all ALUE tasks."""

    def __init__(self, file_path: str):
        """Load and normalize data from JSON file."""
        self.file_path = Path(file_path)
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
        if "examples" in self.raw_data:
            return self._normalize_items(self.raw_data["examples"])
        return []

    def _extract_test_data(self) -> List[Dict[str, Any]]:
        """Extract test data section."""
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
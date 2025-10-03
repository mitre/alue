from pydantic import BaseModel
from enum import Enum

class AnswerChoice(str, Enum):
    """Enum for multiple choice answers"""
    
    A = "A"
    B = "B"
    C = "C"

class MCQAResponse(BaseModel):
    """Schema for Multiple Choice Question Answering response"""
    answer: AnswerChoice
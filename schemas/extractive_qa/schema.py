from pydantic import BaseModel

class ExtractiveQAResponse(BaseModel):
    answer: str

from pydantic import BaseModel
from typing import List

class TailNumberResponse(BaseModel):
    tail_numbers: List[str]
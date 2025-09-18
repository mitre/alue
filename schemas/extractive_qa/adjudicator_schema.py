from typing import Literal

from pydantic import BaseModel


class Score(BaseModel):
    score: Literal[0, 1]  # Restrict the score to only 0 or 1

    class Config:
        title = "Score"

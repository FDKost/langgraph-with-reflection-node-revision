from dataclasses import dataclass, field
from typing import List

@dataclass
class State:
    question: str
    answer: str = ""
    round: int = 0
    verdict: str = ""
    critique: List[str] = field(default_factory=list)

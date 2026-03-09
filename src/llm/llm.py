from typing import Protocol

from src.models import Structure

class LLM(Protocol):

    async def ask(self, system: str, user: str) -> str:
        ...

    async def structured_output(self, question: str, structure: Structure) -> Structure:
        ...

    def estimate_question_tokens(self, system: str, user: str) -> int:
        ...
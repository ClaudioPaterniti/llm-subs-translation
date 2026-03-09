import groq
from groq import AsyncGroq
from pydantic import BaseModel
from typing import TypeVar, Type

from src.models import RetriableException, InvalidJsonException, Structure
import src.logger as logger

class GroqClient:
    def __init__(self, key: str, model: str, prompt: str, config: dict = None):
        self.model = model
        self.prompt = prompt
        self.config = config or {}
        # Groq uses separate clients for sync and async
        self.client = AsyncGroq(api_key=key)

    async def ask(self, system: str, user: str) -> str:

        full_prompt = f"{self.prompt}\n{system}"
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": full_prompt},
                    {"role": "user", "content": user},
                ],
                **self.config
            )
            return response.choices[0].message.content
        except groq.RateLimitError as ex:
            raise RetriableException(str(ex))
        except groq.APIStatusError as ex:
            if ex.status_code in {502, 503, 504}:
                raise RetriableException(str(ex))
            raise ex

    async def structured_output(self, question: str, structure: Type[Structure]) -> Structure:
        try:
            # Groq's SDK supports Pydantic parsing directly via this helper
            completion = await self.client.chat.completions.with_structured_output(structure).create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": question},
                ],
                **self.config
            )

            if completion is None:
                raise InvalidJsonException("Groq response could not be parsed into structure")

            return completion

        except groq.RateLimitError as ex:
            raise RetriableException(str(ex))
        except Exception as ex:
            # Catching general parsing errors or API errors
            if hasattr(ex, 'status_code') and ex.status_code in {502, 503, 504}:
                raise RetriableException(str(ex))
            raise ex

    def estimate_question_tokens(self, system: str, user: str) -> int:
        # Simple heuristic estimation
        total_text = '\n'.join((self.prompt, system, user))
        return int(len(total_text)/4 + 20)
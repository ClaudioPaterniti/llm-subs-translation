from pydantic import BaseModel
from typing import TypeVar

from google import genai
from google.genai.errors import ClientError, ServerError
from google.genai.types import GenerateContentResponse, GenerateContentConfig

from src.models import RetriableException, InvalidJsonException, Structure
import src.logger as logger

class GeminiClient:

    def __init__(self,
            key: str, model: str, prompt: str, config: dict = None):
        self.model = model
        self.prompt = prompt
        self.config = config or {}

        self.client = genai.Client(api_key=key)

    async def ask(self, system: str, user: str) -> str:

        full_prompt = f"{self.prompt}\n{system}"
        config = GenerateContentConfig(
            system_instruction=full_prompt,
            temperature=0.2
        )

        try:
            response = await self.client.aio.models.generate_content(
                model=self.model, contents=user,
                config=config
            )
        except (ClientError, ServerError) as ex:
            if ex.status in {'RESOURCE_EXHAUSTED', 'UNAVAILABLE'}:
                raise RetriableException(ex.message)
            else:
                raise ex

        return response.text


    async def structured_output(self, question: str, structure: Structure) -> Structure:

        config= self.config | {
            "response_mime_type": "application/json",
            "response_schema": structure
        }

        try:
            full_question = self.prompt + '\n' + question
            response = await self.client.aio.models.generate_content(
                model=self.model, contents=full_question,
                config=config
            )
        except (ClientError, ServerError) as ex:
            if ex.status in {'RESOURCE_EXHAUSTED', 'UNAVAILABLE'}:
                raise RetriableException(ex.message)
            else:
                raise ex

        if response.parsed is None:
            raise InvalidJsonException("LLM response could not be parsed")

        return response.parsed

    def estimate_question_tokens(self, system: str, user: str) -> int:
        # Simple heuristic estimation
        total_text = '\n'.join((self.prompt, system, user))
        return int(len(total_text)/4 + 20)
import ollama
from pydantic import BaseModel
from src.models import RetriableException, InvalidJsonException, Structure
import src.logger as logger

class OllamaClient:
    def __init__(self, model: str, prompt: str, config: dict = None):
        self.model = model
        self.prompt = prompt
        self.config = config or {}
        self._client = ollama.AsyncClient()

    async def ask(self, system: str, user: str) -> str:
        full_prompt = f"{self.prompt}\n{system}"

        try:
            response = await self._client.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': full_prompt},
                    {'role': 'user', 'content': user}
                ],
                think=False,
                options=self.config
            )
            return response['message']['content']
        except Exception as ex:
            if "connection refused" in str(ex).lower():
                raise RetriableException("Ollama service is unavailable.")
            raise ex

    async def structured_output(self, question: str, structure: Structure) -> Structure:
        full_prompt = f"{self.prompt}\n{question}"

        try:
            response = await self._client.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': full_prompt}],
                format=structure.model_json_schema(),
                think=False,
                options=self.config
            )

            return structure.model_validate_json(response['message']['content'])

        except Exception as ex:
            # If the model fails to return valid JSON matching the schema
            raise InvalidJsonException(f"LLM response could not be parsed: {ex}")

    def estimate_question_tokens(self, system: str, user: str) -> int:
        # Simple heuristic estimation
        total_text = '\n'.join((self.prompt, system, user))
        return int(len(total_text)/4 + 20)
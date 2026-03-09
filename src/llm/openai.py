import openai
from src.models import RetriableException, InvalidJsonException, Structure

class OpenAIClient:
    def __init__(
            self, model: str, prompt: str, api_key: str = None,
            base_url: str = None, config: dict = None):
        self.model = model
        self.prompt = prompt
        self.config = config or {}
        self._client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key= api_key or "dummy"
        )

    async def ask(self, system: str, user: str) -> str:
        full_system_prompt = f"{self.prompt}\n{system}"

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': full_system_prompt},
                    {'role': 'user', 'content': user}
                ],
                max_tokens=32768,
                temperature=0.7,
                top_p=0.8,
                presence_penalty=1.5,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            return response.choices[0].message.content
        except openai.APIConnectionError as ex:
            raise RetriableException(f"OpenAI service is unreachable: {ex}")
        except Exception as ex:
            raise ex

    async def structured_output(self, question: str, structure: type[Structure]) -> Structure:
        full_prompt = f"{self.prompt}\n{question}"

        try:
            # Using OpenAI's native 'parsed' response for Pydantic models
            completion = await self._client.beta.chat.completions.parse(
                model=self.model,
                messages=[{'role': 'user', 'content': full_prompt}],
                response_format=structure,
                **self.config
            )

            return completion.choices[0].message.parsed

        except Exception as ex:
            # Handles validation errors or API failures
            raise InvalidJsonException(f"OpenAI response could not be parsed: {ex}")

    def estimate_question_tokens(self, system: str, user: str) -> int:
        # Simple heuristic estimation
        total_text = '\n'.join((self.prompt, system, user))
        return int(len(total_text)/4 + 20)
import asyncio

from src.models import *
from src.rate_limiter import RateLimitedLLM
import src.logger as logger

class LineTranslator:

    def __init__(
            self,
            llm: RateLimitedLLM):
        self.llm = llm

    async def __call__(self, filename: str, dialogue: list[str]) -> TranslationOutput:
        translated = await self._split_and_translate(filename, dialogue)

        return TranslationOutput(filename, translated)

    async def _split_and_translate(
            self, chunk_id: str, dialogue: list[str]) -> list[str]:
        results = []
        for i in range(0, len(dialogue), 100):
            batch = dialogue[i : i + 100]
            logger.info(f"{chunk_id}: translating lines {i}-{i + len(batch)} / {len(dialogue)}")
            try:
                async with asyncio.TaskGroup() as tg:
                    tasks = [
                        tg.create_task(self.llm.ask(f"{chunk_id}.{i+idx+1}", '', line))
                        for idx, line in enumerate(batch)
                    ]
                # Collect results for this batch
                results.extend([t.result() for t in tasks])

            except* Exception as exs:
                raise exs.exceptions[0]

        return results
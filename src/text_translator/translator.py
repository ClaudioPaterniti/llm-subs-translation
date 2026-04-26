import asyncio
import traceback
import re

from string import Template
from math import ceil
from itertools import chain

from src.models import *
from src.rate_limiter import RateLimitedLLM
from src.utils import balanced_partition
import src.logger as logger

from importlib import resources

prompt = Template(resources.files(__package__).joinpath("prompt.md").read_text())

class TextTranslator:

    def __init__(
            self,
            llm: RateLimitedLLM,
            chunk_lines: int):
        self.llm = llm
        self.chunk_lines = chunk_lines
        self._prompt = prompt.substitute(lines_per_chunk= self.chunk_lines)

    async def __call__(self, filename: str, dialogue: list[str]) -> TranslationOutput:
        translated = await self._split_and_translate(
            filename, dialogue, self.chunk_lines)

        return TranslationOutput(filename, translated)

    async def _split_and_translate(
            self, chunk_id: str, dialogue: list[str],
            chunk_lines: int, reduce_retry: bool = True) -> list[str]:
        chunks = [
            dialogue[slc]
            for slc in balanced_partition(len(dialogue), max= chunk_lines)]

        chunk_id = f"{chunk_id}.{{}}" if len(chunks) > 1 else chunk_id
        try:
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(
                        self._translate_block(chunk_id.format(i+1), chunk, reduce_retry))
                    for i, chunk in enumerate(chunks)]

        except* Exception as exs:
            raise exs.exceptions[0]

        return list(chain.from_iterable(t.result() for t in tasks))

    async def _translate_block(
            self, chunk_id: str, dialogue: list[str], reduce_retry: bool) -> list[str]:

        text = '\n'.join([f"<line>{line}</line>" for line in dialogue])
        resp = await self.llm.ask(chunk_id, self._prompt, text)
        lines = [line.replace('</line>', '').strip() for line in resp.split('<line>')[1:]]
        if len(lines) != len(dialogue):
            if reduce_retry:
                logger.warning(f"{chunk_id}: response lines number does not match original dialogue, retrying with reduced context")
                chunk_lines = ceil((len(dialogue))/2)
                return await self._split_and_translate(chunk_id, dialogue, chunk_lines, False)
            else:
                raise MisalignmentException(f"{chunk_id}: response lines number does not match original dialogue")
        return lines
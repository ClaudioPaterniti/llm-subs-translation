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
split_regex = re.compile(r'\s*Line\s+\d+\s*-\s*')
part_regex = re.compile(r'^\s*Part\s+\d+\s*$', re.MULTILINE)

class TextTranslator:

    def __init__(
            self,
            llm: RateLimitedLLM,
            chunk_lines: int,
            chunks_per_request: int = 1):
        self.llm = llm
        self.chunk_lines = chunk_lines
        self.chunks_per_request = chunks_per_request

    async def __call__(self, filename: str, dialogue: list[str]) -> TranslationOutput:
        translated = await self._split_and_translate(
            filename, dialogue, self.chunk_lines, self.chunks_per_request)

        return TranslationOutput(filename, translated)

    async def _split_and_translate(
            self, chunk_id: str, dialogue: list[str],
            chunk_lines: int, chunks_per_request: int) -> list[str]:
        chunks = [
            dialogue[slc]
            for slc in balanced_partition(len(dialogue), max= chunk_lines*chunks_per_request)]

        chunk_id = f"{chunk_id}.{{}}" if len(chunks) > 1 else chunk_id
        try:
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(
                        self._translate_block(chunk_id.format(i+1), chunk, chunks_per_request))
                    for i, chunk in enumerate(chunks)]

        except* Exception as exs:
            raise exs.exceptions[0]

        return list(chain.from_iterable(t.result() for t in tasks))

    async def _translate_block(
            self, chunk_id: str, dialogue: list[str], chunks_per_request: int) -> list[str]:
        blocks = []
        for i, slc in enumerate(balanced_partition(len(dialogue), chunks_per_request)):
            blocks.append(f'Part {i+1}\n\n')
            blocks.append('\n'.join([f"Line {h} - {line}" for h, line in enumerate(dialogue[slc])]))
        text = '\n\n'.join(blocks)
        question = prompt.substitute(lines_per_chunk= self.chunk_lines, text= text)
        resp = await self.llm.ask(chunk_id, question)
        resp = part_regex.sub('', resp)
        lines = [line.strip() for line in split_regex.split(resp)][1:]
        if len(lines) != len(dialogue):
            if len(dialogue) > self.chunk_lines*self.chunks_per_request/2:
                logger.warning(f"{chunk_id}: response lines number does not match original dialogue, retrying with reduced context")
                return await self._split_and_translate(chunk_id, dialogue, self.chunk_lines/2, chunks_per_request)
            else:
                raise MisalignmentException(f"{chunk_id}: response lines number does not match original dialogue")
        return lines
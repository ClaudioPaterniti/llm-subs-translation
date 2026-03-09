from typing import Optional, Any, Protocol, TypeVar
from dataclasses import dataclass

from pydantic import BaseModel

Structure = TypeVar('Structure', bound=BaseModel)

class AssIgnore(BaseModel):
    field: str
    values: set[str]
    _field_i: int

class AssSettings(BaseModel):
    keep_formats: Optional[bool] = True
    use_characters: Optional[bool] = True
    remove_complex_lines: Optional[bool] = True
    ignore: Optional[list[AssIgnore]] = None

class Config(BaseModel):
    api: str
    base_url: Optional[str] = None
    key: str = None
    original_language: str
    translate_to: str
    outfile_suffix: str
    model: str = "gemini-2.0-flash-lite"
    translator_type: str = "text"
    lines_per_chunk: int = 500
    chunks_per_request: int = 10
    requests_per_minutes: int = 15
    token_per_minutes: int = 1000000
    max_concurrent_requests: Optional[int] = None
    llm_config: dict[str, Any] = {}
    max_retries: int = 2
    ass_settings: AssSettings
    debug: bool = False

@dataclass
class TranslationOutput:
    name: str
    dialogue: list[str]
    misalignments: list[tuple[int, int]] = None

class MisalignmentException(Exception):
    pass

class RetriableException(Exception):
    pass

class InvalidJsonException(Exception):
    pass

class TranslationFile(Protocol):
    def get_dialogue(self) -> list[str]:
        """Return a simple dialogue as a list of lines"""
        ...

    def map_dialogue_lines(self, lines: list[int]) -> list[int]:
        """Map simple dialogue line numbers to the corresponding lines in the final file"""
        ...

    def get_translation(self, translation: TranslationOutput) -> str:
        """Recompose the final file structure with the translated dialogue"""
        ...

class Translator(Protocol):
    async def __call__(self, filename: str, dialogue: list[str]) -> TranslationOutput: ...

class DialogueChunk(BaseModel):
    from_line: int
    to_line: int
    dialogue: list[str]
    _translated: list[str] = None

class DialogueChunks(BaseModel):
    chunks: list[DialogueChunk]
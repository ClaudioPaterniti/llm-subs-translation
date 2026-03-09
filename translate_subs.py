import os
import sys
import asyncio
import glob
import traceback

from string import Template

from src.models import *
from src.rate_limiter import RateLimitedLLM
from src.translate_file import TranslateFileTask

import src.logger as logger

def translated_path(file_path: str, suffix: str) -> str:
    path, file = os.path.split(file_path)
    filename, ext = os.path.splitext(file)
    out_file_name = f'{filename}{suffix}{ext}'
    full_path = os.path.join(path, out_file_name)
    return full_path

async def translate_file(task: TranslateFileTask, semaphore: asyncio.Semaphore):
    async with semaphore: # avoid files being loded all at once
        try:
            await task()
        except Exception as ex:
            logger.error(f"{task.filename} failed: {ex}", save=True)
            logger.debug(traceback.format_exc())

async def main(llm: RateLimitedLLM, file_paths: list[str], config: Config):
    semaphore = asyncio.Semaphore(config.max_concurrent_requests or config.requests_per_minutes)

    if config.translator_type == 'json':
        from src.json_translator.translator import JsonChunkerTranslator
        translator = JsonChunkerTranslator(llm, config.lines_per_chunk, config.chunks_per_request)
    elif config.translator_type == 'line':
        from src.line_translator.translator import LineTranslator
        translator = LineTranslator(llm)
    else:
        from src.text_translator.translator import TextTranslator
        translator = TextTranslator(llm, config.lines_per_chunk)

    async with asyncio.TaskGroup() as tg:
        for file_path in file_paths:
            out_path = translated_path(file_path, config.outfile_suffix)
            translation_task = TranslateFileTask(translator, file_path, out_path, config.ass_settings)
            tg.create_task(translate_file(translation_task, semaphore))

    print('\n')
    logger.info(f'Terminated - final log:')
    logger.print_final_log()

def build_system_prompt(conf: Config) -> str:
    prompt = [f"You have to translate movie subtitles from {conf.original_language} to {conf.translate_to}."]
    if conf.ass_settings.use_characters:
        prompt.append("Each line will have the pattern '[Character]: text' keep the same structre without translating the character name.")
    if conf.ass_settings.keep_formats:
        prompt.append("Do not translate blocks like '{{format #i}}' between braces or html tags like <tag>.")
    prompt.append("Translate the dialogue and answer only with the translation without adding anything else.")
    return '\n'.join(prompt)


if __name__ == '__main__':
    script_path = os.path.abspath(os.path.split(__file__)[0])

    with (
            open(os.path.join(script_path, 'config.json'), 'r') as config_fp,
            open(os.path.join(script_path, 'user_prompt.md'), 'r') as user_prompt_fp,
        ):
        config = Config.model_validate_json(config_fp.read())
        user_prompt = user_prompt_fp.read()

    system_prompt = build_system_prompt(config)

    if config.key:
        api_key_var = f'{config.key}_key'.upper()
        api_key_file = f'{config.key}.key'
        key = os.environ.get(api_key_var)
        if key is None and os.path.exists(os.path.join(script_path, api_key_file)):
                with open(os.path.join(script_path, api_key_file), 'r') as key_fp:
                    key = key_fp.read()

        if not key:
            logger.error(f"Could not retrieve llm key, populate env variable {api_key_var} or file {api_key_file}")
            sys.exit()
    else:
        key = None

    logger.debug_enabled = config.debug
    prompt = user_prompt + '\n' + system_prompt

    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        folder = glob.escape(sys.argv[1])
        file_paths = glob.glob(f'{folder}/*.ass') + glob.glob(f'{folder}/*.srt')
    else:
        file_paths = [f for f in sys.argv[1:] if f.endswith('.ass') or f.endswith('.srt')]
        folder, _ = os.path.split(file_paths[0])
        folder = glob.escape(folder)

    translated = glob.glob(f'{folder}/*{config.outfile_suffix}.ass') + glob.glob(f'{folder}/*{config.outfile_suffix}.srt')

    to_translate = [f for f in file_paths
                    if not f[:-4].endswith(f'{config.outfile_suffix}')
                    and translated_path(f, config.outfile_suffix) not in translated]

    if not to_translate:
        logger.warning("Found no file to translate, already translated files are ignored.")
        sys.exit()

    if config.api == 'gemini':
        from src.llm.gemini import GeminiClient
        client = GeminiClient(
            key=key,
            model=config.model,
            prompt=prompt,
            config=config.llm_config
        )
        logger.info("Using gemini")
    elif config.api == 'groq':
        from src.llm.groq import GroqClient
        client = GroqClient(
            key=key,
            model=config.model,
            prompt=prompt,
            config=config.llm_config
        )
        logger.info("Using groq")
    elif config.api == 'ollama':
        from src.llm.ollama import OllamaClient
        client = OllamaClient(
            model=config.model,
            prompt=prompt,
            config=config.llm_config
        )
        logger.info("Using ollama")
    elif config.api == 'openai':
        from src.llm.openai import OpenAIClient
        client = OpenAIClient(
            model=config.model,
            prompt=prompt,
            base_url= config.base_url,
            config=config.llm_config
        )
        logger.info("Using openai")
    else:
        logger.error(f"Api {config.api} not supported")
        sys.exit()

    queue = RateLimitedLLM(
        client=client,
        requests_per_minute=config.requests_per_minutes,
        tokens_per_minute=config.token_per_minutes,
        max_retries=config.max_retries,
        max_concurrent_requests=config.max_concurrent_requests,
        verbose= config.translator_type != 'line'
    )

    asyncio.run(main(queue, to_translate, config))
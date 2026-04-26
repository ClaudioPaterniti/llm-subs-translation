import re

from typing import ClassVar

from src.models import TranslationFile, AssSettings
import src.logger as logger

class AssTranslationFile(TranslationFile):
    alpha_ratio_thresh = 0.8

    char_regex: ClassVar[re.Pattern] = re.compile(r'\[[^[\]]+\]:')
    command_regex: ClassVar[re.Pattern] = re.compile(r'\{[^{}]+\}')
    alpha_regex: ClassVar[re.Pattern] =  re.compile(r"[a-zA-Z \"?!,.]")

    def __init__(self, text: str, settings: AssSettings):
        self.settings = settings
        splitted = text.strip().split('[Events]', 1)
        if len(splitted) == 1: raise ValueError("Invalid .ass format, [Events] tag not found")
        subs = [s for s in splitted[1].split('\n') if s.strip()]
        if not subs[0].strip().lower().startswith('format'): raise ValueError("Invalid .ass format, 'Format:' line not found")
        self._header = splitted[0] + '[Events]\n' + subs[0] # from start to 'Format:...' line included
        self._format = {
            s.strip().lower(): i
            for i, s in enumerate(subs[0].replace('Format:', '').split(','))
        }

        self._name_i = self._format.get('name')

        for rule in settings.ignore:
            rule._field_i = self._format.get(rule.field.lower()) # maps field name to field position
        self._ignore = [r for r in settings.ignore if r._field_i is not None]
        self._body = self._parse_body(subs[1:])

        # .ass format commands '{...}' are replaced with dummy tokens '{format 1}' to simplify dialogue for translation
        self._commands: dict[str, str] = {} # maps dummy tokens to original commands for final restore
        self._dialogue_i = self._process_body(self._body)

    def _parse_body(self, lines: list[str]) -> list[list[str]]:
        """
        Returns a list of lists, lines that respects the format are returned as a list of the field
        values, other lines are left as a single string value.
        """
        body = []
        for l in lines:
            splitted = l.split(':', 1)
            if len(splitted) == 2:
                event, value = splitted
                fields = value.split(',', len(self._format)-1)
                if len(fields) == len(self._format):
                    body.append([event] + fields)
                else:
                    body.append([l])
            else:
                body.append([l])

        return body

    def _sub_commands(self, m: re.Match) -> str:
        if not self.settings.keep_formats:
            return ''
        token = f"{{format {len(self._commands)}}}"
        self._commands[token] = m.group(0)
        return token

    def _restore_commands(self, m: re.Match) -> str:
        return self._commands.get(m.group(0), '{}')

    def _process_body(self, body: list[list[str]]) -> list[int]:
        """
        Update the body replacing the commands and logging them into self._commands.
        Returns the indeces of the selected dialogue lines.
        """
        dialogue_i = []

        for i, line in enumerate(body):
            if len(line) == len(self._format) + 1 and line[0].strip().lower() == 'dialogue':
                sub = self.command_regex.sub(self._sub_commands, line[-1].strip())
                ignored = False
                for rule in self._ignore:
                    if line[rule._field_i + 1].strip() in rule.values:
                        ignored = True
                        break
                if ignored: continue
                clean = self.command_regex.sub('', sub).strip()
                if (
                    len(clean) > 1
                    and len(self.alpha_regex.findall(clean))/len(clean) < self.alpha_ratio_thresh
                ):
                    continue
                line[-1] = sub
                dialogue_i.append(i)

        return dialogue_i

    def get_dialogue(self):
        if self._name_i is not None and self.settings.use_characters:
            return [
                f"[{self._body[i][self._name_i + 1] or 'Unknown'}]: {self._body[i][-1]}"
                for i in self._dialogue_i
            ]
        else: return [self._body[i][-1] for i in self._dialogue_i]

    def map_dialogue_lines(self, lines: list[int]) -> list[int]:
        offset = self._header.count('\n') + 1
        return [offset + self._dialogue_i[i] for i in lines]

    def get_translation(self, translation: list[str]):
        if len(self._dialogue_i) != len(translation): raise Exception("Lines count mismatch")
        lines = []
        dialogue_i = 0
        for i, line in enumerate(self._body):
            if len(line) == len(self._format) + 1:
                sub = line[-1]
                if dialogue_i < len(self._dialogue_i) and i == self._dialogue_i[dialogue_i]:
                    sub = translation[dialogue_i]
                    if self._name_i is not None and self.settings.use_characters:
                        sub = self.char_regex.split(sub, maxsplit=1)[-1].strip()
                    sub = self.command_regex.sub(self._restore_commands, sub)
                    dialogue_i += 1
                lines.append(f"{line[0]}:{','.join(line[1:-1])},{sub}")
            else:
                lines.append(line[0])

        return  '\n'.join([self._header] + lines)
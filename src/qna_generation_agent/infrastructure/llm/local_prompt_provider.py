"""Local prompt provider backed by .prompt.md assets."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import Any

from qna_generation_agent.app.logging import get_logger
from qna_generation_agent.application.errors import PermanentError
from qna_generation_agent.application.ports.prompt_provider import (
    Prompt,
    PromptProvider,
)

logger = get_logger(__name__)

_PROMPT_PACKAGE = "qna_generation_agent.infrastructure.llm"
_PROMPT_ASSET_DIR = "prompt_assets"


@dataclass(frozen=True, slots=True)
class _PromptRecord:
    prompt: Prompt
    source_name: str


def _normalize_heading(value: str) -> str:
    return value.strip().lower().replace("_", " ")


def _parse_scalar(value: str) -> Any:
    raw = value.strip()
    if raw in {"true", "false"}:
        return raw == "true"
    if raw.isdigit():
        return int(raw)
    if (raw.startswith('"') and raw.endswith('"')) or (
        raw.startswith("'") and raw.endswith("'")
    ):
        return raw[1:-1]
    return raw


def _parse_frontmatter(lines: list[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        index += 1
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError(f"Invalid frontmatter line: {line!r}")
        key, raw_value = line.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if raw_value:
            metadata[key] = _parse_scalar(raw_value)
            continue

        if index < len(lines) and lines[index].lstrip().startswith("- "):
            items: list[str] = []
            while index < len(lines):
                item_line = lines[index]
                if not item_line.lstrip().startswith("- "):
                    break
                items.append(str(_parse_scalar(item_line.lstrip()[2:])))
                index += 1
            metadata[key] = items
            continue

        metadata[key] = ""
    return metadata


def _split_prompt_sections(body: str) -> tuple[str, str]:
    sections: dict[str, list[str]] = {}
    current_section = ""
    current_lines: list[str] = []

    def _commit() -> None:
        if current_section:
            sections[current_section] = current_lines.copy()

    for line in body.splitlines():
        if line.startswith("## "):
            _commit()
            current_section = _normalize_heading(line[3:])
            current_lines = []
            continue
        current_lines.append(line)

    _commit()

    system_section = sections.get("system prompt") or sections.get("system") or []
    user_section = sections.get("user prompt") or sections.get("user") or []

    system_prompt = "\n".join(system_section).strip()
    user_prompt = "\n".join(user_section).strip()
    if not system_prompt and not user_prompt:
        raise ValueError("Prompt asset is missing system/user sections")
    return system_prompt, user_prompt


def _load_prompt_asset(asset_path: Any) -> _PromptRecord:
    raw_text = asset_path.read_text(encoding="utf-8")
    lines = raw_text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"Prompt asset {asset_path.name} is missing frontmatter")

    try:
        end_index = next(index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration as error:
        raise ValueError(f"Prompt asset {asset_path.name} has unterminated frontmatter") from error

    metadata = _parse_frontmatter(lines[1:end_index])
    name = str(metadata.pop("name"))
    version = str(metadata.pop("version"))
    aliases_raw = metadata.pop("aliases", [])
    aliases = tuple(str(alias) for alias in aliases_raw) if isinstance(aliases_raw, list) else ()
    system_prompt, user_prompt = _split_prompt_sections("\n".join(lines[end_index + 1 :]))

    prompt = Prompt(
        name=name,
        version=version,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        aliases=aliases,
        metadata=metadata,
    )
    return _PromptRecord(prompt=prompt, source_name=asset_path.name)


class LocalPromptProvider(PromptProvider):
    """Prompt provider that loads assets from the package filesystem."""

    def __init__(self) -> None:
        prompt_root = resources.files(_PROMPT_PACKAGE).joinpath(_PROMPT_ASSET_DIR)
        self._prompts: dict[str, _PromptRecord] = {}
        self._aliases: dict[str, str] = {}

        if not prompt_root.is_dir():
            raise PermanentError("Prompt asset directory is missing", directory=_PROMPT_ASSET_DIR)

        for asset_path in sorted(prompt_root.iterdir(), key=lambda entry: entry.name):
            if not asset_path.name.endswith(".prompt.md"):
                continue
            record = _load_prompt_asset(asset_path)
            self._prompts[record.prompt.name] = record
            for alias in record.prompt.aliases:
                self._aliases[alias] = record.prompt.name

        required_prompts = {
            self.ITEM_WRITER_PROMPT_NAME,
            self.OPTIONS_ONLY_WRITER_PROMPT_NAME,
            self.FEEDBACK_WRITER_PROMPT_NAME,
        }
        missing = sorted(name for name in required_prompts if name not in self._prompts)
        if missing:
            raise PermanentError("Missing required prompt assets", missing_prompts=missing)

        logger.info(
            "local_prompt_provider_initialized",
            prompt_count=len(self._prompts),
            aliases=len(self._aliases),
        )

    def _resolve_name(self, name: str) -> str:
        return self._aliases.get(name, name)

    async def get_prompt(self, name: str) -> Prompt:
        resolved_name = self._resolve_name(name)
        record = self._prompts.get(resolved_name)
        if record is None:
            raise PermanentError("Prompt asset not found", prompt_name=name)
        return record.prompt

    async def health_check(self) -> bool:
        return True

    async def shutdown(self) -> None:
        return None

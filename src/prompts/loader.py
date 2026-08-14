"""Load a versioned prompt file, and render it without touching its braces.

CLAUDE.md rule 17: prompts are versioned files in `config/prompts/`, carrying a
version header, logged on every span. **Never inlined in Python.**

WHY THIS EXISTS RATHER THAN `str.format`. Two reasons, and both are load-bearing.

1. **Prompts contain JSON.** `rerank.md` instructs the model to reply with
   `{"scores": [{"index": 0, "score": 0.0}]}`. `str.format` reads every one of
   those braces as a field and raises. Substituting only the DECLARED
   placeholders leaves JSON examples — the most useful part of a
   structured-output prompt — untouched.

2. **A prompt declares its own inputs.** The front matter lists `placeholders`,
   and `render` requires exactly that set: a missing one is an error rather than
   a literal `{question}` reaching a model, and an extra one is an error rather
   than a silently ignored argument. A prompt that quietly renders with a hole in
   it produces a plausible answer to the wrong question.

WHAT WAS HERE BEFORE. `build_rerank_prompt` in `src/retrieval/retriever.py`
built the string in Python and carried a docstring saying it "mirrors
config/prompts/rerank.md version 1". Two copies of one prompt, kept in step by
hand, with the version number written in a comment — exactly the drift rule 17
forbids. `tests/security/test_prompts_are_files.py` now makes that structural.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = REPO_ROOT / "config" / "prompts"

#: `---\n<yaml>\n---\n<body>`
_FRONT_MATTER = re.compile(r"\A---\r?\n(?P<yaml>.*?)\r?\n---\r?\n(?P<body>.*)\Z", re.DOTALL)

#: Fields every prompt must declare. `model_alias` is required even where a
#: caller knows it, because a prompt and the tier it is written for travel
#: together — a drafter prompt sent to the triage model is a silent quality
#: regression, and the file is the only place that pairing is stated.
REQUIRED_KEYS = ("name", "version", "model_alias")


class PromptError(RuntimeError):
    """A prompt file is missing, malformed, or rendered with the wrong inputs."""


@dataclass(frozen=True)
class Prompt:
    """One prompt file: its header, its body, and where it came from."""

    name: str
    version: int
    model_alias: str
    body: str
    path: Path
    placeholders: tuple[str, ...] = ()
    temperature: float | None = None

    @property
    def version_tag(self) -> str:
        """What goes on a span. `rerank@v1`.

        One string rather than two attributes because it is read by humans in a
        trace UI, and because a version with no name is unattributable the moment
        a trace carries more than one prompt.
        """
        return f"{self.name}@v{self.version}"

    def render(self, **values: Any) -> str:
        """Substitute exactly the declared placeholders.

        Refuses on a missing or unexpected key. Braces that are not declared
        placeholders — JSON examples, literal shapes — are left alone.
        """
        supplied = set(values)
        declared = set(self.placeholders)
        if missing := sorted(declared - supplied):
            raise PromptError(
                f"{self.path.name} declares placeholders {sorted(declared)} and "
                f"render() was not given: {missing}. A prompt rendered with a hole "
                f"in it produces a plausible answer to the wrong question."
            )
        if unexpected := sorted(supplied - declared):
            raise PromptError(
                f"{self.path.name} does not declare {unexpected}; declared: "
                f"{sorted(declared)}. Add it to the file's front matter, so the "
                f"prompt states its own inputs."
            )

        rendered = self.body
        for key, value in values.items():
            rendered = rendered.replace("{" + key + "}", str(value))
        return rendered


def _parse(path: Path) -> Prompt:
    if not path.is_file():
        raise PromptError(
            f"no prompt file at {path}. Prompts are versioned files in "
            f"config/prompts/ and are never inlined in Python (CLAUDE.md rule 17)."
        )
    text = path.read_text(encoding="utf-8")
    match = _FRONT_MATTER.match(text)
    if match is None:
        raise PromptError(
            f"{path.name} has no YAML front matter. Expected a '---' delimited "
            f"header carrying at least {list(REQUIRED_KEYS)}."
        )

    header = yaml.safe_load(match.group("yaml")) or {}
    if not isinstance(header, dict):
        raise PromptError(f"{path.name}'s front matter is not a mapping")
    if missing := [key for key in REQUIRED_KEYS if key not in header]:
        raise PromptError(f"{path.name} front matter is missing {missing}")

    placeholders = tuple(header.get("placeholders") or ())
    body = match.group("body").strip()
    if not body:
        raise PromptError(f"{path.name} has a header but no body")

    # Declared-but-absent is a broken file: the caller will be required to supply
    # a value that goes nowhere, which reads as a working prompt and is not one.
    for placeholder in placeholders:
        if "{" + placeholder + "}" not in body:
            raise PromptError(
                f"{path.name} declares placeholder '{placeholder}' but its body "
                f"never uses it. Remove the declaration or use it."
            )

    return Prompt(
        name=str(header["name"]),
        version=int(header["version"]),
        model_alias=str(header["model_alias"]),
        body=body,
        path=path,
        placeholders=placeholders,
        temperature=header.get("temperature"),
    )


@cache
def load_prompt(name: str) -> Prompt:
    """Load `config/prompts/<name>.md`. Cached — prompts do not change at runtime.

    The filename must match the declared `name`, so a file copied to a new
    purpose without editing its header fails here rather than logging a version
    tag that names the prompt it was copied from.
    """
    prompt = _parse(PROMPT_DIR / f"{name}.md")
    if prompt.name != name:
        raise PromptError(
            f"config/prompts/{name}.md declares name '{prompt.name}'. The header "
            f"and the filename must agree, or spans are tagged with the wrong prompt."
        )
    return prompt


def prompt_versions() -> dict[str, str]:
    """Every prompt's version tag, for a run-level span attribute.

    A run records the versions of everything that participated in it, so a
    quality regression can be traced to the exact texts in play — not only to
    the one prompt whose span happened to be examined.
    """
    return {
        path.stem: load_prompt(path.stem).version_tag
        for path in sorted(PROMPT_DIR.glob("*.md"))
        if path.stem != "README"
    }

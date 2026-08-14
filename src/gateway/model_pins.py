"""Read the model pins out of the LiteLLM config and judge whether they are pins.

`ollama/llama3.2` is not a pin. It resolves to whatever `:latest` points at
today, which changes the next time anyone runs `ollama pull` — so two machines,
or the same machine two weeks apart, can silently run different weights behind
the same alias. `:latest` written out explicitly has exactly the same problem;
being visible does not make it stable.

So the rule is: every Ollama reference carries an explicit, versioned tag.
Preflight walks this list rather than checking a hardcoded handful, which means
an alias added in a later phase is covered the day it appears instead of the day
someone remembers to add a check for it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LITELLM_CONFIG = REPO_ROOT / "docker" / "litellm" / "config.yaml"

OLLAMA_PREFIX = "ollama/"

#: Tags that name a moving target rather than a version.
DRIFTING_TAGS = {"latest"}

#: A tag is versioned if it is not one of the drifting names above. Anything
#: with digits or a size suffix (3b, 8b, v1.5, 70b-instruct-q4) qualifies.
_HAS_TAG = re.compile(r"^[^:]+:[^:]+$")


@dataclass(frozen=True)
class GatewayModel:
    """One Ollama-backed alias from the gateway config."""

    alias: str
    #: The full reference as written, e.g. "ollama/llama3.2:3b".
    reference: str
    #: The tag an Ollama host would be asked for, e.g. "llama3.2:3b".
    tag: str

    @property
    def has_explicit_tag(self) -> bool:
        return bool(_HAS_TAG.match(self.tag))

    @property
    def is_drifting(self) -> bool:
        """No tag at all, or a tag that means 'whatever is newest'."""
        if not self.has_explicit_tag:
            return True
        return self.tag.split(":", 1)[1] in DRIFTING_TAGS

    @property
    def pull_command(self) -> str:
        return f"ollama pull {self.tag}"

    def violation(self) -> str | None:
        """Why this reference is not a pin, or None when it is fine."""
        if not self.has_explicit_tag:
            return (
                f"alias '{self.alias}' uses '{self.reference}' with no tag, which resolves "
                f"to :latest and changes on the next pull"
            )
        if self.is_drifting:
            return (
                f"alias '{self.alias}' uses '{self.reference}', and :latest is not a pin — "
                f"it moves whenever the upstream model is republished"
            )
        return None


def parse_gateway_models(path: Path | None = None) -> list[GatewayModel]:
    """Every Ollama-backed alias in the gateway config, in file order.

    Non-Ollama aliases (Groq) are skipped: their versioning is the provider's
    problem and there is no local host to check them against.
    """
    resolved = path or DEFAULT_LITELLM_CONFIG
    with resolved.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    models: list[GatewayModel] = []
    for entry in config.get("model_list") or []:
        alias = str(entry.get("model_name", ""))
        reference = str((entry.get("litellm_params") or {}).get("model", ""))
        if not alias or not reference.startswith(OLLAMA_PREFIX):
            continue
        models.append(
            GatewayModel(
                alias=alias,
                reference=reference,
                tag=reference[len(OLLAMA_PREFIX) :],
            )
        )
    return models


def drifting_models(models: list[GatewayModel]) -> list[GatewayModel]:
    return [model for model in models if model.is_drifting]

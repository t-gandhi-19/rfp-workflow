"""Typed access to `config/embedding.yaml`.

The embedding model tag and the vector-index dimension are the same decision
recorded once. Both the graph schema and the ingest pipeline read them from
here, so they cannot drift apart in the way that produces silently bad
retrieval rather than an error.
"""

from __future__ import annotations

from functools import lru_cache

import yaml
from pydantic import BaseModel, ConfigDict, Field

from src.contracts.thresholds import config_dir


class EmbeddingModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Gateway alias used by application code. Never a raw model name in Python.
    alias: str = Field(min_length=1)
    #: Concrete provider tag the alias must resolve to.
    tag: str = Field(min_length=1)
    #: Vector width, asserted by a real probe embedding in preflight.
    dimensions: int = Field(gt=0)


class VectorIndexConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    label: str = Field(min_length=1)
    property: str = Field(min_length=1)
    similarity: str = Field(min_length=1)


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    model: EmbeddingModelConfig
    index: VectorIndexConfig
    #: Printed verbatim when the model is missing, so the fix needs no thought.
    install_command: str = Field(min_length=1)


@lru_cache(maxsize=1)
def embedding_config() -> EmbeddingConfig:
    """Parsed, validated `embedding.yaml`. Cached after first read."""
    path = config_dir() / "embedding.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data: object = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping")
    return EmbeddingConfig.model_validate(data)


def reload_embedding_config() -> None:
    """Drop the cache so the next read picks up a new ``RFP_CONFIG_DIR``."""
    embedding_config.cache_clear()


def embedding_dimensions() -> int:
    """Vector width the Neo4j index is built for."""
    return embedding_config().model.dimensions

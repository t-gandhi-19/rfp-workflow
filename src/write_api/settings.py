"""write-api configuration, read from the environment."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class WriteApiSettings(BaseSettings):
    """Environment-driven settings. Every field is documented in .env.example."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    keycloak_url: str = "http://keycloak:8080"
    keycloak_realm: str = "rfp"
    # Must equal the `iss` claim exactly. Containers and host callers resolve
    # Keycloak by different names, so this is configured rather than derived.
    keycloak_issuer: str = "http://keycloak:8080/realms/rfp"
    keycloak_audience: str = "rfp-api"

    # Seconds to cache the JWKS. Signing keys rotate rarely; a short lifespan
    # would put a network call on the hot path of every request.
    jwks_cache_seconds: int = 300

    # Clock skew tolerance when validating exp/nbf, in seconds.
    leeway_seconds: int = 10

    git_sha: str = "local-dev"

    @property
    def jwks_uri(self) -> str:
        return f"{self.keycloak_url}/realms/{self.keycloak_realm}/protocol/openid-connect/certs"


@lru_cache(maxsize=1)
def get_settings() -> WriteApiSettings:
    return WriteApiSettings()

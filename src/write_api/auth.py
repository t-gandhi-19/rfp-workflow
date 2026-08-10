"""RS256 bearer-token validation against Keycloak.

Four things are checked on every request, and all four matter:

* **signature** — against the realm's published JWKS, RS256 only. Pinning the
  algorithm is what stops the `alg: none` and HMAC-confusion families of attack.
* **issuer** — a validly-signed token from a different realm is still not ours.
* **audience** — the token must have been minted for `rfp-api`. Without this, a
  token issued for any other resource in the realm would be accepted here.
* **role** — the specific realm role the endpoint requires.

The status codes are distinct on purpose: **401** means "I do not know who you
are", **403** means "I know exactly who you are and you may not do this". A
service account calling an endpoint outside its grant should see the second, and
so should the test suite.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient
from pydantic import BaseModel, ConfigDict

from src.write_api.settings import WriteApiSettings, get_settings

_bearer = HTTPBearer(auto_error=False)


class Principal(BaseModel):
    """The authenticated caller behind a request.

    `subject` is stamped onto every row this request writes, which is what joins
    the database record to the Keycloak audit event and the OTel span.
    """

    model_config = ConfigDict(frozen=True)

    subject: str
    client_id: str
    roles: frozenset[str]

    def has_role(self, role: str) -> bool:
        return role in self.roles


class TokenVerifier:
    """Verifies bearer tokens, caching the realm's signing keys.

    Tests substitute their own `signing_key_resolver` so the whole auth path can
    be exercised against a locally generated keypair with no Keycloak running.
    """

    def __init__(
        self,
        settings: WriteApiSettings,
        signing_key_resolver: Any | None = None,
    ) -> None:
        self._settings = settings
        self._resolver = signing_key_resolver or PyJWKClient(
            settings.jwks_uri,
            cache_keys=True,
            lifespan=settings.jwks_cache_seconds,
        )

    def _signing_key(self, token: str) -> Any:
        return self._resolver.get_signing_key_from_jwt(token).key

    async def verify(self, token: str) -> Principal:
        settings = self._settings
        try:
            # JWKS lookup may hit the network on a cache miss; keep it off the
            # event loop.
            key = await asyncio.to_thread(self._signing_key, token)
            claims: dict[str, Any] = jwt.decode(
                token,
                key=key,
                algorithms=["RS256"],
                audience=settings.keycloak_audience,
                issuer=settings.keycloak_issuer,
                leeway=settings.leeway_seconds,
                options={"require": ["exp", "iat", "sub", "aud", "iss"]},
            )
        except jwt.PyJWTError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {exc}",
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc
        except Exception as exc:  # JWKS fetch failure, malformed header, etc.
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token could not be verified",
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc

        realm_access = claims.get("realm_access") or {}
        roles = realm_access.get("roles") or []
        return Principal(
            subject=str(claims["sub"]),
            client_id=str(claims.get("azp") or claims.get("client_id") or "unknown"),
            roles=frozenset(str(role) for role in roles),
        )


def get_verifier(request: Request) -> TokenVerifier:
    """The app's verifier. Overridden in tests via dependency_overrides."""
    verifier: TokenVerifier = request.app.state.verifier
    return verifier


async def current_principal(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    verifier: Annotated[TokenVerifier, Depends(get_verifier)],
) -> Principal:
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await verifier.verify(credentials.credentials)


def require_role(role: str) -> Any:
    """Dependency factory: 403 unless the caller holds `role`."""

    async def _dependency(
        principal: Annotated[Principal, Depends(current_principal)],
    ) -> Principal:
        if not principal.has_role(role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role}' is required; caller {principal.client_id} does not hold it",
            )
        return principal

    return _dependency


def build_verifier() -> TokenVerifier:
    return TokenVerifier(get_settings())

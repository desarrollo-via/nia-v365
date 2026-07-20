"""Persistencia y renovación OAuth aisladas para una instalación Bitrix."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError
from pymongo import ASCENDING


DEFAULT_OAUTH_TOKEN_URL = "https://oauth.bitrix.info/oauth/token/"
DEFAULT_INSTALLATIONS_COLLECTION = "nia_bitrix_installations"
INSTALLATION_MEMBER_INDEX = "idx_nia_bitrix_installation_member_unique"


def _required_text(value: str, field: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field} no puede estar vacío")
    return cleaned


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("las fechas OAuth deben incluir zona horaria")
    return value.astimezone(timezone.utc)


def _validated_https_url(value: str, field: str) -> str:
    cleaned = _required_text(value, field).rstrip("/")
    parsed = httpx.URL(cleaned)
    if parsed.scheme != "https" or not parsed.host:
        raise ValueError(f"{field} debe ser una URL HTTPS válida")
    if parsed.userinfo or parsed.query or parsed.fragment:
        raise ValueError(f"{field} no admite credenciales, query ni fragmento")
    return cleaned


class BitrixOAuthInstallation(BaseModel):
    """Par durable propio de la instalación; su repr enmascara los secretos."""

    model_config = ConfigDict(extra="ignore")

    member_id: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    client_endpoint: str = Field(min_length=1)
    server_endpoint: str = Field(min_length=1)
    access_token: SecretStr
    refresh_token: SecretStr
    application_token: SecretStr
    expires_at: datetime
    updated_at: datetime
    revision: int = Field(default=1, ge=1)

    def model_post_init(self, __context: Any) -> None:
        self.member_id = _required_text(self.member_id, "member_id")
        self.domain = _required_text(self.domain, "domain").lower()
        self.client_endpoint = _validated_https_url(
            self.client_endpoint,
            "client_endpoint",
        )
        self.server_endpoint = _validated_https_url(
            self.server_endpoint,
            "server_endpoint",
        )
        self.expires_at = _utc(self.expires_at)
        self.updated_at = _utc(self.updated_at)
        if not self.access_token.get_secret_value().strip():
            raise ValueError("access_token no puede estar vacío")
        if not self.refresh_token.get_secret_value().strip():
            raise ValueError("refresh_token no puede estar vacío")
        if not self.application_token.get_secret_value().strip():
            raise ValueError("application_token no puede estar vacío")


class BitrixOAuthTokenPair(BaseModel):
    """Respuesta interna renovada; nunca se devuelve por API ni por resultados."""

    model_config = ConfigDict(extra="ignore")

    member_id: str = Field(min_length=1)
    client_endpoint: str = Field(min_length=1)
    server_endpoint: str = Field(min_length=1)
    access_token: SecretStr
    refresh_token: SecretStr
    expires_at: datetime

    def model_post_init(self, __context: Any) -> None:
        self.member_id = _required_text(self.member_id, "member_id")
        self.client_endpoint = _validated_https_url(
            self.client_endpoint,
            "client_endpoint",
        )
        self.server_endpoint = _validated_https_url(
            self.server_endpoint,
            "server_endpoint",
        )
        self.expires_at = _utc(self.expires_at)
        if not self.access_token.get_secret_value().strip():
            raise ValueError("access_token no puede estar vacío")
        if not self.refresh_token.get_secret_value().strip():
            raise ValueError("refresh_token no puede estar vacío")


class BitrixOAuthStore(Protocol):
    async def ensure_indexes(self) -> None: ...

    async def save_installation(
        self,
        installation: BitrixOAuthInstallation,
    ) -> "OAuthInstallationWriteResult": ...

    async def get_installation(
        self,
        member_id: str,
    ) -> Optional[BitrixOAuthInstallation]: ...

    async def get_installation_by_domain(
        self,
        domain: str,
    ) -> Optional[BitrixOAuthInstallation]: ...

    async def replace_tokens(
        self,
        member_id: str,
        *,
        expected_refresh_token: str,
        token_pair: BitrixOAuthTokenPair,
        now: datetime,
    ) -> bool: ...


class MongoBitrixOAuthStore:
    """Colección exclusiva; la sustitución del par usa compare-and-swap."""

    def __init__(self, collection: Any) -> None:
        self._collection = collection

    async def ensure_indexes(self) -> None:
        await self._collection.create_index(
            [("member_id", ASCENDING)],
            name=INSTALLATION_MEMBER_INDEX,
            unique=True,
        )

    async def save_installation(
        self,
        installation: BitrixOAuthInstallation,
    ) -> "OAuthInstallationWriteResult":
        for _ in range(3):
            current = await self.get_installation(installation.member_id)
            if current is None:
                result = await self._collection.update_one(
                    {"member_id": installation.member_id},
                    {"$setOnInsert": self._document(installation)},
                    upsert=True,
                )
                if result.upserted_id is not None:
                    return OAuthInstallationWriteResult(
                        status=OAuthInstallationWriteStatus.CREATED,
                        revision=installation.revision,
                    )
                continue
            if self._same_credentials(current, installation):
                return OAuthInstallationWriteResult(
                    status=OAuthInstallationWriteStatus.IDEMPOTENT,
                    revision=current.revision,
                )
            result = await self._collection.update_one(
                {
                    "member_id": installation.member_id,
                    "revision": current.revision,
                },
                {
                    "$set": {
                        **self._document(installation),
                        "revision": current.revision + 1,
                    },
                },
                upsert=False,
            )
            if result.modified_count == 1:
                return OAuthInstallationWriteResult(
                    status=OAuthInstallationWriteStatus.REINSTALLED,
                    revision=current.revision + 1,
                )
        raise RuntimeError("oauth_installation_concurrent_update")

    async def get_installation(
        self,
        member_id: str,
    ) -> Optional[BitrixOAuthInstallation]:
        document = await self._collection.find_one(
            {"member_id": _required_text(member_id, "member_id")},
            {"_id": 0},
        )
        if document is None:
            return None
        return BitrixOAuthInstallation.model_validate(document)

    async def get_installation_by_domain(
        self,
        domain: str,
    ) -> Optional[BitrixOAuthInstallation]:
        normalized = _required_text(domain, "domain").lower()
        document = await self._collection.find_one(
            {"domain": normalized},
            {"_id": 0},
            sort=[("updated_at", -1)],
        )
        if document is None:
            return None
        return BitrixOAuthInstallation.model_validate(document)

    async def replace_tokens(
        self,
        member_id: str,
        *,
        expected_refresh_token: str,
        token_pair: BitrixOAuthTokenPair,
        now: datetime,
    ) -> bool:
        member = _required_text(member_id, "member_id")
        expected = _required_text(expected_refresh_token, "expected_refresh_token")
        if token_pair.member_id != member:
            raise ValueError("oauth_member_id_mismatch")
        result = await self._collection.update_one(
            {"member_id": member, "refresh_token": expected},
            {
                "$set": {
                    "client_endpoint": token_pair.client_endpoint,
                    "server_endpoint": token_pair.server_endpoint,
                    "access_token": token_pair.access_token.get_secret_value(),
                    "refresh_token": token_pair.refresh_token.get_secret_value(),
                    "expires_at": token_pair.expires_at,
                    "updated_at": _utc(now),
                },
                "$inc": {"revision": 1},
            },
            upsert=False,
        )
        return result.modified_count == 1

    @staticmethod
    def _document(installation: BitrixOAuthInstallation) -> dict[str, Any]:
        return {
            "member_id": installation.member_id,
            "domain": installation.domain,
            "client_endpoint": installation.client_endpoint,
            "server_endpoint": installation.server_endpoint,
            "access_token": installation.access_token.get_secret_value(),
            "refresh_token": installation.refresh_token.get_secret_value(),
            "application_token": installation.application_token.get_secret_value(),
            "expires_at": installation.expires_at,
            "updated_at": installation.updated_at,
            "revision": installation.revision,
        }

    @staticmethod
    def _same_credentials(
        current: BitrixOAuthInstallation,
        candidate: BitrixOAuthInstallation,
    ) -> bool:
        return (
            current.domain == candidate.domain
            and current.client_endpoint == candidate.client_endpoint
            and current.server_endpoint == candidate.server_endpoint
            and current.access_token.get_secret_value()
            == candidate.access_token.get_secret_value()
            and current.refresh_token.get_secret_value()
            == candidate.refresh_token.get_secret_value()
            and current.application_token.get_secret_value()
            == candidate.application_token.get_secret_value()
            and current.expires_at == candidate.expires_at
        )


class OAuthInstallationWriteStatus(str, Enum):
    CREATED = "created"
    IDEMPOTENT = "idempotent"
    REINSTALLED = "reinstalled"


@dataclass(frozen=True)
class OAuthInstallationWriteResult:
    status: OAuthInstallationWriteStatus
    revision: int


class OAuthRefreshDecision(str, Enum):
    SUCCESS = "success"
    RETRY = "retry"
    FAIL = "fail"


@dataclass(frozen=True)
class OAuthRefreshResult:
    decision: OAuthRefreshDecision
    token_pair: Optional[BitrixOAuthTokenPair] = None
    error_code: Optional[str] = None
    retry_after_seconds: int = 0

    def __post_init__(self) -> None:
        if self.retry_after_seconds < 0:
            raise ValueError("retry_after_seconds no puede ser negativo")
        if self.decision is OAuthRefreshDecision.SUCCESS:
            if self.token_pair is None or self.error_code is not None:
                raise ValueError("success requiere token_pair")
            return
        if self.token_pair is not None or not self.error_code:
            raise ValueError("retry y fail requieren error_code")
        if self.decision is OAuthRefreshDecision.FAIL and self.retry_after_seconds:
            raise ValueError("fail no admite espera")


class BitrixOAuthAccessError(RuntimeError):
    """Error seguro consumible por BitrixClient."""

    def __init__(
        self,
        error_code: str,
        *,
        retryable: bool,
        retry_after_seconds: int = 0,
    ) -> None:
        safe_code = _required_text(error_code, "error_code")
        super().__init__(safe_code)
        self.error_code = safe_code
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds


class BitrixOAuthTransport:
    """Intercambia un refresh token mediante el endpoint oficial inyectable."""

    def __init__(
        self,
        *,
        client_id: str,
        client_secret: str,
        timeout_seconds: float,
        token_url: str = DEFAULT_OAUTH_TOKEN_URL,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._client_id = _required_text(client_id, "client_id")
        self._client_secret = _required_text(client_secret, "client_secret")
        self._token_url = _validated_https_url(token_url, "token_url") + "/"
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds debe ser positivo")
        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.AsyncClient(
            timeout=timeout_seconds
        )

    async def refresh(self, refresh_token: str) -> OAuthRefreshResult:
        token = _required_text(refresh_token, "refresh_token")
        try:
            response = await self._http_client.get(
                self._token_url,
                params={
                    "grant_type": "refresh_token",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "refresh_token": token,
                },
            )
        except httpx.TimeoutException:
            return OAuthRefreshResult(
                OAuthRefreshDecision.RETRY,
                error_code="oauth_timeout",
            )
        except httpx.TransportError:
            return OAuthRefreshResult(
                OAuthRefreshDecision.RETRY,
                error_code="oauth_transport_error",
            )

        retry_after = response.headers.get("Retry-After", "").strip()
        retry_seconds = min(int(retry_after), 86_400) if retry_after.isdigit() else 0
        if response.status_code == 429 or 500 <= response.status_code <= 599:
            return OAuthRefreshResult(
                OAuthRefreshDecision.RETRY,
                error_code="oauth_server_retryable",
                retry_after_seconds=retry_seconds,
            )
        try:
            payload = response.json()
        except ValueError:
            return OAuthRefreshResult(
                OAuthRefreshDecision.FAIL,
                error_code="oauth_invalid_response",
            )
        if not isinstance(payload, dict):
            return OAuthRefreshResult(
                OAuthRefreshDecision.FAIL,
                error_code="oauth_invalid_response",
            )
        if payload.get("error"):
            return OAuthRefreshResult(
                OAuthRefreshDecision.FAIL,
                error_code="oauth_refresh_rejected",
            )
        try:
            pair = BitrixOAuthTokenPair.model_validate(
                {
                    "member_id": payload.get("member_id"),
                    "client_endpoint": payload.get("client_endpoint"),
                    "server_endpoint": payload.get("server_endpoint"),
                    "access_token": payload.get("access_token"),
                    "refresh_token": payload.get("refresh_token"),
                    "expires_at": datetime.fromtimestamp(
                        int(payload.get("expires")),
                        tz=timezone.utc,
                    ),
                }
            )
        except (TypeError, ValueError, ValidationError, OSError):
            return OAuthRefreshResult(
                OAuthRefreshDecision.FAIL,
                error_code="oauth_invalid_response",
            )
        return OAuthRefreshResult(
            OAuthRefreshDecision.SUCCESS,
            token_pair=pair,
        )

    async def close(self) -> None:
        if self._owns_http_client:
            await self._http_client.aclose()


class BitrixOAuthCredentialProvider:
    """Entrega el token retenido y serializa renovaciones por instalación."""

    def __init__(
        self,
        store: BitrixOAuthStore,
        transport: BitrixOAuthTransport,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._store = store
        self._transport = transport
        self._clock = clock
        self._locks: dict[str, asyncio.Lock] = {}

    async def get_access_token(self, member_id: str) -> str:
        installation = await self._required_installation(member_id)
        return installation.access_token.get_secret_value()

    async def refresh_access_token(
        self,
        member_id: str,
        stale_access_token: str,
    ) -> str:
        member = _required_text(member_id, "member_id")
        stale = _required_text(stale_access_token, "stale_access_token")
        async with self._locks.setdefault(member, asyncio.Lock()):
            current = await self._required_installation(member)
            current_access = current.access_token.get_secret_value()
            if current_access != stale:
                return current_access

            old_refresh = current.refresh_token.get_secret_value()
            result = await self._transport.refresh(old_refresh)
            if result.decision is not OAuthRefreshDecision.SUCCESS:
                raise BitrixOAuthAccessError(
                    result.error_code or "oauth_refresh_failed",
                    retryable=result.decision is OAuthRefreshDecision.RETRY,
                    retry_after_seconds=result.retry_after_seconds,
                )
            pair = result.token_pair
            if pair is None or pair.member_id != member:
                raise BitrixOAuthAccessError(
                    "oauth_installation_mismatch",
                    retryable=False,
                )
            applied = await self._store.replace_tokens(
                member,
                expected_refresh_token=old_refresh,
                token_pair=pair,
                now=self._clock(),
            )
            if applied:
                return pair.access_token.get_secret_value()

            winner = await self._required_installation(member)
            winner_access = winner.access_token.get_secret_value()
            if winner_access == stale:
                raise BitrixOAuthAccessError(
                    "oauth_refresh_conflict",
                    retryable=True,
                )
            return winner_access

    async def _required_installation(
        self,
        member_id: str,
    ) -> BitrixOAuthInstallation:
        installation = await self._store.get_installation(member_id)
        if installation is None:
            raise BitrixOAuthAccessError(
                "oauth_installation_not_found",
                retryable=False,
            )
        return installation

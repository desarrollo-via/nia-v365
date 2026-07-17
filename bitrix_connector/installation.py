"""Ingreso seguro y aislado del callback inicial OAuth de Bitrix."""

from __future__ import annotations

import hmac
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Protocol

import httpx
from pydantic import BaseModel, ValidationError

from .config import ConnectorSettings
from .bitrix_client import _remote_error_code, _retry_after_seconds
from .oauth import (
    BitrixOAuthInstallation,
    BitrixOAuthStore,
    OAuthInstallationWriteStatus,
)


class OAuthInstallationStatus(str, Enum):
    INSTALLED = "installed"
    IDEMPOTENT = "idempotent"
    REINSTALLED = "reinstalled"
    INVALID = "invalid"
    REJECTED = "rejected"


class OAuthInstallationResult(BaseModel):
    status: OAuthInstallationStatus
    reason: str
    persisted: bool = False
    revision: Optional[int] = None


class OAuthInstallationPersistenceError(RuntimeError):
    """La instalación era válida, pero el almacén no confirmó la escritura."""


class OAuthInstallationVerificationError(RuntimeError):
    """Bitrix no pudo confirmar temporalmente el token del callback."""


class OAuthTokenVerificationDecision(str, Enum):
    VERIFIED = "verified"
    RETRY = "retry"
    REJECTED = "rejected"


@dataclass(frozen=True)
class OAuthTokenVerificationResult:
    decision: OAuthTokenVerificationDecision
    reason: str
    retry_after_seconds: int = 0


class OAuthCallbackVerifier(Protocol):
    async def verify(
        self,
        *,
        client_endpoint: str,
        access_token: str,
    ) -> OAuthTokenVerificationResult: ...


class BitrixOAuthCallbackVerifier:
    """Comprueba el token mediante la llamada oficial y de lectura app.info."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 10.0,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds debe ser positivo")
        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.AsyncClient(
            timeout=timeout_seconds
        )

    async def verify(
        self,
        *,
        client_endpoint: str,
        access_token: str,
    ) -> OAuthTokenVerificationResult:
        endpoint = f"{client_endpoint.rstrip('/')}/app.info"
        try:
            response = await self._http_client.post(
                endpoint,
                json={"auth": access_token},
            )
        except httpx.TimeoutException:
            return OAuthTokenVerificationResult(
                OAuthTokenVerificationDecision.RETRY,
                "installation_verification_timeout",
            )
        except httpx.TransportError:
            return OAuthTokenVerificationResult(
                OAuthTokenVerificationDecision.RETRY,
                "installation_verification_transport_error",
            )

        status = response.status_code
        remote_code = _remote_error_code(response)
        retry_after = _retry_after_seconds(response)
        if status == 429 or remote_code == "QUERY_LIMIT_EXCEEDED":
            return OAuthTokenVerificationResult(
                OAuthTokenVerificationDecision.RETRY,
                "installation_verification_rate_limited",
                retry_after_seconds=retry_after,
            )
        if status in {408, 425} or 500 <= status <= 599:
            return OAuthTokenVerificationResult(
                OAuthTokenVerificationDecision.RETRY,
                "installation_verification_unavailable",
                retry_after_seconds=retry_after,
            )
        if remote_code is not None or not 200 <= status <= 299:
            return OAuthTokenVerificationResult(
                OAuthTokenVerificationDecision.REJECTED,
                "installation_access_token_rejected",
            )
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if not isinstance(payload, dict) or not isinstance(
            payload.get("result"), dict
        ):
            return OAuthTokenVerificationResult(
                OAuthTokenVerificationDecision.REJECTED,
                "installation_verification_invalid_response",
            )
        return OAuthTokenVerificationResult(
            OAuthTokenVerificationDecision.VERIFIED,
            "installation_access_token_verified",
        )

    async def close(self) -> None:
        if self._owns_http_client:
            await self._http_client.aclose()


def _value(form: Mapping[str, Any], key: str) -> str:
    value = form.get(key, "")
    if hasattr(value, "filename"):
        return ""
    return str(value or "").strip()


def _domain(value: str) -> str:
    return value.lower().removeprefix("https://").rstrip("/")


class BitrixOAuthInstallationService:
    def __init__(
        self,
        store: BitrixOAuthStore,
        verifier: OAuthCallbackVerifier,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._store = store
        self._verifier = verifier
        self._clock = clock

    async def install(
        self,
        raw_form: Mapping[str, Any],
        settings: ConnectorSettings,
    ) -> OAuthInstallationResult:
        event = _value(raw_form, "event").upper()
        if event not in {"", "ONAPPINSTALL"}:
            return self._invalid("unsupported_installation_event")

        if not settings.bitrix_domain:
            return self._rejected("installation_domain_not_configured")

        domain = _domain(_value(raw_form, "auth[domain]"))
        member_id = _value(raw_form, "auth[member_id]")
        client_endpoint = _value(raw_form, "auth[client_endpoint]")
        server_endpoint = _value(raw_form, "auth[server_endpoint]")
        access_token = _value(raw_form, "auth[access_token]")
        refresh_token = _value(raw_form, "auth[refresh_token]")
        application_token = _value(raw_form, "auth[application_token]")

        required = (
            domain,
            member_id,
            client_endpoint,
            server_endpoint,
            access_token,
            refresh_token,
            application_token,
        )
        if not all(required):
            return self._invalid("invalid_installation_payload")
        if not hmac.compare_digest(domain, _domain(settings.bitrix_domain)):
            return self._rejected("installation_domain_mismatch")
        if settings.bitrix_member_id and not hmac.compare_digest(
            member_id,
            settings.bitrix_member_id,
        ):
            return self._rejected("installation_member_id_mismatch")
        if not self._endpoint_matches_domain(client_endpoint, domain):
            return self._rejected("installation_client_endpoint_mismatch")
        if not self._official_server_endpoint(server_endpoint):
            return self._rejected("installation_server_endpoint_mismatch")

        verification = await self._verifier.verify(
            client_endpoint=client_endpoint,
            access_token=access_token,
        )
        if verification.decision is OAuthTokenVerificationDecision.RETRY:
            raise OAuthInstallationVerificationError(verification.reason)
        if verification.decision is OAuthTokenVerificationDecision.REJECTED:
            return self._rejected(verification.reason)

        now = self._clock().astimezone(timezone.utc)
        expires_at = self._expires_at(raw_form, now)
        if expires_at is None or expires_at <= now:
            return self._invalid("invalid_installation_expiry")

        try:
            installation = BitrixOAuthInstallation(
                member_id=member_id,
                domain=domain,
                client_endpoint=client_endpoint,
                server_endpoint=server_endpoint,
                access_token=access_token,
                refresh_token=refresh_token,
                application_token=application_token,
                expires_at=expires_at,
                updated_at=now,
            )
        except (TypeError, ValueError, ValidationError):
            return self._invalid("invalid_installation_payload")

        try:
            write = await self._store.save_installation(installation)
        except Exception as exc:
            raise OAuthInstallationPersistenceError(
                "oauth_installation_storage_unavailable"
            ) from exc

        statuses = {
            OAuthInstallationWriteStatus.CREATED: (
                OAuthInstallationStatus.INSTALLED,
                "installation_stored",
            ),
            OAuthInstallationWriteStatus.IDEMPOTENT: (
                OAuthInstallationStatus.IDEMPOTENT,
                "installation_already_stored",
            ),
            OAuthInstallationWriteStatus.REINSTALLED: (
                OAuthInstallationStatus.REINSTALLED,
                "installation_replaced",
            ),
        }
        status, reason = statuses[write.status]
        return OAuthInstallationResult(
            status=status,
            reason=reason,
            persisted=True,
            revision=write.revision,
        )

    @staticmethod
    def _expires_at(
        raw_form: Mapping[str, Any],
        now: datetime,
    ) -> Optional[datetime]:
        absolute = _value(raw_form, "auth[expires]")
        relative = _value(raw_form, "auth[expires_in]")
        try:
            if absolute:
                return datetime.fromtimestamp(int(absolute), tz=timezone.utc)
            if relative:
                seconds = int(relative)
                if seconds <= 0 or seconds > 604_800:
                    return None
                return now + timedelta(seconds=seconds)
        except (TypeError, ValueError, OSError):
            return None
        return None

    @staticmethod
    def _endpoint_matches_domain(endpoint: str, domain: str) -> bool:
        try:
            parsed = httpx.URL(endpoint)
        except (TypeError, ValueError):
            return False
        return (
            parsed.scheme == "https"
            and (parsed.host or "").lower() == domain
            and not parsed.userinfo
            and not parsed.query
            and not parsed.fragment
        )

    @staticmethod
    def _official_server_endpoint(endpoint: str) -> bool:
        try:
            parsed = httpx.URL(endpoint)
        except (TypeError, ValueError):
            return False
        return (
            parsed.scheme == "https"
            and (parsed.host or "").lower() == "oauth.bitrix.info"
            and not parsed.userinfo
            and not parsed.query
            and not parsed.fragment
        )

    @staticmethod
    def _invalid(reason: str) -> OAuthInstallationResult:
        return OAuthInstallationResult(
            status=OAuthInstallationStatus.INVALID,
            reason=reason,
        )

    @staticmethod
    def _rejected(reason: str) -> OAuthInstallationResult:
        return OAuthInstallationResult(
            status=OAuthInstallationStatus.REJECTED,
            reason=reason,
        )

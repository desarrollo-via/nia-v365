"""Límite HTTP OAuth aislado para enviar mensajes mediante Chatbots 2.0."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .output_review import (
    MAX_BITRIX_MESSAGE_CHARS,
    BitrixMessagePayloadPreview,
)


BITRIX_SEND_PATH = "/rest/imbot.v2.Chat.Message.send"
_RATE_LIMIT_CODES = {"QUERY_LIMIT_EXCEEDED"}
_TOKEN_REFRESH_CODES = {"expired_token"}
_RETRYABLE_API_CODES = {
    "SENDING_FAILED",
    "INTERNAL_SERVER_ERROR",
    "ERROR_UNEXPECTED_ANSWER",
    "OVERLOAD_LIMIT",
}
_PERMANENT_API_CODES = {
    "BOT_TOKEN_NOT_SPECIFIED",
    "BOT_ID_REQUIRED",
    "BOT_NOT_FOUND",
    "BOT_OWNERSHIP_ERROR",
    "ACCESS_DENIED",
    "EMPTY_MESSAGE",
    "NO_AUTH_FOUND",
    "INVALID_REQUEST",
    "ERROR_BATCH_METHOD_NOT_ALLOWED",
    "ERROR_BATCH_LENGTH_EXCEEDED",
    "INVALID_CREDENTIALS",
    "ERROR_MANIFEST_IS_NOT_AVAILABLE",
    "insufficient_scope",
    "user_access_error",
    "PORTAL_DELETED",
}


class BitrixSentMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int = Field(gt=0)
    uuidMap: dict[str, int] = Field(default_factory=dict)


class BitrixSendResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    result: BitrixSentMessage


class BitrixClientDecision(str, Enum):
    SUCCESS = "success"
    RETRY = "retry"
    FAIL = "fail"


@dataclass(frozen=True)
class BitrixClientResult:
    """Resultado seguro sin cuerpos remotos, credenciales o excepciones."""

    decision: BitrixClientDecision
    response: Optional[BitrixSendResponse] = None
    error_code: Optional[str] = None
    http_status: Optional[int] = None
    retry_after_seconds: int = 0

    def __post_init__(self) -> None:
        if self.retry_after_seconds < 0:
            raise ValueError("retry_after_seconds no puede ser negativo")
        if self.decision is BitrixClientDecision.SUCCESS:
            if self.response is None or self.error_code is not None:
                raise ValueError("success requiere response y no admite error_code")
            if self.retry_after_seconds:
                raise ValueError("success no admite espera de reintento")
            return
        if self.response is not None or not self.error_code:
            raise ValueError("retry y fail requieren solo un error_code seguro")
        if (
            self.decision is BitrixClientDecision.FAIL
            and self.retry_after_seconds
        ):
            raise ValueError("fail no admite espera de reintento")

    @classmethod
    def succeeded(
        cls,
        response: BitrixSendResponse,
        *,
        http_status: int,
    ) -> "BitrixClientResult":
        return cls(
            decision=BitrixClientDecision.SUCCESS,
            response=response,
            http_status=http_status,
        )

    @classmethod
    def retryable(
        cls,
        error_code: str,
        *,
        http_status: Optional[int] = None,
        retry_after_seconds: int = 0,
    ) -> "BitrixClientResult":
        return cls(
            decision=BitrixClientDecision.RETRY,
            error_code=error_code,
            http_status=http_status,
            retry_after_seconds=retry_after_seconds,
        )

    @classmethod
    def failed(
        cls,
        error_code: str,
        *,
        http_status: Optional[int] = None,
    ) -> "BitrixClientResult":
        return cls(
            decision=BitrixClientDecision.FAIL,
            error_code=error_code,
            http_status=http_status,
        )


class BitrixAccessTokenProvider(Protocol):
    async def get_access_token(self, member_id: str) -> str: ...

    async def refresh_access_token(
        self,
        member_id: str,
        stale_access_token: str,
    ) -> str: ...


def _validated_portal_url(portal_url: str) -> str:
    cleaned = portal_url.strip().rstrip("/")
    if not cleaned:
        raise ValueError("bitrix_portal_url no puede estar vacía")
    parsed = httpx.URL(cleaned)
    if parsed.scheme != "https" or not parsed.host:
        raise ValueError("bitrix_portal_url debe ser una URL HTTPS válida")
    if parsed.userinfo or parsed.query or parsed.fragment:
        raise ValueError(
            "bitrix_portal_url no admite credenciales, query ni fragmento"
        )
    if parsed.path not in {"", "/"}:
        raise ValueError("bitrix_portal_url debe identificar la raíz del portal")
    return cleaned


def _retry_after_seconds(response: httpx.Response) -> int:
    value = response.headers.get("Retry-After", "").strip()
    if not value.isdigit():
        return 0
    return min(int(value), 86_400)


def _remote_error_code(response: httpx.Response) -> Optional[str]:
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("error")
    if not isinstance(value, str):
        return None
    clean = value.strip()
    return clean or None


class BitrixClient:
    """Publica únicamente mensajes ya aprobados mediante OAuth de aplicación."""

    def __init__(
        self,
        *,
        portal_url: str,
        access_token: Optional[str] = None,
        token_provider: Optional[BitrixAccessTokenProvider] = None,
        member_id: Optional[str] = None,
        timeout_seconds: float,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        token = (access_token or "").strip() or None
        member = (member_id or "").strip() or None
        if (token is None) == (token_provider is None):
            raise ValueError(
                "configure exactamente access_token o token_provider"
            )
        if token_provider is not None and member is None:
            raise ValueError("member_id es obligatorio con token_provider")
        if token is not None and member is not None:
            raise ValueError("member_id solo se usa con token_provider")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds debe ser positivo")

        self._endpoint = f"{_validated_portal_url(portal_url)}{BITRIX_SEND_PATH}"
        self._access_token = token
        self._token_provider = token_provider
        self._member_id = member
        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.AsyncClient(
            timeout=timeout_seconds
        )

    async def send_approved_message(
        self,
        payload: BitrixMessagePayloadPreview,
    ) -> BitrixClientResult:
        message = payload.fields.message
        if not message.strip():
            return BitrixClientResult.failed("bitrix_empty_message")
        if len(message) > MAX_BITRIX_MESSAGE_CHARS:
            return BitrixClientResult.failed("bitrix_message_too_long")

        try:
            access_token = await self._current_access_token()
        except Exception as exc:
            return self._oauth_failure(exc)

        result = await self._send_with_token(payload, access_token)
        if (
            result.error_code != "bitrix_token_expired"
            or self._token_provider is None
        ):
            return result
        try:
            refreshed = await self._token_provider.refresh_access_token(
                self._member_id or "",
                access_token,
            )
        except Exception as exc:
            return self._oauth_failure(exc)
        return await self._send_with_token(payload, refreshed)

    async def _current_access_token(self) -> str:
        if self._token_provider is None:
            return self._access_token or ""
        return await self._token_provider.get_access_token(self._member_id or "")

    @staticmethod
    def _oauth_failure(exc: Exception) -> BitrixClientResult:
        from .oauth import BitrixOAuthAccessError

        if not isinstance(exc, BitrixOAuthAccessError):
            return BitrixClientResult.retryable("bitrix_oauth_unavailable")
        if exc.retryable:
            return BitrixClientResult.retryable(
                f"bitrix_{exc.error_code}",
                retry_after_seconds=exc.retry_after_seconds,
            )
        return BitrixClientResult.failed(f"bitrix_{exc.error_code}")

    async def _send_with_token(
        self,
        payload: BitrixMessagePayloadPreview,
        access_token: str,
    ) -> BitrixClientResult:
        request_json = payload.model_dump(mode="json")
        request_json["auth"] = access_token
        try:
            response = await self._http_client.post(
                self._endpoint,
                json=request_json,
            )
        except httpx.TimeoutException:
            return BitrixClientResult.retryable("bitrix_timeout")
        except httpx.TransportError:
            return BitrixClientResult.retryable("bitrix_transport_error")

        status = response.status_code
        retry_after = _retry_after_seconds(response)
        remote_code = _remote_error_code(response)

        if status == 429 or remote_code in _RATE_LIMIT_CODES:
            return BitrixClientResult.retryable(
                "bitrix_rate_limited",
                http_status=status,
                retry_after_seconds=retry_after,
            )
        if remote_code in _TOKEN_REFRESH_CODES:
            return BitrixClientResult.retryable(
                "bitrix_token_expired",
                http_status=status,
            )
        if remote_code in _PERMANENT_API_CODES:
            return BitrixClientResult.failed(
                "bitrix_api_permanent",
                http_status=status,
            )
        if (
            status in {408, 425}
            or 500 <= status <= 599
            or remote_code in _RETRYABLE_API_CODES
        ):
            return BitrixClientResult.retryable(
                "bitrix_api_retryable",
                http_status=status,
                retry_after_seconds=retry_after,
            )
        if remote_code is not None:
            return BitrixClientResult.failed(
                "bitrix_api_permanent",
                http_status=status,
            )
        if not 200 <= status <= 299:
            return BitrixClientResult.failed(
                "bitrix_http_permanent",
                http_status=status,
            )

        try:
            parsed = BitrixSendResponse.model_validate(response.json())
        except (ValueError, ValidationError):
            return BitrixClientResult.failed(
                "bitrix_invalid_response",
                http_status=status,
            )
        return BitrixClientResult.succeeded(parsed, http_status=status)

    async def close(self) -> None:
        if self._owns_http_client:
            await self._http_client.aclose()

    async def __aenter__(self) -> "BitrixClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

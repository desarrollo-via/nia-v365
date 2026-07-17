"""Descubrimiento de chats piloto mediante una única consulta Bitrix de lectura."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from .bitrix_client import (
    _remote_error_code,
    _retry_after_seconds,
    _validated_portal_url,
)
from .pilot_scope import PilotScopeRule


BITRIX_CRM_CHAT_GET_PATH = "/rest/imopenlines.crm.chat.get"
_RATE_LIMIT_CODES = {"QUERY_LIMIT_EXCEEDED"}
_TOKEN_REFRESH_CODES = {"expired_token"}
_RETRYABLE_API_CODES = {
    "INTERNAL_SERVER_ERROR",
    "ERROR_UNEXPECTED_ANSWER",
    "OVERLOAD_LIMIT",
}
_PERMANENT_API_CODES = {
    "ACCESS_DENIED",
    "ERROR_ARGUMENT",
    "INVALID_REQUEST",
    "NO_AUTH_FOUND",
    "INVALID_CREDENTIALS",
    "insufficient_scope",
    "user_access_error",
    "PORTAL_DELETED",
}


class PilotCrmEntityType(str, Enum):
    LEAD = "lead"
    DEAL = "deal"
    CONTACT = "contact"
    COMPANY = "company"


class PilotDiscoveryRequest(BaseModel):
    """Identidad CRM controlada y datos necesarios para proponer la regla."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    crm_entity_type: PilotCrmEntityType
    crm_entity_id: int = Field(gt=0)
    member_id: Optional[str] = Field(default=None, min_length=1)
    bot_id: Optional[int] = Field(default=None, gt=0)
    active_only: bool = False

    @model_validator(mode="after")
    def validate_optional_rule_identity(self) -> "PilotDiscoveryRequest":
        if (self.member_id is None) != (self.bot_id is None):
            raise ValueError(
                "member_id y bot_id deben suministrarse juntos"
            )
        if self.member_id is not None:
            member = self.member_id.strip()
            if not member:
                raise ValueError("member_id no admite texto vacío")
            object.__setattr__(self, "member_id", member)
        return self


class BitrixCrmChat(BaseModel):
    """Fila mínima y segura devuelta por ``imopenlines.crm.chat.get``."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    chat_id: int = Field(alias="CHAT_ID", gt=0)
    connector_id: str = Field(alias="CONNECTOR_ID", min_length=1)
    connector_title: str = Field(alias="CONNECTOR_TITLE", min_length=1)


class BitrixCrmChatResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    result: tuple[BitrixCrmChat, ...]


class PilotDiscoveryTransportDecision(str, Enum):
    SUCCESS = "success"
    RETRY = "retry"
    FAIL = "fail"


@dataclass(frozen=True)
class PilotDiscoveryTransportResult:
    """Resultado del transporte sin cuerpo remoto, token ni descripción."""

    decision: PilotDiscoveryTransportDecision
    chats: tuple[BitrixCrmChat, ...] = ()
    error_code: Optional[str] = None
    http_status: Optional[int] = None
    retry_after_seconds: int = 0

    def __post_init__(self) -> None:
        if self.retry_after_seconds < 0:
            raise ValueError("retry_after_seconds no puede ser negativo")
        if self.decision is PilotDiscoveryTransportDecision.SUCCESS:
            if self.error_code is not None or self.retry_after_seconds:
                raise ValueError("success no admite error ni reintento")
            return
        if self.chats or not self.error_code:
            raise ValueError("retry y fail requieren solo un error seguro")
        if (
            self.decision is PilotDiscoveryTransportDecision.FAIL
            and self.retry_after_seconds
        ):
            raise ValueError("fail no admite espera de reintento")


class PilotDiscoveryTransport(Protocol):
    async def get_crm_chats(
        self,
        request: PilotDiscoveryRequest,
    ) -> PilotDiscoveryTransportResult: ...


class BitrixPilotDiscoveryClient:
    """
    Transporte con una superficie de solo lectura y endpoint fijo.

    No ofrece una llamada REST genérica ni métodos para registrar bots,
    actualizar Líneas Abiertas, enviar mensajes o modificar el CRM.
    """

    def __init__(
        self,
        *,
        portal_url: str,
        access_token: str,
        timeout_seconds: float,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        token = access_token.strip()
        if not token:
            raise ValueError("access_token no puede estar vacío")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds debe ser positivo")

        self._endpoint = (
            f"{_validated_portal_url(portal_url)}"
            f"{BITRIX_CRM_CHAT_GET_PATH}"
        )
        self._access_token = token
        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.AsyncClient(
            timeout=timeout_seconds
        )

    async def get_crm_chats(
        self,
        request: PilotDiscoveryRequest,
    ) -> PilotDiscoveryTransportResult:
        request_json = {
            "CRM_ENTITY_TYPE": request.crm_entity_type.value,
            "CRM_ENTITY": request.crm_entity_id,
            "ACTIVE_ONLY": "Y" if request.active_only else "N",
            "auth": self._access_token,
        }
        try:
            response = await self._http_client.post(
                self._endpoint,
                json=request_json,
            )
        except httpx.TimeoutException:
            return PilotDiscoveryTransportResult(
                decision=PilotDiscoveryTransportDecision.RETRY,
                error_code="pilot_discovery_timeout",
            )
        except httpx.TransportError:
            return PilotDiscoveryTransportResult(
                decision=PilotDiscoveryTransportDecision.RETRY,
                error_code="pilot_discovery_transport_error",
            )

        status = response.status_code
        retry_after = _retry_after_seconds(response)
        remote_code = _remote_error_code(response)

        if status == 429 or remote_code in _RATE_LIMIT_CODES:
            return PilotDiscoveryTransportResult(
                decision=PilotDiscoveryTransportDecision.RETRY,
                error_code="pilot_discovery_rate_limited",
                http_status=status,
                retry_after_seconds=retry_after,
            )
        if remote_code in _TOKEN_REFRESH_CODES:
            return PilotDiscoveryTransportResult(
                decision=PilotDiscoveryTransportDecision.RETRY,
                error_code="pilot_discovery_token_expired",
                http_status=status,
            )
        if (
            status in {408, 425}
            or 500 <= status <= 599
            or remote_code in _RETRYABLE_API_CODES
        ):
            return PilotDiscoveryTransportResult(
                decision=PilotDiscoveryTransportDecision.RETRY,
                error_code="pilot_discovery_api_retryable",
                http_status=status,
                retry_after_seconds=retry_after,
            )
        if remote_code in _PERMANENT_API_CODES or remote_code is not None:
            return PilotDiscoveryTransportResult(
                decision=PilotDiscoveryTransportDecision.FAIL,
                error_code="pilot_discovery_api_permanent",
                http_status=status,
            )
        if not 200 <= status <= 299:
            return PilotDiscoveryTransportResult(
                decision=PilotDiscoveryTransportDecision.FAIL,
                error_code="pilot_discovery_http_permanent",
                http_status=status,
            )

        try:
            parsed = BitrixCrmChatResponse.model_validate(response.json())
        except (ValueError, ValidationError):
            return PilotDiscoveryTransportResult(
                decision=PilotDiscoveryTransportDecision.FAIL,
                error_code="pilot_discovery_invalid_response",
                http_status=status,
            )
        return PilotDiscoveryTransportResult(
            decision=PilotDiscoveryTransportDecision.SUCCESS,
            chats=parsed.result,
            http_status=status,
        )

    async def close(self) -> None:
        if self._owns_http_client:
            await self._http_client.aclose()

    async def __aenter__(self) -> "BitrixPilotDiscoveryClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


class PilotChatCandidate(BaseModel):
    """Vista segura para escoger una identidad exacta del piloto."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chat_id: int = Field(gt=0)
    dialog_id: str = Field(min_length=1)
    connector_id: str = Field(min_length=1)
    connector_title: str = Field(min_length=1)
    crm_entity_type: PilotCrmEntityType
    crm_entity_id: int = Field(gt=0)
    pilot_rule: Optional[PilotScopeRule] = None


class PilotDiscoveryStatus(str, Enum):
    FOUND = "found"
    EMPTY = "empty"
    RETRY = "retry"
    FAILED = "failed"


class PilotDiscoveryResult(BaseModel):
    """Salida del inspector apta para revisión humana y sin datos del chat."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: PilotDiscoveryStatus
    reason: str
    candidates: tuple[PilotChatCandidate, ...] = ()
    retry_after_seconds: int = Field(default=0, ge=0)


class PilotChatInspector:
    """Normaliza chats CRM sin persistir, enviar o cambiar estado externo."""

    def __init__(self, transport: PilotDiscoveryTransport) -> None:
        self._transport = transport

    async def inspect(
        self,
        request: PilotDiscoveryRequest,
    ) -> PilotDiscoveryResult:
        transport_result = await self._transport.get_crm_chats(request)
        if (
            transport_result.decision
            is PilotDiscoveryTransportDecision.RETRY
        ):
            return PilotDiscoveryResult(
                status=PilotDiscoveryStatus.RETRY,
                reason=transport_result.error_code
                or "pilot_discovery_retryable",
                retry_after_seconds=transport_result.retry_after_seconds,
            )
        if (
            transport_result.decision
            is PilotDiscoveryTransportDecision.FAIL
        ):
            return PilotDiscoveryResult(
                status=PilotDiscoveryStatus.FAILED,
                reason=transport_result.error_code
                or "pilot_discovery_failed",
            )

        candidates = []
        seen_chat_ids: set[int] = set()
        for chat in transport_result.chats:
            if chat.chat_id in seen_chat_ids:
                continue
            seen_chat_ids.add(chat.chat_id)
            dialog_id = f"chat{chat.chat_id}"
            pilot_rule = None
            if request.member_id is not None and request.bot_id is not None:
                pilot_rule = PilotScopeRule(
                    member_id=request.member_id,
                    bot_id=request.bot_id,
                    dialog_id=dialog_id,
                    chat_id=chat.chat_id,
                )
            candidates.append(
                PilotChatCandidate(
                    chat_id=chat.chat_id,
                    dialog_id=dialog_id,
                    connector_id=chat.connector_id,
                    connector_title=chat.connector_title,
                    crm_entity_type=request.crm_entity_type,
                    crm_entity_id=request.crm_entity_id,
                    pilot_rule=pilot_rule,
                )
            )

        if not candidates:
            return PilotDiscoveryResult(
                status=PilotDiscoveryStatus.EMPTY,
                reason="pilot_discovery_no_chats",
            )
        return PilotDiscoveryResult(
            status=PilotDiscoveryStatus.FOUND,
            reason="pilot_discovery_candidates_found",
            candidates=tuple(candidates),
        )

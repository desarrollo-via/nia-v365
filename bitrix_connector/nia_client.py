"""Límite HTTP aislado entre el conector Bitrix y el contrato público de NIA."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from .preflight import NiaTextPayloadPreview


NIA_CHAT_PATH = "/nia/chat"


class NiaChatResponse(BaseModel):
    """Respuesta pública aceptada del endpoint de texto de NIA."""

    model_config = ConfigDict(extra="ignore")

    respuesta: str
    etapa: Optional[str] = None
    items_resultado: Optional[list[Any]] = None
    cliente: Optional[dict[str, Any]] = None


class NiaClientDecision(str, Enum):
    SUCCESS = "success"
    RETRY = "retry"
    FAIL = "fail"


@dataclass(frozen=True)
class NiaClientResult:
    """Resultado seguro: nunca conserva cuerpos ni excepciones remotas."""

    decision: NiaClientDecision
    response: Optional[NiaChatResponse] = None
    error_code: Optional[str] = None
    http_status: Optional[int] = None
    retry_after_seconds: int = 0

    def __post_init__(self) -> None:
        if self.retry_after_seconds < 0:
            raise ValueError("retry_after_seconds no puede ser negativo")

        if self.decision is NiaClientDecision.SUCCESS:
            if self.response is None or self.error_code is not None:
                raise ValueError("success requiere response y no admite error_code")
            if self.retry_after_seconds != 0:
                raise ValueError("success no admite espera de reintento")
            return

        if self.response is not None or not self.error_code:
            raise ValueError("retry y fail requieren solo un error_code seguro")
        if (
            self.decision is NiaClientDecision.FAIL
            and self.retry_after_seconds != 0
        ):
            raise ValueError("fail no admite espera de reintento")

    @classmethod
    def succeeded(
        cls,
        response: NiaChatResponse,
        *,
        http_status: int,
    ) -> "NiaClientResult":
        return cls(
            decision=NiaClientDecision.SUCCESS,
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
    ) -> "NiaClientResult":
        return cls(
            decision=NiaClientDecision.RETRY,
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
    ) -> "NiaClientResult":
        return cls(
            decision=NiaClientDecision.FAIL,
            error_code=error_code,
            http_status=http_status,
        )


def _validated_base_url(base_url: str) -> str:
    cleaned = base_url.strip().rstrip("/")
    if not cleaned:
        raise ValueError("nia_base_url no puede estar vacía")

    parsed = httpx.URL(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.host:
        raise ValueError("nia_base_url debe ser una URL HTTP válida")
    if parsed.userinfo or parsed.query or parsed.fragment:
        raise ValueError("nia_base_url no admite credenciales, query ni fragmento")
    if parsed.path not in {"", "/"}:
        raise ValueError("nia_base_url debe identificar la raíz del servicio")
    return cleaned


def _retry_after_seconds(response: httpx.Response) -> int:
    value = response.headers.get("Retry-After", "").strip()
    if not value.isdigit():
        return 0
    return min(int(value), 86_400)


class NiaClient:
    """
    Envía únicamente payloads de texto ya aprobados al contrato público de NIA.

    El cliente HTTP puede inyectarse para pruebas o composición. Si se inyecta,
    su ciclo de vida pertenece al llamador.
    """

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds debe ser positivo")

        self._endpoint = f"{_validated_base_url(base_url)}{NIA_CHAT_PATH}"
        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.AsyncClient(
            timeout=timeout_seconds
        )

    async def send_approved_text(
        self,
        payload: NiaTextPayloadPreview,
    ) -> NiaClientResult:
        request_json = payload.model_dump(mode="json")

        try:
            response = await self._http_client.post(
                self._endpoint,
                json=request_json,
            )
        except httpx.TimeoutException:
            return NiaClientResult.retryable("nia_timeout")
        except httpx.TransportError:
            return NiaClientResult.retryable("nia_transport_error")

        status = response.status_code
        if status == 429:
            return NiaClientResult.retryable(
                "nia_rate_limited",
                http_status=status,
                retry_after_seconds=_retry_after_seconds(response),
            )
        if status in {408, 425} or 500 <= status <= 599:
            return NiaClientResult.retryable(
                "nia_http_retryable",
                http_status=status,
            )
        if not 200 <= status <= 299:
            return NiaClientResult.failed(
                "nia_http_permanent",
                http_status=status,
            )

        try:
            response_payload = response.json()
        except ValueError:
            return NiaClientResult.failed(
                "nia_invalid_json",
                http_status=status,
            )

        try:
            parsed = NiaChatResponse.model_validate(response_payload)
        except ValidationError:
            return NiaClientResult.failed(
                "nia_invalid_response",
                http_status=status,
            )

        return NiaClientResult.succeeded(parsed, http_status=status)

    async def close(self) -> None:
        if self._owns_http_client:
            await self._http_client.aclose()

    async def __aenter__(self) -> "NiaClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

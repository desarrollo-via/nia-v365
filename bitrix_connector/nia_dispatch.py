"""Despacho durable del payload aprobado hacia un cliente NIA inyectado."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Optional, Protocol

from pydantic import ValidationError

from .models import ConnectorEventRecord
from .nia_client import NiaClientDecision, NiaClientResult
from .output_review import build_output_review
from .preflight import NiaTextPayloadPreview, build_nia_payload_hash
from .storage import ConnectorEventStore
from .worker import ConnectorHandlerResult


class NiaTextSender(Protocol):
    async def send_approved_text(
        self,
        payload: NiaTextPayloadPreview,
    ) -> NiaClientResult: ...


class NiaDispatchWorkerStore:
    """Adapta las transiciones NIA al contrato ya usado por ConnectorWorker."""

    def __init__(self, store: ConnectorEventStore) -> None:
        self._store = store

    async def claim_next(self, **kwargs):
        return await self._store.claim_ready_for_nia(**kwargs)

    async def retry_claim(self, *args, **kwargs):
        return await self._store.retry_nia_claim(*args, **kwargs)

    async def fail_claim(self, *args, **kwargs):
        return await self._store.fail_nia_claim(*args, **kwargs)

    async def complete_claim(self, *args, **kwargs):
        raise RuntimeError("nia_dispatch_success_requires_atomic_response_save")


class NiaDispatchWorkerHandler:
    """Valida el payload durable, invoca el doble NIA y guarda su respuesta."""

    def __init__(
        self,
        store: ConnectorEventStore,
        nia_client: NiaTextSender,
        *,
        lease_owner: str,
        default_retry_after_seconds: int = 30,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        owner = lease_owner.strip()
        if not owner:
            raise ValueError("lease_owner no puede estar vacío")
        if default_retry_after_seconds <= 0:
            raise ValueError("default_retry_after_seconds debe ser positivo")

        self._store = store
        self._nia_client = nia_client
        self._lease_owner = owner
        self._default_retry_after_seconds = default_retry_after_seconds
        self._clock = clock

    async def handle(self, event: ConnectorEventRecord) -> ConnectorHandlerResult:
        payload = self._approved_payload(event)
        if payload is None:
            return ConnectorHandlerResult.failed("nia_approved_payload_invalid")

        result = await self._nia_client.send_approved_text(payload)
        if result.decision is NiaClientDecision.RETRY:
            return ConnectorHandlerResult.retryable(
                result.error_code or "nia_retryable_error",
                retry_after_seconds=(
                    result.retry_after_seconds
                    or self._default_retry_after_seconds
                ),
            )
        if result.decision is NiaClientDecision.FAIL:
            return ConnectorHandlerResult.failed(
                result.error_code or "nia_permanent_error"
            )

        if result.response is None or result.http_status is None:
            return ConnectorHandlerResult.failed("nia_success_result_invalid")

        applied = await self._store.save_nia_response(
            event.event_key,
            self._lease_owner,
            build_output_review(event, result.response),
            http_status=result.http_status,
            now=self._clock(),
        )
        if not applied:
            return ConnectorHandlerResult.lease_lost()
        return ConnectorHandlerResult.applied()

    @staticmethod
    def _approved_payload(
        event: ConnectorEventRecord,
    ) -> Optional[NiaTextPayloadPreview]:
        if event.processing_stage != "nia_dispatch":
            return None
        review = event.preflight_review or {}
        input_decision = event.input_decision or {}
        if input_decision.get("decision") != "approved":
            return None
        if input_decision.get("content_hash") != review.get("content_hash"):
            return None
        payload = review.get("nia_payload_preview")
        if payload is None:
            return None
        try:
            parsed = NiaTextPayloadPreview.model_validate(payload)
        except ValidationError:
            return None
        if build_nia_payload_hash(parsed) != review.get("content_hash"):
            return None
        return parsed

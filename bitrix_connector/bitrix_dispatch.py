"""Despacho durable de salidas aprobadas hacia un cliente Bitrix inyectado."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Optional, Protocol

from pydantic import ValidationError

from .bitrix_client import BitrixClientDecision, BitrixClientResult
from .models import ConnectorEventRecord
from .output_review import (
    BitrixMessagePayloadPreview,
    OutputReview,
    build_output_content_hash,
)
from .storage import ConnectorEventStore
from .worker import ConnectorHandlerResult


class BitrixMessageSender(Protocol):
    async def send_approved_message(
        self,
        payload: BitrixMessagePayloadPreview,
    ) -> BitrixClientResult: ...


class BitrixDispatchWorkerStore:
    """Adapta las transiciones Bitrix al contrato de ConnectorWorker."""

    def __init__(self, store: ConnectorEventStore) -> None:
        self._store = store

    async def claim_next(self, **kwargs):
        return await self._store.claim_ready_for_bitrix(**kwargs)

    async def retry_claim(self, *args, **kwargs):
        return await self._store.retry_bitrix_claim(*args, **kwargs)

    async def fail_claim(self, *args, **kwargs):
        return await self._store.fail_bitrix_claim(*args, **kwargs)

    async def complete_claim(self, *args, **kwargs):
        raise RuntimeError("bitrix_dispatch_success_requires_atomic_sent_save")


class BitrixDispatchWorkerHandler:
    """Valida aprobación y hash antes de usar el cliente Bitrix falso."""

    def __init__(
        self,
        store: ConnectorEventStore,
        bitrix_client: BitrixMessageSender,
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
        self._bitrix_client = bitrix_client
        self._lease_owner = owner
        self._default_retry_after_seconds = default_retry_after_seconds
        self._clock = clock

    async def handle(self, event: ConnectorEventRecord) -> ConnectorHandlerResult:
        if event.outbound_message_id is not None:
            return ConnectorHandlerResult.lease_lost()
        payload = self._approved_payload(event)
        if payload is None:
            return ConnectorHandlerResult.failed(
                "bitrix_approved_payload_invalid"
            )

        result = await self._bitrix_client.send_approved_message(payload)
        if result.decision is BitrixClientDecision.RETRY:
            return ConnectorHandlerResult.retryable(
                result.error_code or "bitrix_retryable_error",
                retry_after_seconds=(
                    result.retry_after_seconds
                    or self._default_retry_after_seconds
                ),
            )
        if result.decision is BitrixClientDecision.FAIL:
            return ConnectorHandlerResult.failed(
                result.error_code or "bitrix_permanent_error"
            )

        if result.response is None or result.http_status is None:
            return ConnectorHandlerResult.failed(
                "bitrix_success_result_invalid"
            )
        applied = await self._store.save_bitrix_sent(
            event.event_key,
            self._lease_owner,
            result.response,
            http_status=result.http_status,
            now=self._clock(),
        )
        if not applied:
            return ConnectorHandlerResult.lease_lost()
        return ConnectorHandlerResult.applied()

    @staticmethod
    def _approved_payload(
        event: ConnectorEventRecord,
    ) -> Optional[BitrixMessagePayloadPreview]:
        if event.processing_stage != "bitrix_dispatch":
            return None
        review_data = event.output_review or {}
        decision = event.output_decision or {}
        if decision.get("decision") != "approved":
            return None
        if decision.get("content_hash") != review_data.get("content_hash"):
            return None
        try:
            review = OutputReview.model_validate(review_data)
        except ValidationError:
            return None
        if review.decision != "ready_for_review":
            return None
        if (
            build_output_content_hash(review.bitrix_payload_preview)
            != review.content_hash
        ):
            return None
        return review.bitrix_payload_preview

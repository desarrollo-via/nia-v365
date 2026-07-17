"""Persistencia exclusiva del conector, independiente de la memoria de NIA."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional, Protocol

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING, ReturnDocument

from .config import ConnectorMode, ConnectorSettings, DEFAULT_EVENTS_COLLECTION
from .bitrix_client import BitrixSendResponse
from .idempotency import build_event_key
from .models import (
    ConnectorEventRecord,
    ConnectorEventStatus,
    NormalizedBitrixEvent,
    StorageWriteResult,
)
from .output_review import OutputReview
from .preflight import TextPreflightReview
from .review import (
    ReviewDecisionOutcome,
    ReviewDecisionResult,
    ReviewInputDecision,
)
from .security import redact_form_data


EVENT_KEY_INDEX = "idx_nia_bitrix_event_key_unique"
STATUS_RECEIVED_INDEX = "idx_nia_bitrix_status_received"
STATUS_LEASE_INDEX = "idx_nia_bitrix_status_lease"


class ConnectorEventStore(Protocol):
    """Contrato mínimo que consumirá posteriormente el servicio conector."""

    async def ensure_indexes(self) -> None: ...

    async def save_received(self, record: ConnectorEventRecord) -> StorageWriteResult: ...

    async def get_by_key(self, event_key: str) -> Optional[dict[str, Any]]: ...

    async def claim_next(
        self,
        *,
        lease_owner: str,
        lease_seconds: int,
        now: Optional[datetime] = None,
    ) -> Optional[ConnectorEventRecord]: ...

    async def renew_claim(
        self,
        event_key: str,
        lease_owner: str,
        *,
        lease_seconds: int,
        now: Optional[datetime] = None,
    ) -> bool: ...

    async def complete_claim(
        self,
        event_key: str,
        lease_owner: str,
        *,
        now: Optional[datetime] = None,
    ) -> bool: ...

    async def retry_claim(
        self,
        event_key: str,
        lease_owner: str,
        *,
        error_code: str,
        retry_after_seconds: int,
        now: Optional[datetime] = None,
    ) -> bool: ...

    async def fail_claim(
        self,
        event_key: str,
        lease_owner: str,
        *,
        error_code: str,
        now: Optional[datetime] = None,
    ) -> bool: ...

    async def save_preflight(
        self,
        event_key: str,
        lease_owner: str,
        review: TextPreflightReview,
        *,
        now: Optional[datetime] = None,
    ) -> bool: ...

    async def list_reviews(
        self,
        *,
        decision: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]: ...

    async def get_review(self, event_key: str) -> Optional[dict[str, Any]]: ...

    async def approve_input(
        self,
        event_key: str,
        *,
        content_hash: str,
        actor: str,
        reason: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> ReviewDecisionResult: ...

    async def reject_input(
        self,
        event_key: str,
        *,
        content_hash: str,
        actor: str,
        reason: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> ReviewDecisionResult: ...

    async def claim_ready_for_nia(
        self,
        *,
        lease_owner: str,
        lease_seconds: int,
        now: Optional[datetime] = None,
    ) -> Optional[ConnectorEventRecord]: ...

    async def save_nia_response(
        self,
        event_key: str,
        lease_owner: str,
        review: OutputReview,
        *,
        http_status: int,
        now: Optional[datetime] = None,
    ) -> bool: ...

    async def retry_nia_claim(
        self,
        event_key: str,
        lease_owner: str,
        *,
        error_code: str,
        retry_after_seconds: int,
        now: Optional[datetime] = None,
    ) -> bool: ...

    async def fail_nia_claim(
        self,
        event_key: str,
        lease_owner: str,
        *,
        error_code: str,
        now: Optional[datetime] = None,
    ) -> bool: ...

    async def list_output_reviews(
        self,
        *,
        decision: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]: ...

    async def get_output_review(
        self,
        event_key: str,
    ) -> Optional[dict[str, Any]]: ...

    async def approve_output(
        self,
        event_key: str,
        *,
        content_hash: str,
        actor: str,
        reason: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> ReviewDecisionResult: ...

    async def reject_output(
        self,
        event_key: str,
        *,
        content_hash: str,
        actor: str,
        reason: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> ReviewDecisionResult: ...

    async def claim_ready_for_bitrix(
        self,
        *,
        lease_owner: str,
        lease_seconds: int,
        now: Optional[datetime] = None,
    ) -> Optional[ConnectorEventRecord]: ...

    async def save_bitrix_sent(
        self,
        event_key: str,
        lease_owner: str,
        response: BitrixSendResponse,
        *,
        http_status: int,
        now: Optional[datetime] = None,
    ) -> bool: ...

    async def retry_bitrix_claim(
        self,
        event_key: str,
        lease_owner: str,
        *,
        error_code: str,
        retry_after_seconds: int,
        now: Optional[datetime] = None,
    ) -> bool: ...

    async def fail_bitrix_claim(
        self,
        event_key: str,
        lease_owner: str,
        *,
        error_code: str,
        now: Optional[datetime] = None,
    ) -> bool: ...


def _serializable_form(form: Mapping[str, Any]) -> dict[str, Any]:
    """Redacta secretos y evita guardar objetos propios del framework web."""
    redacted = redact_form_data(form)
    return {
        str(key): value if isinstance(value, (str, int, float, bool)) or value is None else str(value)
        for key, value in redacted.items()
    }


def _normalized_time(value: Optional[datetime] = None) -> datetime:
    instant = value or datetime.now(timezone.utc)
    if instant.tzinfo is None:
        return instant.replace(tzinfo=timezone.utc)
    return instant


def _required_text(value: str, field_name: str) -> str:
    clean = value.strip()
    if not clean:
        raise ValueError(f"{field_name} no puede estar vacío")
    return clean


def build_received_record(
    event: NormalizedBitrixEvent,
    raw_form: Mapping[str, Any],
    *,
    identity_verified: bool,
    security_reason: str,
    received_at: Optional[datetime] = None,
) -> ConnectorEventRecord:
    """Construye el documento inicial sin conservar tokens del evento."""
    now = _normalized_time(received_at)

    return ConnectorEventRecord(
        event_key=build_event_key(event),
        received_at=now,
        updated_at=now,
        normalized_event=event.model_dump(mode="python"),
        raw_redacted=_serializable_form(raw_form),
        identity_verified=identity_verified,
        security_reason=security_reason,
    )


class MongoConnectorEventStore:
    """Repositorio MongoDB inyectable; su construcción no abre conexiones."""

    def __init__(self, collection: Any) -> None:
        self._collection = collection

    @classmethod
    def from_database(
        cls,
        database: Any,
        collection_name: str = DEFAULT_EVENTS_COLLECTION,
    ) -> "MongoConnectorEventStore":
        clean_name = collection_name.strip()
        if not clean_name:
            raise ValueError("collection_name no puede estar vacío")
        return cls(database[clean_name])

    async def ensure_indexes(self) -> None:
        await self._collection.create_index(
            [("event_key", ASCENDING)],
            unique=True,
            name=EVENT_KEY_INDEX,
        )
        await self._collection.create_index(
            [("status", ASCENDING), ("received_at", ASCENDING)],
            name=STATUS_RECEIVED_INDEX,
        )
        await self._collection.create_index(
            [("status", ASCENDING), ("lease_until", ASCENDING)],
            name=STATUS_LEASE_INDEX,
        )

    async def save_received(self, record: ConnectorEventRecord) -> StorageWriteResult:
        result = await self._collection.update_one(
            {"event_key": record.event_key},
            {"$setOnInsert": record.model_dump(mode="python")},
            upsert=True,
        )
        created = result.upserted_id is not None
        return StorageWriteResult(
            event_key=record.event_key,
            created=created,
            duplicate=not created,
        )

    async def get_by_key(self, event_key: str) -> Optional[dict[str, Any]]:
        return await self._collection.find_one({"event_key": event_key}, {"_id": 0})

    async def claim_next(
        self,
        *,
        lease_owner: str,
        lease_seconds: int,
        now: Optional[datetime] = None,
    ) -> Optional[ConnectorEventRecord]:
        owner = _required_text(lease_owner, "lease_owner")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds debe ser positivo")

        claimed_at = _normalized_time(now)
        lease_until = claimed_at + timedelta(seconds=lease_seconds)

        document = await self._collection.find_one_and_update(
            {
                "$or": [
                    {"status": ConnectorEventStatus.RECEIVED.value},
                    {
                        "status": ConnectorEventStatus.PROCESSING.value,
                        "lease_until": {"$lte": claimed_at},
                        "processing_stage": {
                            "$nin": ["nia_dispatch", "bitrix_dispatch"],
                        },
                    },
                    {
                        "status": ConnectorEventStatus.RETRYABLE_ERROR.value,
                        "lease_until": {"$lte": claimed_at},
                        "processing_stage": {
                            "$nin": ["nia_dispatch", "bitrix_dispatch"],
                        },
                    },
                ],
            },
            {
                "$set": {
                    "status": ConnectorEventStatus.PROCESSING.value,
                    "updated_at": claimed_at,
                    "lease_owner": owner,
                    "lease_until": lease_until,
                    "processing_stage": "preflight",
                    "last_error": None,
                },
                "$inc": {"attempt_count": 1},
            },
            sort=[("received_at", ASCENDING)],
            projection={"_id": 0},
            return_document=ReturnDocument.AFTER,
        )
        if document is None:
            return None
        return ConnectorEventRecord.model_validate(document)

    async def renew_claim(
        self,
        event_key: str,
        lease_owner: str,
        *,
        lease_seconds: int,
        now: Optional[datetime] = None,
    ) -> bool:
        owner = _required_text(lease_owner, "lease_owner")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds debe ser positivo")

        renewed_at = _normalized_time(now)
        result = await self._collection.update_one(
            self._owned_active_claim(event_key, owner, renewed_at),
            {
                "$set": {
                    "updated_at": renewed_at,
                    "lease_until": renewed_at + timedelta(seconds=lease_seconds),
                },
            },
            upsert=False,
        )
        return result.modified_count == 1

    async def complete_claim(
        self,
        event_key: str,
        lease_owner: str,
        *,
        now: Optional[datetime] = None,
    ) -> bool:
        completed_at = _normalized_time(now)
        return await self._finish_claim(
            event_key,
            lease_owner,
            now=completed_at,
            status=ConnectorEventStatus.PENDING,
            lease_until=None,
            last_error=None,
        )

    async def retry_claim(
        self,
        event_key: str,
        lease_owner: str,
        *,
        error_code: str,
        retry_after_seconds: int,
        now: Optional[datetime] = None,
    ) -> bool:
        if retry_after_seconds < 0:
            raise ValueError("retry_after_seconds no puede ser negativo")
        retried_at = _normalized_time(now)
        return await self._finish_claim(
            event_key,
            lease_owner,
            now=retried_at,
            status=ConnectorEventStatus.RETRYABLE_ERROR,
            lease_until=retried_at + timedelta(seconds=retry_after_seconds),
            last_error=_required_text(error_code, "error_code"),
        )

    async def fail_claim(
        self,
        event_key: str,
        lease_owner: str,
        *,
        error_code: str,
        now: Optional[datetime] = None,
    ) -> bool:
        failed_at = _normalized_time(now)
        return await self._finish_claim(
            event_key,
            lease_owner,
            now=failed_at,
            status=ConnectorEventStatus.FAILED,
            lease_until=None,
            last_error=_required_text(error_code, "error_code"),
        )

    async def save_preflight(
        self,
        event_key: str,
        lease_owner: str,
        review: TextPreflightReview,
        *,
        now: Optional[datetime] = None,
    ) -> bool:
        if review.event_key != event_key:
            raise ValueError("preflight_event_key_mismatch")

        saved_at = _normalized_time(now)
        owner = _required_text(lease_owner, "lease_owner")
        result = await self._collection.update_one(
            self._owned_active_claim(event_key, owner, saved_at),
            {
                "$set": {
                    "status": ConnectorEventStatus.NEEDS_INPUT_REVIEW.value,
                    "updated_at": saved_at,
                    "lease_owner": None,
                    "lease_until": None,
                    "processing_stage": None,
                    "last_error": None,
                    "preflight_review": review.model_dump(mode="python"),
                },
            },
            upsert=False,
        )
        return result.modified_count == 1

    async def list_reviews(
        self,
        *,
        decision: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        selector: dict[str, Any] = {
            "status": ConnectorEventStatus.NEEDS_INPUT_REVIEW.value,
        }
        if decision is not None:
            selector["preflight_review.decision"] = decision

        projection = {
            "_id": 0,
            "event_key": 1,
            "status": 1,
            "received_at": 1,
            "updated_at": 1,
            "preflight_review": 1,
        }
        cursor = (
            self._collection.find(selector, projection)
            .sort("updated_at", DESCENDING)
            .limit(limit)
        )
        return [document async for document in cursor]

    async def get_review(self, event_key: str) -> Optional[dict[str, Any]]:
        return await self._collection.find_one(
            {
                "event_key": event_key,
                "status": ConnectorEventStatus.NEEDS_INPUT_REVIEW.value,
            },
            {
                "_id": 0,
                "event_key": 1,
                "status": 1,
                "received_at": 1,
                "updated_at": 1,
                "preflight_review": 1,
            },
        )

    async def approve_input(
        self,
        event_key: str,
        *,
        content_hash: str,
        actor: str,
        reason: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> ReviewDecisionResult:
        return await self._decide_input(
            event_key,
            decision=ReviewInputDecision.APPROVED,
            target_status=ConnectorEventStatus.READY_FOR_NIA,
            content_hash=content_hash,
            actor=actor,
            reason=reason,
            now=now,
        )

    async def reject_input(
        self,
        event_key: str,
        *,
        content_hash: str,
        actor: str,
        reason: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> ReviewDecisionResult:
        return await self._decide_input(
            event_key,
            decision=ReviewInputDecision.REJECTED,
            target_status=ConnectorEventStatus.INPUT_REJECTED,
            content_hash=content_hash,
            actor=actor,
            reason=reason,
            now=now,
        )

    async def claim_ready_for_nia(
        self,
        *,
        lease_owner: str,
        lease_seconds: int,
        now: Optional[datetime] = None,
    ) -> Optional[ConnectorEventRecord]:
        owner = _required_text(lease_owner, "lease_owner")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds debe ser positivo")

        claimed_at = _normalized_time(now)
        lease_until = claimed_at + timedelta(seconds=lease_seconds)
        document = await self._collection.find_one_and_update(
            {
                "$or": [
                    {"status": ConnectorEventStatus.READY_FOR_NIA.value},
                    {
                        "status": ConnectorEventStatus.PROCESSING.value,
                        "processing_stage": "nia_dispatch",
                        "lease_until": {"$lte": claimed_at},
                    },
                    {
                        "status": ConnectorEventStatus.RETRYABLE_ERROR.value,
                        "processing_stage": "nia_dispatch",
                        "lease_until": {"$lte": claimed_at},
                    },
                ],
            },
            {
                "$set": {
                    "status": ConnectorEventStatus.PROCESSING.value,
                    "updated_at": claimed_at,
                    "lease_owner": owner,
                    "lease_until": lease_until,
                    "processing_stage": "nia_dispatch",
                    "last_error": None,
                },
                "$inc": {
                    "attempt_count": 1,
                    "nia_attempt_count": 1,
                },
            },
            sort=[("received_at", ASCENDING)],
            projection={"_id": 0},
            return_document=ReturnDocument.AFTER,
        )
        if document is None:
            return None
        return ConnectorEventRecord.model_validate(document)

    async def save_nia_response(
        self,
        event_key: str,
        lease_owner: str,
        review: OutputReview,
        *,
        http_status: int,
        now: Optional[datetime] = None,
    ) -> bool:
        if review.event_key != event_key:
            raise ValueError("output_review_event_key_mismatch")
        if not 200 <= http_status <= 299:
            raise ValueError("http_status de respuesta NIA debe ser exitoso")

        saved_at = _normalized_time(now)
        result = await self._collection.update_one(
            self._owned_nia_claim(event_key, lease_owner, saved_at),
            {
                "$set": {
                    "status": ConnectorEventStatus.NEEDS_OUTPUT_REVIEW.value,
                    "updated_at": saved_at,
                    "lease_owner": None,
                    "lease_until": None,
                    "processing_stage": None,
                    "last_error": None,
                    "nia_response": review.nia_response.model_dump(mode="python"),
                    "output_review": review.model_dump(mode="python"),
                    "output_decision": None,
                    "nia_dispatch": {
                        "http_status": http_status,
                        "completed_at": saved_at,
                    },
                },
            },
            upsert=False,
        )
        return result.modified_count == 1

    async def list_output_reviews(
        self,
        *,
        decision: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        selector: dict[str, Any] = {
            "status": ConnectorEventStatus.NEEDS_OUTPUT_REVIEW.value,
        }
        if decision is not None:
            selector["output_review.decision"] = decision

        projection = {
            "_id": 0,
            "event_key": 1,
            "status": 1,
            "received_at": 1,
            "updated_at": 1,
            "output_review": 1,
        }
        cursor = (
            self._collection.find(selector, projection)
            .sort("updated_at", DESCENDING)
            .limit(limit)
        )
        return [document async for document in cursor]

    async def get_output_review(
        self,
        event_key: str,
    ) -> Optional[dict[str, Any]]:
        return await self._collection.find_one(
            {
                "event_key": event_key,
                "status": ConnectorEventStatus.NEEDS_OUTPUT_REVIEW.value,
            },
            {
                "_id": 0,
                "event_key": 1,
                "status": 1,
                "received_at": 1,
                "updated_at": 1,
                "output_review": 1,
            },
        )

    async def approve_output(
        self,
        event_key: str,
        *,
        content_hash: str,
        actor: str,
        reason: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> ReviewDecisionResult:
        return await self._decide_output(
            event_key,
            decision=ReviewInputDecision.APPROVED,
            target_status=ConnectorEventStatus.READY_FOR_BITRIX,
            content_hash=content_hash,
            actor=actor,
            reason=reason,
            now=now,
        )

    async def reject_output(
        self,
        event_key: str,
        *,
        content_hash: str,
        actor: str,
        reason: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> ReviewDecisionResult:
        return await self._decide_output(
            event_key,
            decision=ReviewInputDecision.REJECTED,
            target_status=ConnectorEventStatus.OUTPUT_REJECTED,
            content_hash=content_hash,
            actor=actor,
            reason=reason,
            now=now,
        )

    async def claim_ready_for_bitrix(
        self,
        *,
        lease_owner: str,
        lease_seconds: int,
        now: Optional[datetime] = None,
    ) -> Optional[ConnectorEventRecord]:
        owner = _required_text(lease_owner, "lease_owner")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds debe ser positivo")

        claimed_at = _normalized_time(now)
        lease_until = claimed_at + timedelta(seconds=lease_seconds)
        document = await self._collection.find_one_and_update(
            {
                "$or": [
                    {
                        "status": ConnectorEventStatus.READY_FOR_BITRIX.value,
                        "outbound_message_id": None,
                    },
                    {
                        "status": ConnectorEventStatus.PROCESSING.value,
                        "processing_stage": "bitrix_dispatch",
                        "lease_until": {"$lte": claimed_at},
                        "outbound_message_id": None,
                    },
                    {
                        "status": ConnectorEventStatus.RETRYABLE_ERROR.value,
                        "processing_stage": "bitrix_dispatch",
                        "lease_until": {"$lte": claimed_at},
                        "outbound_message_id": None,
                    },
                ],
            },
            {
                "$set": {
                    "status": ConnectorEventStatus.PROCESSING.value,
                    "updated_at": claimed_at,
                    "lease_owner": owner,
                    "lease_until": lease_until,
                    "processing_stage": "bitrix_dispatch",
                    "last_error": None,
                },
                "$inc": {
                    "attempt_count": 1,
                    "bitrix_attempt_count": 1,
                },
            },
            sort=[("received_at", ASCENDING)],
            projection={"_id": 0},
            return_document=ReturnDocument.AFTER,
        )
        if document is None:
            return None
        return ConnectorEventRecord.model_validate(document)

    async def save_bitrix_sent(
        self,
        event_key: str,
        lease_owner: str,
        response: BitrixSendResponse,
        *,
        http_status: int,
        now: Optional[datetime] = None,
    ) -> bool:
        if not 200 <= http_status <= 299:
            raise ValueError("http_status de respuesta Bitrix debe ser exitoso")

        sent_at = _normalized_time(now)
        result = await self._collection.update_one(
            {
                **self._owned_bitrix_claim(event_key, lease_owner, sent_at),
                "outbound_message_id": None,
            },
            {
                "$set": {
                    "status": ConnectorEventStatus.SENT.value,
                    "updated_at": sent_at,
                    "lease_owner": None,
                    "lease_until": None,
                    "processing_stage": None,
                    "last_error": None,
                    "outbound_message_id": response.result.id,
                    "bitrix_dispatch": {
                        "http_status": http_status,
                        "uuid_map": dict(response.result.uuidMap),
                        "sent_at": sent_at,
                    },
                },
            },
            upsert=False,
        )
        return result.modified_count == 1

    async def retry_bitrix_claim(
        self,
        event_key: str,
        lease_owner: str,
        *,
        error_code: str,
        retry_after_seconds: int,
        now: Optional[datetime] = None,
    ) -> bool:
        if retry_after_seconds < 0:
            raise ValueError("retry_after_seconds no puede ser negativo")
        retried_at = _normalized_time(now)
        return await self._finish_bitrix_claim(
            event_key,
            lease_owner,
            now=retried_at,
            status=ConnectorEventStatus.RETRYABLE_ERROR,
            lease_until=retried_at + timedelta(seconds=retry_after_seconds),
            last_error=_required_text(error_code, "error_code"),
        )

    async def fail_bitrix_claim(
        self,
        event_key: str,
        lease_owner: str,
        *,
        error_code: str,
        now: Optional[datetime] = None,
    ) -> bool:
        failed_at = _normalized_time(now)
        return await self._finish_bitrix_claim(
            event_key,
            lease_owner,
            now=failed_at,
            status=ConnectorEventStatus.FAILED,
            lease_until=None,
            last_error=_required_text(error_code, "error_code"),
        )

    async def retry_nia_claim(
        self,
        event_key: str,
        lease_owner: str,
        *,
        error_code: str,
        retry_after_seconds: int,
        now: Optional[datetime] = None,
    ) -> bool:
        if retry_after_seconds < 0:
            raise ValueError("retry_after_seconds no puede ser negativo")
        retried_at = _normalized_time(now)
        return await self._finish_nia_claim(
            event_key,
            lease_owner,
            now=retried_at,
            status=ConnectorEventStatus.RETRYABLE_ERROR,
            lease_until=retried_at + timedelta(seconds=retry_after_seconds),
            last_error=_required_text(error_code, "error_code"),
        )

    async def fail_nia_claim(
        self,
        event_key: str,
        lease_owner: str,
        *,
        error_code: str,
        now: Optional[datetime] = None,
    ) -> bool:
        failed_at = _normalized_time(now)
        return await self._finish_nia_claim(
            event_key,
            lease_owner,
            now=failed_at,
            status=ConnectorEventStatus.FAILED,
            lease_until=None,
            last_error=_required_text(error_code, "error_code"),
        )

    async def _decide_input(
        self,
        event_key: str,
        *,
        decision: ReviewInputDecision,
        target_status: ConnectorEventStatus,
        content_hash: str,
        actor: str,
        reason: Optional[str],
        now: Optional[datetime],
    ) -> ReviewDecisionResult:
        clean_actor = _required_text(actor, "actor")
        clean_reason = (reason or "").strip() or None
        decided_at = _normalized_time(now)
        decision_document = {
            "decision": decision.value,
            "content_hash": content_hash,
            "actor": clean_actor,
            "reason": clean_reason,
            "decided_at": decided_at,
        }
        selector: dict[str, Any] = {
            "event_key": event_key,
            "status": ConnectorEventStatus.NEEDS_INPUT_REVIEW.value,
            "preflight_review.content_hash": content_hash,
            "input_decision": None,
        }
        if decision is ReviewInputDecision.APPROVED:
            selector["preflight_review.decision"] = "ready_for_review"

        projection = {
            "_id": 0,
            "event_key": 1,
            "status": 1,
            "preflight_review": 1,
            "input_decision": 1,
        }
        updated = await self._collection.find_one_and_update(
            selector,
            {
                "$set": {
                    "status": target_status.value,
                    "updated_at": decided_at,
                    "input_decision": decision_document,
                },
            },
            projection=projection,
            return_document=ReturnDocument.AFTER,
        )
        if updated is not None:
            return ReviewDecisionResult(
                event_key=event_key,
                outcome=ReviewDecisionOutcome.APPLIED,
                decision=decision,
                status=target_status.value,
                content_hash=content_hash,
                actor=clean_actor,
                reason=clean_reason,
                decided_at=decided_at,
            )

        current = await self._collection.find_one(
            {"event_key": event_key},
            projection,
        )
        if current is None:
            return ReviewDecisionResult(
                event_key=event_key,
                outcome=ReviewDecisionOutcome.NOT_FOUND,
            )

        existing = current.get("input_decision")
        if existing is not None:
            exact_repeat = (
                existing.get("decision") == decision.value
                and existing.get("content_hash") == content_hash
                and existing.get("actor") == clean_actor
                and existing.get("reason") == clean_reason
            )
            if exact_repeat:
                return ReviewDecisionResult(
                    event_key=event_key,
                    outcome=ReviewDecisionOutcome.IDEMPOTENT,
                    decision=decision,
                    status=str(current["status"]),
                    content_hash=content_hash,
                    actor=clean_actor,
                    reason=clean_reason,
                    decided_at=existing.get("decided_at"),
                )
            return ReviewDecisionResult(
                event_key=event_key,
                outcome=ReviewDecisionOutcome.ALREADY_DECIDED,
            )

        if current.get("status") != ConnectorEventStatus.NEEDS_INPUT_REVIEW.value:
            return ReviewDecisionResult(
                event_key=event_key,
                outcome=ReviewDecisionOutcome.NOT_REVIEWABLE,
            )

        review = current.get("preflight_review") or {}
        if review.get("content_hash") != content_hash:
            return ReviewDecisionResult(
                event_key=event_key,
                outcome=ReviewDecisionOutcome.STALE_HASH,
            )
        if (
            decision is ReviewInputDecision.APPROVED
            and (
                review.get("decision") != "ready_for_review"
                or review.get("nia_payload_preview") is None
            )
        ):
            return ReviewDecisionResult(
                event_key=event_key,
                outcome=ReviewDecisionOutcome.PREFLIGHT_BLOCKED,
            )

        return ReviewDecisionResult(
            event_key=event_key,
            outcome=ReviewDecisionOutcome.NOT_REVIEWABLE,
        )

    async def _decide_output(
        self,
        event_key: str,
        *,
        decision: ReviewInputDecision,
        target_status: ConnectorEventStatus,
        content_hash: str,
        actor: str,
        reason: Optional[str],
        now: Optional[datetime],
    ) -> ReviewDecisionResult:
        clean_actor = _required_text(actor, "actor")
        clean_reason = (reason or "").strip() or None
        decided_at = _normalized_time(now)
        decision_document = {
            "decision": decision.value,
            "content_hash": content_hash,
            "actor": clean_actor,
            "reason": clean_reason,
            "decided_at": decided_at,
        }
        selector: dict[str, Any] = {
            "event_key": event_key,
            "status": ConnectorEventStatus.NEEDS_OUTPUT_REVIEW.value,
            "output_review.content_hash": content_hash,
            "output_decision": None,
        }
        if decision is ReviewInputDecision.APPROVED:
            selector["output_review.decision"] = "ready_for_review"

        projection = {
            "_id": 0,
            "event_key": 1,
            "status": 1,
            "output_review": 1,
            "output_decision": 1,
        }
        updated = await self._collection.find_one_and_update(
            selector,
            {
                "$set": {
                    "status": target_status.value,
                    "updated_at": decided_at,
                    "output_decision": decision_document,
                },
            },
            projection=projection,
            return_document=ReturnDocument.AFTER,
        )
        if updated is not None:
            return ReviewDecisionResult(
                event_key=event_key,
                outcome=ReviewDecisionOutcome.APPLIED,
                decision=decision,
                status=target_status.value,
                content_hash=content_hash,
                actor=clean_actor,
                reason=clean_reason,
                decided_at=decided_at,
            )

        current = await self._collection.find_one(
            {"event_key": event_key},
            projection,
        )
        if current is None:
            return ReviewDecisionResult(
                event_key=event_key,
                outcome=ReviewDecisionOutcome.NOT_FOUND,
            )

        existing = current.get("output_decision")
        if existing is not None:
            exact_repeat = (
                existing.get("decision") == decision.value
                and existing.get("content_hash") == content_hash
                and existing.get("actor") == clean_actor
                and existing.get("reason") == clean_reason
            )
            if exact_repeat:
                return ReviewDecisionResult(
                    event_key=event_key,
                    outcome=ReviewDecisionOutcome.IDEMPOTENT,
                    decision=decision,
                    status=str(current["status"]),
                    content_hash=content_hash,
                    actor=clean_actor,
                    reason=clean_reason,
                    decided_at=existing.get("decided_at"),
                )
            return ReviewDecisionResult(
                event_key=event_key,
                outcome=ReviewDecisionOutcome.ALREADY_DECIDED,
            )

        if current.get("status") != ConnectorEventStatus.NEEDS_OUTPUT_REVIEW.value:
            return ReviewDecisionResult(
                event_key=event_key,
                outcome=ReviewDecisionOutcome.NOT_REVIEWABLE,
            )

        review = current.get("output_review") or {}
        if review.get("content_hash") != content_hash:
            return ReviewDecisionResult(
                event_key=event_key,
                outcome=ReviewDecisionOutcome.STALE_HASH,
            )
        if (
            decision is ReviewInputDecision.APPROVED
            and review.get("decision") != "ready_for_review"
        ):
            return ReviewDecisionResult(
                event_key=event_key,
                outcome=ReviewDecisionOutcome.OUTPUT_BLOCKED,
            )

        return ReviewDecisionResult(
            event_key=event_key,
            outcome=ReviewDecisionOutcome.NOT_REVIEWABLE,
        )

    @staticmethod
    def _owned_active_claim(
        event_key: str,
        lease_owner: str,
        now: datetime,
    ) -> dict[str, Any]:
        return {
            "event_key": event_key,
            "status": ConnectorEventStatus.PROCESSING.value,
            "lease_owner": lease_owner,
            "lease_until": {"$gt": now},
        }

    @staticmethod
    def _owned_nia_claim(
        event_key: str,
        lease_owner: str,
        now: datetime,
    ) -> dict[str, Any]:
        return {
            "event_key": event_key,
            "status": ConnectorEventStatus.PROCESSING.value,
            "processing_stage": "nia_dispatch",
            "lease_owner": _required_text(lease_owner, "lease_owner"),
            "lease_until": {"$gt": now},
        }

    @staticmethod
    def _owned_bitrix_claim(
        event_key: str,
        lease_owner: str,
        now: datetime,
    ) -> dict[str, Any]:
        return {
            "event_key": event_key,
            "status": ConnectorEventStatus.PROCESSING.value,
            "processing_stage": "bitrix_dispatch",
            "lease_owner": _required_text(lease_owner, "lease_owner"),
            "lease_until": {"$gt": now},
        }

    async def _finish_bitrix_claim(
        self,
        event_key: str,
        lease_owner: str,
        *,
        now: datetime,
        status: ConnectorEventStatus,
        lease_until: Optional[datetime],
        last_error: str,
    ) -> bool:
        result = await self._collection.update_one(
            {
                **self._owned_bitrix_claim(event_key, lease_owner, now),
                "outbound_message_id": None,
            },
            {
                "$set": {
                    "status": status.value,
                    "updated_at": now,
                    "lease_owner": None,
                    "lease_until": lease_until,
                    "processing_stage": "bitrix_dispatch",
                    "last_error": last_error,
                },
            },
            upsert=False,
        )
        return result.modified_count == 1

    async def _finish_nia_claim(
        self,
        event_key: str,
        lease_owner: str,
        *,
        now: datetime,
        status: ConnectorEventStatus,
        lease_until: Optional[datetime],
        last_error: str,
    ) -> bool:
        result = await self._collection.update_one(
            self._owned_nia_claim(event_key, lease_owner, now),
            {
                "$set": {
                    "status": status.value,
                    "updated_at": now,
                    "lease_owner": None,
                    "lease_until": lease_until,
                    "processing_stage": "nia_dispatch",
                    "last_error": last_error,
                },
            },
            upsert=False,
        )
        return result.modified_count == 1

    async def _finish_claim(
        self,
        event_key: str,
        lease_owner: str,
        *,
        now: datetime,
        status: ConnectorEventStatus,
        lease_until: Optional[datetime],
        last_error: Optional[str],
    ) -> bool:
        owner = _required_text(lease_owner, "lease_owner")
        result = await self._collection.update_one(
            self._owned_active_claim(event_key, owner, now),
            {
                "$set": {
                    "status": status.value,
                    "updated_at": now,
                    "lease_owner": None,
                    "lease_until": lease_until,
                    "last_error": last_error,
                },
            },
            upsert=False,
        )
        return result.modified_count == 1


class ConnectorStorageConfigurationError(ValueError):
    """La persistencia fue habilitada sin configuración completa."""


@dataclass
class ConnectorStorageResources:
    """Agrupa recursos propios para permitir un cierre explícito."""

    client: Any
    store: MongoConnectorEventStore

    def close(self) -> None:
        self.client.close()


class ConnectorStorageFactory:
    """Construye almacenamiento solo cuando el modo esté realmente habilitado."""

    def __init__(self, client_factory: Any = AsyncIOMotorClient) -> None:
        self._client_factory = client_factory

    async def build(
        self,
        settings: ConnectorSettings,
    ) -> Optional[ConnectorStorageResources]:
        if settings.activation_locked or settings.effective_mode is ConnectorMode.OFF:
            return None

        if not settings.mongo_uri:
            raise ConnectorStorageConfigurationError("mongo_uri_not_configured")
        if not settings.mongo_db:
            raise ConnectorStorageConfigurationError("mongo_db_not_configured")
        if not settings.events_collection:
            raise ConnectorStorageConfigurationError("events_collection_not_configured")

        client = self._client_factory(settings.mongo_uri)
        try:
            database = client[settings.mongo_db]
            store = MongoConnectorEventStore.from_database(
                database,
                settings.events_collection,
            )
            await store.ensure_indexes()
        except Exception:
            client.close()
            raise

        return ConnectorStorageResources(client=client, store=store)

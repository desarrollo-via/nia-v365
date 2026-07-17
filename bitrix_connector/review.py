"""Contratos y lectura segura de la bandeja preflight."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional, Protocol

from pydantic import BaseModel, Field, field_validator

from .output_review import OutputReview
from .preflight import TextPreflightReview


class ReviewStore(Protocol):
    async def list_reviews(
        self,
        *,
        decision: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]: ...

    async def get_review(self, event_key: str) -> Optional[dict]: ...

    async def approve_input(
        self,
        event_key: str,
        *,
        content_hash: str,
        actor: str,
        reason: Optional[str] = None,
    ) -> "ReviewDecisionResult": ...

    async def reject_input(
        self,
        event_key: str,
        *,
        content_hash: str,
        actor: str,
        reason: Optional[str] = None,
    ) -> "ReviewDecisionResult": ...

    async def list_output_reviews(
        self,
        *,
        decision: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]: ...

    async def get_output_review(
        self,
        event_key: str,
    ) -> Optional[dict]: ...

    async def approve_output(
        self,
        event_key: str,
        *,
        content_hash: str,
        actor: str,
        reason: Optional[str] = None,
    ) -> "ReviewDecisionResult": ...

    async def reject_output(
        self,
        event_key: str,
        *,
        content_hash: str,
        actor: str,
        reason: Optional[str] = None,
    ) -> "ReviewDecisionResult": ...


class ReviewListItem(BaseModel):
    event_key: str
    content_hash: str
    status: str
    received_at: datetime
    updated_at: datetime
    decision: str
    message_id: int
    chat_id: int
    dialog_id: str
    normalized_text_length: int
    signals: list[str]
    blocking_reasons: list[str]
    attachment_count: int


class ReviewListResponse(BaseModel):
    items: list[ReviewListItem]
    returned: int


class ReviewDetailResponse(BaseModel):
    event_key: str
    status: str
    received_at: datetime
    updated_at: datetime
    preflight_review: TextPreflightReview


class OutputReviewListItem(BaseModel):
    event_key: str
    content_hash: str
    status: str
    received_at: datetime
    updated_at: datetime
    decision: str
    bot_id: int
    dialog_id: str
    message_length: int
    nia_etapa: Optional[str] = None
    blocking_reasons: list[str]


class OutputReviewListResponse(BaseModel):
    items: list[OutputReviewListItem]
    returned: int


class OutputReviewDetailResponse(BaseModel):
    event_key: str
    status: str
    received_at: datetime
    updated_at: datetime
    output_review: OutputReview


class ReviewInputDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


class ReviewDecisionOutcome(str, Enum):
    APPLIED = "applied"
    IDEMPOTENT = "idempotent"
    STALE_HASH = "stale_hash"
    PREFLIGHT_BLOCKED = "preflight_blocked"
    OUTPUT_BLOCKED = "output_blocked"
    ALREADY_DECIDED = "already_decided"
    NOT_REVIEWABLE = "not_reviewable"
    NOT_FOUND = "not_found"


class ReviewDecisionRequest(BaseModel):
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    actor: str = Field(min_length=1, max_length=100)
    reason: Optional[str] = Field(default=None, max_length=500)

    @field_validator("actor")
    @classmethod
    def clean_actor(cls, value: str) -> str:
        clean = value.strip()
        if not clean:
            raise ValueError("actor no puede estar vacío")
        return clean

    @field_validator("reason")
    @classmethod
    def clean_reason(cls, value: Optional[str]) -> Optional[str]:
        clean = (value or "").strip()
        return clean or None


class ReviewDecisionResult(BaseModel):
    event_key: str
    outcome: ReviewDecisionOutcome
    decision: Optional[ReviewInputDecision] = None
    status: Optional[str] = None
    content_hash: Optional[str] = None
    actor: Optional[str] = None
    reason: Optional[str] = None
    decided_at: Optional[datetime] = None


class ReviewDecisionResponse(BaseModel):
    event_key: str
    decision: ReviewInputDecision
    status: str
    content_hash: str
    actor: str
    reason: Optional[str] = None
    decided_at: datetime
    idempotent: bool


class ConnectorReviewService:
    def __init__(self, store: ReviewStore) -> None:
        self._store = store

    async def list_reviews(
        self,
        *,
        decision: Optional[str] = None,
        limit: int = 50,
    ) -> ReviewListResponse:
        documents = await self._store.list_reviews(
            decision=decision,
            limit=limit,
        )
        items: list[ReviewListItem] = []
        for document in documents:
            review = TextPreflightReview.model_validate(
                document["preflight_review"]
            )
            items.append(
                ReviewListItem(
                    event_key=document["event_key"],
                    content_hash=review.content_hash,
                    status=str(document["status"]),
                    received_at=document["received_at"],
                    updated_at=document["updated_at"],
                    decision=review.decision,
                    message_id=review.normalized_input.message_id,
                    chat_id=review.normalized_input.chat_id,
                    dialog_id=review.normalized_input.dialog_id,
                    normalized_text_length=(
                        review.normalized_input.normalized_text_length
                    ),
                    signals=list(review.preflight_manifest.signals),
                    blocking_reasons=list(
                        review.preflight_manifest.blocking_reasons
                    ),
                    attachment_count=review.preflight_manifest.attachment_count,
                )
            )
        return ReviewListResponse(items=items, returned=len(items))

    async def get_review(self, event_key: str) -> Optional[ReviewDetailResponse]:
        document = await self._store.get_review(event_key)
        if document is None:
            return None
        return ReviewDetailResponse(
            event_key=document["event_key"],
            status=str(document["status"]),
            received_at=document["received_at"],
            updated_at=document["updated_at"],
            preflight_review=TextPreflightReview.model_validate(
                document["preflight_review"]
            ),
        )

    async def approve_input(
        self,
        event_key: str,
        request: ReviewDecisionRequest,
    ) -> ReviewDecisionResult:
        return await self._store.approve_input(
            event_key,
            content_hash=request.content_hash,
            actor=request.actor,
            reason=request.reason,
        )

    async def reject_input(
        self,
        event_key: str,
        request: ReviewDecisionRequest,
    ) -> ReviewDecisionResult:
        return await self._store.reject_input(
            event_key,
            content_hash=request.content_hash,
            actor=request.actor,
            reason=request.reason,
        )

    async def list_output_reviews(
        self,
        *,
        decision: Optional[str] = None,
        limit: int = 50,
    ) -> OutputReviewListResponse:
        documents = await self._store.list_output_reviews(
            decision=decision,
            limit=limit,
        )
        items: list[OutputReviewListItem] = []
        for document in documents:
            review = OutputReview.model_validate(document["output_review"])
            payload = review.bitrix_payload_preview
            items.append(
                OutputReviewListItem(
                    event_key=document["event_key"],
                    content_hash=review.content_hash,
                    status=str(document["status"]),
                    received_at=document["received_at"],
                    updated_at=document["updated_at"],
                    decision=review.decision,
                    bot_id=payload.botId,
                    dialog_id=payload.dialogId,
                    message_length=len(payload.fields.message),
                    nia_etapa=review.nia_response.etapa,
                    blocking_reasons=list(review.blocking_reasons),
                )
            )
        return OutputReviewListResponse(items=items, returned=len(items))

    async def get_output_review(
        self,
        event_key: str,
    ) -> Optional[OutputReviewDetailResponse]:
        document = await self._store.get_output_review(event_key)
        if document is None:
            return None
        return OutputReviewDetailResponse(
            event_key=document["event_key"],
            status=str(document["status"]),
            received_at=document["received_at"],
            updated_at=document["updated_at"],
            output_review=OutputReview.model_validate(
                document["output_review"]
            ),
        )

    async def approve_output(
        self,
        event_key: str,
        request: ReviewDecisionRequest,
    ) -> ReviewDecisionResult:
        return await self._store.approve_output(
            event_key,
            content_hash=request.content_hash,
            actor=request.actor,
            reason=request.reason,
        )

    async def reject_output(
        self,
        event_key: str,
        request: ReviewDecisionRequest,
    ) -> ReviewDecisionResult:
        return await self._store.reject_output(
            event_key,
            content_hash=request.content_hash,
            actor=request.actor,
            reason=request.reason,
        )

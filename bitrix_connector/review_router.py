"""Rutas administrativas protegidas para consulta y decisiones de revisión."""

from __future__ import annotations

from typing import Callable, Literal, Optional, Protocol

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from .config import ConnectorSettings, load_settings
from .review import (
    ReviewDecisionOutcome,
    ReviewDecisionRequest,
    ReviewDecisionResponse,
    ReviewDecisionResult,
    ReviewDetailResponse,
    ReviewListResponse,
    OutputReviewDetailResponse,
    OutputReviewListResponse,
)
from .runtime import ConnectorRuntimeUnavailable
from .security import validate_review_access


class ReviewReader(Protocol):
    async def list_reviews(
        self,
        *,
        decision: Optional[str] = None,
        limit: int = 50,
    ) -> ReviewListResponse: ...

    async def get_review(
        self,
        event_key: str,
    ) -> Optional[ReviewDetailResponse]: ...

    async def approve_input(
        self,
        event_key: str,
        request: ReviewDecisionRequest,
    ) -> ReviewDecisionResult: ...

    async def reject_input(
        self,
        event_key: str,
        request: ReviewDecisionRequest,
    ) -> ReviewDecisionResult: ...

    async def list_output_reviews(
        self,
        *,
        decision: Optional[str] = None,
        limit: int = 50,
    ) -> OutputReviewListResponse: ...

    async def get_output_review(
        self,
        event_key: str,
    ) -> Optional[OutputReviewDetailResponse]: ...

    async def approve_output(
        self,
        event_key: str,
        request: ReviewDecisionRequest,
    ) -> ReviewDecisionResult: ...

    async def reject_output(
        self,
        event_key: str,
        request: ReviewDecisionRequest,
    ) -> ReviewDecisionResult: ...


def create_review_router(
    reader: ReviewReader,
    *,
    settings_loader: Callable[[], ConnectorSettings] = load_settings,
) -> APIRouter:
    router = APIRouter(prefix="/review", tags=["Bitrix Connector Review"])

    def authorize(authorization: str = Header(default="")) -> None:
        decision = validate_review_access(authorization, settings_loader())
        if decision.accepted:
            return
        if decision.reason == "review_token_not_configured":
            raise HTTPException(
                status_code=503,
                detail="review_token_not_configured",
            )
        raise HTTPException(
            status_code=401,
            detail="review_unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )

    def validate_event_key(event_key: str) -> None:
        if len(event_key) != 64 or any(
            character not in "0123456789abcdef" for character in event_key
        ):
            raise HTTPException(status_code=404, detail="review_event_not_found")

    def decision_response(
        result: ReviewDecisionResult,
    ) -> ReviewDecisionResponse:
        if result.outcome in {
            ReviewDecisionOutcome.APPLIED,
            ReviewDecisionOutcome.IDEMPOTENT,
        }:
            return ReviewDecisionResponse(
                event_key=result.event_key,
                decision=result.decision,
                status=result.status,
                content_hash=result.content_hash,
                actor=result.actor,
                reason=result.reason,
                decided_at=result.decided_at,
                idempotent=(
                    result.outcome is ReviewDecisionOutcome.IDEMPOTENT
                ),
            )

        conflicts = {
            ReviewDecisionOutcome.STALE_HASH: "review_hash_mismatch",
            ReviewDecisionOutcome.PREFLIGHT_BLOCKED: "preflight_blocked",
            ReviewDecisionOutcome.OUTPUT_BLOCKED: "output_blocked",
            ReviewDecisionOutcome.ALREADY_DECIDED: "review_already_decided",
        }
        if result.outcome in conflicts:
            raise HTTPException(
                status_code=409,
                detail=conflicts[result.outcome],
            )
        raise HTTPException(status_code=404, detail="review_event_not_found")

    @router.get("", response_model=ReviewListResponse)
    async def list_review_items(
        decision: Optional[Literal["ready_for_review", "blocked"]] = None,
        limit: int = Query(default=50, ge=1, le=100),
        _authorized: None = Depends(authorize),
    ) -> ReviewListResponse:
        try:
            return await reader.list_reviews(
                decision=decision,
                limit=limit,
            )
        except ConnectorRuntimeUnavailable as exc:
            raise HTTPException(
                status_code=503,
                detail="connector_runtime_not_ready",
            ) from exc

    @router.get("/output", response_model=OutputReviewListResponse)
    async def list_output_review_items(
        decision: Optional[Literal["ready_for_review", "blocked"]] = None,
        limit: int = Query(default=50, ge=1, le=100),
        _authorized: None = Depends(authorize),
    ) -> OutputReviewListResponse:
        try:
            return await reader.list_output_reviews(
                decision=decision,
                limit=limit,
            )
        except ConnectorRuntimeUnavailable as exc:
            raise HTTPException(
                status_code=503,
                detail="connector_runtime_not_ready",
            ) from exc

    @router.get(
        "/output/{event_key}",
        response_model=OutputReviewDetailResponse,
    )
    async def get_output_review_item(
        event_key: str,
        _authorized: None = Depends(authorize),
    ) -> OutputReviewDetailResponse:
        validate_event_key(event_key)
        try:
            item = await reader.get_output_review(event_key)
        except ConnectorRuntimeUnavailable as exc:
            raise HTTPException(
                status_code=503,
                detail="connector_runtime_not_ready",
            ) from exc
        if item is None:
            raise HTTPException(status_code=404, detail="review_event_not_found")
        return item

    @router.get("/{event_key}", response_model=ReviewDetailResponse)
    async def get_review_item(
        event_key: str,
        _authorized: None = Depends(authorize),
    ) -> ReviewDetailResponse:
        validate_event_key(event_key)
        try:
            item = await reader.get_review(event_key)
        except ConnectorRuntimeUnavailable as exc:
            raise HTTPException(
                status_code=503,
                detail="connector_runtime_not_ready",
            ) from exc
        if item is None:
            raise HTTPException(status_code=404, detail="review_event_not_found")
        return item

    @router.post(
        "/{event_key}/approve-input",
        response_model=ReviewDecisionResponse,
    )
    async def approve_review_input(
        event_key: str,
        request: ReviewDecisionRequest,
        _authorized: None = Depends(authorize),
    ) -> ReviewDecisionResponse:
        validate_event_key(event_key)
        try:
            result = await reader.approve_input(event_key, request)
        except ConnectorRuntimeUnavailable as exc:
            raise HTTPException(
                status_code=503,
                detail="connector_runtime_not_ready",
            ) from exc
        return decision_response(result)

    @router.post(
        "/{event_key}/approve-output",
        response_model=ReviewDecisionResponse,
    )
    async def approve_review_output(
        event_key: str,
        request: ReviewDecisionRequest,
        _authorized: None = Depends(authorize),
    ) -> ReviewDecisionResponse:
        validate_event_key(event_key)
        try:
            result = await reader.approve_output(event_key, request)
        except ConnectorRuntimeUnavailable as exc:
            raise HTTPException(
                status_code=503,
                detail="connector_runtime_not_ready",
            ) from exc
        return decision_response(result)

    @router.post(
        "/{event_key}/reject-output",
        response_model=ReviewDecisionResponse,
    )
    async def reject_review_output(
        event_key: str,
        request: ReviewDecisionRequest,
        _authorized: None = Depends(authorize),
    ) -> ReviewDecisionResponse:
        validate_event_key(event_key)
        try:
            result = await reader.reject_output(event_key, request)
        except ConnectorRuntimeUnavailable as exc:
            raise HTTPException(
                status_code=503,
                detail="connector_runtime_not_ready",
            ) from exc
        return decision_response(result)

    @router.post(
        "/{event_key}/reject-input",
        response_model=ReviewDecisionResponse,
    )
    async def reject_review_input(
        event_key: str,
        request: ReviewDecisionRequest,
        _authorized: None = Depends(authorize),
    ) -> ReviewDecisionResponse:
        validate_event_key(event_key)
        try:
            result = await reader.reject_input(event_key, request)
        except ConnectorRuntimeUnavailable as exc:
            raise HTTPException(
                status_code=503,
                detail="connector_runtime_not_ready",
            ) from exc
        return decision_response(result)

    return router

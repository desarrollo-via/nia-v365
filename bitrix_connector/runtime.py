"""Ciclo de vida aislado del conector, todavía desconectado de FastAPI."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional

from .config import ConnectorSettings
from .models import ConnectorIngestionResult
from .pilot_scope import PilotScopePolicy
from .review import (
    ConnectorReviewService,
    ReviewDecisionRequest,
    ReviewDecisionResult,
    ReviewDetailResponse,
    ReviewListResponse,
    OutputReviewDetailResponse,
    OutputReviewListResponse,
)
from .service import ConnectorIngestionService
from .storage import ConnectorStorageFactory, ConnectorStorageResources


class ConnectorRuntimeState(str, Enum):
    STOPPED = "stopped"
    INERT = "inert"
    READY = "ready"


class ConnectorRuntimeUnavailable(RuntimeError):
    """El runtime no dispone de un servicio de ingreso activo."""


@dataclass(frozen=True)
class ConnectorRuntimeSnapshot:
    state: ConnectorRuntimeState
    service_available: bool
    resources_available: bool


class ConnectorRuntime:
    """Administra recursos y servicio sin acoplarlos al router o a NIA."""

    def __init__(self, storage_factory: Optional[ConnectorStorageFactory] = None) -> None:
        self._storage_factory = storage_factory or ConnectorStorageFactory()
        self._lock = asyncio.Lock()
        self._state = ConnectorRuntimeState.STOPPED
        self._resources: Optional[ConnectorStorageResources] = None
        self._service: Optional[ConnectorIngestionService] = None
        self._review_service: Optional[ConnectorReviewService] = None

    @property
    def snapshot(self) -> ConnectorRuntimeSnapshot:
        return ConnectorRuntimeSnapshot(
            state=self._state,
            service_available=self._service is not None,
            resources_available=self._resources is not None,
        )

    async def start(self, settings: ConnectorSettings) -> ConnectorRuntimeSnapshot:
        async with self._lock:
            if self._state is not ConnectorRuntimeState.STOPPED:
                return self.snapshot

            if not settings.pilot_configuration_valid:
                self._state = ConnectorRuntimeState.INERT
                return self.snapshot

            pilot_policy = PilotScopePolicy(
                enabled=settings.pilot_enabled,
                rules=settings.pilot_rules,
                emergency_stop=settings.pilot_emergency_stop,
            )
            resources = await self._storage_factory.build(settings)
            if resources is None:
                self._state = ConnectorRuntimeState.INERT
                return self.snapshot

            self._resources = resources
            self._service = ConnectorIngestionService(
                resources.store,
                pilot_policy,
            )
            self._review_service = ConnectorReviewService(resources.store)
            self._state = ConnectorRuntimeState.READY
            return self.snapshot

    async def ingest(
        self,
        raw_form: Mapping[str, Any],
        settings: ConnectorSettings,
    ) -> ConnectorIngestionResult:
        async with self._lock:
            if self._service is None:
                raise ConnectorRuntimeUnavailable("connector_runtime_not_ready")
            return await self._service.ingest(raw_form, settings)

    async def list_reviews(
        self,
        *,
        decision: Optional[str] = None,
        limit: int = 50,
    ) -> ReviewListResponse:
        async with self._lock:
            if self._review_service is None:
                raise ConnectorRuntimeUnavailable("connector_runtime_not_ready")
            return await self._review_service.list_reviews(
                decision=decision,
                limit=limit,
            )

    async def get_review(
        self,
        event_key: str,
    ) -> Optional[ReviewDetailResponse]:
        async with self._lock:
            if self._review_service is None:
                raise ConnectorRuntimeUnavailable("connector_runtime_not_ready")
            return await self._review_service.get_review(event_key)

    async def approve_input(
        self,
        event_key: str,
        request: ReviewDecisionRequest,
    ) -> ReviewDecisionResult:
        async with self._lock:
            if self._review_service is None:
                raise ConnectorRuntimeUnavailable("connector_runtime_not_ready")
            return await self._review_service.approve_input(event_key, request)

    async def reject_input(
        self,
        event_key: str,
        request: ReviewDecisionRequest,
    ) -> ReviewDecisionResult:
        async with self._lock:
            if self._review_service is None:
                raise ConnectorRuntimeUnavailable("connector_runtime_not_ready")
            return await self._review_service.reject_input(event_key, request)

    async def list_output_reviews(
        self,
        *,
        decision: Optional[str] = None,
        limit: int = 50,
    ) -> OutputReviewListResponse:
        async with self._lock:
            if self._review_service is None:
                raise ConnectorRuntimeUnavailable("connector_runtime_not_ready")
            return await self._review_service.list_output_reviews(
                decision=decision,
                limit=limit,
            )

    async def get_output_review(
        self,
        event_key: str,
    ) -> Optional[OutputReviewDetailResponse]:
        async with self._lock:
            if self._review_service is None:
                raise ConnectorRuntimeUnavailable("connector_runtime_not_ready")
            return await self._review_service.get_output_review(event_key)

    async def approve_output(
        self,
        event_key: str,
        request: ReviewDecisionRequest,
    ) -> ReviewDecisionResult:
        async with self._lock:
            if self._review_service is None:
                raise ConnectorRuntimeUnavailable("connector_runtime_not_ready")
            return await self._review_service.approve_output(event_key, request)

    async def reject_output(
        self,
        event_key: str,
        request: ReviewDecisionRequest,
    ) -> ReviewDecisionResult:
        async with self._lock:
            if self._review_service is None:
                raise ConnectorRuntimeUnavailable("connector_runtime_not_ready")
            return await self._review_service.reject_output(event_key, request)

    async def close(self) -> ConnectorRuntimeSnapshot:
        async with self._lock:
            resources = self._resources
            self._resources = None
            self._service = None
            self._review_service = None
            self._state = ConnectorRuntimeState.STOPPED
            if resources is not None:
                resources.close()
            return self.snapshot

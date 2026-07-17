"""Una iteración durable del trabajador, sin bucle ni llamadas externas."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Optional, Protocol

from .models import ConnectorEventRecord
from .storage import ConnectorEventStore


class ConnectorHandlerDecision(str, Enum):
    COMPLETE = "complete"
    RETRY = "retry"
    FAIL = "fail"
    APPLIED = "applied"
    LEASE_LOST = "lease_lost"


@dataclass(frozen=True)
class ConnectorHandlerResult:
    decision: ConnectorHandlerDecision
    error_code: Optional[str] = None
    retry_after_seconds: int = 0

    def __post_init__(self) -> None:
        try:
            decision = ConnectorHandlerDecision(self.decision)
        except ValueError as exc:
            raise ValueError("decisión de manejador no soportada") from exc
        object.__setattr__(self, "decision", decision)

        error_code = self.error_code.strip() if self.error_code else None
        object.__setattr__(self, "error_code", error_code)

        if decision in {
            ConnectorHandlerDecision.COMPLETE,
            ConnectorHandlerDecision.APPLIED,
            ConnectorHandlerDecision.LEASE_LOST,
        }:
            if error_code is not None or self.retry_after_seconds != 0:
                raise ValueError(
                    f"{decision.value} no admite error ni espera de reintento"
                )
            return

        if not error_code:
            raise ValueError("retry y fail requieren error_code")
        if self.retry_after_seconds < 0:
            raise ValueError("retry_after_seconds no puede ser negativo")
        if (
            decision is ConnectorHandlerDecision.FAIL
            and self.retry_after_seconds != 0
        ):
            raise ValueError("fail no admite espera de reintento")

    @classmethod
    def completed(cls) -> "ConnectorHandlerResult":
        return cls(decision=ConnectorHandlerDecision.COMPLETE)

    @classmethod
    def retryable(
        cls,
        error_code: str,
        *,
        retry_after_seconds: int,
    ) -> "ConnectorHandlerResult":
        return cls(
            decision=ConnectorHandlerDecision.RETRY,
            error_code=error_code,
            retry_after_seconds=retry_after_seconds,
        )

    @classmethod
    def failed(cls, error_code: str) -> "ConnectorHandlerResult":
        return cls(
            decision=ConnectorHandlerDecision.FAIL,
            error_code=error_code,
        )

    @classmethod
    def applied(cls) -> "ConnectorHandlerResult":
        return cls(decision=ConnectorHandlerDecision.APPLIED)

    @classmethod
    def lease_lost(cls) -> "ConnectorHandlerResult":
        return cls(decision=ConnectorHandlerDecision.LEASE_LOST)


class ConnectorEventHandler(Protocol):
    async def handle(self, event: ConnectorEventRecord) -> ConnectorHandlerResult: ...


class ConnectorWorkerRunStatus(str, Enum):
    IDLE = "idle"
    COMPLETED = "completed"
    RETRY_SCHEDULED = "retry_scheduled"
    FAILED = "failed"
    LEASE_LOST = "lease_lost"


@dataclass(frozen=True)
class ConnectorWorkerRunResult:
    status: ConnectorWorkerRunStatus
    event_key: Optional[str] = None
    attempt_count: int = 0
    outcome_applied: bool = False


class ConnectorWorker:
    """Coordina exactamente una reclamación y un resultado persistible."""

    def __init__(
        self,
        store: ConnectorEventStore,
        handler: ConnectorEventHandler,
        *,
        lease_owner: str,
        lease_seconds: int,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        owner = lease_owner.strip()
        if not owner:
            raise ValueError("lease_owner no puede estar vacío")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds debe ser positivo")

        self._store = store
        self._handler = handler
        self._lease_owner = owner
        self._lease_seconds = lease_seconds
        self._clock = clock

    async def run_once(self) -> ConnectorWorkerRunResult:
        claimed = await self._store.claim_next(
            lease_owner=self._lease_owner,
            lease_seconds=self._lease_seconds,
            now=self._clock(),
        )
        if claimed is None:
            return ConnectorWorkerRunResult(status=ConnectorWorkerRunStatus.IDLE)

        if claimed.lease_owner != self._lease_owner:
            return ConnectorWorkerRunResult(
                status=ConnectorWorkerRunStatus.LEASE_LOST,
                event_key=claimed.event_key,
                attempt_count=claimed.attempt_count,
            )

        outcome = await self._handler.handle(claimed)

        if outcome.decision is ConnectorHandlerDecision.APPLIED:
            return ConnectorWorkerRunResult(
                status=ConnectorWorkerRunStatus.COMPLETED,
                event_key=claimed.event_key,
                attempt_count=claimed.attempt_count,
                outcome_applied=True,
            )
        if outcome.decision is ConnectorHandlerDecision.LEASE_LOST:
            return ConnectorWorkerRunResult(
                status=ConnectorWorkerRunStatus.LEASE_LOST,
                event_key=claimed.event_key,
                attempt_count=claimed.attempt_count,
            )

        finished_at = self._clock()

        if outcome.decision is ConnectorHandlerDecision.COMPLETE:
            applied = await self._store.complete_claim(
                claimed.event_key,
                self._lease_owner,
                now=finished_at,
            )
            status = ConnectorWorkerRunStatus.COMPLETED
        elif outcome.decision is ConnectorHandlerDecision.RETRY:
            applied = await self._store.retry_claim(
                claimed.event_key,
                self._lease_owner,
                error_code=outcome.error_code or "retryable_handler_error",
                retry_after_seconds=outcome.retry_after_seconds,
                now=finished_at,
            )
            status = ConnectorWorkerRunStatus.RETRY_SCHEDULED
        else:
            applied = await self._store.fail_claim(
                claimed.event_key,
                self._lease_owner,
                error_code=outcome.error_code or "permanent_handler_error",
                now=finished_at,
            )
            status = ConnectorWorkerRunStatus.FAILED

        if not applied:
            status = ConnectorWorkerRunStatus.LEASE_LOST

        return ConnectorWorkerRunResult(
            status=status,
            event_key=claimed.event_key,
            attempt_count=claimed.attempt_count,
            outcome_applied=applied,
        )

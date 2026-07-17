"""Adaptador entre el preflight puro, el lease y el trabajador."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from .models import ConnectorEventRecord
from .preflight import build_text_preflight
from .storage import ConnectorEventStore
from .worker import ConnectorHandlerResult


class TextPreflightWorkerHandler:
    """Construye y persiste una vista preflight bajo el lease vigente."""

    def __init__(
        self,
        store: ConnectorEventStore,
        *,
        lease_owner: str,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        owner = lease_owner.strip()
        if not owner:
            raise ValueError("lease_owner no puede estar vacío")
        self._store = store
        self._lease_owner = owner
        self._clock = clock

    async def handle(self, event: ConnectorEventRecord) -> ConnectorHandlerResult:
        review = build_text_preflight(event)
        applied = await self._store.save_preflight(
            event.event_key,
            self._lease_owner,
            review,
            now=self._clock(),
        )
        if not applied:
            return ConnectorHandlerResult.lease_lost()
        return ConnectorHandlerResult.applied()

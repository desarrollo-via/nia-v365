"""Orquestación de ingreso desacoplada de FastAPI, Motor y NIA."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from pydantic import ValidationError

from .config import ConnectorSettings
from .event_parser import parse_webhook_form
from .idempotency import build_event_key
from .models import ConnectorIngestionResult, ConnectorIngestionStatus
from .pilot_scope import PilotScopePolicy
from .security import validate_webhook_identity
from .storage import ConnectorEventStore, build_received_record


SUPPORTED_EVENTS = frozenset({"ONIMBOTV2MESSAGEADD"})


class ConnectorPersistenceError(RuntimeError):
    """El repositorio no pudo confirmar la recepción del evento."""


class ConnectorIngestionService:
    """Valida y persiste recepciones mediante un repositorio inyectado."""

    def __init__(
        self,
        store: ConnectorEventStore,
        pilot_policy: Optional[PilotScopePolicy] = None,
    ) -> None:
        self._store = store
        self._pilot_policy = pilot_policy

    async def ingest(
        self,
        raw_form: Mapping[str, Any],
        settings: ConnectorSettings,
    ) -> ConnectorIngestionResult:
        try:
            event = parse_webhook_form(raw_form)
        except (TypeError, ValueError, ValidationError):
            return ConnectorIngestionResult(
                status=ConnectorIngestionStatus.INVALID,
                reason="invalid_webhook_payload",
            )

        event_key = build_event_key(event)
        security = validate_webhook_identity(event, settings)

        if event.event not in SUPPORTED_EVENTS:
            return ConnectorIngestionResult(
                status=ConnectorIngestionStatus.IGNORED,
                reason="unsupported_event",
                event_key=event_key,
                identity_verified=security.accepted,
            )

        if not security.accepted:
            return ConnectorIngestionResult(
                status=ConnectorIngestionStatus.IGNORED,
                reason=security.reason,
                event_key=event_key,
            )

        if self._pilot_policy is not None:
            pilot = self._pilot_policy.evaluate(event)
            if not pilot.accepted:
                return ConnectorIngestionResult(
                    status=ConnectorIngestionStatus.IGNORED,
                    reason=pilot.reason,
                    event_key=event_key,
                    identity_verified=True,
                )

        record = build_received_record(
            event,
            raw_form,
            identity_verified=True,
            security_reason=security.reason,
        )
        try:
            write = await self._store.save_received(record)
        except Exception as exc:
            raise ConnectorPersistenceError("connector_storage_unavailable") from exc

        if write.duplicate:
            return ConnectorIngestionResult(
                status=ConnectorIngestionStatus.DUPLICATE,
                reason="event_already_received",
                event_key=write.event_key,
                identity_verified=True,
                duplicate=True,
            )

        return ConnectorIngestionResult(
            status=ConnectorIngestionStatus.STORED,
            reason="event_stored",
            event_key=write.event_key,
            identity_verified=True,
            created=write.created,
        )

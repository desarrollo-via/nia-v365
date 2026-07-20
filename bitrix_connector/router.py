"""Superficie HTTP aislada y bloqueada en ``off``."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from . import CONNECTOR_VERSION
from .config import ConnectorMode, load_settings
from .event_parser import parse_webhook_form
from .idempotency import build_event_key
from .installation_router import create_installation_router
from .installation_status_router import create_installation_status_router
from .models import (
    ConnectorHealth,
    ConnectorIngestionStatus,
    WebhookEventSummary,
    WebhookReceipt,
)
from .runtime import ConnectorRuntime, ConnectorRuntimeUnavailable
from .review_router import create_review_router
from .security import redact_form_data, validate_webhook_identity
from .service import ConnectorPersistenceError


router = APIRouter(prefix="/bitrix-connector", tags=["Bitrix Connector"])
connector_runtime = ConnectorRuntime()
router.include_router(create_installation_router())
router.include_router(create_installation_status_router())
router.include_router(create_review_router(connector_runtime))


async def start_connector_runtime() -> None:
    await connector_runtime.start(load_settings())


async def stop_connector_runtime() -> None:
    await connector_runtime.close()


router.add_event_handler("startup", start_connector_runtime)
router.add_event_handler("shutdown", stop_connector_runtime)


@router.get("/health", response_model=ConnectorHealth)
async def connector_health() -> ConnectorHealth:
    settings = load_settings()
    runtime = connector_runtime.snapshot
    return ConnectorHealth(
        status="ok",
        module="bitrix_connector",
        version=CONNECTOR_VERSION,
        requested_mode=settings.requested_mode,
        effective_mode=settings.effective_mode.value,
        activation_locked=settings.activation_locked,
        external_calls_enabled=settings.external_calls_enabled,
        runtime_state=runtime.state.value,
        runtime_service_available=runtime.service_available,
        runtime_resources_available=runtime.resources_available,
        configured=settings.configured,
        pilot=settings.pilot_summary,
        warnings=list(settings.warnings),
    )


@router.post("/webhook", response_model=WebhookReceipt)
async def bitrix_webhook(request: Request):
    """
    Inspecciona un evento sin persistirlo ni ejecutar acciones externas.

    El endpoint existe para validar el contrato real de Bitrix mientras el
    conector sigue bloqueado en ``off``.
    """
    try:
        incoming_form = await request.form()
        flat_form = {str(key): value for key, value in incoming_form.multi_items()}
        redacted = redact_form_data(flat_form)
        event = parse_webhook_form(flat_form)
    except (TypeError, ValueError, ValidationError):
        return JSONResponse(
            status_code=400,
            content={
                "status": "invalid",
                "reason": "invalid_webhook_payload",
                "persisted": False,
                "nia_called": False,
                "bitrix_written": False,
            },
        )

    settings = load_settings()
    security = validate_webhook_identity(event, settings)
    supported_event = event.event == "ONIMBOTV2MESSAGEADD"
    event_key = build_event_key(event)
    duplicate_detection = "not_persisted"
    persisted = False
    http_status = 200

    if not supported_event:
        status = "ignored"
        reason = "unsupported_event"
    elif not security.accepted:
        status = "ignored"
        reason = security.reason
    elif settings.activation_locked or settings.effective_mode is ConnectorMode.OFF:
        status = "disabled"
        reason = "connector_locked_off"
    else:
        try:
            ingestion = await connector_runtime.ingest(flat_form, settings)
        except ConnectorRuntimeUnavailable:
            status = "disabled"
            reason = "connector_runtime_not_ready"
        except ConnectorPersistenceError:
            status = "retryable_error"
            reason = "connector_storage_unavailable"
            http_status = 503
        else:
            event_key = ingestion.event_key or event_key
            if ingestion.status is ConnectorIngestionStatus.STORED:
                status = "stored"
                reason = ingestion.reason
                duplicate_detection = "unique_created"
                persisted = True
            elif ingestion.status is ConnectorIngestionStatus.DUPLICATE:
                status = "duplicate"
                reason = ingestion.reason
                duplicate_detection = "duplicate"
                persisted = True
            else:
                status = ingestion.status.value
                reason = ingestion.reason

    receipt = WebhookReceipt(
        status=status,
        reason=reason,
        effective_mode=settings.effective_mode.value,
        event_key=event_key,
        identity_verified=security.accepted,
        redacted_secret_fields=sum(value == "[REDACTED]" for value in redacted.values()),
        duplicate_detection=duplicate_detection,
        persisted=persisted,
        nia_called=False,
        bitrix_written=False,
        event_summary=WebhookEventSummary(
            event=event.event,
            bot_id=event.bot_id,
            message_id=event.message_id,
            chat_id=event.chat_id,
            dialog_id=event.dialog_id,
            text_length=len(event.text),
            is_system=event.is_system,
        ),
    )
    if http_status == 503:
        return JSONResponse(
            status_code=http_status,
            headers={"Retry-After": "5"},
            content=receipt.model_dump(),
        )
    return receipt

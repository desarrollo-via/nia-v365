"""Contratos propios del conector; no dependen de modelos de NIA."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class NormalizedBitrixEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    event: str = Field(min_length=1)
    timestamp: Optional[int] = None
    bot_id: int = Field(gt=0)
    bot_code: str = ""
    message_id: int = Field(gt=0)
    message_uuid: str = ""
    chat_id: int = Field(gt=0)
    dialog_id: str = Field(min_length=1)
    author_id: int
    text: str = ""
    is_system: bool = False
    chat_type: str = ""
    entity_type: str = ""
    user_id: int
    user_is_bot: bool = False
    user_is_connector: bool = False
    domain: str = Field(min_length=1)
    member_id: str = Field(min_length=1)
    application_token: Optional[SecretStr] = Field(default=None, exclude=True)


class SecurityDecision(BaseModel):
    accepted: bool
    reason: str


class ConnectorHealth(BaseModel):
    status: str
    module: str
    version: str
    requested_mode: str
    effective_mode: str
    activation_locked: bool
    external_calls_enabled: bool
    runtime_state: str
    runtime_service_available: bool
    runtime_resources_available: bool
    configured: dict[str, bool]
    pilot: dict[str, Any]
    warnings: list[str]


class WebhookEventSummary(BaseModel):
    event: str
    bot_id: int
    message_id: int
    chat_id: int
    dialog_id: str
    text_length: int
    is_system: bool


class WebhookReceipt(BaseModel):
    status: str
    reason: str
    effective_mode: str
    event_key: str
    identity_verified: bool
    redacted_secret_fields: int
    duplicate_detection: str
    persisted: bool
    nia_called: bool
    bitrix_written: bool
    event_summary: WebhookEventSummary


class ConnectorEventStatus(str, Enum):
    """Estados persistibles del ciclo de vida propio del conector."""

    RECEIVED = "received"
    IGNORED = "ignored"
    PENDING = "pending"
    PROCESSING = "processing"
    PREFLIGHT = "preflight"
    NEEDS_INPUT_REVIEW = "needs_input_review"
    INPUT_REJECTED = "input_rejected"
    READY_FOR_NIA = "ready_for_nia"
    NEEDS_OUTPUT_REVIEW = "needs_output_review"
    OUTPUT_REJECTED = "output_rejected"
    READY_FOR_BITRIX = "ready_for_bitrix"
    SHADOWED = "shadowed"
    SENT = "sent"
    RETRYABLE_ERROR = "retryable_error"
    FAILED = "failed"


class ConnectorEventRecord(BaseModel):
    """Documento auditable almacenado exclusivamente por el conector."""

    schema_version: int = 1
    event_key: str = Field(pattern=r"^[a-f0-9]{64}$")
    status: ConnectorEventStatus = ConnectorEventStatus.RECEIVED
    received_at: datetime
    updated_at: datetime
    normalized_event: dict[str, Any]
    raw_redacted: dict[str, Any]
    identity_verified: bool
    security_reason: str
    attempt_count: int = Field(default=0, ge=0)
    nia_attempt_count: int = Field(default=0, ge=0)
    bitrix_attempt_count: int = Field(default=0, ge=0)
    processing_stage: Optional[str] = None
    lease_owner: Optional[str] = None
    lease_until: Optional[datetime] = None
    last_error: Optional[str] = None
    preflight_review: Optional[dict[str, Any]] = None
    input_decision: Optional[dict[str, Any]] = None
    nia_response: Optional[dict[str, Any]] = None
    nia_dispatch: Optional[dict[str, Any]] = None
    output_review: Optional[dict[str, Any]] = None
    output_decision: Optional[dict[str, Any]] = None
    outbound_message_id: Optional[int] = Field(default=None, gt=0)
    bitrix_dispatch: Optional[dict[str, Any]] = None


class StorageWriteResult(BaseModel):
    event_key: str
    created: bool
    duplicate: bool


class ConnectorIngestionStatus(str, Enum):
    STORED = "stored"
    DUPLICATE = "duplicate"
    IGNORED = "ignored"
    INVALID = "invalid"


class ConnectorIngestionResult(BaseModel):
    """Resultado interno sin contenido del cliente ni secretos."""

    status: ConnectorIngestionStatus
    reason: str
    event_key: Optional[str] = None
    identity_verified: bool = False
    created: bool = False
    duplicate: bool = False

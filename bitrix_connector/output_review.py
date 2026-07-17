"""Vista canónica y revisable de una futura salida hacia Bitrix."""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, Field

from .models import ConnectorEventRecord, NormalizedBitrixEvent
from .nia_client import NiaChatResponse


MAX_BITRIX_MESSAGE_CHARS = 20_000


class BitrixMessageFieldsPreview(BaseModel):
    message: str


class BitrixMessagePayloadPreview(BaseModel):
    botId: int = Field(gt=0)
    dialogId: str = Field(min_length=1)
    fields: BitrixMessageFieldsPreview


class OutputReview(BaseModel):
    schema_version: int = 1
    event_key: str
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    decision: str
    blocking_reasons: list[str]
    nia_response: NiaChatResponse
    bitrix_payload_preview: BitrixMessagePayloadPreview


def build_output_content_hash(
    payload: BitrixMessagePayloadPreview,
) -> str:
    serialized = json.dumps(
        {
            "kind": "bitrix_message",
            "payload": payload.model_dump(mode="json"),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def build_output_review(
    event: ConnectorEventRecord,
    response: NiaChatResponse,
) -> OutputReview:
    normalized = NormalizedBitrixEvent.model_validate(event.normalized_event)
    payload = BitrixMessagePayloadPreview(
        botId=normalized.bot_id,
        dialogId=normalized.dialog_id,
        fields=BitrixMessageFieldsPreview(message=response.respuesta),
    )
    blocking_reasons = []
    if not response.respuesta.strip():
        blocking_reasons.append("empty_nia_response")
    if len(response.respuesta) > MAX_BITRIX_MESSAGE_CHARS:
        blocking_reasons.append("bitrix_message_too_long")

    return OutputReview(
        event_key=event.event_key,
        content_hash=build_output_content_hash(payload),
        decision="blocked" if blocking_reasons else "ready_for_review",
        blocking_reasons=blocking_reasons,
        nia_response=response,
        bitrix_payload_preview=payload,
    )

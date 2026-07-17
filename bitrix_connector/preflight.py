"""Preparación pura y revisable de eventos de texto antes de NIA."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any, Optional

from pydantic import BaseModel, Field

from .models import ConnectorEventRecord, NormalizedBitrixEvent
from .security import redact_form_data


MAX_PREFLIGHT_TEXT_CHARS = 12_000
MAX_NIA_SESSION_ID_CHARS = 256

_URL_PATTERN = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
_EMAIL_PATTERN = re.compile(
    r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"
)
_PHONE_PATTERN = re.compile(r"(?<!\w)\+?\d[\d\s().-]{6,}\d(?!\w)")
_SECRET_PATTERN = re.compile(
    r"(?i)(?:\b(?:api[_ -]?key|token|password|secret)\b\s*[:=]\s*\S+"
    r"|\bbearer\s+[A-Za-z0-9._-]{8,}"
    r"|\bsk-[A-Za-z0-9_-]{8,})"
)


class NiaTextPayloadPreview(BaseModel):
    session_id: str
    mensaje: str


class PreflightNormalizedInput(BaseModel):
    event: str
    message_id: int
    chat_id: int
    dialog_id: str
    author_id: int
    member_id: str
    text: str
    original_text_length: int = Field(ge=0)
    normalized_text_length: int = Field(ge=0)


class PreflightManifest(BaseModel):
    transformations: list[str]
    signals: list[str]
    blocking_reasons: list[str]
    attachment_status: str
    attachment_count: int = Field(ge=0)


class TextPreflightReview(BaseModel):
    schema_version: int = 1
    event_key: str
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    decision: str
    raw_redacted: dict[str, Any]
    normalized_input: PreflightNormalizedInput
    preflight_manifest: PreflightManifest
    nia_payload_preview: Optional[NiaTextPayloadPreview] = None


def _canonical_hash(content: dict[str, Any]) -> str:
    serialized = json.dumps(
        content,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def build_preflight_content_hash(
    normalized_input: PreflightNormalizedInput,
    manifest: PreflightManifest,
    payload: Optional[NiaTextPayloadPreview],
) -> str:
    if payload is not None:
        return build_nia_payload_hash(payload)
    else:
        content = {
            "kind": "blocked_preflight",
            "normalized_input": normalized_input.model_dump(mode="json"),
            "blocking_reasons": list(manifest.blocking_reasons),
        }
    return _canonical_hash(content)


def build_nia_payload_hash(payload: NiaTextPayloadPreview) -> str:
    """Hash canónico recalculable justo antes del despacho aprobado."""
    return _canonical_hash(
        {
            "kind": "nia_payload",
            "payload": payload.model_dump(mode="json"),
        }
    )


def _normalize_text(text: str) -> tuple[str, list[str]]:
    transformations: list[str] = []
    had_outer_whitespace = text != text.strip()

    normalized = unicodedata.normalize("NFC", text)
    if normalized != text:
        transformations.append("unicode_nfc")

    line_normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    if line_normalized != normalized:
        transformations.append("line_endings_lf")
    normalized = line_normalized

    without_controls = "".join(
        character
        if character == "\n" or unicodedata.category(character) != "Cc"
        else " "
        for character in normalized
    )
    if without_controls != normalized:
        transformations.append("control_characters_removed")
    normalized = without_controls

    collapsed_lines = [
        re.sub(r"[^\S\n]+", " ", line).strip()
        for line in normalized.split("\n")
    ]
    collapsed = "\n".join(collapsed_lines)
    if collapsed != normalized:
        transformations.append("horizontal_whitespace_collapsed")
    normalized = collapsed

    compact_blank_lines = re.sub(r"\n{3,}", "\n\n", normalized)
    if compact_blank_lines != normalized:
        transformations.append("blank_lines_collapsed")
    normalized = compact_blank_lines

    stripped = normalized.strip()
    if stripped != normalized or had_outer_whitespace:
        transformations.append("outer_whitespace_trimmed")

    return stripped, transformations


def _signals(text: str) -> list[str]:
    signals: list[str] = []
    if _URL_PATTERN.search(text):
        signals.append("url_detected")
    if _EMAIL_PATTERN.search(text):
        signals.append("email_detected")
    if _PHONE_PATTERN.search(text):
        signals.append("phone_detected")
    if _SECRET_PATTERN.search(text):
        signals.append("possible_secret_detected")
    return signals


def _is_open_line(event: NormalizedBitrixEvent) -> bool:
    chat_type = event.chat_type.strip().lower()
    entity_type = event.entity_type.strip().upper()
    return chat_type in {"openchannel", "openline"} or entity_type == "LINES"


def build_text_preflight(record: ConnectorEventRecord) -> TextPreflightReview:
    """Construye vistas inmutables sin persistir ni ejecutar llamadas externas."""
    event = NormalizedBitrixEvent.model_validate(record.normalized_event)
    raw_redacted = redact_form_data(record.raw_redacted)
    original_text = str(
        raw_redacted.get("data[message][text]", event.text)
    )
    normalized_text, transformations = _normalize_text(original_text)
    signals = _signals(normalized_text)
    blocking_reasons: list[str] = []

    if event.is_system:
        blocking_reasons.append("system_message")
    if event.user_is_bot:
        blocking_reasons.append("bot_message")
    if not _is_open_line(event):
        blocking_reasons.append("unsupported_chat_context")
    if not normalized_text:
        blocking_reasons.append("empty_text")
    if len(normalized_text) > MAX_PREFLIGHT_TEXT_CHARS:
        blocking_reasons.append("text_too_long")
    if "possible_secret_detected" in signals:
        blocking_reasons.append("possible_secret_detected")

    session_id = f"bitrix:{event.member_id}:{event.dialog_id}"
    if len(session_id) > MAX_NIA_SESSION_ID_CHARS:
        blocking_reasons.append("session_id_too_long")

    payload = None
    if not blocking_reasons:
        payload = NiaTextPayloadPreview(
            session_id=session_id,
            mensaje=normalized_text,
        )

    normalized_input = PreflightNormalizedInput(
        event=event.event,
        message_id=event.message_id,
        chat_id=event.chat_id,
        dialog_id=event.dialog_id,
        author_id=event.author_id,
        member_id=event.member_id,
        text=normalized_text,
        original_text_length=len(original_text),
        normalized_text_length=len(normalized_text),
    )
    manifest = PreflightManifest(
        transformations=transformations,
        signals=signals,
        blocking_reasons=blocking_reasons,
        attachment_status="not_present",
        attachment_count=0,
    )

    return TextPreflightReview(
        event_key=record.event_key,
        content_hash=build_preflight_content_hash(
            normalized_input,
            manifest,
            payload,
        ),
        decision="blocked" if blocking_reasons else "ready_for_review",
        raw_redacted=raw_redacted,
        normalized_input=normalized_input,
        preflight_manifest=manifest,
        nia_payload_preview=payload,
    )

"""Normaliza el form-data plano enviado por los webhooks de Bitrix24."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .models import NormalizedBitrixEvent


def _value(form: Mapping[str, Any], key: str, default: Any = "") -> Any:
    value = form.get(key, default)
    if hasattr(value, "filename"):
        return default
    return value


def _to_int(value: Any, *, default: Optional[int] = None) -> Optional[int]:
    if value in (None, ""):
        return default
    try:
        return int(float(str(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Valor entero inválido: {value!r}") from exc


def _to_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "y", "yes"}


def parse_webhook_form(form: Mapping[str, Any]) -> NormalizedBitrixEvent:
    """Convierte claves PHP ``data[x][y]`` en un contrato tipado."""
    return NormalizedBitrixEvent(
        event=str(_value(form, "event")).strip().upper(),
        timestamp=_to_int(_value(form, "ts"), default=None),
        bot_id=_to_int(_value(form, "data[bot][id]"), default=0) or 0,
        bot_code=str(_value(form, "data[bot][code]")).strip(),
        message_id=_to_int(_value(form, "data[message][id]"), default=0) or 0,
        message_uuid=str(_value(form, "data[message][uuid]")).strip(),
        chat_id=_to_int(_value(form, "data[message][chatId]"), default=0) or 0,
        dialog_id=str(_value(form, "data[chat][dialogId]")).strip(),
        author_id=_to_int(_value(form, "data[message][authorId]"), default=0) or 0,
        text=str(_value(form, "data[message][text]")).strip(),
        is_system=_to_bool(_value(form, "data[message][isSystem]")),
        chat_type=str(_value(form, "data[chat][type]")).strip(),
        entity_type=str(_value(form, "data[chat][entityType]")).strip(),
        user_id=_to_int(_value(form, "data[user][id]"), default=0) or 0,
        user_is_bot=_to_bool(_value(form, "data[user][bot]")),
        user_is_connector=_to_bool(_value(form, "data[user][connector]")),
        domain=str(_value(form, "auth[domain]")).strip().lower(),
        member_id=str(_value(form, "auth[member_id]")).strip(),
        application_token=str(_value(form, "auth[application_token]")).strip() or None,
    )

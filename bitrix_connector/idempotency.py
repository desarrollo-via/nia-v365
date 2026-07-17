"""Identidad determinista; la detección persistida llegará en otra fase."""

import hashlib

from .models import NormalizedBitrixEvent


def build_event_key(event: NormalizedBitrixEvent) -> str:
    """Genera la misma clave para la misma instalación, bot y mensaje."""
    identity = f"{event.member_id}:{event.bot_id}:{event.message_id}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()

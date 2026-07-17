"""Validaciones de identidad y redacción propias del conector."""

from __future__ import annotations

import hmac
from typing import Any, Mapping

from .config import ConnectorSettings
from .models import NormalizedBitrixEvent, SecurityDecision


_SECRET_LEAVES = {
    "access_token",
    "refresh_token",
    "application_token",
    "bottoken",
}


def _leaf_name(key: str) -> str:
    return key.rsplit("[", 1)[-1].rstrip("]").lower()


def redact_form_data(form: Mapping[str, Any]) -> dict[str, Any]:
    """Copia form-data sustituyendo secretos conocidos por ``[REDACTED]``."""
    redacted: dict[str, Any] = {}
    for key, value in form.items():
        redacted[str(key)] = "[REDACTED]" if _leaf_name(str(key)) in _SECRET_LEAVES else value
    return redacted


def validate_webhook_identity(
    event: NormalizedBitrixEvent,
    settings: ConnectorSettings,
) -> SecurityDecision:
    """Compara portal, instalación y token sin activar ninguna acción externa."""
    if not settings.bitrix_domain or not settings.bitrix_member_id:
        return SecurityDecision(accepted=False, reason="installation_identity_not_configured")
    if not settings.bitrix_application_token:
        return SecurityDecision(accepted=False, reason="application_token_not_configured")

    expected_domain = settings.bitrix_domain.lower().removeprefix("https://").rstrip("/")
    received_domain = event.domain.lower().removeprefix("https://").rstrip("/")
    if not hmac.compare_digest(received_domain, expected_domain):
        return SecurityDecision(accepted=False, reason="domain_mismatch")
    if not hmac.compare_digest(event.member_id, settings.bitrix_member_id):
        return SecurityDecision(accepted=False, reason="member_id_mismatch")

    received_token = event.application_token.get_secret_value() if event.application_token else ""
    if not hmac.compare_digest(received_token, settings.bitrix_application_token):
        return SecurityDecision(accepted=False, reason="application_token_mismatch")

    return SecurityDecision(accepted=True, reason="identity_verified")


def validate_review_access(
    authorization: str,
    settings: ConnectorSettings,
) -> SecurityDecision:
    """Valida un Bearer administrativo sin revelar el token configurado."""
    if not settings.review_token:
        return SecurityDecision(accepted=False, reason="review_token_not_configured")

    scheme, separator, received_token = (authorization or "").partition(" ")
    if (
        not separator
        or scheme.lower() != "bearer"
        or not received_token.strip()
        or not hmac.compare_digest(received_token.strip(), settings.review_token)
    ):
        return SecurityDecision(accepted=False, reason="review_unauthorized")

    return SecurityDecision(accepted=True, reason="review_authorized")

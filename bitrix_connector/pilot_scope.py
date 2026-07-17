"""Compuerta pura para limitar pruebas a chats Bitrix explícitos."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import NormalizedBitrixEvent


def _normalized_time(value: datetime, field: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{field} debe incluir zona horaria")
    return value.astimezone(timezone.utc)


class PilotScopeRule(BaseModel):
    """Identidad exacta de un único chat autorizado para el piloto."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    member_id: str = Field(min_length=1)
    bot_id: int = Field(gt=0)
    dialog_id: str = Field(min_length=1)
    chat_id: int = Field(gt=0)
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None

    @model_validator(mode="after")
    def validate_window(self) -> "PilotScopeRule":
        member = self.member_id.strip()
        dialog = self.dialog_id.strip()
        if not member or not dialog:
            raise ValueError("la identidad piloto no admite textos vacíos")
        object.__setattr__(self, "member_id", member)
        object.__setattr__(self, "dialog_id", dialog)

        start = (
            _normalized_time(self.valid_from, "valid_from")
            if self.valid_from
            else None
        )
        end = (
            _normalized_time(self.valid_until, "valid_until")
            if self.valid_until
            else None
        )
        if start and end and end <= start:
            raise ValueError("valid_until debe ser posterior a valid_from")
        object.__setattr__(self, "valid_from", start)
        object.__setattr__(self, "valid_until", end)
        return self

    def matches_identity(self, event: NormalizedBitrixEvent) -> bool:
        return (
            event.member_id == self.member_id
            and event.bot_id == self.bot_id
            and event.dialog_id == self.dialog_id
            and event.chat_id == self.chat_id
        )

    def active_at(self, now: datetime) -> bool:
        return (
            (self.valid_from is None or now >= self.valid_from)
            and (self.valid_until is None or now < self.valid_until)
        )


class PilotScopeDecision(BaseModel):
    accepted: bool
    reason: str


class PilotScopePolicy:
    """
    Autoriza únicamente identidades exactas y falla cerrada cuando está activa.

    Es ortogonal a ``off/review/shadow/active``: no habilita ningún modo.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        rules: Iterable[PilotScopeRule] = (),
        emergency_stop: bool = False,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._enabled = bool(enabled)
        self._rules = tuple(rules)
        self._emergency_stop = bool(emergency_stop)
        self._clock = clock

    def evaluate(self, event: NormalizedBitrixEvent) -> PilotScopeDecision:
        if not self._enabled:
            return PilotScopeDecision(
                accepted=True,
                reason="pilot_scope_disabled",
            )
        if self._emergency_stop:
            return PilotScopeDecision(
                accepted=False,
                reason="pilot_emergency_stop",
            )
        if not self._rules:
            return PilotScopeDecision(
                accepted=False,
                reason="pilot_scope_empty",
            )

        now = _normalized_time(self._clock(), "pilot_clock")
        matching = [
            rule
            for rule in self._rules
            if rule.matches_identity(event)
        ]
        if not matching:
            return PilotScopeDecision(
                accepted=False,
                reason="outside_pilot_scope",
            )
        if any(rule.active_at(now) for rule in matching):
            return PilotScopeDecision(
                accepted=True,
                reason="pilot_scope_allowed",
            )
        if all(
            rule.valid_from is not None and now < rule.valid_from
            for rule in matching
        ):
            reason = "pilot_scope_not_started"
        else:
            reason = "pilot_scope_expired"
        return PilotScopeDecision(accepted=False, reason=reason)

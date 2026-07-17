import unittest
import json
from datetime import datetime, timedelta, timezone

from pydantic import ValidationError

from bitrix_connector.event_parser import parse_webhook_form
from bitrix_connector.config import load_settings
from bitrix_connector.pilot_scope import PilotScopePolicy, PilotScopeRule


NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)


def event_form():
    return {
        "event": "ONIMBOTV2MESSAGEADD",
        "data[bot][id]": "456",
        "data[message][id]": "789",
        "data[message][chatId]": "5",
        "data[message][authorId]": "27",
        "data[message][text]": "Prueba controlada",
        "data[chat][dialogId]": "chat5",
        "data[user][id]": "27",
        "auth[domain]": "portal.bitrix24.test",
        "auth[member_id]": "member-123",
        "auth[application_token]": "secret-token",
    }


def exact_rule(**overrides):
    data = {
        "member_id": "member-123",
        "bot_id": 456,
        "dialog_id": "chat5",
        "chat_id": 5,
    }
    data.update(overrides)
    return PilotScopeRule(**data)


class PilotScopePolicyTests(unittest.TestCase):
    def setUp(self):
        self.event = parse_webhook_form(event_form())

    def test_disabled_policy_does_not_change_normal_flow(self):
        decision = PilotScopePolicy(
            enabled=False,
            emergency_stop=True,
        ).evaluate(self.event)

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reason, "pilot_scope_disabled")

    def test_enabled_policy_requires_exact_four_part_identity(self):
        allowed = PilotScopePolicy(
            enabled=True,
            rules=[exact_rule()],
            clock=lambda: NOW,
        ).evaluate(self.event)

        self.assertTrue(allowed.accepted)
        self.assertEqual(allowed.reason, "pilot_scope_allowed")

        mutations = {
            "member_id": "member-other",
            "bot_id": 999,
            "dialog_id": "chat99",
            "chat_id": 99,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                denied = PilotScopePolicy(
                    enabled=True,
                    rules=[exact_rule(**{field: value})],
                    clock=lambda: NOW,
                ).evaluate(self.event)
                self.assertFalse(denied.accepted)
                self.assertEqual(denied.reason, "outside_pilot_scope")

    def test_empty_scope_and_emergency_stop_fail_closed(self):
        empty = PilotScopePolicy(
            enabled=True,
            clock=lambda: NOW,
        ).evaluate(self.event)
        stopped = PilotScopePolicy(
            enabled=True,
            rules=[exact_rule()],
            emergency_stop=True,
            clock=lambda: NOW,
        ).evaluate(self.event)

        self.assertFalse(empty.accepted)
        self.assertEqual(empty.reason, "pilot_scope_empty")
        self.assertFalse(stopped.accepted)
        self.assertEqual(stopped.reason, "pilot_emergency_stop")

    def test_time_window_is_start_inclusive_and_end_exclusive(self):
        rule = exact_rule(
            valid_from=NOW,
            valid_until=NOW + timedelta(minutes=30),
        )

        before = PilotScopePolicy(
            enabled=True,
            rules=[rule],
            clock=lambda: NOW - timedelta(seconds=1),
        ).evaluate(self.event)
        at_start = PilotScopePolicy(
            enabled=True,
            rules=[rule],
            clock=lambda: NOW,
        ).evaluate(self.event)
        before_end = PilotScopePolicy(
            enabled=True,
            rules=[rule],
            clock=lambda: NOW + timedelta(minutes=30, seconds=-1),
        ).evaluate(self.event)
        at_end = PilotScopePolicy(
            enabled=True,
            rules=[rule],
            clock=lambda: NOW + timedelta(minutes=30),
        ).evaluate(self.event)

        self.assertEqual(before.reason, "pilot_scope_not_started")
        self.assertTrue(at_start.accepted)
        self.assertTrue(before_end.accepted)
        self.assertEqual(at_end.reason, "pilot_scope_expired")

    def test_one_active_exact_rule_is_enough_among_multiple_windows(self):
        policy = PilotScopePolicy(
            enabled=True,
            rules=[
                exact_rule(valid_until=NOW - timedelta(seconds=1)),
                exact_rule(
                    valid_from=NOW - timedelta(minutes=1),
                    valid_until=NOW + timedelta(minutes=1),
                ),
                exact_rule(dialog_id="chat-other"),
            ],
            clock=lambda: NOW,
        )

        decision = policy.evaluate(self.event)

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reason, "pilot_scope_allowed")

    def test_invalid_rules_and_naive_clock_are_rejected(self):
        with self.assertRaises(ValidationError):
            exact_rule(
                valid_from=NOW,
                valid_until=NOW,
            )
        with self.assertRaisesRegex(ValueError, "pilot_clock"):
            PilotScopePolicy(
                enabled=True,
                rules=[exact_rule()],
                clock=lambda: datetime(2026, 7, 16, 12, 0),
            ).evaluate(self.event)


class PilotScopeConfigTests(unittest.TestCase):
    def test_defaults_are_disabled_stopped_and_valid(self):
        settings = load_settings({})

        self.assertFalse(settings.pilot_enabled)
        self.assertTrue(settings.pilot_emergency_stop)
        self.assertEqual(settings.pilot_rules, ())
        self.assertTrue(settings.pilot_configuration_valid)
        self.assertEqual(
            settings.pilot_summary,
            {
                "enabled": False,
                "emergency_stop": True,
                "rule_count": 0,
                "configuration_valid": True,
            },
        )

    def test_strict_json_builds_exact_rules_without_exposing_them_in_summary(self):
        settings = load_settings(
            {
                "NIA_BITRIX_PILOT_ENABLED": "true",
                "NIA_BITRIX_PILOT_EMERGENCY_STOP": "false",
                "NIA_BITRIX_PILOT_RULES_JSON": json.dumps(
                    [
                        {
                            "member_id": "member-123",
                            "bot_id": 456,
                            "dialog_id": "chat5",
                            "chat_id": 5,
                            "valid_from": "2026-07-16T12:00:00Z",
                            "valid_until": "2026-07-16T13:00:00Z",
                        }
                    ]
                ),
            }
        )

        self.assertTrue(settings.pilot_enabled)
        self.assertFalse(settings.pilot_emergency_stop)
        self.assertTrue(settings.pilot_configuration_valid)
        self.assertEqual(len(settings.pilot_rules), 1)
        self.assertEqual(settings.pilot_rules[0].dialog_id, "chat5")
        self.assertEqual(settings.pilot_summary["rule_count"], 1)
        self.assertNotIn("chat5", repr(settings.pilot_summary))
        self.assertEqual(settings.warnings, ())

    def test_enabled_empty_or_invalid_configuration_fails_closed(self):
        cases = (
            (
                {
                    "NIA_BITRIX_PILOT_ENABLED": "true",
                    "NIA_BITRIX_PILOT_RULES_JSON": "[]",
                },
                "pilot_scope_enabled_without_rules",
            ),
            (
                {
                    "NIA_BITRIX_PILOT_ENABLED": "sometimes",
                },
                "invalid_pilot_enabled",
            ),
            (
                {
                    "NIA_BITRIX_PILOT_EMERGENCY_STOP": "maybe",
                },
                "invalid_pilot_emergency_stop",
            ),
            (
                {
                    "NIA_BITRIX_PILOT_ENABLED": "true",
                    "NIA_BITRIX_PILOT_RULES_JSON": "{invalid",
                },
                "invalid_pilot_rules_json",
            ),
        )
        for environ, warning in cases:
            with self.subTest(warning=warning):
                settings = load_settings(environ)
                self.assertIn(warning, settings.warnings)
                self.assertFalse(settings.pilot_configuration_valid)


if __name__ == "__main__":
    unittest.main()

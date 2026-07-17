import hashlib
import json
import unittest
from datetime import datetime, timezone

from bitrix_connector.event_parser import parse_webhook_form
from bitrix_connector.preflight import (
    MAX_PREFLIGHT_TEXT_CHARS,
    build_text_preflight,
)
from bitrix_connector.storage import build_received_record


def event_form(text="Necesito una bomba"):
    return {
        "event": "ONIMBOTV2MESSAGEADD",
        "data[bot][id]": "456",
        "data[message][id]": "789",
        "data[message][chatId]": "5",
        "data[message][authorId]": "27",
        "data[message][text]": text,
        "data[message][isSystem]": "0",
        "data[chat][dialogId]": "chat5",
        "data[chat][type]": "openChannel",
        "data[chat][entityType]": "LINES",
        "data[user][id]": "27",
        "data[user][bot]": "0",
        "data[user][connector]": "1",
        "auth[domain]": "viaindustrial.bitrix24.es",
        "auth[member_id]": "member-123",
        "auth[application_token]": "secret-token",
        "data[bot][auth][access_token]": "oauth-secret",
        "data[bot][auth][refresh_token]": "refresh-secret",
    }


def event_record(form):
    return build_received_record(
        parse_webhook_form(form),
        form,
        identity_verified=True,
        security_reason="identity_verified",
        received_at=datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc),
    )


class TextPreflightTests(unittest.TestCase):
    def test_builds_exact_text_payload_and_empty_attachment_manifest(self):
        review = build_text_preflight(event_record(event_form()))

        self.assertEqual(review.decision, "ready_for_review")
        self.assertEqual(
            review.nia_payload_preview.model_dump(),
            {
                "session_id": "bitrix:member-123:chat5",
                "mensaje": "Necesito una bomba",
            },
        )
        self.assertEqual(review.preflight_manifest.attachment_status, "not_present")
        self.assertEqual(review.preflight_manifest.attachment_count, 0)
        self.assertEqual(review.preflight_manifest.blocking_reasons, [])

    def test_normalizes_unicode_lines_spaces_and_control_characters(self):
        form = event_form("  Cafe\u0301\t industrial\r\n\r\n\r\nBomba\x00  10 HP  ")

        review = build_text_preflight(event_record(form))

        self.assertEqual(review.normalized_input.text, "Café industrial\n\nBomba 10 HP")
        self.assertIn("unicode_nfc", review.preflight_manifest.transformations)
        self.assertIn("line_endings_lf", review.preflight_manifest.transformations)
        self.assertIn(
            "control_characters_removed",
            review.preflight_manifest.transformations,
        )
        self.assertIn(
            "horizontal_whitespace_collapsed",
            review.preflight_manifest.transformations,
        )
        self.assertIn(
            "blank_lines_collapsed",
            review.preflight_manifest.transformations,
        )
        self.assertIn(
            "outer_whitespace_trimmed",
            review.preflight_manifest.transformations,
        )
        self.assertGreater(
            review.normalized_input.original_text_length,
            review.normalized_input.normalized_text_length,
        )
        self.assertEqual(
            review.nia_payload_preview.mensaje,
            review.normalized_input.text,
        )

    def test_flags_commercial_contact_data_without_blocking(self):
        text = (
            "Cotizar en https://viaindustrial.com para compras@cliente.com "
            "o llamar al +57 300 123 4567"
        )

        review = build_text_preflight(event_record(event_form(text)))

        self.assertEqual(review.decision, "ready_for_review")
        self.assertEqual(
            review.preflight_manifest.signals,
            ["url_detected", "email_detected", "phone_detected"],
        )
        self.assertEqual(review.nia_payload_preview.mensaje, text)

    def test_possible_secret_is_visible_but_not_added_to_nia_payload(self):
        form = event_form("Mi api_key = sk-example123456789")

        review = build_text_preflight(event_record(form))
        serialized = review.model_dump_json()

        self.assertEqual(review.decision, "blocked")
        self.assertIn(
            "possible_secret_detected",
            review.preflight_manifest.blocking_reasons,
        )
        self.assertIsNone(review.nia_payload_preview)
        self.assertNotIn("secret-token", serialized)
        self.assertNotIn("oauth-secret", serialized)
        self.assertNotIn("refresh-secret", serialized)
        self.assertEqual(
            review.raw_redacted["auth[application_token]"],
            "[REDACTED]",
        )

    def test_empty_and_oversized_text_are_blocked_without_truncation(self):
        empty = build_text_preflight(event_record(event_form(" \r\n\t ")))
        long_text = "A" * (MAX_PREFLIGHT_TEXT_CHARS + 1)
        oversized = build_text_preflight(event_record(event_form(long_text)))

        self.assertIn("empty_text", empty.preflight_manifest.blocking_reasons)
        self.assertIsNone(empty.nia_payload_preview)
        self.assertIn(
            "text_too_long",
            oversized.preflight_manifest.blocking_reasons,
        )
        self.assertEqual(oversized.normalized_input.text, long_text)
        self.assertIsNone(oversized.nia_payload_preview)

    def test_system_bot_and_non_open_line_messages_are_blocked(self):
        form = event_form()
        form["data[message][isSystem]"] = "1"
        form["data[user][bot]"] = "1"
        form["data[chat][type]"] = "private"
        form["data[chat][entityType]"] = "PRIVATE"

        review = build_text_preflight(event_record(form))

        self.assertEqual(review.decision, "blocked")
        self.assertEqual(
            review.preflight_manifest.blocking_reasons,
            ["system_message", "bot_message", "unsupported_chat_context"],
        )
        self.assertIsNone(review.nia_payload_preview)

    def test_phone_id_is_omitted_until_contact_identity_is_resolved(self):
        review = build_text_preflight(event_record(event_form()))
        payload = review.nia_payload_preview.model_dump()

        self.assertNotIn("phone_id", payload)
        self.assertEqual(payload["session_id"], "bitrix:member-123:chat5")

    def test_ready_hash_is_canonical_for_exact_nia_payload(self):
        first = build_text_preflight(
            event_record(event_form("Necesito una bomba"))
        )
        second = build_text_preflight(
            event_record(event_form("  Necesito   una bomba  "))
        )
        canonical = json.dumps(
            {
                "kind": "nia_payload",
                "payload": {
                    "session_id": "bitrix:member-123:chat5",
                    "mensaje": "Necesito una bomba",
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

        self.assertEqual(
            first.content_hash,
            hashlib.sha256(canonical).hexdigest(),
        )
        self.assertEqual(first.content_hash, second.content_hash)


if __name__ == "__main__":
    unittest.main()

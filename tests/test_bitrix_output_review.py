import unittest
from datetime import datetime, timezone

from bitrix_connector.event_parser import parse_webhook_form
from bitrix_connector.nia_client import NiaChatResponse
from bitrix_connector.output_review import (
    build_output_content_hash,
    build_output_review,
)
from bitrix_connector.storage import build_received_record


def event_record():
    form = {
        "event": "ONIMBOTV2MESSAGEADD",
        "data[bot][id]": "456",
        "data[message][id]": "789",
        "data[message][chatId]": "5",
        "data[message][authorId]": "27",
        "data[message][text]": "Necesito una bomba",
        "data[chat][dialogId]": "chat5",
        "data[chat][type]": "openChannel",
        "data[chat][entityType]": "LINES",
        "data[user][id]": "27",
        "auth[domain]": "viaindustrial.bitrix24.es",
        "auth[member_id]": "member-123",
        "auth[application_token]": "secret-token",
    }
    return build_received_record(
        parse_webhook_form(form),
        form,
        identity_verified=True,
        security_reason="identity_verified",
        received_at=datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc),
    )


class OutputReviewTests(unittest.TestCase):
    def test_builds_exact_future_bitrix_payload_from_contract_response(self):
        response = NiaChatResponse(
            respuesta="Tenemos una opción disponible.",
            etapa="resultado",
            items_resultado=[{"codigo": "123456"}],
            cliente={"id": "contact-27"},
        )

        review = build_output_review(event_record(), response)

        self.assertEqual(review.decision, "ready_for_review")
        self.assertEqual(review.blocking_reasons, [])
        self.assertEqual(review.nia_response, response)
        self.assertEqual(
            review.bitrix_payload_preview.model_dump(),
            {
                "botId": 456,
                "dialogId": "chat5",
                "fields": {"message": "Tenemos una opción disponible."},
            },
        )
        self.assertEqual(
            review.content_hash,
            build_output_content_hash(review.bitrix_payload_preview),
        )

    def test_hash_covers_exact_message_and_destination(self):
        first = build_output_review(
            event_record(),
            NiaChatResponse(respuesta="Respuesta A"),
        )
        changed_message = build_output_review(
            event_record(),
            NiaChatResponse(respuesta="Respuesta B"),
        )
        changed_event = event_record().model_copy(
            update={
                "normalized_event": {
                    **event_record().normalized_event,
                    "dialog_id": "chat6",
                }
            }
        )
        changed_destination = build_output_review(
            changed_event,
            NiaChatResponse(respuesta="Respuesta A"),
        )

        self.assertNotEqual(first.content_hash, changed_message.content_hash)
        self.assertNotEqual(
            first.content_hash,
            changed_destination.content_hash,
        )

    def test_empty_response_is_visible_but_blocked(self):
        review = build_output_review(
            event_record(),
            NiaChatResponse(respuesta=" \n "),
        )

        self.assertEqual(review.decision, "blocked")
        self.assertEqual(review.blocking_reasons, ["empty_nia_response"])
        self.assertEqual(
            review.bitrix_payload_preview.fields.message,
            " \n ",
        )

    def test_oversized_response_is_not_silently_truncated(self):
        message = "x" * 20_001
        review = build_output_review(
            event_record(),
            NiaChatResponse(respuesta=message),
        )

        self.assertEqual(review.decision, "blocked")
        self.assertIn("bitrix_message_too_long", review.blocking_reasons)
        self.assertEqual(
            review.bitrix_payload_preview.fields.message,
            message,
        )


if __name__ == "__main__":
    unittest.main()

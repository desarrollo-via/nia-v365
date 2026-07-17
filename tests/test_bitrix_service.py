import unittest
from datetime import datetime, timezone

from bitrix_connector.config import load_settings
from bitrix_connector.models import ConnectorIngestionStatus, StorageWriteResult
from bitrix_connector.pilot_scope import PilotScopePolicy, PilotScopeRule
from bitrix_connector.service import ConnectorIngestionService, ConnectorPersistenceError


def event_form():
    return {
        "event": "ONIMBOTV2MESSAGEADD",
        "ts": "1772093963",
        "data[bot][id]": "456",
        "data[message][id]": "789",
        "data[message][chatId]": "5",
        "data[message][authorId]": "27",
        "data[message][text]": "Necesito una bomba",
        "data[chat][dialogId]": "chat5",
        "data[user][id]": "27",
        "auth[domain]": "viaindustrial.bitrix24.es",
        "auth[member_id]": "member-123",
        "auth[application_token]": "secret-token",
        "data[bot][auth][access_token]": "oauth-secret",
    }


def connector_settings():
    return load_settings({
        "NIA_BITRIX_DOMAIN": "viaindustrial.bitrix24.es",
        "NIA_BITRIX_MEMBER_ID": "member-123",
        "NIA_BITRIX_APPLICATION_TOKEN": "secret-token",
    })


class RecordingStore:
    def __init__(self):
        self.records = []
        self.keys = set()

    async def ensure_indexes(self):
        return None

    async def save_received(self, record):
        self.records.append(record)
        duplicate = record.event_key in self.keys
        self.keys.add(record.event_key)
        return StorageWriteResult(
            event_key=record.event_key,
            created=not duplicate,
            duplicate=duplicate,
        )

    async def get_by_key(self, event_key):
        return next((record for record in self.records if record.event_key == event_key), None)


class FailingStore(RecordingStore):
    async def save_received(self, record):
        raise RuntimeError("mongodb://user:secret@internal.invalid")


class ConnectorIngestionServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = RecordingStore()
        self.service = ConnectorIngestionService(self.store)
        self.settings = connector_settings()

    async def test_valid_event_is_normalized_secured_and_stored(self):
        result = await self.service.ingest(event_form(), self.settings)

        self.assertEqual(result.status, ConnectorIngestionStatus.STORED)
        self.assertTrue(result.created)
        self.assertFalse(result.duplicate)
        self.assertTrue(result.identity_verified)
        self.assertEqual(len(self.store.records), 1)
        stored = self.store.records[0]
        self.assertEqual(stored.normalized_event["text"], "Necesito una bomba")
        self.assertNotIn("application_token", stored.normalized_event)
        self.assertEqual(stored.raw_redacted["auth[application_token]"], "[REDACTED]")
        self.assertEqual(stored.raw_redacted["data[bot][auth][access_token]"], "[REDACTED]")

    async def test_repeated_event_is_reported_as_duplicate(self):
        first = await self.service.ingest(event_form(), self.settings)
        second = await self.service.ingest(event_form(), self.settings)

        self.assertEqual(first.status, ConnectorIngestionStatus.STORED)
        self.assertEqual(second.status, ConnectorIngestionStatus.DUPLICATE)
        self.assertTrue(second.duplicate)
        self.assertFalse(second.created)
        self.assertEqual(first.event_key, second.event_key)

    async def test_exact_pilot_scope_allows_storage(self):
        policy = PilotScopePolicy(
            enabled=True,
            rules=[
                PilotScopeRule(
                    member_id="member-123",
                    bot_id=456,
                    dialog_id="chat5",
                    chat_id=5,
                )
            ],
            clock=lambda: datetime(
                2026,
                7,
                16,
                12,
                0,
                tzinfo=timezone.utc,
            ),
        )
        service = ConnectorIngestionService(self.store, policy)

        result = await service.ingest(event_form(), self.settings)

        self.assertEqual(result.status, ConnectorIngestionStatus.STORED)
        self.assertEqual(len(self.store.records), 1)

    async def test_event_outside_pilot_scope_never_reaches_storage(self):
        policy = PilotScopePolicy(
            enabled=True,
            rules=[
                PilotScopeRule(
                    member_id="member-123",
                    bot_id=456,
                    dialog_id="chat-controlled",
                    chat_id=99,
                )
            ],
        )
        service = ConnectorIngestionService(self.store, policy)

        result = await service.ingest(event_form(), self.settings)

        self.assertEqual(result.status, ConnectorIngestionStatus.IGNORED)
        self.assertEqual(result.reason, "outside_pilot_scope")
        self.assertTrue(result.identity_verified)
        self.assertEqual(self.store.records, [])

    async def test_invalid_identity_is_ignored_without_storage_call(self):
        form = event_form()
        form["auth[application_token]"] = "attacker-token"

        result = await self.service.ingest(form, self.settings)

        self.assertEqual(result.status, ConnectorIngestionStatus.IGNORED)
        self.assertEqual(result.reason, "application_token_mismatch")
        self.assertFalse(result.identity_verified)
        self.assertEqual(self.store.records, [])
        self.assertNotIn("attacker-token", result.model_dump_json())

    async def test_unsupported_event_is_ignored_without_storage_call(self):
        form = event_form()
        form["event"] = "ONIMBOTV2MESSAGEUPDATE"

        result = await self.service.ingest(form, self.settings)

        self.assertEqual(result.status, ConnectorIngestionStatus.IGNORED)
        self.assertEqual(result.reason, "unsupported_event")
        self.assertTrue(result.identity_verified)
        self.assertEqual(self.store.records, [])

    async def test_invalid_payload_is_reported_without_event_key_or_storage(self):
        form = event_form()
        form.pop("data[message][id]")

        result = await self.service.ingest(form, self.settings)

        self.assertEqual(result.status, ConnectorIngestionStatus.INVALID)
        self.assertEqual(result.reason, "invalid_webhook_payload")
        self.assertIsNone(result.event_key)
        self.assertEqual(self.store.records, [])

    async def test_storage_error_is_not_hidden(self):
        service = ConnectorIngestionService(FailingStore())

        with self.assertRaisesRegex(
            ConnectorPersistenceError,
            "connector_storage_unavailable",
        ) as raised:
            await service.ingest(event_form(), self.settings)

        self.assertIsInstance(raised.exception.__cause__, RuntimeError)
        self.assertNotIn("secret", str(raised.exception))


if __name__ == "__main__":
    unittest.main()

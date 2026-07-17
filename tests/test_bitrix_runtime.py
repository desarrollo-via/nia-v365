import unittest
import json
from dataclasses import replace

from bitrix_connector.config import ConnectorMode, load_settings
from bitrix_connector.models import ConnectorIngestionStatus, StorageWriteResult
from bitrix_connector.runtime import (
    ConnectorRuntime,
    ConnectorRuntimeState,
    ConnectorRuntimeUnavailable,
)
from bitrix_connector.storage import ConnectorStorageFactory, ConnectorStorageResources


def event_form():
    return {
        "event": "ONIMBOTV2MESSAGEADD",
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
    }


def connector_settings():
    return load_settings({
        "NIA_BITRIX_DOMAIN": "viaindustrial.bitrix24.es",
        "NIA_BITRIX_MEMBER_ID": "member-123",
        "NIA_BITRIX_APPLICATION_TOKEN": "secret-token",
        "MONGO_URI": "mongodb://unused.invalid",
    })


def enabled_pilot_settings(*, emergency_stop=False, rules=None):
    rules = rules if rules is not None else [
        {
            "member_id": "member-123",
            "bot_id": 456,
            "dialog_id": "chat5",
            "chat_id": 5,
        }
    ]
    return replace(
        load_settings(
            {
                "NIA_BITRIX_DOMAIN": "viaindustrial.bitrix24.es",
                "NIA_BITRIX_MEMBER_ID": "member-123",
                "NIA_BITRIX_APPLICATION_TOKEN": "secret-token",
                "NIA_BITRIX_PILOT_ENABLED": "true",
                "NIA_BITRIX_PILOT_EMERGENCY_STOP":
                "true" if emergency_stop else "false",
                "NIA_BITRIX_PILOT_RULES_JSON": json.dumps(rules),
            }
        ),
        effective_mode=ConnectorMode.REVIEW,
        activation_locked=False,
    )


class FakeClient:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


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
        return next((item for item in self.records if item.event_key == event_key), None)


class FakeFactory:
    def __init__(self, resources=None, error=None):
        self.resources = resources
        self.error = error
        self.build_calls = 0

    async def build(self, settings):
        self.build_calls += 1
        if self.error:
            raise self.error
        return self.resources


class ConnectorRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_enabled_pilot_fails_before_storage_factory(self):
        factory = FakeFactory(
            ConnectorStorageResources(
                client=FakeClient(),
                store=RecordingStore(),
            )
        )
        runtime = ConnectorRuntime(factory)
        invalid = replace(
            load_settings(
                {
                    "NIA_BITRIX_PILOT_ENABLED": "true",
                    "NIA_BITRIX_PILOT_RULES_JSON": "[]",
                }
            ),
            effective_mode=ConnectorMode.REVIEW,
            activation_locked=False,
        )

        snapshot = await runtime.start(invalid)

        self.assertEqual(snapshot.state, ConnectorRuntimeState.INERT)
        self.assertFalse(snapshot.service_available)
        self.assertFalse(snapshot.resources_available)
        self.assertEqual(factory.build_calls, 0)

    async def test_invalid_pilot_text_fails_closed_even_when_disabled(self):
        factory = FakeFactory(
            ConnectorStorageResources(
                client=FakeClient(),
                store=RecordingStore(),
            )
        )
        runtime = ConnectorRuntime(factory)
        invalid = replace(
            load_settings(
                {
                    "NIA_BITRIX_PILOT_ENABLED": "false",
                    "NIA_BITRIX_PILOT_RULES_JSON": "{invalid",
                }
            ),
            effective_mode=ConnectorMode.REVIEW,
            activation_locked=False,
        )

        snapshot = await runtime.start(invalid)

        self.assertEqual(snapshot.state, ConnectorRuntimeState.INERT)
        self.assertEqual(factory.build_calls, 0)

    async def test_runtime_injects_exact_pilot_policy(self):
        client = FakeClient()
        store = RecordingStore()
        factory = FakeFactory(
            ConnectorStorageResources(client=client, store=store)
        )
        runtime = ConnectorRuntime(factory)
        settings = enabled_pilot_settings()
        outside = event_form()
        outside["data[chat][dialogId]"] = "chat-other"
        outside["data[message][chatId]"] = "99"

        snapshot = await runtime.start(settings)
        allowed = await runtime.ingest(event_form(), settings)
        rejected = await runtime.ingest(outside, settings)

        self.assertEqual(snapshot.state, ConnectorRuntimeState.READY)
        self.assertEqual(allowed.status, ConnectorIngestionStatus.STORED)
        self.assertEqual(rejected.status, ConnectorIngestionStatus.IGNORED)
        self.assertEqual(rejected.reason, "outside_pilot_scope")
        self.assertEqual(len(store.records), 1)

    async def test_runtime_emergency_stop_blocks_exact_chat(self):
        store = RecordingStore()
        runtime = ConnectorRuntime(
            FakeFactory(
                ConnectorStorageResources(
                    client=FakeClient(),
                    store=store,
                )
            )
        )
        settings = enabled_pilot_settings(emergency_stop=True)

        await runtime.start(settings)
        result = await runtime.ingest(event_form(), settings)

        self.assertEqual(result.status, ConnectorIngestionStatus.IGNORED)
        self.assertEqual(result.reason, "pilot_emergency_stop")
        self.assertEqual(store.records, [])

    async def test_off_runtime_is_inert_without_creating_client(self):
        client_calls = []

        def forbidden_client_factory(uri):
            client_calls.append(uri)
            raise AssertionError("client must not be created in off")

        runtime = ConnectorRuntime(ConnectorStorageFactory(forbidden_client_factory))
        snapshot = await runtime.start(connector_settings())

        self.assertEqual(snapshot.state, ConnectorRuntimeState.INERT)
        self.assertFalse(snapshot.service_available)
        self.assertFalse(snapshot.resources_available)
        self.assertEqual(client_calls, [])
        with self.assertRaisesRegex(ConnectorRuntimeUnavailable, "not_ready"):
            await runtime.ingest(event_form(), connector_settings())

    async def test_ready_runtime_builds_service_and_ingests(self):
        client = FakeClient()
        store = RecordingStore()
        factory = FakeFactory(ConnectorStorageResources(client=client, store=store))
        runtime = ConnectorRuntime(factory)

        snapshot = await runtime.start(connector_settings())
        result = await runtime.ingest(event_form(), connector_settings())

        self.assertEqual(snapshot.state, ConnectorRuntimeState.READY)
        self.assertTrue(snapshot.service_available)
        self.assertEqual(result.status, ConnectorIngestionStatus.STORED)
        self.assertEqual(len(store.records), 1)

    async def test_start_is_idempotent(self):
        resources = ConnectorStorageResources(client=FakeClient(), store=RecordingStore())
        factory = FakeFactory(resources)
        runtime = ConnectorRuntime(factory)

        first = await runtime.start(connector_settings())
        second = await runtime.start(connector_settings())

        self.assertEqual(first, second)
        self.assertEqual(factory.build_calls, 1)

    async def test_close_releases_resources_once_and_returns_to_stopped(self):
        client = FakeClient()
        factory = FakeFactory(
            ConnectorStorageResources(client=client, store=RecordingStore())
        )
        runtime = ConnectorRuntime(factory)
        await runtime.start(connector_settings())

        first = await runtime.close()
        second = await runtime.close()

        self.assertEqual(first.state, ConnectorRuntimeState.STOPPED)
        self.assertEqual(second.state, ConnectorRuntimeState.STOPPED)
        self.assertFalse(first.service_available)
        self.assertEqual(client.close_calls, 1)
        with self.assertRaises(ConnectorRuntimeUnavailable):
            await runtime.ingest(event_form(), connector_settings())

    async def test_failed_start_remains_stopped_and_can_retry(self):
        factory = FakeFactory(error=RuntimeError("factory unavailable"))
        runtime = ConnectorRuntime(factory)

        with self.assertRaisesRegex(RuntimeError, "factory unavailable"):
            await runtime.start(connector_settings())
        self.assertEqual(runtime.snapshot.state, ConnectorRuntimeState.STOPPED)

        factory.error = None
        factory.resources = ConnectorStorageResources(
            client=FakeClient(),
            store=RecordingStore(),
        )
        snapshot = await runtime.start(connector_settings())

        self.assertEqual(snapshot.state, ConnectorRuntimeState.READY)
        self.assertEqual(factory.build_calls, 2)


if __name__ == "__main__":
    unittest.main()

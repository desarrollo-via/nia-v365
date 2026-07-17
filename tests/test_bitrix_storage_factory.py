import unittest
from dataclasses import replace

from bitrix_connector.config import (
    ConnectorMode,
    DEFAULT_EVENTS_COLLECTION,
    DEFAULT_INSTALLATIONS_COLLECTION,
    load_settings,
)
from bitrix_connector.storage import (
    ConnectorStorageConfigurationError,
    ConnectorStorageFactory,
)


class FakeCollection:
    def __init__(self, fail_indexes=False):
        self.fail_indexes = fail_indexes
        self.indexes = []

    async def create_index(self, keys, **options):
        if self.fail_indexes:
            raise RuntimeError("index failure")
        self.indexes.append((keys, options))
        return options["name"]


class FakeDatabase:
    def __init__(self, collection):
        self.collection = collection
        self.requested_collection = None

    def __getitem__(self, name):
        self.requested_collection = name
        return self.collection


class FakeClient:
    def __init__(self, uri, *, fail_indexes=False):
        self.uri = uri
        self.closed = False
        self.collection = FakeCollection(fail_indexes=fail_indexes)
        self.database = FakeDatabase(self.collection)
        self.requested_database = None

    def __getitem__(self, name):
        self.requested_database = name
        return self.database

    def close(self):
        self.closed = True


def enabled_settings(**overrides):
    base = load_settings({
        "MONGO_URI": "mongodb://example.invalid",
        "MONGO_DB": "shared_nia",
    })
    return replace(
        base,
        requested_mode="review",
        effective_mode=ConnectorMode.REVIEW,
        activation_locked=False,
        **overrides,
    )


class ConnectorStorageConfigTests(unittest.TestCase):
    def test_shared_mongo_values_and_exclusive_default_collection(self):
        settings = load_settings({
            "MONGO_URI": " mongodb://shared.invalid ",
            "MONGO_DB": " shared_nia ",
        })

        self.assertEqual(settings.mongo_uri, "mongodb://shared.invalid")
        self.assertEqual(settings.mongo_db, "shared_nia")
        self.assertEqual(settings.events_collection, DEFAULT_EVENTS_COLLECTION)
        self.assertEqual(
            settings.installations_collection,
            DEFAULT_INSTALLATIONS_COLLECTION,
        )
        self.assertTrue(all(settings.storage_configured.values()))

    def test_connector_specific_values_take_precedence(self):
        settings = load_settings({
            "MONGO_URI": "mongodb://shared.invalid",
            "MONGO_DB": "shared_nia",
            "NIA_BITRIX_MONGO_URI": "mongodb://connector.invalid",
            "NIA_BITRIX_MONGO_DB": "connector_db",
            "NIA_BITRIX_EVENTS_COLLECTION": "connector_events",
            "NIA_BITRIX_INSTALLATIONS_COLLECTION": "connector_installations",
            "NIA_BITRIX_CLIENT_ID": "local.client",
            "NIA_BITRIX_CLIENT_SECRET": "client-secret",
        })

        self.assertEqual(settings.mongo_uri, "mongodb://connector.invalid")
        self.assertEqual(settings.mongo_db, "connector_db")
        self.assertEqual(settings.events_collection, "connector_events")
        self.assertEqual(
            settings.installations_collection,
            "connector_installations",
        )
        self.assertTrue(all(settings.oauth_configured.values()))


class ConnectorStorageFactoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_effective_off_never_invokes_client_factory(self):
        calls = []

        def forbidden_factory(uri):
            calls.append(uri)
            raise AssertionError("client factory must not be called")

        settings = load_settings({"MONGO_URI": "mongodb://unused.invalid"})
        resources = await ConnectorStorageFactory(forbidden_factory).build(settings)

        self.assertIsNone(resources)
        self.assertEqual(calls, [])

    async def test_activation_lock_blocks_even_a_simulated_review_mode(self):
        calls = []
        settings = replace(
            load_settings({"MONGO_URI": "mongodb://unused.invalid"}),
            effective_mode=ConnectorMode.REVIEW,
            activation_locked=True,
        )

        resources = await ConnectorStorageFactory(lambda uri: calls.append(uri)).build(settings)

        self.assertIsNone(resources)
        self.assertEqual(calls, [])

    async def test_enabled_simulation_builds_exclusive_store_and_indexes(self):
        clients = []

        def fake_factory(uri):
            client = FakeClient(uri)
            clients.append(client)
            return client

        settings = enabled_settings(events_collection="custom_events")
        resources = await ConnectorStorageFactory(fake_factory).build(settings)

        self.assertIsNotNone(resources)
        client = clients[0]
        self.assertEqual(client.uri, "mongodb://example.invalid")
        self.assertEqual(client.requested_database, "shared_nia")
        self.assertEqual(client.database.requested_collection, "custom_events")
        self.assertEqual(len(client.collection.indexes), 3)
        resources.close()
        self.assertTrue(client.closed)

    async def test_missing_uri_fails_before_client_creation(self):
        calls = []
        settings = enabled_settings(mongo_uri=None)

        with self.assertRaisesRegex(ConnectorStorageConfigurationError, "mongo_uri_not_configured"):
            await ConnectorStorageFactory(lambda uri: calls.append(uri)).build(settings)

        self.assertEqual(calls, [])

    async def test_index_failure_closes_simulated_client(self):
        client = FakeClient("mongodb://example.invalid", fail_indexes=True)
        settings = enabled_settings()

        with self.assertRaisesRegex(RuntimeError, "index failure"):
            await ConnectorStorageFactory(lambda uri: client).build(settings)

        self.assertTrue(client.closed)


if __name__ == "__main__":
    unittest.main()

import unittest

from bitrix_connector.config import load_settings
from bitrix_connector.installation_factory import (
    OAuthInstallationConfigurationError,
    OAuthInstallationFactory,
)


class FakeCollection:
    def __init__(self, *, fail_indexes=False):
        self.fail_indexes = fail_indexes
        self.indexes = []

    async def create_index(self, keys, **options):
        if self.fail_indexes:
            raise RuntimeError("private index failure")
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


class FakeVerifier:
    def __init__(self):
        self.closed = False

    async def verify(self, **kwargs):
        raise AssertionError("la factoría no debe verificar durante build")

    async def close(self):
        self.closed = True


def installation_settings(**overrides):
    values = {
        "NIA_BITRIX_INSTALLATION_ENABLED": "true",
        "NIA_BITRIX_DOMAIN": "portal.bitrix24.test",
        "MONGO_URI": "mongodb://example.invalid",
        "MONGO_DB": "nia",
        "NIA_BITRIX_INSTALLATIONS_COLLECTION": "oauth_installations",
    }
    values.update(overrides)
    return load_settings(values)


class OAuthInstallationFactoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_disabled_window_never_creates_client(self):
        calls = []
        factory = OAuthInstallationFactory(
            client_factory=lambda uri: calls.append(uri),
            verifier_factory=FakeVerifier,
        )

        with self.assertRaisesRegex(
            OAuthInstallationConfigurationError,
            "installation_window_disabled",
        ):
            await factory.build(load_settings({}))

        self.assertEqual(calls, [])

    async def test_enabled_window_builds_only_oauth_store_and_closes(self):
        clients = []
        verifiers = []

        def client_factory(uri):
            client = FakeClient(uri)
            clients.append(client)
            return client

        def verifier_factory():
            verifier = FakeVerifier()
            verifiers.append(verifier)
            return verifier

        resources = await OAuthInstallationFactory(
            client_factory=client_factory,
            verifier_factory=verifier_factory,
        ).build(installation_settings())

        client = clients[0]
        self.assertEqual(client.requested_database, "nia")
        self.assertEqual(
            client.database.requested_collection,
            "oauth_installations",
        )
        self.assertEqual(len(client.collection.indexes), 1)
        self.assertEqual(
            client.collection.indexes[0][1]["name"],
            "idx_nia_bitrix_installation_member_unique",
        )
        self.assertFalse(client.closed)
        self.assertFalse(verifiers[0].closed)

        await resources.close()

        self.assertTrue(client.closed)
        self.assertTrue(verifiers[0].closed)

    async def test_index_failure_closes_client_without_verifier(self):
        client = FakeClient(
            "mongodb://example.invalid",
            fail_indexes=True,
        )
        verifier_calls = []

        with self.assertRaisesRegex(RuntimeError, "private index failure"):
            await OAuthInstallationFactory(
                client_factory=lambda uri: client,
                verifier_factory=lambda: verifier_calls.append(True),
            ).build(installation_settings())

        self.assertTrue(client.closed)
        self.assertEqual(verifier_calls, [])

    def test_boolean_is_strict_and_safe_by_default(self):
        default = load_settings({})
        invalid = load_settings(
            {"NIA_BITRIX_INSTALLATION_ENABLED": "sometimes"}
        )

        self.assertFalse(default.installation_enabled)
        self.assertTrue(default.installation_configuration_valid)
        self.assertFalse(invalid.installation_enabled)
        self.assertFalse(invalid.installation_configuration_valid)
        self.assertIn("invalid_installation_enabled", invalid.warnings)


if __name__ == "__main__":
    unittest.main()

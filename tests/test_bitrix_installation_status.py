import unittest
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bitrix_connector.config import load_settings
from bitrix_connector.installation_status import (
    OAuthInstallationStatusResponse,
    OAuthInstallationStatusService,
    OAuthInstallationStatusStorageUnavailable,
    OAuthInstallationStatusStoredDocumentInvalid,
)
from bitrix_connector.installation_status_factory import (
    OAuthInstallationStatusFactory,
)
from bitrix_connector.installation_status_router import (
    create_installation_status_router,
)
from bitrix_connector.oauth import BitrixOAuthInstallation


START = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)


def installed():
    return BitrixOAuthInstallation(
        member_id="private-member-id",
        domain="portal.bitrix24.test",
        client_endpoint="https://portal.bitrix24.test/rest/",
        server_endpoint="https://oauth.bitrix.info/rest/",
        access_token="private-access-token",
        refresh_token="private-refresh-token",
        application_token="private-application-token",
        expires_at=START + timedelta(hours=1),
        updated_at=START,
        revision=2,
    )


def diagnostic_settings(*, review_token="review-secret", **overrides):
    values = {
        "NIA_BITRIX_DOMAIN": "portal.bitrix24.test",
        "NIA_BITRIX_REVIEW_TOKEN": review_token,
        "MONGO_URI": "mongodb://example.invalid",
        "MONGO_DB": "nia",
        "NIA_BITRIX_INSTALLATIONS_COLLECTION": "oauth_installations",
        "NIA_BITRIX_INSTALLATION_ENABLED": "false",
    }
    values.update(overrides)
    return load_settings(values)


class MemoryStatusStore:
    def __init__(self, current=None):
        self.current = current
        self.domains = []

    async def get_installation_by_domain(self, domain):
        self.domains.append(domain)
        if self.current is None or self.current.domain != domain.lower():
            return None
        return self.current.model_copy(deep=True)


class RecordingReader:
    def __init__(self, response):
        self.response = response
        self.domains = []

    async def get_status(self, domain):
        self.domains.append(domain)
        return self.response


class FailingStore:
    def __init__(self, error):
        self.error = error

    async def get_installation_by_domain(self, domain):
        raise self.error


class FailingReader:
    def __init__(self, error):
        self.error = error

    async def get_status(self, domain):
        raise self.error


class FakeCollection:
    pass


class FakeDatabase:
    def __init__(self, collection):
        self.collection = collection
        self.requested_collection = None

    def __getitem__(self, name):
        self.requested_collection = name
        return self.collection


class FakeClient:
    def __init__(self, uri):
        self.uri = uri
        self.closed = False
        self.collection = FakeCollection()
        self.database = FakeDatabase(self.collection)
        self.requested_database = None

    def __getitem__(self, name):
        self.requested_database = name
        return self.database

    def close(self):
        self.closed = True


class InstallationStatusServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_present_status_contains_only_boolean_token_evidence(self):
        store = MemoryStatusStore(installed())
        result = await OAuthInstallationStatusService(store).get_status(
            "portal.bitrix24.test"
        )
        rendered = result.model_dump_json()

        self.assertEqual(result.status, "installed")
        self.assertTrue(result.installation_present)
        self.assertTrue(result.access_token_present)
        self.assertTrue(result.refresh_token_present)
        self.assertTrue(result.application_token_present)
        self.assertEqual(result.revision, 2)
        self.assertNotIn("private-member-id", rendered)
        self.assertNotIn("private-access-token", rendered)
        self.assertNotIn("private-refresh-token", rendered)
        self.assertNotIn("private-application-token", rendered)

    async def test_missing_status_is_explicit_and_empty(self):
        result = await OAuthInstallationStatusService(
            MemoryStatusStore()
        ).get_status("portal.bitrix24.test")

        self.assertEqual(result.status, "not_found")
        self.assertFalse(result.installation_present)
        self.assertIsNone(result.revision)
        self.assertIsNone(result.updated_at)
        self.assertIsNone(result.expires_at)

    async def test_storage_failure_is_classified_without_leaking_cause(self):
        with self.assertRaises(OAuthInstallationStatusStorageUnavailable) as caught:
            await OAuthInstallationStatusService(
                FailingStore(RuntimeError("mongodb://user:secret@private.invalid"))
            ).get_status("portal.bitrix24.test")

        self.assertEqual(
            str(caught.exception),
            "oauth_installation_storage_unavailable",
        )
        self.assertNotIn("secret", str(caught.exception))

    async def test_invalid_document_is_classified_without_leaking_cause(self):
        with self.assertRaises(
            OAuthInstallationStatusStoredDocumentInvalid
        ) as caught:
            await OAuthInstallationStatusService(
                FailingStore(ValueError("private malformed token document"))
            ).get_status("portal.bitrix24.test")

        self.assertEqual(
            str(caught.exception),
            "stored_oauth_installation_invalid",
        )
        self.assertNotIn("private", str(caught.exception))


class InstallationStatusFactoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_factory_reads_while_installation_window_is_closed(self):
        clients = []

        def client_factory(uri):
            client = FakeClient(uri)
            clients.append(client)
            return client

        resources = await OAuthInstallationStatusFactory(
            client_factory=client_factory
        ).build(diagnostic_settings())

        self.assertEqual(clients[0].requested_database, "nia")
        self.assertEqual(
            clients[0].database.requested_collection,
            "oauth_installations",
        )
        self.assertFalse(clients[0].closed)
        await resources.close()
        self.assertTrue(clients[0].closed)


class InstallationStatusRouterTests(unittest.TestCase):
    def app(self, reader, settings):
        app = FastAPI()
        app.include_router(
            create_installation_status_router(
                reader,
                settings_loader=lambda: settings,
            ),
            prefix="/bitrix-connector",
        )
        return app

    def test_route_requires_configured_bearer_token(self):
        response = OAuthInstallationStatusResponse(
            status="not_found",
            installation_present=False,
            access_token_present=False,
            refresh_token_present=False,
            application_token_present=False,
        )
        reader = RecordingReader(response)

        unconfigured = TestClient(
            self.app(reader, diagnostic_settings(review_token=""))
        ).get("/bitrix-connector/installation-status")
        unauthorized = TestClient(
            self.app(reader, diagnostic_settings())
        ).get(
            "/bitrix-connector/installation-status",
            headers={"Authorization": "Bearer wrong-secret"},
        )

        self.assertEqual(unconfigured.status_code, 503)
        self.assertEqual(
            unconfigured.json()["detail"],
            "review_token_not_configured",
        )
        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(reader.domains, [])

    def test_authorized_route_returns_safe_status_only(self):
        response = OAuthInstallationStatusResponse(
            status="installed",
            installation_present=True,
            access_token_present=True,
            refresh_token_present=True,
            application_token_present=True,
            revision=2,
            updated_at=START,
            expires_at=START + timedelta(hours=1),
        )
        reader = RecordingReader(response)
        result = TestClient(
            self.app(reader, diagnostic_settings())
        ).get(
            "/bitrix-connector/installation-status",
            headers={"Authorization": "Bearer review-secret"},
        )

        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["status"], "installed")
        self.assertEqual(result.json()["revision"], 2)
        self.assertEqual(reader.domains, ["portal.bitrix24.test"])
        for secret in (
            "private-member-id",
            "private-access-token",
            "private-refresh-token",
            "private-application-token",
            "review-secret",
        ):
            self.assertNotIn(secret, result.text)

    def test_authorized_route_distinguishes_safe_failure_categories(self):
        scenarios = (
            (
                OAuthInstallationStatusStorageUnavailable(
                    "mongodb://user:secret@private.invalid"
                ),
                "installation_diagnostic_storage_unavailable",
            ),
            (
                OAuthInstallationStatusStoredDocumentInvalid(
                    "private malformed token document"
                ),
                "installation_diagnostic_stored_document_invalid",
            ),
        )

        for error, expected_detail in scenarios:
            with self.subTest(expected_detail=expected_detail):
                result = TestClient(
                    self.app(FailingReader(error), diagnostic_settings())
                ).get(
                    "/bitrix-connector/installation-status",
                    headers={"Authorization": "Bearer review-secret"},
                )

                self.assertEqual(result.status_code, 503)
                self.assertEqual(result.json()["detail"], expected_detail)
                for secret in (
                    "private",
                    "mongodb://",
                    "review-secret",
                ):
                    self.assertNotIn(secret, result.text)


if __name__ == "__main__":
    unittest.main()

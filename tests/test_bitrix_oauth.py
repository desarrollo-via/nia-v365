import asyncio
import unittest
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from urllib.parse import parse_qs

import httpx

from bitrix_connector.bitrix_client import BitrixClient, BitrixClientDecision
from bitrix_connector.oauth import (
    INSTALLATION_MEMBER_INDEX,
    BitrixOAuthAccessError,
    BitrixOAuthCredentialProvider,
    BitrixOAuthInstallation,
    BitrixOAuthTokenPair,
    BitrixOAuthTransport,
    MongoBitrixOAuthStore,
    OAuthRefreshDecision,
    OAuthRefreshResult,
)
from bitrix_connector.output_review import (
    BitrixMessageFieldsPreview,
    BitrixMessagePayloadPreview,
)


MEMBER_ID = "member-123"
START = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)


def installation():
    return BitrixOAuthInstallation(
        member_id=MEMBER_ID,
        domain="portal.bitrix24.test",
        client_endpoint="https://portal.bitrix24.test/rest/",
        server_endpoint="https://oauth.bitrix.info/rest/",
        access_token="old-access-secret",
        refresh_token="old-refresh-secret",
        application_token="event-application-secret",
        expires_at=START + timedelta(hours=1),
        updated_at=START,
    )


def renewed_pair():
    return BitrixOAuthTokenPair(
        member_id=MEMBER_ID,
        client_endpoint="https://portal.bitrix24.test/rest/",
        server_endpoint="https://oauth.bitrix.info/rest/",
        access_token="new-access-secret",
        refresh_token="new-refresh-secret",
        expires_at=START + timedelta(hours=2),
    )


class MemoryOAuthStore:
    def __init__(self, current=None):
        self.current = current
        self.replace_calls = []

    async def get_installation(self, member_id):
        if self.current is None or self.current.member_id != member_id:
            return None
        return self.current.model_copy(deep=True)

    async def replace_tokens(
        self,
        member_id,
        *,
        expected_refresh_token,
        token_pair,
        now,
    ):
        self.replace_calls.append((member_id, expected_refresh_token, token_pair))
        if (
            self.current is None
            or self.current.refresh_token.get_secret_value()
            != expected_refresh_token
        ):
            return False
        self.current = BitrixOAuthInstallation(
            member_id=member_id,
            domain=self.current.domain,
            client_endpoint=token_pair.client_endpoint,
            server_endpoint=token_pair.server_endpoint,
            access_token=token_pair.access_token,
            refresh_token=token_pair.refresh_token,
            application_token=self.current.application_token,
            expires_at=token_pair.expires_at,
            updated_at=now,
            revision=self.current.revision + 1,
        )
        return True


class CountingRefreshTransport:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def refresh(self, refresh_token):
        self.calls.append(refresh_token)
        await asyncio.sleep(0)
        return self.result


class FakeMongoCollection:
    def __init__(self):
        self.document = None
        self.indexes = []

    async def create_index(self, keys, **options):
        self.indexes.append((keys, options))
        return options["name"]

    async def update_one(self, selector, update, *, upsert):
        if upsert:
            if self.document is not None:
                return SimpleNamespace(modified_count=0, upserted_id=None)
            self.document = deepcopy(update["$setOnInsert"])
            return SimpleNamespace(modified_count=0, upserted_id="memory-1")
        if self.document is None:
            return SimpleNamespace(modified_count=0)
        if any(self.document.get(key) != value for key, value in selector.items()):
            return SimpleNamespace(modified_count=0)
        self.document.update(deepcopy(update.get("$set", {})))
        for field, increment in update.get("$inc", {}).items():
            self.document[field] += increment
        return SimpleNamespace(modified_count=1)

    async def find_one(self, selector, projection, sort=None):
        if self.document is None:
            return None
        if any(self.document.get(key) != value for key, value in selector.items()):
            return None
        return deepcopy(self.document)


class OAuthStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_install_is_idempotent_and_reinstall_increments_revision(self):
        collection = FakeMongoCollection()
        store = MongoBitrixOAuthStore(collection)
        original = installation()
        replacement = BitrixOAuthInstallation(
            member_id=MEMBER_ID,
            domain="portal.bitrix24.test",
            client_endpoint="https://portal.bitrix24.test/rest/",
            server_endpoint="https://oauth.bitrix.info/rest/",
            access_token="replacement-access-secret",
            refresh_token="replacement-refresh-secret",
            application_token="event-application-secret",
            expires_at=START + timedelta(hours=3),
            updated_at=START + timedelta(minutes=5),
        )

        created = await store.save_installation(original)
        repeated = await store.save_installation(original)
        reinstalled = await store.save_installation(replacement)
        stored = await store.get_installation(MEMBER_ID)

        self.assertEqual(created.status.value, "created")
        self.assertEqual(created.revision, 1)
        self.assertEqual(repeated.status.value, "idempotent")
        self.assertEqual(repeated.revision, 1)
        self.assertEqual(reinstalled.status.value, "reinstalled")
        self.assertEqual(reinstalled.revision, 2)
        self.assertEqual(stored.revision, 2)
        self.assertEqual(
            stored.access_token.get_secret_value(),
            "replacement-access-secret",
        )

    async def test_installation_is_durable_and_refresh_replaces_pair_atomically(self):
        collection = FakeMongoCollection()
        store = MongoBitrixOAuthStore(collection)
        await store.ensure_indexes()
        await store.save_installation(installation())

        before = await store.get_installation(MEMBER_ID)
        applied = await store.replace_tokens(
            MEMBER_ID,
            expected_refresh_token="old-refresh-secret",
            token_pair=renewed_pair(),
            now=START + timedelta(minutes=1),
        )
        stale = await store.replace_tokens(
            MEMBER_ID,
            expected_refresh_token="old-refresh-secret",
            token_pair=renewed_pair(),
            now=START + timedelta(minutes=2),
        )
        after = await store.get_installation(MEMBER_ID)

        self.assertEqual(collection.indexes[0][1]["name"], INSTALLATION_MEMBER_INDEX)
        self.assertTrue(collection.indexes[0][1]["unique"])
        self.assertEqual(before.revision, 1)
        self.assertTrue(applied)
        self.assertFalse(stale)
        self.assertEqual(after.revision, 2)
        self.assertEqual(
            after.access_token.get_secret_value(),
            "new-access-secret",
        )
        self.assertEqual(
            after.refresh_token.get_secret_value(),
            "new-refresh-secret",
        )
        self.assertNotIn("old-access-secret", repr(before))
        self.assertNotIn("old-refresh-secret", repr(before))

    async def test_installation_can_be_read_by_domain_without_exposing_it(self):
        collection = FakeMongoCollection()
        store = MongoBitrixOAuthStore(collection)
        await store.save_installation(installation())

        stored = await store.get_installation_by_domain(
            "PORTAL.BITRIX24.TEST"
        )
        missing = await store.get_installation_by_domain(
            "other.bitrix24.test"
        )

        self.assertEqual(stored.member_id, MEMBER_ID)
        self.assertIsNone(missing)
        self.assertNotIn("old-access-secret", repr(stored))
        self.assertNotIn("old-refresh-secret", repr(stored))


class OAuthTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_refresh_uses_official_contract_and_masks_result_secrets(self):
        captured = {}

        async def handler(request):
            captured["method"] = request.method
            captured["url"] = str(request.url)
            return httpx.Response(
                200,
                json={
                    "access_token": "new-access-secret",
                    "refresh_token": "new-refresh-secret",
                    "client_endpoint": "https://portal.bitrix24.test/rest/",
                    "server_endpoint": "https://oauth.bitrix.info/rest/",
                    "expires": int((START + timedelta(hours=1)).timestamp()),
                    "member_id": MEMBER_ID,
                },
            )

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(http_client.aclose)
        transport = BitrixOAuthTransport(
            client_id="local.client",
            client_secret="client-secret",
            timeout_seconds=3,
            http_client=http_client,
        )

        result = await transport.refresh("old-refresh-secret")
        query = parse_qs(httpx.URL(captured["url"]).query.decode())

        self.assertEqual(result.decision, OAuthRefreshDecision.SUCCESS)
        self.assertEqual(captured["method"], "GET")
        self.assertEqual(query["grant_type"], ["refresh_token"])
        self.assertEqual(query["client_id"], ["local.client"])
        self.assertEqual(query["client_secret"], ["client-secret"])
        self.assertEqual(query["refresh_token"], ["old-refresh-secret"])
        self.assertNotIn("new-access-secret", repr(result))
        self.assertNotIn("new-refresh-secret", repr(result))

    async def test_transport_errors_are_safe_and_classified(self):
        responses = (
            httpx.Response(429, headers={"Retry-After": "90"}, text="secret"),
            httpx.Response(400, json={"error": "invalid_grant"}),
            httpx.Response(200, text="not-json-secret"),
        )
        expected = (
            (OAuthRefreshDecision.RETRY, "oauth_server_retryable", 90),
            (OAuthRefreshDecision.FAIL, "oauth_refresh_rejected", 0),
            (OAuthRefreshDecision.FAIL, "oauth_invalid_response", 0),
        )
        for response, outcome in zip(responses, expected):
            with self.subTest(outcome=outcome):
                async def handler(request, current=response):
                    return current

                client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
                self.addAsyncCleanup(client.aclose)
                transport = BitrixOAuthTransport(
                    client_id="local.client",
                    client_secret="client-secret",
                    timeout_seconds=3,
                    http_client=client,
                )
                result = await transport.refresh("old-refresh-secret")
                self.assertEqual(
                    (result.decision, result.error_code, result.retry_after_seconds),
                    outcome,
                )
                self.assertNotIn("secret", repr(result))


class OAuthProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_refreshes_share_one_rotation(self):
        store = MemoryOAuthStore(installation())
        transport = CountingRefreshTransport(
            OAuthRefreshResult(
                OAuthRefreshDecision.SUCCESS,
                token_pair=renewed_pair(),
            )
        )
        provider = BitrixOAuthCredentialProvider(
            store,
            transport,
            clock=lambda: START + timedelta(minutes=1),
        )

        tokens = await asyncio.gather(
            *[
                provider.refresh_access_token(MEMBER_ID, "old-access-secret")
                for _ in range(8)
            ]
        )

        self.assertEqual(tokens, ["new-access-secret"] * 8)
        self.assertEqual(transport.calls, ["old-refresh-secret"])
        self.assertEqual(len(store.replace_calls), 1)
        self.assertEqual(store.current.revision, 2)

    async def test_missing_or_rejected_installation_uses_safe_errors(self):
        missing = BitrixOAuthCredentialProvider(
            MemoryOAuthStore(),
            CountingRefreshTransport(None),
        )
        with self.assertRaises(BitrixOAuthAccessError) as raised:
            await missing.get_access_token(MEMBER_ID)
        self.assertEqual(raised.exception.error_code, "oauth_installation_not_found")
        self.assertNotIn(MEMBER_ID, repr(raised.exception))

        rejected = BitrixOAuthCredentialProvider(
            MemoryOAuthStore(installation()),
            CountingRefreshTransport(
                OAuthRefreshResult(
                    OAuthRefreshDecision.FAIL,
                    error_code="oauth_refresh_rejected",
                )
            ),
        )
        with self.assertRaises(BitrixOAuthAccessError) as raised:
            await rejected.refresh_access_token(MEMBER_ID, "old-access-secret")
        self.assertFalse(raised.exception.retryable)
        self.assertNotIn("old-access-secret", repr(raised.exception))
        self.assertNotIn("old-refresh-secret", repr(raised.exception))


class BitrixClientOAuthIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_expired_token_is_rotated_once_and_original_request_is_retried(self):
        store = MemoryOAuthStore(installation())
        transport = CountingRefreshTransport(
            OAuthRefreshResult(
                OAuthRefreshDecision.SUCCESS,
                token_pair=renewed_pair(),
            )
        )
        provider = BitrixOAuthCredentialProvider(store, transport)
        calls = []

        async def handler(request):
            payload = __import__("json").loads(request.content)
            calls.append(payload)
            if payload["auth"] == "old-access-secret":
                return httpx.Response(
                    401,
                    json={"error": "expired_token"},
                )
            return httpx.Response(200, json={"result": {"id": 987}})

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(http_client.aclose)
        client = BitrixClient(
            portal_url="https://portal.bitrix24.test",
            token_provider=provider,
            member_id=MEMBER_ID,
            timeout_seconds=3,
            http_client=http_client,
        )
        payload = BitrixMessagePayloadPreview(
            botId=456,
            dialogId="chat5",
            fields=BitrixMessageFieldsPreview(message="Respuesta aprobada"),
        )

        result = await client.send_approved_message(payload)

        self.assertEqual(result.decision, BitrixClientDecision.SUCCESS)
        self.assertEqual(result.response.result.id, 987)
        self.assertEqual(
            [call["auth"] for call in calls],
            ["old-access-secret", "new-access-secret"],
        )
        self.assertEqual(transport.calls, ["old-refresh-secret"])
        self.assertNotIn("old-access-secret", repr(result))
        self.assertNotIn("new-access-secret", repr(result))


if __name__ == "__main__":
    unittest.main()

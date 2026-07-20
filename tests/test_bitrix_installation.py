import unittest
from dataclasses import replace
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bitrix_connector.config import ConnectorMode, load_settings
from bitrix_connector.installation import (
    BitrixOAuthCallbackVerifier,
    BitrixOAuthInstallationService,
    OAuthInstallationPersistenceError,
    OAuthInstallationResult,
    OAuthInstallationStatus,
    OAuthInstallationVerificationError,
    OAuthTokenVerificationDecision,
    OAuthTokenVerificationResult,
)
from bitrix_connector.installation_router import create_installation_router
from bitrix_connector.oauth import (
    OAuthInstallationWriteResult,
    OAuthInstallationWriteStatus,
)


NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)


def settings(*, locked=False):
    return replace(
        load_settings(
            {
                "NIA_BITRIX_DOMAIN": "portal.bitrix24.test",
                "NIA_BITRIX_MEMBER_ID": "member-123",
                "NIA_BITRIX_INSTALLATION_ENABLED": "true",
            }
        ),
        effective_mode=ConnectorMode.REVIEW,
        activation_locked=locked,
    )


def installation_form(**overrides):
    form = {
        "event": "ONAPPINSTALL",
        "auth[domain]": "portal.bitrix24.test",
        "auth[member_id]": "member-123",
        "auth[client_endpoint]": "https://portal.bitrix24.test/rest/",
        "auth[server_endpoint]": "https://oauth.bitrix.info/rest/",
        "auth[access_token]": "access-secret",
        "auth[refresh_token]": "refresh-secret",
        "auth[application_token]": "application-secret",
        "auth[expires]": str(int(NOW.timestamp()) + 3600),
    }
    form.update(overrides)
    return form


class InstallationStore:
    def __init__(self, *, status=OAuthInstallationWriteStatus.CREATED):
        self.status = status
        self.installations = []

    async def save_installation(self, installation):
        self.installations.append(installation)
        revision = 2 if self.status is OAuthInstallationWriteStatus.REINSTALLED else 1
        return OAuthInstallationWriteResult(
            status=self.status,
            revision=revision,
        )


class FailingInstallationStore(InstallationStore):
    async def save_installation(self, installation):
        raise RuntimeError("mongodb://user:secret@private.invalid")


class CallbackVerifier:
    def __init__(self, result=None):
        self.result = result or OAuthTokenVerificationResult(
            OAuthTokenVerificationDecision.VERIFIED,
            "installation_access_token_verified",
        )
        self.calls = []

    async def verify(self, *, client_endpoint, access_token):
        self.calls.append((client_endpoint, access_token))
        return self.result


class InstallationServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_valid_callback_is_stored_without_exposing_secrets(self):
        store = InstallationStore()
        service = BitrixOAuthInstallationService(
            store,
            CallbackVerifier(),
            clock=lambda: NOW,
        )

        result = await service.install(installation_form(), settings())

        self.assertEqual(result.status, OAuthInstallationStatus.INSTALLED)
        self.assertTrue(result.persisted)
        self.assertEqual(result.revision, 1)
        self.assertEqual(len(store.installations), 1)
        saved = store.installations[0]
        self.assertEqual(saved.domain, "portal.bitrix24.test")
        self.assertEqual(saved.member_id, "member-123")
        serialized = result.model_dump_json()
        for secret in (
            "access-secret",
            "refresh-secret",
            "application-secret",
        ):
            self.assertNotIn(secret, serialized)
            self.assertNotIn(secret, repr(saved))

    async def test_idempotent_and_reinstalled_results_remain_safe(self):
        for write_status, expected in (
            (
                OAuthInstallationWriteStatus.IDEMPOTENT,
                OAuthInstallationStatus.IDEMPOTENT,
            ),
            (
                OAuthInstallationWriteStatus.REINSTALLED,
                OAuthInstallationStatus.REINSTALLED,
            ),
        ):
            with self.subTest(write_status=write_status):
                service = BitrixOAuthInstallationService(
                    InstallationStore(status=write_status),
                    CallbackVerifier(),
                    clock=lambda: NOW,
                )
                result = await service.install(installation_form(), settings())
                self.assertEqual(result.status, expected)
                self.assertTrue(result.persisted)
                self.assertNotIn("secret", result.model_dump_json())

    async def test_invalid_or_other_installation_never_reaches_store(self):
        cases = (
            (
                {"auth[refresh_token]": ""},
                OAuthInstallationStatus.INVALID,
                "invalid_installation_payload",
            ),
            (
                {"auth[domain]": "other.bitrix24.test"},
                OAuthInstallationStatus.REJECTED,
                "installation_domain_mismatch",
            ),
            (
                {"auth[member_id]": "member-other"},
                OAuthInstallationStatus.REJECTED,
                "installation_member_id_mismatch",
            ),
            (
                {
                    "auth[client_endpoint]":
                    "https://other.bitrix24.test/rest/"
                },
                OAuthInstallationStatus.REJECTED,
                "installation_client_endpoint_mismatch",
            ),
            (
                {
                    "auth[server_endpoint]":
                    "https://attacker.invalid/rest/"
                },
                OAuthInstallationStatus.REJECTED,
                "installation_server_endpoint_mismatch",
            ),
        )
        for overrides, expected_status, reason in cases:
            with self.subTest(reason=reason):
                store = InstallationStore()
                service = BitrixOAuthInstallationService(
                    store,
                    CallbackVerifier(),
                    clock=lambda: NOW,
                )
                result = await service.install(
                    installation_form(**overrides),
                    settings(),
                )
                self.assertEqual(result.status, expected_status)
                self.assertEqual(result.reason, reason)
                self.assertEqual(store.installations, [])

    async def test_storage_failure_is_safe_and_retryable_at_http_boundary(self):
        service = BitrixOAuthInstallationService(
            FailingInstallationStore(),
            CallbackVerifier(),
            clock=lambda: NOW,
        )
        with self.assertRaises(OAuthInstallationPersistenceError) as raised:
            await service.install(installation_form(), settings())
        self.assertNotIn("secret", repr(raised.exception))
        self.assertNotIn("private.invalid", repr(raised.exception))

    async def test_token_is_verified_before_credentials_are_stored(self):
        store = InstallationStore()
        verifier = CallbackVerifier(
            OAuthTokenVerificationResult(
                OAuthTokenVerificationDecision.REJECTED,
                "installation_access_token_rejected",
            )
        )
        service = BitrixOAuthInstallationService(store, verifier)

        result = await service.install(installation_form(), settings())

        self.assertEqual(result.status, OAuthInstallationStatus.REJECTED)
        self.assertEqual(result.reason, "installation_access_token_rejected")
        self.assertEqual(store.installations, [])
        self.assertEqual(
            verifier.calls,
            [("https://portal.bitrix24.test/rest/", "access-secret")],
        )

    async def test_retryable_verification_never_reaches_store(self):
        store = InstallationStore()
        verifier = CallbackVerifier(
            OAuthTokenVerificationResult(
                OAuthTokenVerificationDecision.RETRY,
                "installation_verification_timeout",
            )
        )
        service = BitrixOAuthInstallationService(store, verifier)

        with self.assertRaises(OAuthInstallationVerificationError):
            await service.install(installation_form(), settings())
        self.assertEqual(store.installations, [])


class OAuthCallbackVerifierTests(unittest.IsolatedAsyncioTestCase):
    async def make_verifier(self, handler):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(client.aclose)
        return BitrixOAuthCallbackVerifier(http_client=client)

    async def test_app_info_success_verifies_token_without_exposing_it(self):
        captured = {}

        async def handler(request):
            captured["url"] = str(request.url)
            captured["body"] = request.content.decode()
            return httpx.Response(
                200,
                json={"result": {"INSTALLED": True, "VERSION": 1}},
            )

        verifier = await self.make_verifier(handler)
        result = await verifier.verify(
            client_endpoint="https://portal.bitrix24.test/rest/",
            access_token="access-secret",
        )

        self.assertEqual(
            result.decision,
            OAuthTokenVerificationDecision.VERIFIED,
        )
        self.assertEqual(
            captured["url"],
            "https://portal.bitrix24.test/rest/app.info",
        )
        self.assertIn("access-secret", captured["body"])
        self.assertNotIn("access-secret", repr(result))

    async def test_app_info_errors_are_safe_and_classified(self):
        cases = (
            (
                httpx.Response(401, json={"error": "expired_token"}),
                OAuthTokenVerificationDecision.REJECTED,
            ),
            (
                httpx.Response(429, text="private", headers={"Retry-After": "8"}),
                OAuthTokenVerificationDecision.RETRY,
            ),
            (
                httpx.Response(503, text="private"),
                OAuthTokenVerificationDecision.RETRY,
            ),
            (
                httpx.Response(200, json={"result": []}),
                OAuthTokenVerificationDecision.REJECTED,
            ),
        )
        for response, decision in cases:
            with self.subTest(status=response.status_code):
                async def handler(request, current=response):
                    return current

                verifier = await self.make_verifier(handler)
                result = await verifier.verify(
                    client_endpoint="https://portal.bitrix24.test/rest/",
                    access_token="access-secret",
                )
                self.assertEqual(result.decision, decision)
                self.assertNotIn("access-secret", repr(result))
                self.assertNotIn("private", repr(result))


class RecordingInstaller:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    async def install(self, raw_form, connector_settings):
        self.calls.append((raw_form, connector_settings))
        if self.error is not None:
            raise self.error
        return self.result


def test_app(installer, connector_settings):
    app = FastAPI()
    app.include_router(
        create_installation_router(
            installer,
            settings_loader=lambda: connector_settings,
        ),
        prefix="/bitrix-connector",
    )
    return app


class InstallationRouterTests(unittest.TestCase):
    def test_route_is_independent_from_conversation_activation_lock(self):
        installer = RecordingInstaller()
        installer.result = OAuthInstallationResult(
            status=OAuthInstallationStatus.INSTALLED,
            reason="installation_stored",
            persisted=True,
            revision=1,
        )
        response = TestClient(
            test_app(installer, settings(locked=True))
        ).post(
            "/bitrix-connector/installation",
            data=installation_form(),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(installer.calls), 1)

    def test_disabled_installation_window_rejects_before_parsing(self):
        installer = RecordingInstaller()
        disabled = replace(settings(), installation_enabled=False)
        response = TestClient(test_app(installer, disabled)).post(
            "/bitrix-connector/installation",
            data={"auth[access_token]": "attacker-secret"},
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["reason"], "installation_route_disabled")
        self.assertEqual(installer.calls, [])
        self.assertNotIn("attacker-secret", response.text)

    def test_unlocked_route_persists_and_returns_only_safe_summary(self):
        installer = RecordingInstaller(
            result=OAuthInstallationResult(
                status=OAuthInstallationStatus.INSTALLED,
                reason="installation_stored",
                persisted=True,
                revision=1,
            )
        )
        response = TestClient(test_app(installer, settings())).post(
            "/bitrix-connector/installation",
            data=installation_form(),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "installed",
                "reason": "installation_stored",
                "persisted": True,
                "revision": 1,
            },
        )
        self.assertEqual(len(installer.calls), 1)
        for secret in (
            "access-secret",
            "refresh-secret",
            "application-secret",
        ):
            self.assertNotIn(secret, response.text)

    def test_callback_log_contains_only_safe_outcome(self):
        installer = RecordingInstaller(
            result=OAuthInstallationResult(
                status=OAuthInstallationStatus.INSTALLED,
                reason="installation_stored",
                persisted=True,
                revision=1,
            )
        )
        with self.assertLogs(
            "nia.bitrix_connector.installation",
            level="INFO",
        ) as captured:
            response = TestClient(test_app(installer, settings())).post(
                "/bitrix-connector/installation",
                data=installation_form(),
            )

        log_output = "\n".join(captured.output)
        self.assertEqual(response.status_code, 200)
        self.assertIn("status=installed", log_output)
        self.assertIn("reason=installation_stored", log_output)
        self.assertIn("persisted=True", log_output)
        for secret in (
            "access-secret",
            "refresh-secret",
            "application-secret",
        ):
            self.assertNotIn(secret, log_output)

    def test_invalid_rejected_and_storage_failure_have_stable_statuses(self):
        scenarios = (
            (
                OAuthInstallationResult(
                    status=OAuthInstallationStatus.INVALID,
                    reason="invalid_installation_payload",
                ),
                None,
                400,
            ),
            (
                OAuthInstallationResult(
                    status=OAuthInstallationStatus.REJECTED,
                    reason="installation_domain_mismatch",
                ),
                None,
                403,
            ),
            (
                None,
                OAuthInstallationPersistenceError(
                    "oauth_installation_storage_unavailable"
                ),
                503,
            ),
        )
        for result, error, expected_status in scenarios:
            with self.subTest(expected_status=expected_status):
                installer = RecordingInstaller(result=result, error=error)
                response = TestClient(test_app(installer, settings())).post(
                    "/bitrix-connector/installation",
                    data=installation_form(),
                )
                self.assertEqual(response.status_code, expected_status)
                self.assertNotIn("access-secret", response.text)
                self.assertNotIn("refresh-secret", response.text)


if __name__ == "__main__":
    unittest.main()

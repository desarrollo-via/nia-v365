import os
import importlib
import unittest
from dataclasses import replace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bitrix_connector.config import ConnectorMode, load_settings
from bitrix_connector.event_parser import parse_webhook_form
from bitrix_connector.idempotency import build_event_key
from bitrix_connector.router import router
from bitrix_connector.models import StorageWriteResult
from bitrix_connector.runtime import ConnectorRuntime, ConnectorRuntimeState
from bitrix_connector.security import redact_form_data, validate_webhook_identity
from bitrix_connector.storage import ConnectorStorageFactory, ConnectorStorageResources


connector_router_module = importlib.import_module("bitrix_connector.router")


class RuntimeTestClient:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class RuntimeTestStore:
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


class FailingRuntimeTestStore(RuntimeTestStore):
    async def save_received(self, record):
        raise RuntimeError("mongodb://user:secret@internal.invalid")


class RuntimeTestFactory:
    def __init__(self, resources):
        self.resources = resources

    async def build(self, settings):
        return self.resources


class ConnectorConfigTests(unittest.TestCase):
    def test_default_is_locked_off(self):
        settings = load_settings({})

        self.assertEqual(settings.requested_mode, "off")
        self.assertEqual(settings.effective_mode, ConnectorMode.OFF)
        self.assertTrue(settings.activation_locked)
        self.assertFalse(settings.external_calls_enabled)

    def test_requested_active_remains_effectively_off(self):
        settings = load_settings({"NIA_BITRIX_MODE": "active"})

        self.assertEqual(settings.requested_mode, "active")
        self.assertEqual(settings.effective_mode, ConnectorMode.OFF)
        self.assertIn("activation_locked_by_skeleton", settings.warnings)

    def test_invalid_mode_falls_back_to_effective_off(self):
        settings = load_settings({"NIA_BITRIX_MODE": "invalid"})

        self.assertEqual(settings.effective_mode, ConnectorMode.OFF)
        self.assertIn("invalid_requested_mode", settings.warnings)


class EventParserTests(unittest.TestCase):
    def setUp(self):
        self.form = {
            "event": "ONIMBOTV2MESSAGEADD",
            "ts": "1772093963",
            "data[bot][id]": "456",
            "data[bot][code]": "nia_bitrix",
            "data[message][id]": "789",
            "data[message][uuid]": "uuid-789",
            "data[message][chatId]": "5",
            "data[message][authorId]": "27",
            "data[message][text]": " Necesito una bomba ",
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
        }

    def test_parses_flat_php_style_form(self):
        event = parse_webhook_form(self.form)

        self.assertEqual(event.event, "ONIMBOTV2MESSAGEADD")
        self.assertEqual(event.bot_id, 456)
        self.assertEqual(event.message_id, 789)
        self.assertEqual(event.dialog_id, "chat5")
        self.assertEqual(event.text, "Necesito una bomba")
        self.assertFalse(event.is_system)
        self.assertTrue(event.user_is_connector)
        self.assertEqual(event.application_token.get_secret_value(), "secret-token")

    def test_redacts_top_level_and_nested_tokens(self):
        redacted = redact_form_data(self.form)

        self.assertEqual(redacted["auth[application_token]"], "[REDACTED]")
        self.assertEqual(redacted["data[bot][auth][access_token]"], "[REDACTED]")
        self.assertEqual(redacted["auth[domain]"], "viaindustrial.bitrix24.es")

    def test_validates_installed_identity(self):
        event = parse_webhook_form(self.form)
        settings = load_settings({
            "NIA_BITRIX_DOMAIN": "https://viaindustrial.bitrix24.es/",
            "NIA_BITRIX_MEMBER_ID": "member-123",
            "NIA_BITRIX_APPLICATION_TOKEN": "secret-token",
        })

        decision = validate_webhook_identity(event, settings)

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reason, "identity_verified")

    def test_event_key_is_deterministic_without_claiming_persistence(self):
        first = parse_webhook_form(self.form)
        second = parse_webhook_form(dict(self.form))
        changed = dict(self.form)
        changed["data[message][id]"] = "790"

        self.assertEqual(build_event_key(first), build_event_key(second))
        self.assertNotEqual(build_event_key(first), build_event_key(parse_webhook_form(changed)))


class ConnectorRouterTests(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(router)
        self.client = TestClient(self.app)
        self.env = {
            "NIA_BITRIX_MODE": "active",
            "NIA_BITRIX_DOMAIN": "viaindustrial.bitrix24.es",
            "NIA_BITRIX_MEMBER_ID": "member-123",
            "NIA_BITRIX_APPLICATION_TOKEN": "secret-token",
            "NIA_BITRIX_REVIEW_TOKEN": "review-secret",
            "MONGO_URI": "mongodb://mongo-user:mongo-password@example.invalid/nia",
        }
        self.form = {
            "event": "ONIMBOTV2MESSAGEADD",
            "ts": "1772093963",
            "data[bot][id]": "456",
            "data[bot][code]": "nia_bitrix",
            "data[message][id]": "789",
            "data[message][uuid]": "uuid-789",
            "data[message][chatId]": "5",
            "data[message][authorId]": "27",
            "data[message][text]": "Necesito una bomba",
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
        }

    def test_health_is_inert_and_does_not_expose_secrets(self):
        with patch.dict(os.environ, self.env, clear=True):
            with TestClient(self.app) as client:
                response = client.get("/bitrix-connector/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["effective_mode"], "off")
        self.assertTrue(payload["activation_locked"])
        self.assertFalse(payload["external_calls_enabled"])
        self.assertEqual(payload["runtime_state"], "inert")
        self.assertFalse(payload["runtime_service_available"])
        self.assertFalse(payload["runtime_resources_available"])
        self.assertTrue(all(payload["configured"].values()))
        self.assertEqual(
            payload["pilot"],
            {
                "enabled": False,
                "emergency_stop": True,
                "rule_count": 0,
                "configuration_valid": True,
            },
        )
        self.assertNotIn("secret-token", response.text)
        self.assertNotIn("review-secret", response.text)
        self.assertNotIn("mongo-password", response.text)
        self.assertNotIn("MONGO_URI", response.text)

    def test_mounted_review_route_is_protected_and_runtime_gated(self):
        with patch.dict(os.environ, self.env, clear=True):
            unauthorized = self.client.get("/bitrix-connector/review")
            authorized = self.client.get(
                "/bitrix-connector/review",
                headers={"Authorization": "Bearer review-secret"},
            )

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(authorized.status_code, 503)
        self.assertEqual(
            authorized.json()["detail"],
            "connector_runtime_not_ready",
        )
        self.assertNotIn("review-secret", authorized.text)

    def test_mounted_installation_route_is_disabled_by_default(self):
        with patch.dict(os.environ, self.env, clear=True):
            response = self.client.post(
                "/bitrix-connector/installation",
                data={"auth[access_token]": "must-not-be-parsed"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["reason"],
            "installation_route_disabled",
        )
        self.assertNotIn("must-not-be-parsed", response.text)

    def test_router_lifecycle_off_never_creates_client_and_closes_runtime(self):
        client_calls = []

        def forbidden_client_factory(uri):
            client_calls.append(uri)
            raise AssertionError("client must not be created")

        runtime = ConnectorRuntime(ConnectorStorageFactory(forbidden_client_factory))
        app = FastAPI()
        app.include_router(router)

        with patch.object(connector_router_module, "connector_runtime", runtime):
            with patch.dict(os.environ, self.env, clear=True):
                with TestClient(app) as client:
                    response = client.get("/bitrix-connector/health")
                    self.assertEqual(response.json()["runtime_state"], "inert")
                    self.assertEqual(runtime.snapshot.state, ConnectorRuntimeState.INERT)

        self.assertEqual(client_calls, [])
        self.assertEqual(runtime.snapshot.state, ConnectorRuntimeState.STOPPED)

    def test_ready_runtime_stores_then_detects_duplicate(self):
        runtime_client = RuntimeTestClient()
        store = RuntimeTestStore()
        runtime = ConnectorRuntime(
            RuntimeTestFactory(
                ConnectorStorageResources(client=runtime_client, store=store)
            )
        )
        unlocked = replace(
            load_settings(self.env),
            requested_mode="review",
            effective_mode=ConnectorMode.REVIEW,
            activation_locked=False,
        )
        app = FastAPI()
        app.include_router(router)

        with patch.object(connector_router_module, "connector_runtime", runtime):
            with patch.object(
                connector_router_module,
                "load_settings",
                return_value=unlocked,
            ):
                with TestClient(app) as client:
                    first = client.post("/bitrix-connector/webhook", data=self.form)
                    second = client.post("/bitrix-connector/webhook", data=self.form)
                    health = client.get("/bitrix-connector/health")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()["status"], "stored")
        self.assertEqual(first.json()["reason"], "event_stored")
        self.assertEqual(first.json()["duplicate_detection"], "unique_created")
        self.assertTrue(first.json()["persisted"])
        self.assertFalse(first.json()["nia_called"])
        self.assertFalse(first.json()["bitrix_written"])
        self.assertEqual(second.json()["status"], "duplicate")
        self.assertEqual(second.json()["reason"], "event_already_received")
        self.assertEqual(second.json()["duplicate_detection"], "duplicate")
        self.assertTrue(second.json()["persisted"])
        self.assertEqual(first.json()["event_key"], second.json()["event_key"])
        self.assertEqual(health.json()["runtime_state"], "ready")
        self.assertEqual(len(store.records), 2)
        self.assertEqual(runtime_client.close_calls, 1)
        self.assertNotIn("secret-token", first.text)
        self.assertNotIn("oauth-secret", first.text)

    def test_unlocked_but_inert_runtime_does_not_persist(self):
        runtime = ConnectorRuntime(RuntimeTestFactory(None))
        unlocked = replace(
            load_settings(self.env),
            requested_mode="review",
            effective_mode=ConnectorMode.REVIEW,
            activation_locked=False,
        )
        app = FastAPI()
        app.include_router(router)

        with patch.object(connector_router_module, "connector_runtime", runtime):
            with patch.object(
                connector_router_module,
                "load_settings",
                return_value=unlocked,
            ):
                with TestClient(app) as client:
                    response = client.post(
                        "/bitrix-connector/webhook",
                        data=self.form,
                    )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "disabled")
        self.assertEqual(response.json()["reason"], "connector_runtime_not_ready")
        self.assertEqual(response.json()["duplicate_detection"], "not_persisted")
        self.assertFalse(response.json()["persisted"])

    def test_storage_failure_returns_safe_retryable_503(self):
        runtime_client = RuntimeTestClient()
        runtime = ConnectorRuntime(
            RuntimeTestFactory(
                ConnectorStorageResources(
                    client=runtime_client,
                    store=FailingRuntimeTestStore(),
                )
            )
        )
        unlocked = replace(
            load_settings(self.env),
            requested_mode="review",
            effective_mode=ConnectorMode.REVIEW,
            activation_locked=False,
        )
        app = FastAPI()
        app.include_router(router)

        with patch.object(connector_router_module, "connector_runtime", runtime):
            with patch.object(
                connector_router_module,
                "load_settings",
                return_value=unlocked,
            ):
                with TestClient(app) as client:
                    response = client.post(
                        "/bitrix-connector/webhook",
                        data=self.form,
                    )

        payload = response.json()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers["retry-after"], "5")
        self.assertEqual(payload["status"], "retryable_error")
        self.assertEqual(payload["reason"], "connector_storage_unavailable")
        self.assertEqual(payload["duplicate_detection"], "not_persisted")
        self.assertFalse(payload["persisted"])
        self.assertFalse(payload["nia_called"])
        self.assertFalse(payload["bitrix_written"])
        self.assertEqual(payload["event_key"], build_event_key(parse_webhook_form(self.form)))
        self.assertEqual(runtime_client.close_calls, 1)
        self.assertNotIn("user:secret", response.text)
        self.assertNotIn("internal.invalid", response.text)

    def test_valid_webhook_is_verified_but_remains_disabled(self):
        with patch.dict(os.environ, self.env, clear=True):
            response = self.client.post("/bitrix-connector/webhook", data=self.form)

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "disabled")
        self.assertEqual(payload["reason"], "connector_locked_off")
        self.assertEqual(payload["effective_mode"], "off")
        self.assertTrue(payload["identity_verified"])
        self.assertEqual(payload["duplicate_detection"], "not_persisted")
        self.assertFalse(payload["persisted"])
        self.assertFalse(payload["nia_called"])
        self.assertFalse(payload["bitrix_written"])
        self.assertNotIn("secret-token", response.text)
        self.assertNotIn("oauth-secret", response.text)

    def test_repeated_webhook_returns_same_key_without_persisting(self):
        with patch.dict(os.environ, self.env, clear=True):
            first = self.client.post("/bitrix-connector/webhook", data=self.form).json()
            second = self.client.post("/bitrix-connector/webhook", data=self.form).json()

        self.assertEqual(first["event_key"], second["event_key"])
        self.assertFalse(first["persisted"])
        self.assertEqual(first["duplicate_detection"], "not_persisted")

    def test_invalid_token_is_ignored_without_echoing_it(self):
        invalid = dict(self.form)
        invalid["auth[application_token]"] = "attacker-token"

        with patch.dict(os.environ, self.env, clear=True):
            response = self.client.post("/bitrix-connector/webhook", data=invalid)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ignored")
        self.assertEqual(response.json()["reason"], "application_token_mismatch")
        self.assertNotIn("attacker-token", response.text)

    def test_missing_event_token_is_ignored(self):
        missing = dict(self.form)
        missing.pop("auth[application_token]")

        with patch.dict(os.environ, self.env, clear=True):
            response = self.client.post("/bitrix-connector/webhook", data=missing)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ignored")
        self.assertEqual(response.json()["reason"], "application_token_mismatch")

    def test_missing_configured_token_is_ignored(self):
        incomplete_env = dict(self.env)
        incomplete_env.pop("NIA_BITRIX_APPLICATION_TOKEN")

        with patch.dict(os.environ, incomplete_env, clear=True):
            response = self.client.post("/bitrix-connector/webhook", data=self.form)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ignored")
        self.assertEqual(response.json()["reason"], "application_token_not_configured")

    def test_unsupported_event_is_ignored(self):
        unsupported = dict(self.form)
        unsupported["event"] = "ONIMBOTV2MESSAGEUPDATE"

        with patch.dict(os.environ, self.env, clear=True):
            response = self.client.post("/bitrix-connector/webhook", data=unsupported)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ignored")
        self.assertEqual(response.json()["reason"], "unsupported_event")

    def test_missing_required_identity_is_invalid_and_never_echoes_secrets(self):
        malformed = dict(self.form)
        malformed.pop("data[message][id]")

        with patch.dict(os.environ, self.env, clear=True):
            response = self.client.post("/bitrix-connector/webhook", data=malformed)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["status"], "invalid")
        self.assertFalse(response.json()["persisted"])
        self.assertNotIn("secret-token", response.text)


if __name__ == "__main__":
    unittest.main()

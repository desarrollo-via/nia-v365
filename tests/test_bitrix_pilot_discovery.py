import json
import unittest

import httpx
from pydantic import ValidationError

from bitrix_connector.pilot_discovery import (
    BitrixCrmChat,
    BitrixPilotDiscoveryClient,
    PilotChatInspector,
    PilotCrmEntityType,
    PilotDiscoveryRequest,
    PilotDiscoveryStatus,
    PilotDiscoveryTransportDecision,
    PilotDiscoveryTransportResult,
)


def discovery_request(**overrides):
    values = {
        "crm_entity_type": "deal",
        "crm_entity_id": 663001,
        "member_id": "member-123",
        "bot_id": 456,
        "active_only": False,
    }
    values.update(overrides)
    return PilotDiscoveryRequest(**values)


class FakeDiscoveryTransport:
    def __init__(self, result):
        self.result = result
        self.requests = []

    async def get_crm_chats(self, request):
        self.requests.append(request)
        return self.result


class BitrixPilotDiscoveryClientTests(unittest.IsolatedAsyncioTestCase):
    async def make_client(self, handler):
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(http_client.aclose)
        return BitrixPilotDiscoveryClient(
            portal_url="https://portal.bitrix24.test/",
            access_token="oauth-secret-token",
            timeout_seconds=3,
            http_client=http_client,
        )

    async def test_uses_only_documented_read_endpoint_and_exact_payload(self):
        captured = {}

        async def handler(request):
            captured["method"] = request.method
            captured["url"] = str(request.url)
            captured["json"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "result": [
                        {
                            "CHAT_ID": "1763",
                            "CONNECTOR_ID": "whatsapp",
                            "CONNECTOR_TITLE": "WhatsApp",
                            "ignored": "extra",
                        }
                    ]
                },
            )

        client = await self.make_client(handler)
        result = await client.get_crm_chats(discovery_request())

        self.assertEqual(
            result.decision,
            PilotDiscoveryTransportDecision.SUCCESS,
        )
        self.assertEqual(result.chats[0].chat_id, 1763)
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(
            captured["url"],
            "https://portal.bitrix24.test/rest/imopenlines.crm.chat.get",
        )
        self.assertEqual(
            captured["json"],
            {
                "CRM_ENTITY_TYPE": "deal",
                "CRM_ENTITY": 663001,
                "ACTIVE_ONLY": "N",
                "auth": "oauth-secret-token",
            },
        )

    async def test_all_documented_crm_types_and_active_filter_are_supported(self):
        captured = []

        async def handler(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json={"result": []})

        client = await self.make_client(handler)
        for entity_type in PilotCrmEntityType:
            await client.get_crm_chats(
                discovery_request(
                    crm_entity_type=entity_type,
                    active_only=True,
                )
            )

        self.assertEqual(
            [item["CRM_ENTITY_TYPE"] for item in captured],
            ["lead", "deal", "contact", "company"],
        )
        self.assertTrue(
            all(item["ACTIVE_ONLY"] == "Y" for item in captured)
        )

    async def test_remote_failures_are_classified_and_redacted(self):
        cases = (
            (
                httpx.Response(
                    429,
                    headers={"Retry-After": "12"},
                    text="cliente privado",
                ),
                PilotDiscoveryTransportDecision.RETRY,
                "pilot_discovery_rate_limited",
                12,
            ),
            (
                httpx.Response(
                    401,
                    json={
                        "error": "expired_token",
                        "error_description": "oauth-secret-token",
                    },
                ),
                PilotDiscoveryTransportDecision.RETRY,
                "pilot_discovery_token_expired",
                0,
            ),
            (
                httpx.Response(
                    403,
                    json={
                        "error": "ACCESS_DENIED",
                        "error_description": "negociación privada",
                    },
                ),
                PilotDiscoveryTransportDecision.FAIL,
                "pilot_discovery_api_permanent",
                0,
            ),
            (
                httpx.Response(503, text="infraestructura interna"),
                PilotDiscoveryTransportDecision.RETRY,
                "pilot_discovery_api_retryable",
                0,
            ),
        )
        for response, decision, code, retry_after in cases:
            with self.subTest(code=code):
                async def handler(request, current=response):
                    return current

                client = await self.make_client(handler)
                result = await client.get_crm_chats(discovery_request())
                self.assertEqual(result.decision, decision)
                self.assertEqual(result.error_code, code)
                self.assertEqual(
                    result.retry_after_seconds,
                    retry_after,
                )
                representation = repr(result)
                self.assertNotIn("oauth-secret-token", representation)
                self.assertNotIn("privada", representation)
                self.assertNotIn("interna", representation)

    async def test_transport_exceptions_are_retryable_and_redacted(self):
        exceptions = (
            httpx.ReadTimeout("oauth-secret-token"),
            httpx.ConnectError("https://private.example/path"),
        )
        expected = (
            "pilot_discovery_timeout",
            "pilot_discovery_transport_error",
        )
        for exception, code in zip(exceptions, expected):
            with self.subTest(code=code):
                async def handler(request, current=exception):
                    current.request = request
                    raise current

                client = await self.make_client(handler)
                result = await client.get_crm_chats(discovery_request())
                self.assertEqual(
                    result.decision,
                    PilotDiscoveryTransportDecision.RETRY,
                )
                self.assertEqual(result.error_code, code)
                self.assertNotIn("oauth-secret-token", repr(result))
                self.assertNotIn("private.example", repr(result))

    async def test_malformed_success_fails_closed(self):
        responses = (
            httpx.Response(200, text="oauth-secret-token"),
            httpx.Response(200, json={"result": {}}),
            httpx.Response(
                200,
                json={
                    "result": [
                        {
                            "CHAT_ID": "not-an-id",
                            "CONNECTOR_ID": "whatsapp",
                            "CONNECTOR_TITLE": "WhatsApp",
                        }
                    ]
                },
            ),
        )
        for response in responses:
            async def handler(request, current=response):
                return current

            client = await self.make_client(handler)
            result = await client.get_crm_chats(discovery_request())
            self.assertEqual(
                result.decision,
                PilotDiscoveryTransportDecision.FAIL,
            )
            self.assertEqual(
                result.error_code,
                "pilot_discovery_invalid_response",
            )
            self.assertNotIn("oauth-secret-token", repr(result))

    def test_configuration_and_request_validation_fail_closed(self):
        for values in (
            {"crm_entity_id": 0},
            {"member_id": ""},
            {"bot_id": 0},
            {"crm_entity_type": "invoice"},
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValidationError):
                    discovery_request(**values)

        for values in (
            {"portal_url": "http://portal.bitrix24.test"},
            {"access_token": ""},
            {"timeout_seconds": 0},
        ):
            parameters = {
                "portal_url": "https://portal.bitrix24.test",
                "access_token": "token",
                "timeout_seconds": 3,
            }
            parameters.update(values)
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    BitrixPilotDiscoveryClient(**parameters)


class PilotChatInspectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_builds_exact_safe_candidates_and_deduplicates(self):
        chat = BitrixCrmChat.model_validate(
            {
                "CHAT_ID": "1763",
                "CONNECTOR_ID": "whatsapp",
                "CONNECTOR_TITLE": "WhatsApp",
            }
        )
        transport = FakeDiscoveryTransport(
            PilotDiscoveryTransportResult(
                decision=PilotDiscoveryTransportDecision.SUCCESS,
                chats=(chat, chat),
                http_status=200,
            )
        )

        result = await PilotChatInspector(transport).inspect(
            discovery_request()
        )

        self.assertEqual(result.status, PilotDiscoveryStatus.FOUND)
        self.assertEqual(len(result.candidates), 1)
        candidate = result.candidates[0]
        self.assertEqual(candidate.chat_id, 1763)
        self.assertEqual(candidate.dialog_id, "chat1763")
        self.assertEqual(candidate.crm_entity_type, PilotCrmEntityType.DEAL)
        self.assertEqual(candidate.crm_entity_id, 663001)
        self.assertEqual(
            candidate.pilot_rule.model_dump(mode="json"),
            {
                "member_id": "member-123",
                "bot_id": 456,
                "dialog_id": "chat1763",
                "chat_id": 1763,
                "valid_from": None,
                "valid_until": None,
            },
        )
        serialized = result.model_dump_json()
        self.assertNotIn("oauth", serialized)
        self.assertNotIn("mensaje", serialized)
        self.assertNotIn("telefono", serialized)

    async def test_discovery_without_bot_identity_does_not_invent_rule(self):
        chat = BitrixCrmChat.model_validate(
            {
                "CHAT_ID": "1763",
                "CONNECTOR_ID": "whatsapp",
                "CONNECTOR_TITLE": "WhatsApp",
            }
        )
        transport = FakeDiscoveryTransport(
            PilotDiscoveryTransportResult(
                decision=PilotDiscoveryTransportDecision.SUCCESS,
                chats=(chat,),
                http_status=200,
            )
        )

        result = await PilotChatInspector(transport).inspect(
            discovery_request(member_id=None, bot_id=None)
        )

        self.assertEqual(result.status, PilotDiscoveryStatus.FOUND)
        self.assertIsNone(result.candidates[0].pilot_rule)
        self.assertEqual(result.candidates[0].dialog_id, "chat1763")

    async def test_empty_retry_and_failure_remain_explicit(self):
        cases = (
            (
                PilotDiscoveryTransportResult(
                    decision=PilotDiscoveryTransportDecision.SUCCESS,
                    http_status=200,
                ),
                PilotDiscoveryStatus.EMPTY,
                "pilot_discovery_no_chats",
            ),
            (
                PilotDiscoveryTransportResult(
                    decision=PilotDiscoveryTransportDecision.RETRY,
                    error_code="pilot_discovery_rate_limited",
                    retry_after_seconds=7,
                ),
                PilotDiscoveryStatus.RETRY,
                "pilot_discovery_rate_limited",
            ),
            (
                PilotDiscoveryTransportResult(
                    decision=PilotDiscoveryTransportDecision.FAIL,
                    error_code="pilot_discovery_api_permanent",
                ),
                PilotDiscoveryStatus.FAILED,
                "pilot_discovery_api_permanent",
            ),
        )
        for transport_result, status, reason in cases:
            with self.subTest(status=status):
                result = await PilotChatInspector(
                    FakeDiscoveryTransport(transport_result)
                ).inspect(discovery_request())
                self.assertEqual(result.status, status)
                self.assertEqual(result.reason, reason)

    def test_public_surface_has_no_mutating_bitrix_operation(self):
        public_methods = {
            name
            for name in dir(BitrixPilotDiscoveryClient)
            if not name.startswith("_")
        }
        self.assertEqual(
            public_methods,
            {"close", "get_crm_chats"},
        )

    def test_optional_rule_identity_must_be_complete(self):
        for values in (
            {"member_id": "member-123", "bot_id": None},
            {"member_id": None, "bot_id": 456},
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValidationError):
                    discovery_request(**values)


if __name__ == "__main__":
    unittest.main()

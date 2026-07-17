import json
import unittest

import httpx

from bitrix_connector.bitrix_client import (
    BitrixClient,
    BitrixClientDecision,
    BitrixClientResult,
)
from bitrix_connector.output_review import (
    BitrixMessageFieldsPreview,
    BitrixMessagePayloadPreview,
)


def approved_payload(message="Respuesta aprobada"):
    return BitrixMessagePayloadPreview(
        botId=456,
        dialogId="chat5",
        fields=BitrixMessageFieldsPreview(message=message),
    )


class BitrixClientTests(unittest.IsolatedAsyncioTestCase):
    async def make_client(self, handler):
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(http_client.aclose)
        return BitrixClient(
            portal_url="https://portal.bitrix24.test/",
            access_token="oauth-secret-token",
            timeout_seconds=3,
            http_client=http_client,
        )

    async def test_success_uses_official_oauth_endpoint_and_contract(self):
        captured = {}

        async def handler(request):
            captured["method"] = request.method
            captured["url"] = str(request.url)
            captured["json"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "result": {
                        "id": 789,
                        "uuidMap": {"forward-a": 790},
                        "ignored": "extra",
                    },
                    "time": {"duration": 0.1},
                },
            )

        client = await self.make_client(handler)
        result = await client.send_approved_message(approved_payload())

        self.assertEqual(result.decision, BitrixClientDecision.SUCCESS)
        self.assertEqual(result.response.result.id, 789)
        self.assertEqual(result.response.result.uuidMap, {"forward-a": 790})
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(
            captured["url"],
            "https://portal.bitrix24.test/rest/"
            "imbot.v2.Chat.Message.send",
        )
        self.assertEqual(
            captured["json"],
            {
                "botId": 456,
                "dialogId": "chat5",
                "fields": {"message": "Respuesta aprobada"},
                "auth": "oauth-secret-token",
            },
        )
        self.assertNotIn("botToken", captured["json"])

    async def test_invalid_messages_fail_before_transport(self):
        calls = []

        async def handler(request):
            calls.append(request)
            raise AssertionError("no debe existir llamada HTTP")

        client = await self.make_client(handler)
        empty = await client.send_approved_message(approved_payload(" \n "))
        oversized = await client.send_approved_message(
            approved_payload("x" * 20_001)
        )

        self.assertEqual(empty.error_code, "bitrix_empty_message")
        self.assertEqual(oversized.error_code, "bitrix_message_too_long")
        self.assertEqual(calls, [])

    async def test_timeout_is_retryable_and_redacted(self):
        async def handler(request):
            raise httpx.ReadTimeout(
                "oauth-secret-token",
                request=request,
            )

        client = await self.make_client(handler)
        result = await client.send_approved_message(approved_payload())

        self.assertEqual(result.decision, BitrixClientDecision.RETRY)
        self.assertEqual(result.error_code, "bitrix_timeout")
        self.assertNotIn("oauth-secret-token", repr(result))
        self.assertNotIn("Respuesta aprobada", repr(result))

    async def test_transport_error_is_retryable_and_redacted(self):
        async def handler(request):
            raise httpx.ConnectError(
                "https://user:password@private.example/token",
                request=request,
            )

        client = await self.make_client(handler)
        result = await client.send_approved_message(approved_payload())

        self.assertEqual(result.decision, BitrixClientDecision.RETRY)
        self.assertEqual(result.error_code, "bitrix_transport_error")
        self.assertNotIn("password", repr(result))
        self.assertNotIn("private.example", repr(result))

    async def test_http_429_is_retryable_with_bounded_retry_after(self):
        async def handler(request):
            return httpx.Response(
                429,
                headers={"Retry-After": "999999"},
                text="cuerpo sensible",
            )

        client = await self.make_client(handler)
        result = await client.send_approved_message(approved_payload())

        self.assertEqual(result.decision, BitrixClientDecision.RETRY)
        self.assertEqual(result.error_code, "bitrix_rate_limited")
        self.assertEqual(result.retry_after_seconds, 86_400)
        self.assertNotIn("sensible", repr(result))

    async def test_api_rate_code_is_retryable(self):
        async def handler(request):
            return httpx.Response(
                503,
                json={
                    "error": "QUERY_LIMIT_EXCEEDED",
                    "error_description": "detalle sensible",
                },
            )

        client = await self.make_client(handler)
        result = await client.send_approved_message(approved_payload())

        self.assertEqual(result.decision, BitrixClientDecision.RETRY)
        self.assertEqual(result.error_code, "bitrix_rate_limited")
        self.assertNotIn("sensible", repr(result))

    async def test_expired_token_is_separate_retryable_result(self):
        async def handler(request):
            return httpx.Response(
                401,
                json={
                    "error": "expired_token",
                    "error_description": "oauth-secret-token",
                },
            )

        client = await self.make_client(handler)
        result = await client.send_approved_message(approved_payload())

        self.assertEqual(result.decision, BitrixClientDecision.RETRY)
        self.assertEqual(result.error_code, "bitrix_token_expired")
        self.assertEqual(result.http_status, 401)
        self.assertNotIn("oauth-secret-token", repr(result))

    async def test_documented_temporary_errors_are_retryable(self):
        for status, code in (
            (400, "SENDING_FAILED"),
            (500, "INTERNAL_SERVER_ERROR"),
            (500, "ERROR_UNEXPECTED_ANSWER"),
            (503, "OVERLOAD_LIMIT"),
        ):
            with self.subTest(code=code):
                async def handler(request, current_status=status, current=code):
                    return httpx.Response(
                        current_status,
                        json={"error": current, "error_description": "privado"},
                    )

                client = await self.make_client(handler)
                result = await client.send_approved_message(
                    approved_payload()
                )
                self.assertEqual(result.decision, BitrixClientDecision.RETRY)
                self.assertEqual(result.error_code, "bitrix_api_retryable")

    async def test_documented_permanent_errors_are_not_retried(self):
        for status, code in (
            (400, "BOT_NOT_FOUND"),
            (403, "ACCESS_DENIED"),
            (400, "EMPTY_MESSAGE"),
            (403, "insufficient_scope"),
            (500, "PORTAL_DELETED"),
        ):
            with self.subTest(code=code):
                async def handler(request, current_status=status, current=code):
                    return httpx.Response(
                        current_status,
                        json={"error": current, "error_description": "privado"},
                    )

                client = await self.make_client(handler)
                result = await client.send_approved_message(
                    approved_payload()
                )
                self.assertEqual(result.decision, BitrixClientDecision.FAIL)
                self.assertEqual(result.error_code, "bitrix_api_permanent")
                self.assertNotIn("privado", repr(result))

    async def test_invalid_success_response_is_permanent_and_redacted(self):
        responses = (
            httpx.Response(200, text="oauth-secret-token"),
            httpx.Response(200, json={"result": {}}),
            httpx.Response(200, json={"result": {"id": 0}}),
            httpx.Response(200, json={"result": {"id": "not-an-id"}}),
        )
        for response in responses:
            with self.subTest(body=response.text):
                async def handler(request, current=response):
                    return current

                client = await self.make_client(handler)
                result = await client.send_approved_message(
                    approved_payload()
                )
                self.assertEqual(result.decision, BitrixClientDecision.FAIL)
                self.assertEqual(
                    result.error_code,
                    "bitrix_invalid_response",
                )
                self.assertNotIn("oauth-secret-token", repr(result))

    async def test_injected_client_is_not_closed(self):
        async def handler(request):
            return httpx.Response(200, json={"result": {"id": 789}})

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = BitrixClient(
            portal_url="https://portal.bitrix24.test",
            access_token="oauth-secret-token",
            timeout_seconds=3,
            http_client=http_client,
        )

        await client.close()
        self.assertFalse(http_client.is_closed)
        await http_client.aclose()

    def test_configuration_and_result_invariants(self):
        for portal_url in (
            "",
            "http://portal.bitrix24.test",
            "https://user:password@portal.bitrix24.test",
            "https://portal.bitrix24.test/rest",
            "https://portal.bitrix24.test?auth=secret",
        ):
            with self.subTest(portal_url=portal_url):
                with self.assertRaises(ValueError):
                    BitrixClient(
                        portal_url=portal_url,
                        access_token="token",
                        timeout_seconds=3,
                    )
        with self.assertRaises(ValueError):
            BitrixClient(
                portal_url="https://portal.bitrix24.test",
                access_token="",
                timeout_seconds=3,
            )
        with self.assertRaises(ValueError):
            BitrixClient(
                portal_url="https://portal.bitrix24.test",
                access_token="token",
                timeout_seconds=0,
            )
        with self.assertRaises(ValueError):
            BitrixClientResult(decision=BitrixClientDecision.SUCCESS)
        with self.assertRaises(ValueError):
            BitrixClientResult(
                decision=BitrixClientDecision.FAIL,
                error_code="permanent",
                retry_after_seconds=1,
            )

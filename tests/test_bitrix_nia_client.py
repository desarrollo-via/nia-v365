import json
import unittest

import httpx

from bitrix_connector.nia_client import (
    NiaClient,
    NiaClientDecision,
    NiaClientResult,
)
from bitrix_connector.preflight import NiaTextPayloadPreview


def approved_payload():
    return NiaTextPayloadPreview(
        session_id="bitrix:member-123:chat5",
        mensaje="Necesito una bomba",
    )


class NiaClientTests(unittest.IsolatedAsyncioTestCase):
    async def make_client(self, handler):
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(http_client.aclose)
        return NiaClient(
            base_url="https://nia.example.test/",
            timeout_seconds=3,
            http_client=http_client,
        )

    async def test_success_sends_only_approved_payload_to_exact_endpoint(self):
        captured = {}

        async def handler(request):
            captured["method"] = request.method
            captured["url"] = str(request.url)
            captured["json"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "respuesta": "Claro, ¿qué caudal necesita?",
                    "etapa": "preguntas_tecnicas",
                    "items_resultado": [{"codigo": "123456"}],
                    "cliente": {"id": "contact-27"},
                    "campo_interno_ignorado": "no forma parte del contrato",
                },
            )

        client = await self.make_client(handler)
        result = await client.send_approved_text(approved_payload())

        self.assertEqual(result.decision, NiaClientDecision.SUCCESS)
        self.assertEqual(result.http_status, 200)
        self.assertEqual(
            result.response.respuesta,
            "Claro, ¿qué caudal necesita?",
        )
        self.assertEqual(result.response.etapa, "preguntas_tecnicas")
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["url"], "https://nia.example.test/nia/chat")
        self.assertEqual(
            captured["json"],
            {
                "session_id": "bitrix:member-123:chat5",
                "mensaje": "Necesito una bomba",
            },
        )

    async def test_timeout_is_retryable_without_exception_details(self):
        async def handler(request):
            raise httpx.ReadTimeout(
                "token-super-secreto",
                request=request,
            )

        client = await self.make_client(handler)
        result = await client.send_approved_text(approved_payload())

        self.assertEqual(result.decision, NiaClientDecision.RETRY)
        self.assertEqual(result.error_code, "nia_timeout")
        self.assertNotIn("secreto", repr(result))
        self.assertNotIn("Necesito una bomba", repr(result))

    async def test_transport_failure_is_retryable_and_redacted(self):
        async def handler(request):
            raise httpx.ConnectError(
                "https://user:password@private.example/token",
                request=request,
            )

        client = await self.make_client(handler)
        result = await client.send_approved_text(approved_payload())

        self.assertEqual(result.decision, NiaClientDecision.RETRY)
        self.assertEqual(result.error_code, "nia_transport_error")
        self.assertNotIn("password", repr(result))
        self.assertNotIn("private.example", repr(result))

    async def test_429_is_retryable_and_accepts_bounded_retry_after(self):
        async def handler(request):
            return httpx.Response(
                429,
                headers={"Retry-After": "999999"},
                text="mensaje y token que nunca deben conservarse",
            )

        client = await self.make_client(handler)
        result = await client.send_approved_text(approved_payload())

        self.assertEqual(result.decision, NiaClientDecision.RETRY)
        self.assertEqual(result.error_code, "nia_rate_limited")
        self.assertEqual(result.http_status, 429)
        self.assertEqual(result.retry_after_seconds, 86_400)
        self.assertNotIn("nunca deben", repr(result))

    async def test_server_and_temporary_http_errors_are_retryable(self):
        for status in (408, 425, 500, 503, 599):
            with self.subTest(status=status):
                async def handler(request, current=status):
                    return httpx.Response(current, text="cuerpo sensible")

                client = await self.make_client(handler)
                result = await client.send_approved_text(approved_payload())
                self.assertEqual(result.decision, NiaClientDecision.RETRY)
                self.assertEqual(result.error_code, "nia_http_retryable")
                self.assertEqual(result.http_status, status)

    async def test_other_http_errors_are_permanent(self):
        for status in (400, 401, 403, 404, 422):
            with self.subTest(status=status):
                async def handler(request, current=status):
                    return httpx.Response(current, text="detalle sensible")

                client = await self.make_client(handler)
                result = await client.send_approved_text(approved_payload())
                self.assertEqual(result.decision, NiaClientDecision.FAIL)
                self.assertEqual(result.error_code, "nia_http_permanent")
                self.assertEqual(result.http_status, status)
                self.assertNotIn("detalle sensible", repr(result))

    async def test_invalid_json_and_contract_are_permanent_and_redacted(self):
        responses = (
            (httpx.Response(200, text="token-secreto"), "nia_invalid_json"),
            (
                httpx.Response(
                    200,
                    json={"respuesta_incorrecta": "dato privado"},
                ),
                "nia_invalid_response",
            ),
            (
                httpx.Response(200, json={"respuesta": {"texto": "privado"}}),
                "nia_invalid_response",
            ),
        )

        for response, expected_code in responses:
            with self.subTest(expected_code=expected_code):
                async def handler(request, current=response):
                    return current

                client = await self.make_client(handler)
                result = await client.send_approved_text(approved_payload())
                self.assertEqual(result.decision, NiaClientDecision.FAIL)
                self.assertEqual(result.error_code, expected_code)
                self.assertNotIn("privado", repr(result))
                self.assertNotIn("secreto", repr(result))

    async def test_injected_client_is_not_closed_by_nia_client(self):
        async def handler(request):
            return httpx.Response(200, json={"respuesta": "ok"})

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = NiaClient(
            base_url="https://nia.example.test",
            timeout_seconds=3,
            http_client=http_client,
        )

        await client.close()
        self.assertFalse(http_client.is_closed)
        await http_client.aclose()

    def test_configuration_and_result_invariants_reject_ambiguous_states(self):
        with self.assertRaises(ValueError):
            NiaClient(base_url="", timeout_seconds=3)
        with self.assertRaises(ValueError):
            NiaClient(
                base_url="https://user:password@nia.example.test",
                timeout_seconds=3,
            )
        with self.assertRaises(ValueError):
            NiaClient(
                base_url="https://nia.example.test?token=secret",
                timeout_seconds=3,
            )
        with self.assertRaises(ValueError):
            NiaClient(
                base_url="https://nia.example.test/api",
                timeout_seconds=3,
            )
        with self.assertRaises(ValueError):
            NiaClient(
                base_url="https://nia.example.test",
                timeout_seconds=0,
            )
        with self.assertRaises(ValueError):
            NiaClientResult(decision=NiaClientDecision.SUCCESS)
        with self.assertRaises(ValueError):
            NiaClientResult(
                decision=NiaClientDecision.FAIL,
                error_code="permanent",
                retry_after_seconds=1,
            )

import unittest
from datetime import datetime, timedelta, timezone

from bitrix_connector.event_parser import parse_webhook_form
from bitrix_connector.models import ConnectorEventStatus
from bitrix_connector.nia_client import NiaChatResponse, NiaClientResult
from bitrix_connector.nia_dispatch import (
    NiaDispatchWorkerHandler,
    NiaDispatchWorkerStore,
)
from bitrix_connector.preflight import build_text_preflight
from bitrix_connector.storage import build_received_record
from bitrix_connector.worker import ConnectorWorker, ConnectorWorkerRunStatus


def approved_claimed_record():
    form = {
        "event": "ONIMBOTV2MESSAGEADD",
        "data[bot][id]": "456",
        "data[message][id]": "789",
        "data[message][chatId]": "5",
        "data[message][authorId]": "27",
        "data[message][text]": "Necesito una bomba",
        "data[chat][dialogId]": "chat5",
        "data[chat][type]": "openChannel",
        "data[chat][entityType]": "LINES",
        "data[user][id]": "27",
        "auth[domain]": "viaindustrial.bitrix24.es",
        "auth[member_id]": "member-123",
        "auth[application_token]": "secret-token",
    }
    received = build_received_record(
        parse_webhook_form(form),
        form,
        identity_verified=True,
        security_reason="identity_verified",
        received_at=datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc),
    )
    review = build_text_preflight(received)
    return received.model_copy(
        update={
            "status": ConnectorEventStatus.PROCESSING,
            "attempt_count": 2,
            "nia_attempt_count": 1,
            "processing_stage": "nia_dispatch",
            "lease_owner": "nia-worker",
            "lease_until": datetime(2026, 7, 16, 12, 1, tzinfo=timezone.utc),
            "preflight_review": review.model_dump(mode="python"),
            "input_decision": {
                "decision": "approved",
                "content_hash": review.content_hash,
                "actor": "hugo",
            },
        }
    )


class FakeNiaClient:
    def __init__(self, result):
        self.result = result
        self.payloads = []

    async def send_approved_text(self, payload):
        self.payloads.append(payload)
        return self.result


class NiaDispatchStore:
    def __init__(self, record, *, save_result=True):
        self.record = record
        self.save_result = save_result
        self.calls = []
        self.saved_response = None

    async def claim_ready_for_nia(
        self,
        *,
        lease_owner,
        lease_seconds,
        now=None,
    ):
        self.calls.append(("claim_nia", lease_owner, lease_seconds, now))
        record, self.record = self.record, None
        return record

    async def save_nia_response(
        self,
        event_key,
        lease_owner,
        response,
        *,
        http_status,
        now=None,
    ):
        self.calls.append(
            (
                "save_nia_response",
                event_key,
                lease_owner,
                http_status,
                now,
            )
        )
        if self.save_result:
            self.saved_response = response
        return self.save_result

    async def retry_nia_claim(
        self,
        event_key,
        lease_owner,
        *,
        error_code,
        retry_after_seconds,
        now=None,
    ):
        self.calls.append(
            (
                "retry_nia",
                event_key,
                lease_owner,
                error_code,
                retry_after_seconds,
                now,
            )
        )
        return self.save_result

    async def fail_nia_claim(
        self,
        event_key,
        lease_owner,
        *,
        error_code,
        now=None,
    ):
        self.calls.append(
            ("fail_nia", event_key, lease_owner, error_code, now)
        )
        return self.save_result


class NiaDispatchWorkerTests(unittest.IsolatedAsyncioTestCase):
    def make_worker(self, store, nia_client):
        start = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
        handler = NiaDispatchWorkerHandler(
            store,
            nia_client,
            lease_owner="nia-worker",
            default_retry_after_seconds=30,
            clock=lambda: start + timedelta(seconds=5),
        )
        return ConnectorWorker(
            NiaDispatchWorkerStore(store),
            handler,
            lease_owner="nia-worker",
            lease_seconds=60,
            clock=lambda: start,
        )

    async def test_success_persists_response_without_generic_completion(self):
        store = NiaDispatchStore(approved_claimed_record())
        client = FakeNiaClient(
            NiaClientResult.succeeded(
                NiaChatResponse(
                    respuesta="¿Qué caudal necesita?",
                    etapa="preguntas_tecnicas",
                ),
                http_status=200,
            )
        )

        result = await self.make_worker(store, client).run_once()

        self.assertEqual(result.status, ConnectorWorkerRunStatus.COMPLETED)
        self.assertTrue(result.outcome_applied)
        self.assertEqual(
            [call[0] for call in store.calls],
            ["claim_nia", "save_nia_response"],
        )
        self.assertEqual(len(client.payloads), 1)
        self.assertEqual(
            client.payloads[0].model_dump(),
            {
                "session_id": "bitrix:member-123:chat5",
                "mensaje": "Necesito una bomba",
            },
        )
        self.assertEqual(
            store.saved_response.nia_response.respuesta,
            "¿Qué caudal necesita?",
        )
        self.assertEqual(
            store.saved_response.bitrix_payload_preview.model_dump(),
            {
                "botId": 456,
                "dialogId": "chat5",
                "fields": {"message": "¿Qué caudal necesita?"},
            },
        )

    async def test_retryable_result_uses_nia_retry_transition(self):
        store = NiaDispatchStore(approved_claimed_record())
        client = FakeNiaClient(
            NiaClientResult.retryable(
                "nia_rate_limited",
                http_status=429,
                retry_after_seconds=45,
            )
        )

        result = await self.make_worker(store, client).run_once()

        self.assertEqual(
            result.status,
            ConnectorWorkerRunStatus.RETRY_SCHEDULED,
        )
        self.assertEqual(store.calls[1][0], "retry_nia")
        self.assertEqual(store.calls[1][3], "nia_rate_limited")
        self.assertEqual(store.calls[1][4], 45)

    async def test_retry_without_server_delay_uses_safe_default(self):
        store = NiaDispatchStore(approved_claimed_record())
        client = FakeNiaClient(NiaClientResult.retryable("nia_timeout"))

        await self.make_worker(store, client).run_once()

        self.assertEqual(store.calls[1][0], "retry_nia")
        self.assertEqual(store.calls[1][4], 30)

    async def test_permanent_result_uses_nia_failure_transition(self):
        store = NiaDispatchStore(approved_claimed_record())
        client = FakeNiaClient(
            NiaClientResult.failed(
                "nia_http_permanent",
                http_status=422,
            )
        )

        result = await self.make_worker(store, client).run_once()

        self.assertEqual(result.status, ConnectorWorkerRunStatus.FAILED)
        self.assertEqual(store.calls[1][0], "fail_nia")
        self.assertEqual(store.calls[1][3], "nia_http_permanent")

    async def test_invalid_or_unapproved_payload_fails_without_calling_nia(self):
        record = approved_claimed_record().model_copy(
            update={"input_decision": None}
        )
        store = NiaDispatchStore(record)
        client = FakeNiaClient(
            NiaClientResult.succeeded(
                NiaChatResponse(respuesta="no debe usarse"),
                http_status=200,
            )
        )

        result = await self.make_worker(store, client).run_once()

        self.assertEqual(result.status, ConnectorWorkerRunStatus.FAILED)
        self.assertEqual(client.payloads, [])
        self.assertEqual(store.calls[1][0], "fail_nia")
        self.assertEqual(
            store.calls[1][3],
            "nia_approved_payload_invalid",
        )

    async def test_payload_changed_after_approval_fails_before_nia(self):
        record = approved_claimed_record()
        changed_review = dict(record.preflight_review)
        changed_review["nia_payload_preview"] = {
            "session_id": "bitrix:member-123:chat5",
            "mensaje": "Payload alterado después de aprobar",
        }
        record = record.model_copy(
            update={"preflight_review": changed_review}
        )
        store = NiaDispatchStore(record)
        client = FakeNiaClient(
            NiaClientResult.succeeded(
                NiaChatResponse(respuesta="no debe usarse"),
                http_status=200,
            )
        )

        result = await self.make_worker(store, client).run_once()

        self.assertEqual(result.status, ConnectorWorkerRunStatus.FAILED)
        self.assertEqual(client.payloads, [])
        self.assertEqual(
            store.calls[1][3],
            "nia_approved_payload_invalid",
        )

    async def test_lost_lease_during_response_save_is_visible(self):
        store = NiaDispatchStore(
            approved_claimed_record(),
            save_result=False,
        )
        client = FakeNiaClient(
            NiaClientResult.succeeded(
                NiaChatResponse(respuesta="respuesta válida"),
                http_status=200,
            )
        )

        result = await self.make_worker(store, client).run_once()

        self.assertEqual(result.status, ConnectorWorkerRunStatus.LEASE_LOST)
        self.assertFalse(result.outcome_applied)
        self.assertIsNone(store.saved_response)

    def test_configuration_rejects_ambiguous_retry_policy(self):
        store = NiaDispatchStore(approved_claimed_record())
        client = FakeNiaClient(NiaClientResult.retryable("nia_timeout"))

        with self.assertRaises(ValueError):
            NiaDispatchWorkerHandler(
                store,
                client,
                lease_owner="",
            )
        with self.assertRaises(ValueError):
            NiaDispatchWorkerHandler(
                store,
                client,
                lease_owner="nia-worker",
                default_retry_after_seconds=0,
            )


if __name__ == "__main__":
    unittest.main()

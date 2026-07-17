import unittest
from datetime import datetime, timedelta, timezone

from bitrix_connector.bitrix_client import (
    BitrixClientResult,
    BitrixSendResponse,
)
from bitrix_connector.bitrix_dispatch import (
    BitrixDispatchWorkerHandler,
    BitrixDispatchWorkerStore,
)
from bitrix_connector.event_parser import parse_webhook_form
from bitrix_connector.models import ConnectorEventStatus
from bitrix_connector.nia_client import NiaChatResponse
from bitrix_connector.output_review import build_output_review
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
    review = build_output_review(
        received,
        NiaChatResponse(
            respuesta="¿Qué caudal necesita?",
            etapa="preguntas_tecnicas",
        ),
    )
    return received.model_copy(
        update={
            "status": ConnectorEventStatus.PROCESSING,
            "attempt_count": 3,
            "bitrix_attempt_count": 1,
            "processing_stage": "bitrix_dispatch",
            "lease_owner": "bitrix-worker",
            "lease_until": datetime(2026, 7, 16, 12, 1, tzinfo=timezone.utc),
            "output_review": review.model_dump(mode="python"),
            "output_decision": {
                "decision": "approved",
                "content_hash": review.content_hash,
                "actor": "hugo",
            },
        }
    )


class FakeBitrixClient:
    def __init__(self, result):
        self.result = result
        self.payloads = []

    async def send_approved_message(self, payload):
        self.payloads.append(payload)
        return self.result


class BitrixDispatchStore:
    def __init__(self, record, *, apply_result=True):
        self.record = record
        self.apply_result = apply_result
        self.calls = []
        self.saved_response = None

    async def claim_ready_for_bitrix(
        self,
        *,
        lease_owner,
        lease_seconds,
        now=None,
    ):
        self.calls.append(("claim_bitrix", lease_owner, lease_seconds, now))
        record, self.record = self.record, None
        return record

    async def save_bitrix_sent(
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
                "save_bitrix_sent",
                event_key,
                lease_owner,
                http_status,
                now,
            )
        )
        if self.apply_result:
            self.saved_response = response
        return self.apply_result

    async def retry_bitrix_claim(
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
                "retry_bitrix",
                event_key,
                lease_owner,
                error_code,
                retry_after_seconds,
                now,
            )
        )
        return self.apply_result

    async def fail_bitrix_claim(
        self,
        event_key,
        lease_owner,
        *,
        error_code,
        now=None,
    ):
        self.calls.append(
            ("fail_bitrix", event_key, lease_owner, error_code, now)
        )
        return self.apply_result


class BitrixDispatchWorkerTests(unittest.IsolatedAsyncioTestCase):
    def make_worker(self, store, client):
        start = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
        handler = BitrixDispatchWorkerHandler(
            store,
            client,
            lease_owner="bitrix-worker",
            default_retry_after_seconds=30,
            clock=lambda: start + timedelta(seconds=5),
        )
        return ConnectorWorker(
            BitrixDispatchWorkerStore(store),
            handler,
            lease_owner="bitrix-worker",
            lease_seconds=60,
            clock=lambda: start,
        )

    async def test_success_persists_message_id_without_generic_completion(self):
        store = BitrixDispatchStore(approved_claimed_record())
        client = FakeBitrixClient(
            BitrixClientResult.succeeded(
                BitrixSendResponse.model_validate(
                    {"result": {"id": 987, "uuidMap": {}}}
                ),
                http_status=200,
            )
        )

        result = await self.make_worker(store, client).run_once()

        self.assertEqual(result.status, ConnectorWorkerRunStatus.COMPLETED)
        self.assertTrue(result.outcome_applied)
        self.assertEqual(
            [call[0] for call in store.calls],
            ["claim_bitrix", "save_bitrix_sent"],
        )
        self.assertEqual(
            client.payloads[0].model_dump(),
            {
                "botId": 456,
                "dialogId": "chat5",
                "fields": {"message": "¿Qué caudal necesita?"},
            },
        )
        self.assertEqual(store.saved_response.result.id, 987)

    async def test_retryable_result_uses_bitrix_retry_transition(self):
        store = BitrixDispatchStore(approved_claimed_record())
        client = FakeBitrixClient(
            BitrixClientResult.retryable(
                "bitrix_rate_limited",
                http_status=429,
                retry_after_seconds=45,
            )
        )

        result = await self.make_worker(store, client).run_once()

        self.assertEqual(
            result.status,
            ConnectorWorkerRunStatus.RETRY_SCHEDULED,
        )
        self.assertEqual(store.calls[1][0], "retry_bitrix")
        self.assertEqual(store.calls[1][3], "bitrix_rate_limited")
        self.assertEqual(store.calls[1][4], 45)

    async def test_retry_without_remote_delay_uses_default(self):
        store = BitrixDispatchStore(approved_claimed_record())
        client = FakeBitrixClient(
            BitrixClientResult.retryable("bitrix_timeout")
        )

        await self.make_worker(store, client).run_once()

        self.assertEqual(store.calls[1][0], "retry_bitrix")
        self.assertEqual(store.calls[1][4], 30)

    async def test_permanent_result_uses_bitrix_failure_transition(self):
        store = BitrixDispatchStore(approved_claimed_record())
        client = FakeBitrixClient(
            BitrixClientResult.failed(
                "bitrix_api_permanent",
                http_status=403,
            )
        )

        result = await self.make_worker(store, client).run_once()

        self.assertEqual(result.status, ConnectorWorkerRunStatus.FAILED)
        self.assertEqual(store.calls[1][0], "fail_bitrix")
        self.assertEqual(store.calls[1][3], "bitrix_api_permanent")

    async def test_changed_hash_fails_before_client_call(self):
        record = approved_claimed_record()
        changed = dict(record.output_review)
        changed["bitrix_payload_preview"] = {
            "botId": 456,
            "dialogId": "chat5",
            "fields": {"message": "Salida alterada"},
        }
        record = record.model_copy(update={"output_review": changed})
        store = BitrixDispatchStore(record)
        client = FakeBitrixClient(
            BitrixClientResult.failed("no_debe_usarse")
        )

        result = await self.make_worker(store, client).run_once()

        self.assertEqual(result.status, ConnectorWorkerRunStatus.FAILED)
        self.assertEqual(client.payloads, [])
        self.assertEqual(
            store.calls[1][3],
            "bitrix_approved_payload_invalid",
        )

    async def test_existing_outbound_id_prevents_client_call(self):
        record = approved_claimed_record().model_copy(
            update={"outbound_message_id": 987}
        )
        store = BitrixDispatchStore(record)
        client = FakeBitrixClient(
            BitrixClientResult.failed("no_debe_usarse")
        )

        result = await self.make_worker(store, client).run_once()

        self.assertEqual(result.status, ConnectorWorkerRunStatus.LEASE_LOST)
        self.assertEqual(client.payloads, [])
        self.assertEqual([call[0] for call in store.calls], ["claim_bitrix"])

    async def test_lost_lease_during_sent_save_is_visible(self):
        store = BitrixDispatchStore(
            approved_claimed_record(),
            apply_result=False,
        )
        client = FakeBitrixClient(
            BitrixClientResult.succeeded(
                BitrixSendResponse.model_validate(
                    {"result": {"id": 987}}
                ),
                http_status=200,
            )
        )

        result = await self.make_worker(store, client).run_once()

        self.assertEqual(result.status, ConnectorWorkerRunStatus.LEASE_LOST)
        self.assertFalse(result.outcome_applied)
        self.assertIsNone(store.saved_response)

    def test_configuration_rejects_ambiguous_retry_policy(self):
        store = BitrixDispatchStore(approved_claimed_record())
        client = FakeBitrixClient(
            BitrixClientResult.retryable("bitrix_timeout")
        )
        with self.assertRaises(ValueError):
            BitrixDispatchWorkerHandler(
                store,
                client,
                lease_owner="",
            )
        with self.assertRaises(ValueError):
            BitrixDispatchWorkerHandler(
                store,
                client,
                lease_owner="bitrix-worker",
                default_retry_after_seconds=0,
            )

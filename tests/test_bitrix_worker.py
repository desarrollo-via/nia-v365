import unittest
from datetime import datetime, timedelta, timezone

from bitrix_connector.event_parser import parse_webhook_form
from bitrix_connector.models import ConnectorEventStatus
from bitrix_connector.storage import build_received_record
from bitrix_connector.worker import (
    ConnectorHandlerResult,
    ConnectorWorker,
    ConnectorWorkerRunStatus,
)


def event_form():
    return {
        "event": "ONIMBOTV2MESSAGEADD",
        "data[bot][id]": "456",
        "data[message][id]": "789",
        "data[message][chatId]": "5",
        "data[message][authorId]": "27",
        "data[message][text]": "Necesito una bomba",
        "data[chat][dialogId]": "chat5",
        "data[user][id]": "27",
        "auth[domain]": "viaindustrial.bitrix24.es",
        "auth[member_id]": "member-123",
        "auth[application_token]": "secret-token",
    }


def claimed_record(owner="worker-a"):
    received = build_received_record(
        parse_webhook_form(event_form()),
        event_form(),
        identity_verified=True,
        security_reason="identity_verified",
        received_at=datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc),
    )
    return received.model_copy(
        update={
            "status": ConnectorEventStatus.PROCESSING,
            "attempt_count": 1,
            "lease_owner": owner,
            "lease_until": datetime(2026, 7, 16, 12, 1, tzinfo=timezone.utc),
        }
    )


class StepClock:
    def __init__(self, *values):
        self.values = list(values)

    def __call__(self):
        return self.values.pop(0)


class RecordingHandler:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.events = []

    async def handle(self, event):
        self.events.append(event)
        if self.error is not None:
            raise self.error
        return self.result


class WorkerStore:
    def __init__(self, claimed=None):
        self.claimed = claimed
        self.calls = []
        self.apply_result = True

    async def claim_next(self, *, lease_owner, lease_seconds, now=None):
        self.calls.append(("claim", lease_owner, lease_seconds, now))
        claimed, self.claimed = self.claimed, None
        return claimed

    async def complete_claim(self, event_key, lease_owner, *, now=None):
        self.calls.append(("complete", event_key, lease_owner, now))
        return self.apply_result

    async def retry_claim(
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
                "retry",
                event_key,
                lease_owner,
                error_code,
                retry_after_seconds,
                now,
            )
        )
        return self.apply_result

    async def fail_claim(
        self,
        event_key,
        lease_owner,
        *,
        error_code,
        now=None,
    ):
        self.calls.append(("fail", event_key, lease_owner, error_code, now))
        return self.apply_result


class ConnectorWorkerTests(unittest.IsolatedAsyncioTestCase):
    def make_worker(self, store, handler):
        start = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
        return ConnectorWorker(
            store,
            handler,
            lease_owner="worker-a",
            lease_seconds=30,
            clock=StepClock(start, start + timedelta(seconds=5)),
        )

    async def test_idle_does_not_call_handler(self):
        store = WorkerStore()
        handler = RecordingHandler(ConnectorHandlerResult.completed())

        result = await self.make_worker(store, handler).run_once()

        self.assertEqual(result.status, ConnectorWorkerRunStatus.IDLE)
        self.assertFalse(result.outcome_applied)
        self.assertEqual(handler.events, [])
        self.assertEqual(store.calls[0][0], "claim")

    async def test_success_completes_owned_claim(self):
        record = claimed_record()
        store = WorkerStore(record)
        handler = RecordingHandler(ConnectorHandlerResult.completed())

        result = await self.make_worker(store, handler).run_once()

        self.assertEqual(result.status, ConnectorWorkerRunStatus.COMPLETED)
        self.assertTrue(result.outcome_applied)
        self.assertEqual(result.event_key, record.event_key)
        self.assertEqual(len(handler.events), 1)
        self.assertEqual(store.calls[1][0], "complete")
        self.assertEqual(store.calls[1][2], "worker-a")

    async def test_retryable_result_schedules_retry(self):
        record = claimed_record()
        store = WorkerStore(record)
        handler = RecordingHandler(
            ConnectorHandlerResult.retryable(
                "temporary_preflight_error",
                retry_after_seconds=20,
            )
        )

        result = await self.make_worker(store, handler).run_once()

        self.assertEqual(result.status, ConnectorWorkerRunStatus.RETRY_SCHEDULED)
        self.assertTrue(result.outcome_applied)
        self.assertEqual(store.calls[1][0], "retry")
        self.assertEqual(store.calls[1][3], "temporary_preflight_error")
        self.assertEqual(store.calls[1][4], 20)

    async def test_permanent_result_marks_failure(self):
        record = claimed_record()
        store = WorkerStore(record)
        handler = RecordingHandler(
            ConnectorHandlerResult.failed("unsupported_event_content")
        )

        result = await self.make_worker(store, handler).run_once()

        self.assertEqual(result.status, ConnectorWorkerRunStatus.FAILED)
        self.assertTrue(result.outcome_applied)
        self.assertEqual(store.calls[1][0], "fail")
        self.assertEqual(store.calls[1][3], "unsupported_event_content")

    async def test_lost_lease_does_not_report_requested_outcome(self):
        record = claimed_record()
        store = WorkerStore(record)
        store.apply_result = False
        handler = RecordingHandler(ConnectorHandlerResult.completed())

        result = await self.make_worker(store, handler).run_once()

        self.assertEqual(result.status, ConnectorWorkerRunStatus.LEASE_LOST)
        self.assertFalse(result.outcome_applied)
        self.assertEqual(store.calls[1][0], "complete")

    async def test_unexpected_handler_error_propagates_without_closing_claim(self):
        store = WorkerStore(claimed_record())
        handler = RecordingHandler(error=RuntimeError("unclassified failure"))

        with self.assertRaisesRegex(RuntimeError, "unclassified failure"):
            await self.make_worker(store, handler).run_once()

        self.assertEqual([call[0] for call in store.calls], ["claim"])

    async def test_mismatched_claim_owner_is_not_processed(self):
        store = WorkerStore(claimed_record(owner="worker-b"))
        handler = RecordingHandler(ConnectorHandlerResult.completed())

        result = await self.make_worker(store, handler).run_once()

        self.assertEqual(result.status, ConnectorWorkerRunStatus.LEASE_LOST)
        self.assertEqual(handler.events, [])
        self.assertEqual([call[0] for call in store.calls], ["claim"])


class ConnectorHandlerResultTests(unittest.TestCase):
    def test_invalid_decision_payloads_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "requieren error_code"):
            ConnectorHandlerResult.retryable("", retry_after_seconds=10)
        with self.assertRaisesRegex(ValueError, "no puede ser negativo"):
            ConnectorHandlerResult.retryable("temporary", retry_after_seconds=-1)
        with self.assertRaisesRegex(ValueError, "no admite espera"):
            ConnectorHandlerResult(
                decision="fail",
                error_code="permanent",
                retry_after_seconds=1,
            )


if __name__ == "__main__":
    unittest.main()

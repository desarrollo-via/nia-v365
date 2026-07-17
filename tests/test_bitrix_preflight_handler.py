import unittest
from datetime import datetime, timedelta, timezone

from bitrix_connector.event_parser import parse_webhook_form
from bitrix_connector.models import ConnectorEventStatus
from bitrix_connector.preflight_handler import TextPreflightWorkerHandler
from bitrix_connector.storage import build_received_record
from bitrix_connector.worker import ConnectorWorker, ConnectorWorkerRunStatus


def claimed_record(text="Necesito una bomba"):
    form = {
        "event": "ONIMBOTV2MESSAGEADD",
        "data[bot][id]": "456",
        "data[message][id]": "789",
        "data[message][chatId]": "5",
        "data[message][authorId]": "27",
        "data[message][text]": text,
        "data[chat][dialogId]": "chat5",
        "data[chat][type]": "openChannel",
        "data[chat][entityType]": "LINES",
        "data[user][id]": "27",
        "auth[domain]": "viaindustrial.bitrix24.es",
        "auth[member_id]": "member-123",
        "auth[application_token]": "secret-token",
    }
    record = build_received_record(
        parse_webhook_form(form),
        form,
        identity_verified=True,
        security_reason="identity_verified",
        received_at=datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc),
    )
    return record.model_copy(
        update={
            "status": ConnectorEventStatus.PROCESSING,
            "attempt_count": 1,
            "lease_owner": "worker-a",
            "lease_until": datetime(2026, 7, 16, 12, 1, tzinfo=timezone.utc),
        }
    )


class IntegratedPreflightStore:
    def __init__(self, record, *, apply_result=True):
        self.record = record
        self.apply_result = apply_result
        self.calls = []
        self.saved_review = None

    async def claim_next(self, *, lease_owner, lease_seconds, now=None):
        self.calls.append(("claim", lease_owner, lease_seconds, now))
        record, self.record = self.record, None
        return record

    async def save_preflight(
        self,
        event_key,
        lease_owner,
        review,
        *,
        now=None,
    ):
        self.calls.append(("save_preflight", event_key, lease_owner, now))
        if self.apply_result:
            self.saved_review = review
        return self.apply_result

    async def complete_claim(self, *args, **kwargs):
        raise AssertionError("preflight aplicado no debe ejecutar complete_claim")

    async def retry_claim(self, *args, **kwargs):
        raise AssertionError("preflight aplicado no debe ejecutar retry_claim")

    async def fail_claim(self, *args, **kwargs):
        raise AssertionError("preflight aplicado no debe ejecutar fail_claim")


class PreflightWorkerIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def make_worker(self, store):
        start = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
        handler = TextPreflightWorkerHandler(
            store,
            lease_owner="worker-a",
            clock=lambda: start + timedelta(seconds=5),
        )
        return ConnectorWorker(
            store,
            handler,
            lease_owner="worker-a",
            lease_seconds=30,
            clock=lambda: start,
        )

    async def test_allowed_text_is_persisted_for_input_review(self):
        store = IntegratedPreflightStore(claimed_record())

        result = await self.make_worker(store).run_once()

        self.assertEqual(result.status, ConnectorWorkerRunStatus.COMPLETED)
        self.assertTrue(result.outcome_applied)
        self.assertEqual(
            [call[0] for call in store.calls],
            ["claim", "save_preflight"],
        )
        self.assertEqual(store.saved_review.decision, "ready_for_review")
        self.assertIsNotNone(store.saved_review.nia_payload_preview)

    async def test_blocked_text_is_persisted_without_nia_payload(self):
        store = IntegratedPreflightStore(
            claimed_record("Mi token = sk-example123456789")
        )

        result = await self.make_worker(store).run_once()

        self.assertEqual(result.status, ConnectorWorkerRunStatus.COMPLETED)
        self.assertTrue(result.outcome_applied)
        self.assertEqual(store.saved_review.decision, "blocked")
        self.assertIsNone(store.saved_review.nia_payload_preview)
        self.assertIn(
            "possible_secret_detected",
            store.saved_review.preflight_manifest.blocking_reasons,
        )

    async def test_lost_lease_is_reported_without_second_transition(self):
        store = IntegratedPreflightStore(claimed_record(), apply_result=False)

        result = await self.make_worker(store).run_once()

        self.assertEqual(result.status, ConnectorWorkerRunStatus.LEASE_LOST)
        self.assertFalse(result.outcome_applied)
        self.assertEqual(
            [call[0] for call in store.calls],
            ["claim", "save_preflight"],
        )
        self.assertIsNone(store.saved_review)


if __name__ == "__main__":
    unittest.main()

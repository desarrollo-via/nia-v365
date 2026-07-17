import asyncio
import unittest
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from bitrix_connector.event_parser import parse_webhook_form
from bitrix_connector.bitrix_client import BitrixSendResponse
from bitrix_connector.models import ConnectorEventStatus
from bitrix_connector.nia_client import NiaChatResponse
from bitrix_connector.output_review import build_output_review
from bitrix_connector.preflight import build_text_preflight
from bitrix_connector.review import ReviewDecisionOutcome
from bitrix_connector.storage import (
    DEFAULT_EVENTS_COLLECTION,
    EVENT_KEY_INDEX,
    STATUS_LEASE_INDEX,
    STATUS_RECEIVED_INDEX,
    MongoConnectorEventStore,
    build_received_record,
)


class FakeCursor:
    def __init__(self, documents):
        self.documents = documents

    def sort(self, field, direction):
        self.documents.sort(
            key=lambda document: document[field],
            reverse=direction < 0,
        )
        return self

    def limit(self, count):
        self.documents = self.documents[:count]
        return self

    def __aiter__(self):
        self._index = 0
        return self

    async def __anext__(self):
        if self._index >= len(self.documents):
            raise StopAsyncIteration
        document = self.documents[self._index]
        self._index += 1
        return deepcopy(document)


class FakeCollection:
    def __init__(self):
        self.documents = {}
        self.indexes = []
        self.claim_calls = []

    async def create_index(self, keys, **options):
        self.indexes.append((keys, options))
        return options["name"]

    @staticmethod
    def _field_value(document, field):
        value = document
        for part in field.split("."):
            if not isinstance(value, dict):
                return None
            value = value.get(part)
        return value

    @staticmethod
    def _matches(document, selector):
        if "$or" in selector:
            return any(FakeCollection._matches(document, item) for item in selector["$or"])
        for field, expected in selector.items():
            actual = FakeCollection._field_value(document, field)
            if isinstance(expected, dict):
                if "$lte" in expected and not actual <= expected["$lte"]:
                    return False
                if "$gt" in expected and not actual > expected["$gt"]:
                    return False
                if "$ne" in expected and actual == expected["$ne"]:
                    return False
                if "$nin" in expected and actual in expected["$nin"]:
                    return False
            elif actual != expected:
                return False
        return True

    @staticmethod
    def _apply_update(document, update):
        document.update(deepcopy(update.get("$set", {})))
        for field, increment in update.get("$inc", {}).items():
            document[field] += increment

    @staticmethod
    def _project(document, projection):
        included = [
            field
            for field, enabled in projection.items()
            if enabled and field != "_id"
        ]
        if not included:
            return deepcopy(document)
        return {
            field: deepcopy(document[field])
            for field in included
            if field in document
        }

    async def update_one(self, selector, update, *, upsert):
        event_key = selector["event_key"]
        if upsert:
            if event_key in self.documents:
                return SimpleNamespace(upserted_id=None, modified_count=0)
            self.documents[event_key] = deepcopy(update["$setOnInsert"])
            return SimpleNamespace(
                upserted_id=f"fake-{len(self.documents)}",
                modified_count=0,
            )

        document = self.documents.get(event_key)
        if document is None or not self._matches(document, selector):
            return SimpleNamespace(upserted_id=None, modified_count=0)
        self._apply_update(document, update)
        return SimpleNamespace(upserted_id=None, modified_count=1)

    async def find_one(self, selector, projection):
        document = next(
            (
                item
                for item in self.documents.values()
                if self._matches(item, selector)
            ),
            None,
        )
        return self._project(document, projection) if document else None

    def find(self, selector, projection):
        documents = [
            self._project(document, projection)
            for document in self.documents.values()
            if self._matches(document, selector)
        ]
        return FakeCursor(documents)

    async def find_one_and_update(
        self,
        selector,
        update,
        *,
        sort=None,
        projection=None,
        return_document,
    ):
        self.claim_calls.append(
            {
                "selector": deepcopy(selector),
                "update": deepcopy(update),
                "sort": deepcopy(sort),
                "projection": deepcopy(projection),
                "return_document": return_document,
            }
        )
        candidates = [
            document
            for document in self.documents.values()
            if self._matches(document, selector)
        ]

        if not candidates:
            return None

        if sort:
            received_at_field = sort[0][0]
            document = min(
                candidates,
                key=lambda item: item[received_at_field],
            )
        else:
            document = candidates[0]
        self._apply_update(document, update)
        return self._project(document, projection or {"_id": 0})


class FakeDatabase:
    def __init__(self):
        self.requested_collection = None
        self.collection = FakeCollection()

    def __getitem__(self, name):
        self.requested_collection = name
        return self.collection


def event_form():
    return {
        "event": "ONIMBOTV2MESSAGEADD",
        "ts": "1772093963",
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
        "data[bot][auth][access_token]": "oauth-secret",
        "data[bot][auth][refresh_token]": "refresh-secret",
    }


class StorageRecordTests(unittest.TestCase):
    def test_record_is_auditable_and_contains_no_tokens(self):
        form = event_form()
        event = parse_webhook_form(form)
        instant = datetime(2026, 7, 15, 15, 0, tzinfo=timezone.utc)

        record = build_received_record(
            event,
            form,
            identity_verified=True,
            security_reason="identity_verified",
            received_at=instant,
        )
        serialized = record.model_dump_json()

        self.assertEqual(record.status, ConnectorEventStatus.RECEIVED)
        self.assertEqual(record.received_at, instant)
        self.assertEqual(record.attempt_count, 0)
        self.assertIsNone(record.lease_owner)
        self.assertNotIn("application_token", record.normalized_event)
        self.assertNotIn("secret-token", serialized)
        self.assertNotIn("oauth-secret", serialized)
        self.assertNotIn("refresh-secret", serialized)
        self.assertEqual(record.raw_redacted["auth[application_token]"], "[REDACTED]")

    def test_status_contract_exposes_review_and_terminal_states(self):
        values = {status.value for status in ConnectorEventStatus}

        self.assertIn("needs_input_review", values)
        self.assertIn("needs_output_review", values)
        self.assertIn("ready_for_bitrix", values)
        self.assertIn("sent", values)
        self.assertIn("failed", values)


class MongoStorageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.database = FakeDatabase()
        self.store = MongoConnectorEventStore.from_database(self.database)
        form = event_form()
        self.record = build_received_record(
            parse_webhook_form(form),
            form,
            identity_verified=True,
            security_reason="identity_verified",
        )

    async def prepare_review(
        self,
        record=None,
        *,
        lease_owner="worker-a",
    ):
        record = record or self.record
        await self.store.save_received(record)
        claimed_at = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
        claimed = await self.store.claim_next(
            lease_owner=lease_owner,
            lease_seconds=30,
            now=claimed_at,
        )
        review = build_text_preflight(claimed)
        applied = await self.store.save_preflight(
            record.event_key,
            lease_owner,
            review,
            now=claimed_at + timedelta(seconds=5),
        )
        self.assertTrue(applied)
        return review, claimed_at

    async def prepare_approved(self, record=None):
        record = record or self.record
        review, claimed_at = await self.prepare_review(record)
        result = await self.store.approve_input(
            record.event_key,
            content_hash=review.content_hash,
            actor="hugo",
            now=claimed_at + timedelta(seconds=10),
        )
        self.assertEqual(result.outcome, ReviewDecisionOutcome.APPLIED)
        return review, claimed_at

    async def test_uses_exclusive_default_collection(self):
        self.assertEqual(self.database.requested_collection, DEFAULT_EVENTS_COLLECTION)

    async def test_creates_unique_and_operational_indexes_without_ttl(self):
        await self.store.ensure_indexes()

        by_name = {options["name"]: (keys, options) for keys, options in self.database.collection.indexes}
        self.assertTrue(by_name[EVENT_KEY_INDEX][1]["unique"])
        self.assertEqual(by_name[EVENT_KEY_INDEX][0], [("event_key", 1)])
        self.assertIn(STATUS_RECEIVED_INDEX, by_name)
        self.assertIn(STATUS_LEASE_INDEX, by_name)
        self.assertTrue(all("expireAfterSeconds" not in options for _, options in self.database.collection.indexes))

    async def test_first_save_creates_and_duplicate_does_not_overwrite(self):
        first = await self.store.save_received(self.record)
        stored_before = await self.store.get_by_key(self.record.event_key)

        changed = self.record.model_copy(update={"security_reason": "changed"})
        second = await self.store.save_received(changed)
        stored_after = await self.store.get_by_key(self.record.event_key)

        self.assertTrue(first.created)
        self.assertFalse(first.duplicate)
        self.assertFalse(second.created)
        self.assertTrue(second.duplicate)
        self.assertEqual(stored_before, stored_after)
        self.assertEqual(stored_after["security_reason"], "identity_verified")

    async def test_missing_event_returns_none(self):
        self.assertIsNone(await self.store.get_by_key("0" * 64))

    async def test_claim_transitions_received_to_processing_with_lease(self):
        await self.store.save_received(self.record)
        claimed_at = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)

        claimed = await self.store.claim_next(
            lease_owner="worker-a",
            lease_seconds=30,
            now=claimed_at,
        )

        self.assertIsNotNone(claimed)
        self.assertEqual(claimed.event_key, self.record.event_key)
        self.assertEqual(claimed.status, ConnectorEventStatus.PROCESSING)
        self.assertEqual(claimed.updated_at, claimed_at)
        self.assertEqual(claimed.lease_owner, "worker-a")
        self.assertEqual(claimed.lease_until, claimed_at + timedelta(seconds=30))
        self.assertEqual(claimed.attempt_count, 1)
        self.assertIsNone(claimed.last_error)

    async def test_two_workers_cannot_claim_the_same_active_lease(self):
        await self.store.save_received(self.record)
        claimed_at = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)

        claims = await asyncio.gather(
            self.store.claim_next(
                lease_owner="worker-a",
                lease_seconds=30,
                now=claimed_at,
            ),
            self.store.claim_next(
                lease_owner="worker-b",
                lease_seconds=30,
                now=claimed_at,
            ),
        )

        claimed = [claim for claim in claims if claim is not None]
        self.assertEqual(len(claimed), 1)
        self.assertEqual(claimed[0].event_key, self.record.event_key)
        self.assertEqual(len(self.database.collection.claim_calls), 2)

    async def test_expired_processing_lease_can_be_reclaimed(self):
        await self.store.save_received(self.record)
        first_at = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
        first = await self.store.claim_next(
            lease_owner="worker-a",
            lease_seconds=30,
            now=first_at,
        )

        before_expiry = await self.store.claim_next(
            lease_owner="worker-b",
            lease_seconds=30,
            now=first_at + timedelta(seconds=29),
        )
        reclaimed_at = first_at + timedelta(seconds=31)
        reclaimed = await self.store.claim_next(
            lease_owner="worker-b",
            lease_seconds=45,
            now=reclaimed_at,
        )

        self.assertIsNotNone(first)
        self.assertIsNone(before_expiry)
        self.assertIsNotNone(reclaimed)
        self.assertEqual(reclaimed.event_key, first.event_key)
        self.assertEqual(reclaimed.attempt_count, 2)
        self.assertEqual(reclaimed.lease_owner, "worker-b")
        self.assertEqual(reclaimed.updated_at, reclaimed_at)
        self.assertEqual(reclaimed.lease_until, reclaimed_at + timedelta(seconds=45))

    async def test_claim_rejects_non_positive_lease(self):
        with self.assertRaisesRegex(ValueError, "lease_seconds debe ser positivo"):
            await self.store.claim_next(lease_owner="worker-a", lease_seconds=0)

    async def test_owner_can_renew_active_claim_but_other_worker_cannot(self):
        await self.store.save_received(self.record)
        claimed_at = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
        await self.store.claim_next(
            lease_owner="worker-a",
            lease_seconds=30,
            now=claimed_at,
        )

        rejected = await self.store.renew_claim(
            self.record.event_key,
            "worker-b",
            lease_seconds=60,
            now=claimed_at + timedelta(seconds=10),
        )
        renewed = await self.store.renew_claim(
            self.record.event_key,
            "worker-a",
            lease_seconds=60,
            now=claimed_at + timedelta(seconds=10),
        )
        stored = await self.store.get_by_key(self.record.event_key)

        self.assertFalse(rejected)
        self.assertTrue(renewed)
        self.assertEqual(stored["lease_owner"], "worker-a")
        self.assertEqual(
            stored["lease_until"],
            claimed_at + timedelta(seconds=70),
        )

    async def test_expired_owner_cannot_close_reassigned_claim(self):
        await self.store.save_received(self.record)
        first_at = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
        await self.store.claim_next(
            lease_owner="worker-a",
            lease_seconds=30,
            now=first_at,
        )
        reassigned_at = first_at + timedelta(seconds=31)
        await self.store.claim_next(
            lease_owner="worker-b",
            lease_seconds=30,
            now=reassigned_at,
        )

        stale_renew = await self.store.renew_claim(
            self.record.event_key,
            "worker-a",
            lease_seconds=30,
            now=reassigned_at + timedelta(seconds=1),
        )
        stale_complete = await self.store.complete_claim(
            self.record.event_key,
            "worker-a",
            now=reassigned_at + timedelta(seconds=1),
        )
        current_complete = await self.store.complete_claim(
            self.record.event_key,
            "worker-b",
            now=reassigned_at + timedelta(seconds=1),
        )
        stored = await self.store.get_by_key(self.record.event_key)

        self.assertFalse(stale_renew)
        self.assertFalse(stale_complete)
        self.assertTrue(current_complete)
        self.assertEqual(stored["status"], ConnectorEventStatus.PENDING.value)
        self.assertIsNone(stored["lease_owner"])
        self.assertIsNone(stored["lease_until"])

    async def test_retry_waits_until_due_and_can_then_fail_under_new_owner(self):
        await self.store.save_received(self.record)
        first_at = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
        await self.store.claim_next(
            lease_owner="worker-a",
            lease_seconds=30,
            now=first_at,
        )
        retried_at = first_at + timedelta(seconds=10)

        retried = await self.store.retry_claim(
            self.record.event_key,
            "worker-a",
            error_code="temporary_storage_error",
            retry_after_seconds=20,
            now=retried_at,
        )
        early = await self.store.claim_next(
            lease_owner="worker-b",
            lease_seconds=30,
            now=retried_at + timedelta(seconds=19),
        )
        due = await self.store.claim_next(
            lease_owner="worker-b",
            lease_seconds=30,
            now=retried_at + timedelta(seconds=20),
        )
        stale_fail = await self.store.fail_claim(
            self.record.event_key,
            "worker-a",
            error_code="stale_worker_error",
            now=retried_at + timedelta(seconds=21),
        )
        failed = await self.store.fail_claim(
            self.record.event_key,
            "worker-b",
            error_code="permanent_processing_error",
            now=retried_at + timedelta(seconds=21),
        )
        stored = await self.store.get_by_key(self.record.event_key)

        self.assertTrue(retried)
        self.assertIsNone(early)
        self.assertIsNotNone(due)
        self.assertFalse(stale_fail)
        self.assertTrue(failed)
        self.assertEqual(stored["status"], ConnectorEventStatus.FAILED.value)
        self.assertEqual(stored["last_error"], "permanent_processing_error")
        self.assertIsNone(stored["lease_owner"])
        self.assertIsNone(stored["lease_until"])

    async def test_preflight_is_saved_and_releases_owned_lease_for_review(self):
        await self.store.save_received(self.record)
        claimed_at = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
        claimed = await self.store.claim_next(
            lease_owner="worker-a",
            lease_seconds=30,
            now=claimed_at,
        )
        review = build_text_preflight(claimed)

        applied = await self.store.save_preflight(
            self.record.event_key,
            "worker-a",
            review,
            now=claimed_at + timedelta(seconds=5),
        )
        stored = await self.store.get_by_key(self.record.event_key)

        self.assertTrue(applied)
        self.assertEqual(
            stored["status"],
            ConnectorEventStatus.NEEDS_INPUT_REVIEW.value,
        )
        self.assertIsNone(stored["lease_owner"])
        self.assertIsNone(stored["lease_until"])
        self.assertEqual(
            stored["preflight_review"]["event_key"],
            self.record.event_key,
        )
        self.assertEqual(
            stored["preflight_review"]["decision"],
            "ready_for_review",
        )

    async def test_preflight_rejects_wrong_owner_and_event_key_mismatch(self):
        await self.store.save_received(self.record)
        claimed_at = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
        claimed = await self.store.claim_next(
            lease_owner="worker-a",
            lease_seconds=30,
            now=claimed_at,
        )
        review = build_text_preflight(claimed)

        rejected = await self.store.save_preflight(
            self.record.event_key,
            "worker-b",
            review,
            now=claimed_at + timedelta(seconds=5),
        )
        stored = await self.store.get_by_key(self.record.event_key)

        self.assertFalse(rejected)
        self.assertEqual(stored["status"], ConnectorEventStatus.PROCESSING)
        self.assertIsNone(stored["preflight_review"])
        with self.assertRaisesRegex(ValueError, "preflight_event_key_mismatch"):
            await self.store.save_preflight(
                "0" * 64,
                "worker-a",
                review,
                now=claimed_at + timedelta(seconds=5),
            )

    async def test_preflight_rejects_expired_lease_without_writing_view(self):
        await self.store.save_received(self.record)
        claimed_at = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
        claimed = await self.store.claim_next(
            lease_owner="worker-a",
            lease_seconds=5,
            now=claimed_at,
        )
        review = build_text_preflight(claimed)

        applied = await self.store.save_preflight(
            self.record.event_key,
            "worker-a",
            review,
            now=claimed_at + timedelta(seconds=6),
        )
        stored = await self.store.get_by_key(self.record.event_key)

        self.assertFalse(applied)
        self.assertEqual(stored["status"], ConnectorEventStatus.PROCESSING)
        self.assertEqual(stored["lease_owner"], "worker-a")
        self.assertIsNone(stored["preflight_review"])

    async def test_review_queries_only_return_review_projection(self):
        await self.store.save_received(self.record)
        claimed_at = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
        claimed = await self.store.claim_next(
            lease_owner="worker-a",
            lease_seconds=30,
            now=claimed_at,
        )
        review = build_text_preflight(claimed)
        await self.store.save_preflight(
            self.record.event_key,
            "worker-a",
            review,
            now=claimed_at + timedelta(seconds=5),
        )

        listed = await self.store.list_reviews(
            decision="ready_for_review",
            limit=10,
        )
        blocked = await self.store.list_reviews(
            decision="blocked",
            limit=10,
        )
        detail = await self.store.get_review(self.record.event_key)

        self.assertEqual(len(listed), 1)
        self.assertEqual(blocked, [])
        self.assertEqual(
            set(listed[0]),
            {
                "event_key",
                "status",
                "received_at",
                "updated_at",
                "preflight_review",
            },
        )
        self.assertEqual(set(detail), set(listed[0]))
        self.assertNotIn("normalized_event", listed[0])
        self.assertNotIn("lease_owner", listed[0])

    async def test_review_detail_excludes_non_review_states(self):
        await self.store.save_received(self.record)

        self.assertIsNone(await self.store.get_review(self.record.event_key))

    async def test_approval_is_atomic_and_exact_repeat_is_idempotent(self):
        review, started_at = await self.prepare_review()
        decided_at = started_at + timedelta(seconds=10)

        applied = await self.store.approve_input(
            self.record.event_key,
            content_hash=review.content_hash,
            actor="hugo",
            reason="Contenido verificado",
            now=decided_at,
        )
        repeated = await self.store.approve_input(
            self.record.event_key,
            content_hash=review.content_hash,
            actor="hugo",
            reason="Contenido verificado",
            now=decided_at + timedelta(seconds=1),
        )
        stored = await self.store.get_by_key(self.record.event_key)

        self.assertEqual(applied.outcome, ReviewDecisionOutcome.APPLIED)
        self.assertEqual(repeated.outcome, ReviewDecisionOutcome.IDEMPOTENT)
        self.assertEqual(
            stored["status"],
            ConnectorEventStatus.READY_FOR_NIA.value,
        )
        self.assertEqual(stored["input_decision"]["decision"], "approved")
        self.assertEqual(
            stored["input_decision"]["content_hash"],
            review.content_hash,
        )
        self.assertEqual(stored["input_decision"]["decided_at"], decided_at)

    async def test_stale_hash_does_not_apply_decision(self):
        await self.prepare_review()

        result = await self.store.approve_input(
            self.record.event_key,
            content_hash="0" * 64,
            actor="hugo",
            now=datetime(2026, 7, 16, 12, 0, 10, tzinfo=timezone.utc),
        )
        stored = await self.store.get_by_key(self.record.event_key)

        self.assertEqual(result.outcome, ReviewDecisionOutcome.STALE_HASH)
        self.assertEqual(
            stored["status"],
            ConnectorEventStatus.NEEDS_INPUT_REVIEW.value,
        )
        self.assertIsNone(stored["input_decision"])

    async def test_blocked_preflight_cannot_be_approved_but_can_be_rejected(self):
        form = event_form()
        form["data[message][id]"] = "790"
        form["data[message][text]"] = "Mi token = sk-example123456789"
        record = build_received_record(
            parse_webhook_form(form),
            form,
            identity_verified=True,
            security_reason="identity_verified",
        )
        review, started_at = await self.prepare_review(record)
        self.assertEqual(review.decision, "blocked")

        blocked = await self.store.approve_input(
            record.event_key,
            content_hash=review.content_hash,
            actor="hugo",
            now=started_at + timedelta(seconds=10),
        )
        rejected = await self.store.reject_input(
            record.event_key,
            content_hash=review.content_hash,
            actor="hugo",
            reason="Contiene una credencial",
            now=started_at + timedelta(seconds=11),
        )
        repeated = await self.store.reject_input(
            record.event_key,
            content_hash=review.content_hash,
            actor="hugo",
            reason="Contiene una credencial",
            now=started_at + timedelta(seconds=12),
        )
        conflicting = await self.store.approve_input(
            record.event_key,
            content_hash=review.content_hash,
            actor="hugo",
            now=started_at + timedelta(seconds=13),
        )

        self.assertEqual(
            blocked.outcome,
            ReviewDecisionOutcome.PREFLIGHT_BLOCKED,
        )
        self.assertEqual(rejected.outcome, ReviewDecisionOutcome.APPLIED)
        self.assertEqual(repeated.outcome, ReviewDecisionOutcome.IDEMPOTENT)
        self.assertEqual(
            conflicting.outcome,
            ReviewDecisionOutcome.ALREADY_DECIDED,
        )

    async def test_nia_claim_only_selects_ready_or_due_nia_work(self):
        _, started_at = await self.prepare_approved()
        other_form = event_form()
        other_form["data[message][id]"] = "900"
        other_record = build_received_record(
            parse_webhook_form(other_form),
            other_form,
            identity_verified=True,
            security_reason="identity_verified",
        )
        await self.store.save_received(other_record)

        claimed = await self.store.claim_ready_for_nia(
            lease_owner="nia-worker",
            lease_seconds=45,
            now=started_at + timedelta(seconds=15),
        )
        no_second = await self.store.claim_ready_for_nia(
            lease_owner="nia-worker-2",
            lease_seconds=45,
            now=started_at + timedelta(seconds=15),
        )

        self.assertEqual(claimed.event_key, self.record.event_key)
        self.assertEqual(claimed.processing_stage, "nia_dispatch")
        self.assertEqual(claimed.nia_attempt_count, 1)
        self.assertIsNone(no_second)
        untouched = await self.store.get_by_key(other_record.event_key)
        self.assertEqual(untouched["status"], ConnectorEventStatus.RECEIVED.value)

    async def test_nia_retry_is_not_claimed_by_preflight_worker(self):
        _, started_at = await self.prepare_approved()
        claimed_at = started_at + timedelta(seconds=15)
        claimed = await self.store.claim_ready_for_nia(
            lease_owner="nia-worker",
            lease_seconds=30,
            now=claimed_at,
        )
        retried_at = claimed_at + timedelta(seconds=5)
        applied = await self.store.retry_nia_claim(
            self.record.event_key,
            "nia-worker",
            error_code="nia_timeout",
            retry_after_seconds=20,
            now=retried_at,
        )

        preflight_claim = await self.store.claim_next(
            lease_owner="preflight-worker",
            lease_seconds=30,
            now=retried_at + timedelta(seconds=20),
        )
        early_nia = await self.store.claim_ready_for_nia(
            lease_owner="nia-worker-2",
            lease_seconds=30,
            now=retried_at + timedelta(seconds=19),
        )
        due_nia = await self.store.claim_ready_for_nia(
            lease_owner="nia-worker-2",
            lease_seconds=30,
            now=retried_at + timedelta(seconds=20),
        )

        self.assertTrue(applied)
        self.assertIsNone(preflight_claim)
        self.assertIsNone(early_nia)
        self.assertIsNotNone(due_nia)
        self.assertEqual(due_nia.processing_stage, "nia_dispatch")
        self.assertEqual(due_nia.nia_attempt_count, 2)

    async def test_valid_nia_response_is_saved_atomically_for_output_review(self):
        _, started_at = await self.prepare_approved()
        claimed_at = started_at + timedelta(seconds=15)
        claimed = await self.store.claim_ready_for_nia(
            lease_owner="nia-worker",
            lease_seconds=30,
            now=claimed_at,
        )
        response = NiaChatResponse(
            respuesta="¿Qué caudal necesita?",
            etapa="preguntas_tecnicas",
        )
        output_review = build_output_review(claimed, response)

        stale = await self.store.save_nia_response(
            self.record.event_key,
            "otro-worker",
            output_review,
            http_status=200,
            now=claimed_at + timedelta(seconds=5),
        )
        applied = await self.store.save_nia_response(
            self.record.event_key,
            "nia-worker",
            output_review,
            http_status=200,
            now=claimed_at + timedelta(seconds=5),
        )
        stored = await self.store.get_by_key(self.record.event_key)

        self.assertFalse(stale)
        self.assertTrue(applied)
        self.assertEqual(
            stored["status"],
            ConnectorEventStatus.NEEDS_OUTPUT_REVIEW.value,
        )
        self.assertEqual(
            stored["nia_response"],
            {
                "respuesta": "¿Qué caudal necesita?",
                "etapa": "preguntas_tecnicas",
                "items_resultado": None,
                "cliente": None,
            },
        )
        self.assertEqual(stored["nia_dispatch"]["http_status"], 200)
        self.assertIsNone(stored["processing_stage"])
        self.assertIsNone(stored["lease_owner"])
        self.assertIsNone(stored["lease_until"])
        self.assertEqual(
            stored["output_review"]["content_hash"],
            output_review.content_hash,
        )
        self.assertEqual(
            stored["output_review"]["bitrix_payload_preview"],
            {
                "botId": 456,
                "dialogId": "chat5",
                "fields": {"message": "¿Qué caudal necesita?"},
            },
        )

    async def prepare_output_review(
        self,
        *,
        response_text="¿Qué caudal necesita?",
    ):
        _, started_at = await self.prepare_approved()
        claimed_at = started_at + timedelta(seconds=15)
        claimed = await self.store.claim_ready_for_nia(
            lease_owner="nia-worker",
            lease_seconds=30,
            now=claimed_at,
        )
        review = build_output_review(
            claimed,
            NiaChatResponse(
                respuesta=response_text,
                etapa="preguntas_tecnicas",
            ),
        )
        applied = await self.store.save_nia_response(
            self.record.event_key,
            "nia-worker",
            review,
            http_status=200,
            now=claimed_at + timedelta(seconds=5),
        )
        self.assertTrue(applied)
        return review, claimed_at

    async def prepare_ready_for_bitrix(self):
        review, claimed_at = await self.prepare_output_review()
        result = await self.store.approve_output(
            self.record.event_key,
            content_hash=review.content_hash,
            actor="hugo",
            now=claimed_at + timedelta(seconds=25),
        )
        self.assertEqual(result.outcome, ReviewDecisionOutcome.APPLIED)
        return review, claimed_at

    async def test_output_queries_use_review_only_projection(self):
        review, _ = await self.prepare_output_review()

        listed = await self.store.list_output_reviews(
            decision="ready_for_review",
            limit=10,
        )
        blocked = await self.store.list_output_reviews(
            decision="blocked",
            limit=10,
        )
        detail = await self.store.get_output_review(self.record.event_key)

        self.assertEqual(len(listed), 1)
        self.assertEqual(blocked, [])
        self.assertEqual(
            set(listed[0]),
            {
                "event_key",
                "status",
                "received_at",
                "updated_at",
                "output_review",
            },
        )
        self.assertEqual(detail, listed[0])
        self.assertEqual(
            detail["output_review"]["content_hash"],
            review.content_hash,
        )
        self.assertNotIn("normalized_event", detail)
        self.assertNotIn("preflight_review", detail)
        self.assertNotIn("lease_owner", detail)

    async def test_output_approval_is_atomic_and_repeat_is_idempotent(self):
        review, started_at = await self.prepare_output_review()
        decided_at = started_at + timedelta(seconds=25)

        applied = await self.store.approve_output(
            self.record.event_key,
            content_hash=review.content_hash,
            actor="hugo",
            reason="Respuesta verificada",
            now=decided_at,
        )
        repeated = await self.store.approve_output(
            self.record.event_key,
            content_hash=review.content_hash,
            actor="hugo",
            reason="Respuesta verificada",
            now=decided_at + timedelta(seconds=1),
        )
        conflicting = await self.store.reject_output(
            self.record.event_key,
            content_hash=review.content_hash,
            actor="hugo",
            reason="Cambio de opinión",
            now=decided_at + timedelta(seconds=2),
        )
        stored = await self.store.get_by_key(self.record.event_key)

        self.assertEqual(applied.outcome, ReviewDecisionOutcome.APPLIED)
        self.assertEqual(repeated.outcome, ReviewDecisionOutcome.IDEMPOTENT)
        self.assertEqual(
            conflicting.outcome,
            ReviewDecisionOutcome.ALREADY_DECIDED,
        )
        self.assertEqual(
            stored["status"],
            ConnectorEventStatus.READY_FOR_BITRIX.value,
        )
        self.assertEqual(stored["output_decision"]["decision"], "approved")
        self.assertEqual(
            stored["output_decision"]["content_hash"],
            review.content_hash,
        )

    async def test_output_stale_hash_does_not_change_document(self):
        await self.prepare_output_review()

        result = await self.store.approve_output(
            self.record.event_key,
            content_hash="0" * 64,
            actor="hugo",
        )
        stored = await self.store.get_by_key(self.record.event_key)

        self.assertEqual(result.outcome, ReviewDecisionOutcome.STALE_HASH)
        self.assertEqual(
            stored["status"],
            ConnectorEventStatus.NEEDS_OUTPUT_REVIEW.value,
        )
        self.assertIsNone(stored["output_decision"])

    async def test_empty_output_cannot_be_approved_but_can_be_rejected(self):
        review, _ = await self.prepare_output_review(response_text="   ")
        self.assertEqual(review.decision, "blocked")
        self.assertIn("empty_nia_response", review.blocking_reasons)

        blocked = await self.store.approve_output(
            self.record.event_key,
            content_hash=review.content_hash,
            actor="hugo",
        )
        rejected = await self.store.reject_output(
            self.record.event_key,
            content_hash=review.content_hash,
            actor="hugo",
            reason="Respuesta vacía",
        )
        stored = await self.store.get_by_key(self.record.event_key)

        self.assertEqual(
            blocked.outcome,
            ReviewDecisionOutcome.OUTPUT_BLOCKED,
        )
        self.assertEqual(rejected.outcome, ReviewDecisionOutcome.APPLIED)
        self.assertEqual(
            stored["status"],
            ConnectorEventStatus.OUTPUT_REJECTED.value,
        )

    async def test_nia_permanent_failure_requires_owned_active_dispatch(self):
        _, started_at = await self.prepare_approved()
        claimed_at = started_at + timedelta(seconds=15)
        await self.store.claim_ready_for_nia(
            lease_owner="nia-worker",
            lease_seconds=5,
            now=claimed_at,
        )

        expired = await self.store.fail_nia_claim(
            self.record.event_key,
            "nia-worker",
            error_code="nia_http_permanent",
            now=claimed_at + timedelta(seconds=6),
        )
        stored = await self.store.get_by_key(self.record.event_key)

        self.assertFalse(expired)
        self.assertEqual(stored["status"], ConnectorEventStatus.PROCESSING.value)
        self.assertIsNone(stored["nia_response"])

    async def test_bitrix_claim_only_selects_ready_unsent_work(self):
        _, started_at = await self.prepare_ready_for_bitrix()
        claimed_at = started_at + timedelta(seconds=30)

        claimed = await self.store.claim_ready_for_bitrix(
            lease_owner="bitrix-worker",
            lease_seconds=45,
            now=claimed_at,
        )
        second = await self.store.claim_ready_for_bitrix(
            lease_owner="bitrix-worker-2",
            lease_seconds=45,
            now=claimed_at,
        )

        self.assertEqual(claimed.event_key, self.record.event_key)
        self.assertEqual(claimed.processing_stage, "bitrix_dispatch")
        self.assertEqual(claimed.bitrix_attempt_count, 1)
        self.assertIsNone(claimed.outbound_message_id)
        self.assertIsNone(second)

    async def test_bitrix_retry_is_isolated_from_other_workers(self):
        _, started_at = await self.prepare_ready_for_bitrix()
        claimed_at = started_at + timedelta(seconds=30)
        await self.store.claim_ready_for_bitrix(
            lease_owner="bitrix-worker",
            lease_seconds=30,
            now=claimed_at,
        )
        retried_at = claimed_at + timedelta(seconds=5)
        applied = await self.store.retry_bitrix_claim(
            self.record.event_key,
            "bitrix-worker",
            error_code="bitrix_timeout",
            retry_after_seconds=20,
            now=retried_at,
        )

        preflight_claim = await self.store.claim_next(
            lease_owner="preflight-worker",
            lease_seconds=30,
            now=retried_at + timedelta(seconds=20),
        )
        nia_claim = await self.store.claim_ready_for_nia(
            lease_owner="nia-worker",
            lease_seconds=30,
            now=retried_at + timedelta(seconds=20),
        )
        early = await self.store.claim_ready_for_bitrix(
            lease_owner="bitrix-worker-2",
            lease_seconds=30,
            now=retried_at + timedelta(seconds=19),
        )
        due = await self.store.claim_ready_for_bitrix(
            lease_owner="bitrix-worker-2",
            lease_seconds=30,
            now=retried_at + timedelta(seconds=20),
        )

        self.assertTrue(applied)
        self.assertIsNone(preflight_claim)
        self.assertIsNone(nia_claim)
        self.assertIsNone(early)
        self.assertIsNotNone(due)
        self.assertEqual(due.bitrix_attempt_count, 2)

    async def test_sent_message_id_is_persisted_and_prevents_reclaim(self):
        _, started_at = await self.prepare_ready_for_bitrix()
        claimed_at = started_at + timedelta(seconds=30)
        await self.store.claim_ready_for_bitrix(
            lease_owner="bitrix-worker",
            lease_seconds=30,
            now=claimed_at,
        )
        response = BitrixSendResponse.model_validate(
            {
                "result": {
                    "id": 987,
                    "uuidMap": {"forward-a": 988},
                }
            }
        )

        stale = await self.store.save_bitrix_sent(
            self.record.event_key,
            "otro-worker",
            response,
            http_status=200,
            now=claimed_at + timedelta(seconds=5),
        )
        applied = await self.store.save_bitrix_sent(
            self.record.event_key,
            "bitrix-worker",
            response,
            http_status=200,
            now=claimed_at + timedelta(seconds=5),
        )
        stored = await self.store.get_by_key(self.record.event_key)
        reclaimed = await self.store.claim_ready_for_bitrix(
            lease_owner="bitrix-worker-2",
            lease_seconds=30,
            now=claimed_at + timedelta(seconds=6),
        )

        self.assertFalse(stale)
        self.assertTrue(applied)
        self.assertEqual(stored["status"], ConnectorEventStatus.SENT.value)
        self.assertEqual(stored["outbound_message_id"], 987)
        self.assertEqual(
            stored["bitrix_dispatch"]["uuid_map"],
            {"forward-a": 988},
        )
        self.assertIsNone(stored["processing_stage"])
        self.assertIsNone(stored["lease_owner"])
        self.assertIsNone(reclaimed)

        self.database.collection.documents[self.record.event_key][
            "status"
        ] = ConnectorEventStatus.READY_FOR_BITRIX.value
        still_not_reclaimed = await self.store.claim_ready_for_bitrix(
            lease_owner="bitrix-worker-3",
            lease_seconds=30,
            now=claimed_at + timedelta(seconds=7),
        )
        self.assertIsNone(still_not_reclaimed)

    async def test_bitrix_failure_requires_owned_active_unsent_claim(self):
        _, started_at = await self.prepare_ready_for_bitrix()
        claimed_at = started_at + timedelta(seconds=30)
        await self.store.claim_ready_for_bitrix(
            lease_owner="bitrix-worker",
            lease_seconds=5,
            now=claimed_at,
        )

        expired = await self.store.fail_bitrix_claim(
            self.record.event_key,
            "bitrix-worker",
            error_code="bitrix_api_permanent",
            now=claimed_at + timedelta(seconds=6),
        )
        stored = await self.store.get_by_key(self.record.event_key)

        self.assertFalse(expired)
        self.assertEqual(stored["status"], ConnectorEventStatus.PROCESSING.value)
        self.assertIsNone(stored["outbound_message_id"])


if __name__ == "__main__":
    unittest.main()

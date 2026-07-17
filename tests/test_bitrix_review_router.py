import unittest
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bitrix_connector.config import load_settings
from bitrix_connector.event_parser import parse_webhook_form
from bitrix_connector.models import ConnectorEventStatus
from bitrix_connector.nia_client import NiaChatResponse
from bitrix_connector.output_review import build_output_review
from bitrix_connector.preflight import build_text_preflight
from bitrix_connector.review import (
    ConnectorReviewService,
    ReviewDecisionOutcome,
    ReviewDecisionResult,
    ReviewInputDecision,
)
from bitrix_connector.review_router import create_review_router
from bitrix_connector.runtime import ConnectorRuntimeUnavailable
from bitrix_connector.storage import build_received_record


def review_document(text="Necesito una bomba"):
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
        "data[bot][auth][access_token]": "oauth-secret",
    }
    instant = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
    record = build_received_record(
        parse_webhook_form(form),
        form,
        identity_verified=True,
        security_reason="identity_verified",
        received_at=instant,
    )
    review = build_text_preflight(record)
    return {
        "event_key": record.event_key,
        "status": ConnectorEventStatus.NEEDS_INPUT_REVIEW.value,
        "received_at": instant,
        "updated_at": instant,
        "preflight_review": review.model_dump(mode="python"),
    }


def output_review_document(text="Respuesta comercial revisable"):
    form = {
        "event": "ONIMBOTV2MESSAGEADD",
        "data[bot][id]": "456",
        "data[message][id]": "790",
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
    instant = datetime(2026, 7, 16, 12, 10, tzinfo=timezone.utc)
    record = build_received_record(
        parse_webhook_form(form),
        form,
        identity_verified=True,
        security_reason="identity_verified",
        received_at=instant,
    )
    review = build_output_review(
        record,
        NiaChatResponse(
            respuesta=text,
            etapa="preguntas_tecnicas",
            cliente={"id": "contact-27"},
        ),
    )
    return {
        "event_key": record.event_key,
        "status": ConnectorEventStatus.NEEDS_OUTPUT_REVIEW.value,
        "received_at": instant,
        "updated_at": instant,
        "output_review": review.model_dump(mode="python"),
    }


class FakeReviewStore:
    def __init__(self, documents, output_documents=None):
        self.documents = documents
        self.output_documents = output_documents or []
        self.calls = []
        self.next_decision_result = None

    async def list_reviews(self, *, decision=None, limit=50):
        self.calls.append(("list", decision, limit))
        documents = self.documents
        if decision is not None:
            documents = [
                document
                for document in documents
                if document["preflight_review"]["decision"] == decision
            ]
        return documents[:limit]

    async def get_review(self, event_key):
        self.calls.append(("get", event_key))
        return next(
            (
                document
                for document in self.documents
                if document["event_key"] == event_key
            ),
            None,
        )

    async def approve_input(
        self,
        event_key,
        *,
        content_hash,
        actor,
        reason=None,
    ):
        self.calls.append(
            ("approve", event_key, content_hash, actor, reason)
        )
        return self.next_decision_result or self._applied_result(
            event_key,
            ReviewInputDecision.APPROVED,
            ConnectorEventStatus.READY_FOR_NIA.value,
            content_hash,
            actor,
            reason,
        )

    async def reject_input(
        self,
        event_key,
        *,
        content_hash,
        actor,
        reason=None,
    ):
        self.calls.append(
            ("reject", event_key, content_hash, actor, reason)
        )
        return self.next_decision_result or self._applied_result(
            event_key,
            ReviewInputDecision.REJECTED,
            ConnectorEventStatus.INPUT_REJECTED.value,
            content_hash,
            actor,
            reason,
        )

    async def list_output_reviews(self, *, decision=None, limit=50):
        self.calls.append(("list_output", decision, limit))
        documents = self.output_documents
        if decision is not None:
            documents = [
                document
                for document in documents
                if document["output_review"]["decision"] == decision
            ]
        return documents[:limit]

    async def get_output_review(self, event_key):
        self.calls.append(("get_output", event_key))
        return next(
            (
                document
                for document in self.output_documents
                if document["event_key"] == event_key
            ),
            None,
        )

    async def approve_output(
        self,
        event_key,
        *,
        content_hash,
        actor,
        reason=None,
    ):
        self.calls.append(
            ("approve_output", event_key, content_hash, actor, reason)
        )
        return self.next_decision_result or self._applied_result(
            event_key,
            ReviewInputDecision.APPROVED,
            ConnectorEventStatus.READY_FOR_BITRIX.value,
            content_hash,
            actor,
            reason,
        )

    async def reject_output(
        self,
        event_key,
        *,
        content_hash,
        actor,
        reason=None,
    ):
        self.calls.append(
            ("reject_output", event_key, content_hash, actor, reason)
        )
        return self.next_decision_result or self._applied_result(
            event_key,
            ReviewInputDecision.REJECTED,
            ConnectorEventStatus.OUTPUT_REJECTED.value,
            content_hash,
            actor,
            reason,
        )

    @staticmethod
    def _applied_result(
        event_key,
        decision,
        status,
        content_hash,
        actor,
        reason,
    ):
        return ReviewDecisionResult(
            event_key=event_key,
            outcome=ReviewDecisionOutcome.APPLIED,
            decision=decision,
            status=status,
            content_hash=content_hash,
            actor=actor,
            reason=reason,
            decided_at=datetime(2026, 7, 16, 12, 5, tzinfo=timezone.utc),
        )


class UnavailableReader:
    async def list_reviews(self, **kwargs):
        raise ConnectorRuntimeUnavailable("connector_runtime_not_ready")

    async def get_review(self, event_key):
        raise ConnectorRuntimeUnavailable("connector_runtime_not_ready")

    async def approve_input(self, event_key, request):
        raise ConnectorRuntimeUnavailable("connector_runtime_not_ready")

    async def reject_input(self, event_key, request):
        raise ConnectorRuntimeUnavailable("connector_runtime_not_ready")

    async def list_output_reviews(self, **kwargs):
        raise ConnectorRuntimeUnavailable("connector_runtime_not_ready")

    async def get_output_review(self, event_key):
        raise ConnectorRuntimeUnavailable("connector_runtime_not_ready")

    async def approve_output(self, event_key, request):
        raise ConnectorRuntimeUnavailable("connector_runtime_not_ready")

    async def reject_output(self, event_key, request):
        raise ConnectorRuntimeUnavailable("connector_runtime_not_ready")


class ReviewRouterTests(unittest.TestCase):
    def setUp(self):
        self.document = review_document()
        self.output_document = output_review_document()
        self.store = FakeReviewStore(
            [self.document],
            [self.output_document],
        )
        self.service = ConnectorReviewService(self.store)
        self.settings = load_settings(
            {"NIA_BITRIX_REVIEW_TOKEN": "review-secret"}
        )
        app = FastAPI()
        app.include_router(
            create_review_router(
                self.service,
                settings_loader=lambda: self.settings,
            )
        )
        self.client = TestClient(app)
        self.headers = {"Authorization": "Bearer review-secret"}

    def test_missing_or_wrong_bearer_is_rejected_before_repository(self):
        missing = self.client.get("/review")
        wrong = self.client.get(
            "/review",
            headers={"Authorization": "Bearer attacker-token"},
        )

        self.assertEqual(missing.status_code, 401)
        self.assertEqual(wrong.status_code, 401)
        self.assertEqual(wrong.json()["detail"], "review_unauthorized")
        self.assertEqual(wrong.headers["www-authenticate"], "Bearer")
        self.assertEqual(self.store.calls, [])
        self.assertNotIn("attacker-token", wrong.text)
        self.assertNotIn("review-secret", wrong.text)

    def test_unconfigured_review_token_returns_503_without_repository(self):
        app = FastAPI()
        app.include_router(
            create_review_router(
                self.service,
                settings_loader=lambda: load_settings({}),
            )
        )

        response = TestClient(app).get(
            "/review",
            headers={"Authorization": "Bearer anything"},
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"],
            "review_token_not_configured",
        )
        self.assertEqual(self.store.calls, [])

    def test_authorized_list_is_filtered_and_contains_no_message_text(self):
        response = self.client.get(
            "/review?decision=ready_for_review&limit=1",
            headers=self.headers,
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["returned"], 1)
        self.assertEqual(len(payload["items"]), 1)
        self.assertEqual(payload["items"][0]["decision"], "ready_for_review")
        self.assertEqual(
            payload["items"][0]["content_hash"],
            self.document["preflight_review"]["content_hash"],
        )
        self.assertEqual(self.store.calls, [("list", "ready_for_review", 1)])
        self.assertNotIn("preflight_review", response.text)
        self.assertNotIn("Necesito una bomba", response.text)
        self.assertNotIn("secret-token", response.text)

    def test_detail_returns_separated_views_without_credentials(self):
        response = self.client.get(
            f"/review/{self.document['event_key']}",
            headers=self.headers,
        )

        self.assertEqual(response.status_code, 200)
        review = response.json()["preflight_review"]
        self.assertIn("raw_redacted", review)
        self.assertIn("content_hash", review)
        self.assertIn("normalized_input", review)
        self.assertIn("preflight_manifest", review)
        self.assertIn("nia_payload_preview", review)
        self.assertEqual(
            review["raw_redacted"]["auth[application_token]"],
            "[REDACTED]",
        )
        self.assertNotIn("secret-token", response.text)
        self.assertNotIn("oauth-secret", response.text)

    def test_output_list_is_summarized_without_message_text(self):
        response = self.client.get(
            "/review/output?decision=ready_for_review&limit=1",
            headers=self.headers,
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["returned"], 1)
        item = payload["items"][0]
        self.assertEqual(item["bot_id"], 456)
        self.assertEqual(item["dialog_id"], "chat5")
        self.assertEqual(
            item["message_length"],
            len("Respuesta comercial revisable"),
        )
        self.assertEqual(item["nia_etapa"], "preguntas_tecnicas")
        self.assertNotIn("Respuesta comercial revisable", response.text)
        self.assertNotIn("bitrix_payload_preview", response.text)
        self.assertEqual(
            self.store.calls,
            [("list_output", "ready_for_review", 1)],
        )

    def test_output_detail_exposes_exact_separated_views(self):
        event_key = self.output_document["event_key"]
        response = self.client.get(
            f"/review/output/{event_key}",
            headers=self.headers,
        )

        self.assertEqual(response.status_code, 200)
        review = response.json()["output_review"]
        self.assertEqual(
            review["nia_response"]["respuesta"],
            "Respuesta comercial revisable",
        )
        self.assertEqual(
            review["bitrix_payload_preview"],
            {
                "botId": 456,
                "dialogId": "chat5",
                "fields": {"message": "Respuesta comercial revisable"},
            },
        )
        self.assertEqual(
            review["content_hash"],
            self.output_document["output_review"]["content_hash"],
        )
        self.assertNotIn("secret-token", response.text)

    def test_output_approve_and_reject_use_exact_hash(self):
        event_key = self.output_document["event_key"]
        content_hash = self.output_document["output_review"]["content_hash"]
        body = {
            "content_hash": content_hash,
            "actor": " hugo ",
            "reason": " Salida verificada ",
        }

        approved = self.client.post(
            f"/review/{event_key}/approve-output",
            headers=self.headers,
            json=body,
        )
        rejected = self.client.post(
            f"/review/{event_key}/reject-output",
            headers=self.headers,
            json=body,
        )

        self.assertEqual(approved.status_code, 200)
        self.assertEqual(
            approved.json()["status"],
            ConnectorEventStatus.READY_FOR_BITRIX.value,
        )
        self.assertEqual(rejected.status_code, 200)
        self.assertEqual(
            rejected.json()["status"],
            ConnectorEventStatus.OUTPUT_REJECTED.value,
        )
        self.assertEqual(self.store.calls[0][0], "approve_output")
        self.assertEqual(self.store.calls[0][3], "hugo")
        self.assertEqual(self.store.calls[1][0], "reject_output")

    def test_output_blocked_conflict_is_safe(self):
        event_key = self.output_document["event_key"]
        content_hash = self.output_document["output_review"]["content_hash"]
        self.store.next_decision_result = ReviewDecisionResult(
            event_key=event_key,
            outcome=ReviewDecisionOutcome.OUTPUT_BLOCKED,
        )

        response = self.client.post(
            f"/review/{event_key}/approve-output",
            headers=self.headers,
            json={"content_hash": content_hash, "actor": "hugo"},
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"], "output_blocked")

    def test_missing_or_invalid_event_key_returns_404(self):
        invalid = self.client.get("/review/not-a-key", headers=self.headers)
        missing = self.client.get("/review/" + "0" * 64, headers=self.headers)

        self.assertEqual(invalid.status_code, 404)
        self.assertEqual(missing.status_code, 404)

    def test_router_is_read_only(self):
        response = self.client.post("/review", headers=self.headers)

        self.assertEqual(response.status_code, 405)

    def test_approve_and_reject_use_authenticated_exact_hash(self):
        event_key = self.document["event_key"]
        content_hash = self.document["preflight_review"]["content_hash"]
        body = {
            "content_hash": content_hash,
            "actor": " hugo ",
            "reason": " Verificado ",
        }

        approved = self.client.post(
            f"/review/{event_key}/approve-input",
            headers=self.headers,
            json=body,
        )
        rejected = self.client.post(
            f"/review/{event_key}/reject-input",
            headers=self.headers,
            json=body,
        )

        self.assertEqual(approved.status_code, 200)
        self.assertEqual(approved.json()["decision"], "approved")
        self.assertFalse(approved.json()["idempotent"])
        self.assertEqual(rejected.status_code, 200)
        self.assertEqual(rejected.json()["decision"], "rejected")
        self.assertEqual(self.store.calls[0][0], "approve")
        self.assertEqual(self.store.calls[0][3], "hugo")
        self.assertEqual(self.store.calls[0][4], "Verificado")
        self.assertEqual(self.store.calls[1][0], "reject")

    def test_decision_conflicts_have_stable_safe_responses(self):
        event_key = self.document["event_key"]
        body = {
            "content_hash": self.document["preflight_review"]["content_hash"],
            "actor": "hugo",
        }
        cases = [
            (
                ReviewDecisionOutcome.STALE_HASH,
                "review_hash_mismatch",
            ),
            (
                ReviewDecisionOutcome.PREFLIGHT_BLOCKED,
                "preflight_blocked",
            ),
            (
                ReviewDecisionOutcome.ALREADY_DECIDED,
                "review_already_decided",
            ),
        ]

        for outcome, detail in cases:
            self.store.next_decision_result = ReviewDecisionResult(
                event_key=event_key,
                outcome=outcome,
            )
            response = self.client.post(
                f"/review/{event_key}/approve-input",
                headers=self.headers,
                json=body,
            )
            self.assertEqual(response.status_code, 409)
            self.assertEqual(response.json()["detail"], detail)

    def test_exact_repeated_decision_is_reported_idempotent(self):
        event_key = self.document["event_key"]
        content_hash = self.document["preflight_review"]["content_hash"]
        self.store.next_decision_result = ReviewDecisionResult(
            event_key=event_key,
            outcome=ReviewDecisionOutcome.IDEMPOTENT,
            decision=ReviewInputDecision.APPROVED,
            status=ConnectorEventStatus.READY_FOR_NIA.value,
            content_hash=content_hash,
            actor="hugo",
            decided_at=datetime(2026, 7, 16, 12, 5, tzinfo=timezone.utc),
        )

        response = self.client.post(
            f"/review/{event_key}/approve-input",
            headers=self.headers,
            json={"content_hash": content_hash, "actor": "hugo"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["idempotent"])

    def test_unauthorized_decision_never_reaches_repository(self):
        event_key = self.document["event_key"]
        response = self.client.post(
            f"/review/{event_key}/approve-input",
            headers={"Authorization": "Bearer wrong-token"},
            json={
                "content_hash": self.document["preflight_review"][
                    "content_hash"
                ],
                "actor": "hugo",
            },
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(self.store.calls, [])

    def test_unauthorized_invalid_body_is_rejected_before_validation(self):
        event_key = self.document["event_key"]

        response = self.client.post(
            f"/review/{event_key}/approve-input",
            json={},
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "review_unauthorized")
        self.assertEqual(self.store.calls, [])

    def test_runtime_unavailable_returns_safe_503(self):
        app = FastAPI()
        app.include_router(
            create_review_router(
                UnavailableReader(),
                settings_loader=lambda: self.settings,
            )
        )
        client = TestClient(app)

        listed = client.get("/review", headers=self.headers)
        detailed = client.get("/review/" + "0" * 64, headers=self.headers)
        decided = client.post(
            "/review/" + "0" * 64 + "/approve-input",
            headers=self.headers,
            json={
                "content_hash": "0" * 64,
                "actor": "hugo",
            },
        )
        output_list = client.get("/review/output", headers=self.headers)
        output_detail = client.get(
            "/review/output/" + "0" * 64,
            headers=self.headers,
        )
        output_decided = client.post(
            "/review/" + "0" * 64 + "/approve-output",
            headers=self.headers,
            json={
                "content_hash": "0" * 64,
                "actor": "hugo",
            },
        )

        self.assertEqual(listed.status_code, 503)
        self.assertEqual(detailed.status_code, 503)
        self.assertEqual(decided.status_code, 503)
        self.assertEqual(output_list.status_code, 503)
        self.assertEqual(output_detail.status_code, 503)
        self.assertEqual(output_decided.status_code, 503)
        self.assertEqual(
            listed.json()["detail"],
            "connector_runtime_not_ready",
        )


if __name__ == "__main__":
    unittest.main()

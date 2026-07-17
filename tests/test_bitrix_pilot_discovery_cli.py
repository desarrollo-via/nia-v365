import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import AsyncMock, patch

from bitrix_connector.pilot_discovery_cli import build_parser, main


class PilotDiscoveryCliTests(unittest.TestCase):
    def test_parser_exposes_only_read_discovery_inputs(self):
        parser = build_parser()
        destinations = {
            action.dest
            for action in parser._actions
            if action.dest != "help"
        }

        self.assertEqual(
            destinations,
            {
                "portal_url",
                "deal_id",
                "member_id",
                "bot_id",
                "timeout_seconds",
            },
        )

    def test_cli_reads_token_hidden_and_prints_only_safe_result(self):
        safe_result = {
            "status": "found",
            "reason": "pilot_discovery_candidates_found",
            "candidates": [
                {
                    "chat_id": 1763,
                    "dialog_id": "chat1763",
                    "connector_id": "whatsapp",
                    "connector_title": "WhatsApp",
                    "crm_entity_type": "deal",
                    "crm_entity_id": 663001,
                    "pilot_rule": None,
                }
            ],
            "retry_after_seconds": 0,
        }
        token_reader = unittest.mock.Mock(
            return_value="oauth-secret-token"
        )
        output = io.StringIO()

        with patch(
            "bitrix_connector.pilot_discovery_cli."
            "execute_read_only_discovery",
            new=AsyncMock(return_value=safe_result),
        ) as execute:
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "--portal-url",
                        "https://portal.bitrix24.test",
                        "--deal-id",
                        "663001",
                    ],
                    token_reader=token_reader,
                )

        self.assertEqual(exit_code, 0)
        token_reader.assert_called_once()
        execute.assert_awaited_once_with(
            portal_url="https://portal.bitrix24.test",
            deal_id=663001,
            access_token="oauth-secret-token",
            member_id=None,
            bot_id=None,
            timeout_seconds=10.0,
        )
        self.assertIn('"dialog_id": "chat1763"', output.getvalue())
        self.assertNotIn("oauth-secret-token", output.getvalue())

    def test_partial_rule_identity_stops_before_token_prompt(self):
        token_reader = unittest.mock.Mock()

        with self.assertRaises(SystemExit):
            main(
                [
                    "--portal-url",
                    "https://portal.bitrix24.test",
                    "--deal-id",
                    "663001",
                    "--member-id",
                    "member-123",
                ],
                token_reader=token_reader,
            )

        token_reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()

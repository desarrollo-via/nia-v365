"""Comando local y explícito para una consulta piloto de solo lectura."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
from collections.abc import Callable, Sequence
from typing import Optional

from .pilot_discovery import (
    BitrixPilotDiscoveryClient,
    PilotChatInspector,
    PilotDiscoveryRequest,
    PilotDiscoveryStatus,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Consulta los chats de una negociación mediante "
            "imopenlines.crm.chat.get; no modifica Bitrix."
        )
    )
    parser.add_argument("--portal-url", required=True)
    parser.add_argument("--deal-id", required=True, type=int)
    parser.add_argument("--member-id")
    parser.add_argument("--bot-id", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    return parser


async def execute_read_only_discovery(
    *,
    portal_url: str,
    deal_id: int,
    access_token: str,
    member_id: Optional[str] = None,
    bot_id: Optional[int] = None,
    timeout_seconds: float = 10.0,
) -> dict[str, object]:
    request = PilotDiscoveryRequest(
        crm_entity_type="deal",
        crm_entity_id=deal_id,
        member_id=member_id,
        bot_id=bot_id,
        active_only=False,
    )
    async with BitrixPilotDiscoveryClient(
        portal_url=portal_url,
        access_token=access_token,
        timeout_seconds=timeout_seconds,
    ) as client:
        result = await PilotChatInspector(client).inspect(request)
    return result.model_dump(mode="json")


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    token_reader: Callable[[str], str] = getpass.getpass,
) -> int:
    args = build_parser().parse_args(argv)
    if (args.member_id is None) != (args.bot_id is None):
        raise SystemExit(
            "--member-id y --bot-id deben suministrarse juntos"
        )

    access_token = token_reader("OAuth access token (entrada oculta): ")
    output = asyncio.run(
        execute_read_only_discovery(
            portal_url=args.portal_url,
            deal_id=args.deal_id,
            access_token=access_token,
            member_id=args.member_id,
            bot_id=args.bot_id,
            timeout_seconds=args.timeout_seconds,
        )
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if output["status"] in {
        PilotDiscoveryStatus.FOUND.value,
        PilotDiscoveryStatus.EMPTY.value,
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())

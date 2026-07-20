"""Recursos efímeros para consultar el estado OAuth persistido."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient

from .config import ConnectorSettings
from .installation_status import OAuthInstallationStatusService
from .oauth import MongoBitrixOAuthStore


class OAuthInstallationStatusConfigurationError(ValueError):
    """El diagnóstico no tiene una configuración de lectura completa."""


@dataclass
class OAuthInstallationStatusResources:
    client: Any
    service: OAuthInstallationStatusService

    async def close(self) -> None:
        self.client.close()


class OAuthInstallationStatusFactory:
    def __init__(self, client_factory: Any = AsyncIOMotorClient) -> None:
        self._client_factory = client_factory

    async def build(
        self,
        settings: ConnectorSettings,
    ) -> OAuthInstallationStatusResources:
        if not settings.bitrix_domain:
            raise OAuthInstallationStatusConfigurationError(
                "installation_domain_not_configured"
            )
        if not settings.mongo_uri:
            raise OAuthInstallationStatusConfigurationError(
                "installation_mongo_uri_not_configured"
            )
        if not settings.mongo_db:
            raise OAuthInstallationStatusConfigurationError(
                "installation_mongo_db_not_configured"
            )
        if not settings.installations_collection:
            raise OAuthInstallationStatusConfigurationError(
                "installation_collection_not_configured"
            )

        client = self._client_factory(settings.mongo_uri)
        try:
            collection = client[settings.mongo_db][
                settings.installations_collection
            ]
            service = OAuthInstallationStatusService(
                MongoBitrixOAuthStore(collection)
            )
            return OAuthInstallationStatusResources(
                client=client,
                service=service,
            )
        except Exception:
            client.close()
            raise

"""Construcción aislada del almacenamiento usado solo durante instalación."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from motor.motor_asyncio import AsyncIOMotorClient

from .config import ConnectorSettings
from .installation import (
    BitrixOAuthCallbackVerifier,
    BitrixOAuthInstallationService,
)
from .oauth import MongoBitrixOAuthStore


class OAuthInstallationConfigurationError(ValueError):
    """La ventana de instalación no tiene configuración segura completa."""


@dataclass
class OAuthInstallationResources:
    client: Any
    verifier: BitrixOAuthCallbackVerifier
    installer: BitrixOAuthInstallationService

    async def close(self) -> None:
        await self.verifier.close()
        self.client.close()


class OAuthInstallationFactory:
    """Abre recursos exclusivamente para el callback y los cierra al terminar."""

    def __init__(
        self,
        client_factory: Any = AsyncIOMotorClient,
        verifier_factory: Callable[[], BitrixOAuthCallbackVerifier] = (
            BitrixOAuthCallbackVerifier
        ),
    ) -> None:
        self._client_factory = client_factory
        self._verifier_factory = verifier_factory

    async def build(
        self,
        settings: ConnectorSettings,
    ) -> OAuthInstallationResources:
        if (
            not settings.installation_enabled
            or not settings.installation_configuration_valid
        ):
            raise OAuthInstallationConfigurationError(
                "installation_window_disabled"
            )
        if not settings.bitrix_domain:
            raise OAuthInstallationConfigurationError(
                "installation_domain_not_configured"
            )
        if not settings.mongo_uri:
            raise OAuthInstallationConfigurationError(
                "installation_mongo_uri_not_configured"
            )
        if not settings.mongo_db:
            raise OAuthInstallationConfigurationError(
                "installation_mongo_db_not_configured"
            )
        if not settings.installations_collection:
            raise OAuthInstallationConfigurationError(
                "installation_collection_not_configured"
            )

        client = self._client_factory(settings.mongo_uri)
        verifier = None
        try:
            collection = client[settings.mongo_db][
                settings.installations_collection
            ]
            store = MongoBitrixOAuthStore(collection)
            await store.ensure_indexes()
            verifier = self._verifier_factory()
            installer = BitrixOAuthInstallationService(store, verifier)
            return OAuthInstallationResources(
                client=client,
                verifier=verifier,
                installer=installer,
            )
        except Exception:
            if verifier is not None:
                await verifier.close()
            client.close()
            raise

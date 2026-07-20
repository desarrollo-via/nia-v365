"""Ruta administrativa protegida para confirmar una instalación OAuth."""

from __future__ import annotations

from typing import Callable, Optional, Protocol

from fastapi import APIRouter, Depends, Header, HTTPException

from .config import ConnectorSettings, load_settings
from .installation_status import OAuthInstallationStatusResponse
from .installation_status_factory import (
    OAuthInstallationStatusConfigurationError,
    OAuthInstallationStatusFactory,
    OAuthInstallationStatusResources,
)
from .security import validate_review_access


class OAuthInstallationStatusReader(Protocol):
    async def get_status(
        self,
        domain: str,
    ) -> OAuthInstallationStatusResponse: ...


def create_installation_status_router(
    reader: Optional[OAuthInstallationStatusReader] = None,
    *,
    settings_loader: Callable[[], ConnectorSettings] = load_settings,
    resources_factory: Optional[OAuthInstallationStatusFactory] = None,
) -> APIRouter:
    router = APIRouter(tags=["Bitrix Connector Installation Diagnostics"])
    factory = resources_factory or OAuthInstallationStatusFactory()

    def authorize(authorization: str = Header(default="")) -> None:
        decision = validate_review_access(authorization, settings_loader())
        if decision.accepted:
            return
        if decision.reason == "review_token_not_configured":
            raise HTTPException(
                status_code=503,
                detail="review_token_not_configured",
            )
        raise HTTPException(
            status_code=401,
            detail="review_unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )

    @router.get(
        "/installation-status",
        response_model=OAuthInstallationStatusResponse,
    )
    async def installation_status(
        _authorized: None = Depends(authorize),
    ) -> OAuthInstallationStatusResponse:
        settings = settings_loader()
        resources: Optional[OAuthInstallationStatusResources] = None
        selected_reader = reader
        if selected_reader is None:
            try:
                resources = await factory.build(settings)
                selected_reader = resources.service
            except OAuthInstallationStatusConfigurationError as exc:
                raise HTTPException(
                    status_code=503,
                    detail="installation_diagnostic_not_configured",
                ) from exc
            except Exception as exc:
                raise HTTPException(
                    status_code=503,
                    detail="installation_diagnostic_unavailable",
                ) from exc

        try:
            return await selected_reader.get_status(settings.bitrix_domain or "")
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="installation_diagnostic_unavailable",
            ) from exc
        finally:
            if resources is not None:
                await resources.close()

    return router

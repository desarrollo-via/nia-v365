"""Callback OAuth público, verificable y habilitable independientemente."""

from __future__ import annotations

import logging
from typing import Callable, Optional, Protocol

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .config import ConnectorSettings, load_settings
from .installation import (
    OAuthInstallationPersistenceError,
    OAuthInstallationResult,
    OAuthInstallationStatus,
    OAuthInstallationVerificationError,
)
from .installation_factory import (
    OAuthInstallationConfigurationError,
    OAuthInstallationFactory,
    OAuthInstallationResources,
)


logger = logging.getLogger("nia.bitrix_connector.installation")


def _log_outcome(
    *,
    status: str,
    reason: str,
    persisted: bool = False,
    revision: Optional[int] = None,
) -> None:
    logger.info(
        "installation_callback status=%s reason=%s persisted=%s revision=%s",
        status,
        reason,
        persisted,
        revision,
    )


class OAuthInstaller(Protocol):
    async def install(
        self,
        raw_form,
        settings: ConnectorSettings,
    ) -> OAuthInstallationResult: ...


def create_installation_router(
    installer: Optional[OAuthInstaller] = None,
    *,
    settings_loader: Callable[[], ConnectorSettings] = load_settings,
    resources_factory: Optional[OAuthInstallationFactory] = None,
) -> APIRouter:
    router = APIRouter(tags=["Bitrix Connector Installation"])
    factory = resources_factory or OAuthInstallationFactory()

    @router.post("/installation", response_model=OAuthInstallationResult)
    async def install_oauth(
        request: Request,
    ):
        settings = settings_loader()
        if not settings.installation_configuration_valid:
            _log_outcome(
                status="rejected",
                reason="installation_configuration_invalid",
            )
            return JSONResponse(
                status_code=503,
                content={
                    "status": "rejected",
                    "reason": "installation_configuration_invalid",
                    "persisted": False,
                    "revision": None,
                },
            )
        if not settings.installation_enabled:
            _log_outcome(
                status="rejected",
                reason="installation_route_disabled",
            )
            return JSONResponse(
                status_code=503,
                content={
                    "status": "rejected",
                    "reason": "installation_route_disabled",
                    "persisted": False,
                    "revision": None,
                },
            )

        try:
            incoming = await request.form()
            flat_form = {
                str(key): value
                for key, value in incoming.multi_items()
            }
        except Exception:
            _log_outcome(
                status="invalid",
                reason="invalid_installation_payload",
            )
            return JSONResponse(
                status_code=400,
                content={
                    "status": "invalid",
                    "reason": "invalid_installation_payload",
                    "persisted": False,
                    "revision": None,
                },
            )

        resources: Optional[OAuthInstallationResources] = None
        selected_installer = installer
        if selected_installer is None:
            try:
                resources = await factory.build(settings)
                selected_installer = resources.installer
            except OAuthInstallationConfigurationError:
                _log_outcome(
                    status="rejected",
                    reason="installation_service_not_configured",
                )
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "rejected",
                        "reason": "installation_service_not_configured",
                        "persisted": False,
                        "revision": None,
                    },
                )
            except Exception:
                _log_outcome(
                    status="rejected",
                    reason="oauth_installation_storage_unavailable",
                )
                return JSONResponse(
                    status_code=503,
                    headers={"Retry-After": "5"},
                    content={
                        "status": "rejected",
                        "reason": "oauth_installation_storage_unavailable",
                        "persisted": False,
                        "revision": None,
                    },
                )

        try:
            result = await selected_installer.install(flat_form, settings)
        except (
            OAuthInstallationPersistenceError,
            OAuthInstallationVerificationError,
        ):
            _log_outcome(
                status="rejected",
                reason="oauth_installation_temporarily_unavailable",
            )
            return JSONResponse(
                status_code=503,
                headers={"Retry-After": "5"},
                content={
                    "status": "rejected",
                    "reason": "oauth_installation_temporarily_unavailable",
                    "persisted": False,
                    "revision": None,
                },
            )
        except Exception:
            _log_outcome(
                status="invalid",
                reason="invalid_installation_payload",
            )
            return JSONResponse(
                status_code=400,
                content={
                    "status": "invalid",
                    "reason": "invalid_installation_payload",
                    "persisted": False,
                    "revision": None,
                },
            )
        finally:
            if resources is not None:
                await resources.close()

        _log_outcome(
            status=result.status.value,
            reason=result.reason,
            persisted=result.persisted,
            revision=result.revision,
        )
        if result.status is OAuthInstallationStatus.INVALID:
            return JSONResponse(status_code=400, content=result.model_dump(mode="json"))
        if result.status is OAuthInstallationStatus.REJECTED:
            return JSONResponse(status_code=403, content=result.model_dump(mode="json"))
        return result

    return router

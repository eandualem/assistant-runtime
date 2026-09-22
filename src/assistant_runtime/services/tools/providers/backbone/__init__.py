"""The agent-backbone provider: peers, rooms, reminders, activity, workgroups, repositories.

Enabled by ``BACKBONE_URL``. Instances capture their configured URL and the
environment credential when this provider is built.
"""

from __future__ import annotations

import os
from functools import partial
from typing import TYPE_CHECKING, Any

from assistant_runtime.services.tools.providers.backbone._client import backbone_request
from assistant_runtime.services.tools.providers.backbone.activity import BackboneActivity
from assistant_runtime.services.tools.providers.backbone.peers import BackbonePeers
from assistant_runtime.services.tools.providers.backbone.reminders import BackboneReminders
from assistant_runtime.services.tools.providers.backbone.repositories import (
    BackboneRepositories,
)
from assistant_runtime.services.tools.providers.backbone.rooms import BackboneRooms
from assistant_runtime.services.tools.providers.backbone.workgroups import BackboneWorkgroups

if TYPE_CHECKING:
    from assistant_runtime.services.tools.providers.config import ProvidersConfig


def build_backbone_providers(config: ProvidersConfig) -> dict[str, Any]:
    """Provider objects by capability for a configured backbone."""
    request = partial(
        backbone_request,
        base_url=config.backbone_url,
        api_key=os.environ.get("BACKBONE_API_KEY", ""),
    )
    return {
        "peers": BackbonePeers(
            infrastructure_sessions=config.backbone_infrastructure_sessions,
            state_dir=config.agent_state_dir,
            request=request,
        ),
        "rooms": BackboneRooms(request=request),
        "reminders": BackboneReminders(request=request),
        "activity": BackboneActivity(request=request),
        "workgroups": BackboneWorkgroups(request=request),
        "repositories": BackboneRepositories(request=request),
    }


__all__ = ["build_backbone_providers"]

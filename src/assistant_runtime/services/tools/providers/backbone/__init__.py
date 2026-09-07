"""The agent-backbone provider: peers, rooms, reminders, activity, workgroups, repositories.

Enabled by ``BACKBONE_URL``. The HTTP client reads the URL and key at call
time so a ``.env`` loaded during startup is honoured.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
    return {
        "peers": BackbonePeers(
            infrastructure_sessions=config.backbone_infrastructure_sessions,
            state_dir=config.agent_state_dir,
        ),
        "rooms": BackboneRooms(),
        "reminders": BackboneReminders(),
        "activity": BackboneActivity(),
        "workgroups": BackboneWorkgroups(),
        "repositories": BackboneRepositories(),
    }


__all__ = ["build_backbone_providers"]

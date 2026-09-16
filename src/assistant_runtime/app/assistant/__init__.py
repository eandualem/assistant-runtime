"""Assistant module — sessions, prompt assembly, per-request agent setup.

The streaming pipeline and the routes build on this module through the
names exported here; the ``_``-prefixed files behind them are private.
"""

from assistant_runtime.app.assistant._prompt_builder import mcp_connections_fragment
from assistant_runtime.app.assistant._serialization import (
    MessageRecord,
    SteeringRecord,
    assistant_record_to_flat_messages,
    build_assistant_message_content,
    build_steering_request,
    merge_display_messages,
    path_records_to_model_history,
    sanitize_image_tool_returns,
    tree_messages_to_tree,
)
from assistant_runtime.app.assistant._session_store import SessionStore
from assistant_runtime.app.assistant._stale_tools import (
    ActionStatus,
    find_tool_entry,
    resolve_tool_entry,
)
from assistant_runtime.app.assistant.config import AssistantConfig
from assistant_runtime.app.assistant.deps import AssistantServiceDep
from assistant_runtime.app.assistant.exceptions import (
    AgentRunError,
    AssistantError,
    PromptBuildError,
    SessionError,
)
from assistant_runtime.app.assistant.interface import AssistantService
from assistant_runtime.app.assistant.models import AssistantRequest, AssistantResult

__all__ = [
    "ActionStatus",
    "AgentRunError",
    "AssistantConfig",
    "AssistantError",
    "AssistantRequest",
    "AssistantResult",
    "AssistantService",
    "AssistantServiceDep",
    "MessageRecord",
    "PromptBuildError",
    "SessionError",
    "SessionStore",
    "SteeringRecord",
    "assistant_record_to_flat_messages",
    "build_assistant_message_content",
    "build_steering_request",
    "find_tool_entry",
    "mcp_connections_fragment",
    "merge_display_messages",
    "path_records_to_model_history",
    "resolve_tool_entry",
    "sanitize_image_tool_returns",
    "tree_messages_to_tree",
]

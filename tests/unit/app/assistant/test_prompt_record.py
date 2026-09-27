"""The prompt record: a snapshot of the stable text plus the dynamic fragments."""

from assistant_runtime.app.assistant._prompt_builder import build_system_prompt
from assistant_runtime.app.assistant.models import PromptResult
from assistant_runtime.app.assistant.prompt_record import prompt_record, prompt_text
from assistant_runtime.artifacts import ArtifactDefinition, AssistantProfile
from assistant_runtime.services.tools.models import ToolSet

PROFILE = AssistantProfile(
    artifacts=(ArtifactDefinition(name="instructions", required=True, default="i"),)
)


def _prompt(**kwargs) -> PromptResult:
    return build_system_prompt(
        available_tools=ToolSet(),
        artifacts={"instructions": "Help the owner"},
        profile=PROFILE,
        artifact_extras=[("owner.preferences", "Short answers")],
        **kwargs,
    )


def test_record_and_snapshot_rebuild_the_exact_text():
    prompt = _prompt(
        session_context={"working_memory": {"active_goal": "Ship"}},
        host_context={"version": 1, "view": {"name": "board"}},
    )
    record, snapshot = prompt_record(
        prompt, profile="lead", subject=None, artifact_versions={"instructions": 3}
    )
    assert snapshot == "Help the owner\n\nShort answers"
    assert [name for name, _ in record["dynamic"]] == ["datetime", "host_context", "working_memory"]
    assert prompt_text(record, snapshot) == prompt.content
    assert record["artifact_versions"] == {"instructions": 3}
    assert [f["name"] for f in record["fragments"]][:2] == ["instructions", "owner.preferences"]


def test_an_appended_instruction_is_kept_as_the_suffix():
    prompt = _prompt(session_context={})
    appended = PromptResult(
        content=prompt.content + "\n\nSelect one action.", fragments=prompt.fragments
    )
    record, snapshot = prompt_record(appended, profile="lead", subject=None, artifact_versions={})
    assert record["suffix"] == "\n\nSelect one action."
    assert prompt_text(record, snapshot) == appended.content


def test_the_snapshot_is_the_same_from_turn_to_turn():
    first, _ = prompt_record(
        _prompt(session_context={}), profile="p", subject=None, artifact_versions={}
    )
    second, _ = prompt_record(
        _prompt(session_context={}, host_context={"version": 1, "view": {"name": "other"}}),
        profile="p",
        subject=None,
        artifact_versions={},
    )
    assert first["snapshot_hash"] == second["snapshot_hash"]

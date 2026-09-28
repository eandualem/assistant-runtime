"""The recorded prompt of an assistant message is exactly what the model received."""

from assistant_runtime.app.assistant.models import AssistantRequest
from assistant_runtime.app.assistant.prompt_record import prompt_text
from assistant_runtime.services.artifacts.models import Actor

from .test_execution import assert_terminal


def request(**overrides):
    return AssistantRequest.model_validate(
        {"id": "user-1", "session_id": "compat", "parent_id": None, "content": "Help", **overrides}
    )


async def test_the_record_rebuilds_the_instructions_the_model_received(runtime, script):
    received = []
    stream = script.stream

    async def capture(messages, info):
        received.append(info.instructions)
        async for frame in stream(messages, info):
            yield frame

    script.stream = capture
    script.steps = [["Done."]]
    await runtime.artifacts.update("instructions", "Answer briefly.", actor=Actor("host"))
    assert_terminal([e async for e in runtime.streaming.stream_message(request())])

    assistant = (await runtime.sessions.get_message_path("compat"))[-1]
    record = assistant["prompt"]
    snapshot = await runtime.sessions.prompt_snapshot(record["snapshot_hash"])
    assert received[-1] == "Answer briefly."
    assert prompt_text(record, snapshot) == received[-1]
    assert record["profile"] == "neutral"
    assert record["subject"] is None
    assert record["artifact_versions"] == {"instructions": 1}
    assert record["dynamic"] == []

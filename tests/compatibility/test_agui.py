"""AG-UI protocol on the shared turn pipeline.

An AG-UI client posts ``RunAgentInput`` bodies and reads SSE. The runtime is
real (``create_app`` with the FunctionModel script); the events are decoded
with the ``ag_ui`` protocol models, as a protocol-compatible frontend would.
"""

from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import TypeAdapter

core = pytest.importorskip("ag_ui.core")
from pydantic_ai.messages import ToolReturnPart  # noqa: E402

from assistant_runtime.main import create_app  # noqa: E402
from assistant_runtime.services.llm.interface import LlmService  # noqa: E402

from .test_execution import calls  # noqa: E402

EVENTS = TypeAdapter(core.Event)

SELECT_ITEM = {
    "name": "select_item",
    "description": "Select an item in the host application.",
    "parameters": {
        "type": "object",
        "properties": {"item": {"type": "string"}},
        "required": ["item"],
    },
}


def run_input(messages, *, thread="thread-1", run="run-1", tools=(SELECT_ITEM,), **extra):
    return {
        "threadId": thread,
        "runId": run,
        "state": {},
        "messages": messages,
        "tools": list(tools),
        "context": [],
        "forwardedProps": {},
        **extra,
    }


def user(text, *, message_id="user-1"):
    return {"id": message_id, "role": "user", "content": text}


def tool_message(call_id, content, *, message_id="tool-1", error=None):
    message = {"id": message_id, "role": "tool", "content": content, "toolCallId": call_id}
    if error is not None:
        message["error"] = error
    return message


def decode(body: str):
    events = []
    for block in body.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                events.append(EVENTS.validate_json(line[len("data: ") :]))
    return events


def types(events):
    return [e.type.value for e in events]


@pytest.fixture
async def client(isolated_services, script, monkeypatch):
    monkeypatch.setattr(LlmService, "_resolve_agent_model", lambda self, name: script.model())
    app = create_app(settings=isolated_services)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app), base_url="http://test") as http,
    ):
        http.app = app
        yield http


async def post(client, body, **kwargs):
    response = await client.post(
        "/api/agui", json=body, headers={"accept": "text/event-stream"}, **kwargs
    )
    return response, decode(response.text) if response.status_code == 200 else None


async def test_streamed_conversation_and_frontend_tool_round_trip(client, script):
    script.steps = [
        [calls(("select_item", '{"item":"sample"}', "host-1"))],
        ["Selected ", "sample."],
    ]

    response, events = await post(client, run_input([user("Pick sample")]))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert types(events)[0] == "RUN_STARTED"
    assert types(events)[-1] == "RUN_FINISHED"
    assert events[0].thread_id == "thread-1"
    assert events[0].run_id == "run-1"
    tool_events = [e for e in events if e.type.value.startswith("TOOL_CALL")]
    assert types(tool_events) == ["TOOL_CALL_START", "TOOL_CALL_ARGS", "TOOL_CALL_END"]
    assert tool_events[0].tool_call_name == "select_item"
    assert tool_events[0].tool_call_id == "host-1"
    assert json.loads(tool_events[1].delta) == {"item": "sample"}
    sessions = client.app.state.assistant_service.get_session_store()
    ctx = sessions.get_context("thread-1")
    assert ctx["pending_tool_call_id"] == "host-1"

    # The frontend ran the tool and resends the transcript with its result.
    response, events = await post(
        client,
        run_input(
            [
                user("Pick sample"),
                {
                    "id": "assistant-1",
                    "role": "assistant",
                    "content": "",
                    "toolCalls": [
                        {
                            "id": "host-1",
                            "type": "function",
                            "function": {"name": "select_item", "arguments": '{"item":"sample"}'},
                        }
                    ],
                },
                tool_message("host-1", '{"selected": true}'),
            ],
            run="run-2",
        ),
    )

    assert response.status_code == 200
    kinds = types(events)
    assert kinds[0] == "RUN_STARTED"
    assert kinds[-1] == "RUN_FINISHED"
    assert "RUN_ERROR" not in kinds
    text = "".join(e.delta for e in events if e.type.value == "TEXT_MESSAGE_CONTENT")
    assert text == "Selected sample."
    returns = [p for m in script.requests[-1] for p in m.parts if isinstance(p, ToolReturnPart)]
    assert [(p.tool_call_id, p.content, p.outcome) for p in returns] == [
        ("host-1", {"selected": True}, "success")
    ]
    assert not ctx.get("pending_tool_call_id")
    assert len(await sessions.get_message_path("thread-1")) == 2
    # The resent transcript was not replayed: the model saw one user prompt.
    assert sum("Pick sample" in str(m) for m in script.requests[-1]) == 1


async def test_two_frontend_tool_results_in_one_run(client, script):
    script.steps = [
        [
            calls(
                ("select_item", '{"item":"a"}', "host-1"), ("select_item", '{"item":"b"}', "host-2")
            )
        ],
        ["Both."],
    ]
    response, events = await post(client, run_input([user("Pick both")]))
    assert response.status_code == 200
    starts = [e for e in events if e.type.value == "TOOL_CALL_START"]
    assert [e.tool_call_id for e in starts] == ["host-1", "host-2"]

    response, events = await post(
        client,
        run_input(
            [
                user("Pick both"),
                tool_message("host-1", '{"ok": 1}'),
                tool_message("host-2", '{"ok": 2}', message_id="tool-2"),
            ],
            run="run-2",
        ),
    )
    assert response.status_code == 200
    kinds = types(events)
    assert kinds[-1] == "RUN_FINISHED"
    assert "RUN_ERROR" not in kinds
    assert "".join(e.delta for e in events if e.type.value == "TEXT_MESSAGE_CONTENT") == "Both."
    returns = [p for m in script.requests[-1] for p in m.parts if isinstance(p, ToolReturnPart)]
    assert [(p.tool_call_id, p.content) for p in returns] == [
        ("host-1", {"ok": 1}),
        ("host-2", {"ok": 2}),
    ]
    sessions = client.app.state.assistant_service.get_session_store()
    assert not sessions.get_context("thread-1").get("pending_tool_call_id")


async def test_failed_frontend_tool_and_duplicate_result(client, script):
    script.steps = [[calls(("select_item", '{"item":"x"}', "host-1"))], ["Understood."]]
    await post(client, run_input([user("Pick x")]))

    response, events = await post(
        client,
        run_input(
            [user("Pick x"), tool_message("host-1", "item not found", error="not found")],
            run="run-2",
        ),
    )
    assert response.status_code == 200
    assert "RUN_ERROR" not in types(events)
    returns = [p for m in script.requests[-1] for p in m.parts if isinstance(p, ToolReturnPart)]
    assert [(p.tool_call_id, p.outcome) for p in returns] == [("host-1", "failed")]

    response, events = await post(
        client,
        run_input(
            [user("Pick x"), tool_message("host-1", "again", message_id="tool-2")],
            run="run-3",
        ),
    )
    assert response.status_code == 200
    kinds = types(events)
    assert kinds[0] == "RUN_STARTED"
    assert kinds[-1] == "RUN_ERROR"
    assert "already recorded" in events[-1].message
    assert len(script.requests) == 2


async def test_second_user_message_appends_to_the_active_leaf(client, script):
    script.steps = [["First."], ["Second."]]
    await post(client, run_input([user("One")], tools=()))
    response, events = await post(
        client, run_input([user("One"), user("Two", message_id="user-2")], run="run-2", tools=())
    )

    assert response.status_code == 200
    assert "".join(e.delta for e in events if e.type.value == "TEXT_MESSAGE_CONTENT") == "Second."
    sessions = client.app.state.assistant_service.get_session_store()
    assert [m["id"] for m in await sessions.get_message_path("thread-1")][::2] == [
        "user-1",
        "user-2",
    ]


async def test_state_context_and_forwarded_config_reach_the_turn(client, script):
    script.steps = [["Ok."]]
    body = run_input(
        [user("Hi")],
        state={"cart": {"items": 2}},
        context=[{"description": "locale", "value": "en"}],
        forwardedProps={"config": {"max_turns": 3}},
    )
    response, _ = await post(client, body)

    assert response.status_code == 200
    ctx = client.app.state.assistant_service.get_session_store().get_context("thread-1")
    host_context = ctx["last_host_context"]
    assert host_context["extensions"]["state"] == {"cart": {"items": 2}}
    assert host_context["background"] == {"locale": "en"}
    assert [a["name"] for a in host_context["actions"]] == ["select_item"]
    assert ctx["last_request_config"].max_turns == 3


@pytest.mark.parametrize(
    ("body", "detail"),
    [
        ({"threadId": "t"}, "runId"),
        (run_input([]), "no messages"),
        (
            run_input([{"id": "a", "role": "assistant", "content": "hi"}]),
            "must end with a user or tool message",
        ),
    ],
    ids=["invalid-input", "no-messages", "assistant-last"],
)
async def test_unmappable_input_is_a_422(client, body, detail):
    response, _ = await post(client, body)
    assert response.status_code == 422
    assert detail in json.dumps(response.json())


async def test_oversized_run_input_is_a_413(client, monkeypatch):
    from assistant_runtime.app.routes import agui

    monkeypatch.setattr(agui, "MAX_BODY_BYTES", 64)
    response, _ = await post(client, run_input([user("x" * 200)]))
    assert response.status_code == 413
    response, _ = (
        await client.post("/api/agui", content=b"{}", headers={"content-length": "100000000"}),
        None,
    )
    assert response.status_code == 413


async def test_missing_extra_is_a_501(client, monkeypatch):
    from assistant_runtime.app.routes import agui

    def missing():
        raise ImportError("ag_ui")

    monkeypatch.setattr(agui, "_load_bridge", missing)
    response, _ = await post(client, run_input([user("Hi")]))
    assert response.status_code == 501
    assert "ag-ui" in response.json()["detail"]

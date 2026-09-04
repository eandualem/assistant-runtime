"""The ``assistant-runtime`` command: parser, chat rendering, doctor helpers."""

from __future__ import annotations

import io
from typing import Any

import pytest

from assistant_runtime.cli import build_parser, main
from assistant_runtime.cli.chat import (
    MAX_HOST_TOOL_ROUNDS,
    NO_HOST_RESULT,
    TurnRenderer,
    build_request,
    run_turn,
)
from assistant_runtime.cli.doctor import FAIL, OK, WARN, configured_providers, run_checks


class TestParser:
    def test_no_command_prints_help(self, capsys):
        assert main([]) == 0
        assert "chat" in capsys.readouterr().out

    def test_version_command(self, capsys):
        assert main(["version"]) == 0
        assert capsys.readouterr().out.strip()

    def test_chat_defaults(self):
        args = build_parser().parse_args(["chat"])
        assert args.model is None
        assert args.session is None
        assert not args.show_thinking

    def test_serve_binds_loopback_by_default(self):
        args = build_parser().parse_args(["serve"])
        assert args.host == "127.0.0.1"
        assert args.port == 7100


class TestTurnRenderer:
    def _render(self, events: list[dict[str, Any]], **kw: Any) -> tuple[str, TurnRenderer]:
        out = io.StringIO()
        renderer = TurnRenderer(out, **kw)
        renderer.start_turn()
        for event in events:
            renderer.handle(event)
        return out.getvalue(), renderer

    def test_text_streams_inline(self):
        text, _ = self._render(
            [
                {"type": "text_delta", "content": "Hel"},
                {"type": "text_delta", "content": "lo"},
                {"type": "final_response", "content": "Hello", "streamed": True},
            ]
        )
        assert text == "Hello\n"

    def test_unstreamed_final_is_printed(self):
        text, renderer = self._render(
            [{"type": "final_response", "content": "Hi", "streamed": False}]
        )
        assert text == "Hi\n"
        assert renderer.final_content == "Hi"

    def test_thinking_hidden_by_default(self):
        text, _ = self._render([{"type": "thinking_delta", "content": "hmm"}])
        assert text == ""

    def test_thinking_shown_on_request(self):
        text, _ = self._render([{"type": "thinking_delta", "content": "hmm"}], show_thinking=True)
        assert text == "hmm"

    def test_tool_call_gets_its_own_line(self):
        text, _ = self._render(
            [
                {"type": "text_delta", "content": "Checking"},
                {"type": "tool_call", "tool_name": "get_time", "arguments": {"tz": "UTC"}},
                {
                    "type": "tool_result",
                    "tool_name": "get_time",
                    "output": "12:00",
                    "duration_ms": 3.2,
                },
            ]
        )
        assert text == 'Checking\n[tool] get_time({"tz": "UTC"})\n[tool] get_time done in 3.2ms\n'

    def test_error_is_recorded(self):
        text, renderer = self._render([{"type": "error", "message": "rate limited"}])
        assert renderer.error == "rate limited"
        assert "[error] rate limited" in text

    def test_pending_tool_call_is_captured(self):
        _, renderer = self._render(
            [
                {
                    "type": "final_response",
                    "content": None,
                    "streamed": False,
                    "pending_tool_call": {"call_id": "c1", "tool_name": "navigate"},
                }
            ]
        )
        assert renderer.pending_tool_call == {"call_id": "c1", "tool_name": "navigate"}


class FakeStreaming:
    """Yields scripted event lists, one per stream_message call, recording requests."""

    def __init__(self, scripts: list[list[dict[str, Any]]]) -> None:
        self._scripts = list(scripts)
        self.requests: list[Any] = []

    async def stream_message(self, request):
        self.requests.append(request)
        script = self._scripts.pop(0) if self._scripts else []
        for event in script:
            yield event


def _completed(content: str, **extra: Any) -> list[dict[str, Any]]:
    return [
        {"type": "agent_status", "status": "started"},
        {"type": "final_response", "content": content, "streamed": False, **extra},
        {"type": "agent_status", "status": "completed"},
    ]


class TestRunTurn:
    async def test_plain_turn_returns_final_text(self):
        streaming = FakeStreaming([_completed("done")])
        renderer = TurnRenderer(io.StringIO())
        result = await run_turn(streaming, build_request("s1", "hi", None), renderer)
        assert result == "done"
        assert len(streaming.requests) == 1

    async def test_model_override_is_sent_as_request_config(self):
        request = build_request("s1", "hi", "anthropic:claude-sonnet-5")
        assert request.config is not None
        assert request.config.default_model == "anthropic:claude-sonnet-5"

    async def test_host_tool_is_answered_with_an_error_continuation(self):
        pending = {"call_id": "call-1", "tool_name": "navigate", "arguments": {"page": "x"}}
        streaming = FakeStreaming(
            [_completed(None, pending_tool_call=pending), _completed("after")]
        )
        out = io.StringIO()
        result = await run_turn(streaming, build_request("s1", "go", None), TurnRenderer(out))
        assert result == "after"
        continuation = streaming.requests[1]
        assert continuation.tool_call_id == "call-1"
        assert continuation.tool_result == NO_HOST_RESULT
        assert continuation.session_id == "s1"
        assert "navigate" in out.getvalue()

    async def test_host_tool_loop_is_bounded(self):
        pending = {"call_id": "c", "tool_name": "navigate", "arguments": {}}
        scripts = [
            _completed(None, pending_tool_call=pending) for _ in range(MAX_HOST_TOOL_ROUNDS + 2)
        ]
        streaming = FakeStreaming(scripts)
        renderer = TurnRenderer(io.StringIO())
        result = await run_turn(streaming, build_request("s1", "go", None), renderer)
        assert result is None
        assert len(streaming.requests) == MAX_HOST_TOOL_ROUNDS + 1

    async def test_error_event_ends_the_turn(self):
        streaming = FakeStreaming(
            [
                [
                    {"type": "error", "message": "no key"},
                    {"type": "agent_status", "status": "completed"},
                ]
            ]
        )
        result = await run_turn(
            streaming, build_request("s1", "hi", None), TurnRenderer(io.StringIO())
        )
        assert result is None


class TestDoctorHelpers:
    def test_configured_providers_reads_keys(self):
        env = {"ANTHROPIC_API_KEY": "sk-ant", "OPENAI_API_KEY": " ", "LLM__PROVIDERS_JSON": ""}
        assert configured_providers(env) == ["anthropic"]

    def test_providers_json_counts_as_a_provider(self):
        assert configured_providers({"LLM__PROVIDERS_JSON": "[{}]"}) == ["providers-json"]

    def test_run_checks_folds_exceptions_into_fail_lines(self):
        def _ok():
            return OK, "fine"

        def _many():
            return [(WARN, "a"), (OK, "b")]

        def _boom():
            raise RuntimeError("nope")

        lines = run_checks([_ok, _many, _boom])
        assert lines[:3] == [(OK, "fine"), (WARN, "a"), (OK, "b")]
        assert lines[3] == (FAIL, "boom: nope")


@pytest.mark.parametrize("argv", [["chat", "--bogus"], ["serve", "--port", "x"]])
def test_bad_arguments_exit_2(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2

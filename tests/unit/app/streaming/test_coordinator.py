"""Tests for EventCoordinator — per-request dedup and tracking."""

import pytest

from assistant_runtime.app.streaming._coordinator import EventCoordinator
from assistant_runtime.app.streaming._event_builder import (
    make_text_delta_event,
    make_thinking_delta_event,
)
from assistant_runtime.app.streaming.exceptions import EventLimitError


class TestEventCoordinatorInit:
    def test_initial_state(self):
        coord = EventCoordinator(max_events=100)
        assert coord.event_count == 0

    def test_max_events_stored(self):
        coord = EventCoordinator(max_events=50)
        assert coord._max_events == 50


class TestCoordinatorStarted:
    def test_try_started_first_time(self):
        coord = EventCoordinator(max_events=100)
        event = coord.try_started()
        assert event is not None
        assert event["type"] == "agent_status"
        assert event["status"] == "started"

    def test_try_started_duplicate_suppressed(self):
        coord = EventCoordinator(max_events=100)
        coord.try_started()
        event = coord.try_started()
        assert event is None

    def test_try_started_increments_count(self):
        coord = EventCoordinator(max_events=100)
        coord.try_started()
        assert coord.event_count == 1


class TestCoordinatorCompleted:
    def test_try_completed_first_time(self):
        coord = EventCoordinator(max_events=100)
        event = coord.try_completed()
        assert event is not None
        assert event["status"] == "completed"

    def test_try_completed_duplicate_suppressed(self):
        coord = EventCoordinator(max_events=100)
        coord.try_completed()
        event = coord.try_completed()
        assert event is None


class TestCoordinatorFinalResponse:
    def test_try_final_response_first_time(self):
        coord = EventCoordinator(max_events=100)
        event = coord.try_final_response("Hello", "model-1")
        assert event is not None
        assert event["type"] == "final_response"
        assert event["content"] == "Hello"
        assert event["model"] == "model-1"

    def test_try_final_response_null_content(self):
        coord = EventCoordinator(max_events=100)
        event = coord.try_final_response(None, "model-1")
        assert event["content"] is None

    def test_try_final_response_duplicate_suppressed(self):
        coord = EventCoordinator(max_events=100)
        coord.try_final_response("Hello", "model-1")
        event = coord.try_final_response("Duplicate", "model-1")
        assert event is None


class TestCoordinatorTrack:
    def test_track_increments_count(self):
        coord = EventCoordinator(max_events=100)
        coord.track(make_text_delta_event("hi"))
        assert coord.event_count == 1

    def test_track_returns_event(self):
        coord = EventCoordinator(max_events=100)
        event = coord.track(make_thinking_delta_event("thinking"))
        assert event["type"] == "thinking_delta"

    def test_multiple_tracks(self):
        coord = EventCoordinator(max_events=100)
        for i in range(10):
            coord.track(make_text_delta_event(f"chunk {i}"))
        assert coord.event_count == 10


class TestCoordinatorEventLimit:
    def test_exceeds_limit_raises(self):
        coord = EventCoordinator(max_events=3)
        coord.track(make_text_delta_event("1"))
        coord.track(make_text_delta_event("2"))
        coord.track(make_text_delta_event("3"))
        with pytest.raises(EventLimitError):
            coord.track(make_text_delta_event("4"))

    def test_exact_limit_ok(self):
        coord = EventCoordinator(max_events=3)
        coord.track(make_text_delta_event("1"))
        coord.track(make_text_delta_event("2"))
        coord.track(make_text_delta_event("3"))
        assert coord.event_count == 3

    def test_terminal_events_bypass_limit(self):
        """try_completed() and try_final_response() bypass the event limit."""
        coord = EventCoordinator(max_events=2)
        coord.try_started()
        coord.track(make_text_delta_event("text"))
        # At limit — regular track would raise, but terminal events must not
        completed = coord.try_completed()
        assert completed is not None
        assert completed["status"] == "completed"


class TestCoordinatorFullProtocol:
    def test_full_lifecycle(self):
        coord = EventCoordinator(max_events=100)
        started = coord.try_started()
        assert started is not None
        coord.track(make_thinking_delta_event("thinking"))
        coord.track(make_text_delta_event("text"))
        final = coord.try_final_response("result", "model-1")
        assert final is not None
        completed = coord.try_completed()
        assert completed is not None
        assert coord.event_count == 5


class TestCoordinatorTrackDebug:
    def test_track_debug_returns_event(self):
        coord = EventCoordinator(max_events=100)
        event = coord.track_debug({"type": "debug_request", "data": "test"})
        assert event["type"] == "debug_request"

    def test_track_debug_increments_debug_count(self):
        coord = EventCoordinator(max_events=100)
        coord.track_debug({"type": "debug_request"})
        coord.track_debug({"type": "debug_system_prompt"})
        assert coord.debug_event_count == 2

    def test_track_debug_does_not_increment_event_count(self):
        coord = EventCoordinator(max_events=100)
        coord.track_debug({"type": "debug_request"})
        assert coord.event_count == 0

    def test_track_debug_never_raises_limit_error(self):
        coord = EventCoordinator(max_events=1)
        coord.track(make_text_delta_event("x"))  # fills the limit
        # This should NOT raise even though we're over the regular event limit
        event = coord.track_debug({"type": "debug_request"})
        assert event is not None
        assert coord.debug_event_count == 1

    def test_debug_event_count_initial(self):
        coord = EventCoordinator(max_events=100)
        assert coord.debug_event_count == 0


class TestCoordinatorStreamingFlags:
    def test_initial_state_false(self):
        coord = EventCoordinator(max_events=100)
        assert coord.has_streamed_text is False
        assert coord.has_streamed_thinking is False

    def test_emit_text_delta_sets_flag(self):
        coord = EventCoordinator(max_events=100)
        event = coord.emit_text_delta("hello")
        assert coord.has_streamed_text is True
        assert event["type"] == "text_delta"
        assert event["content"] == "hello"
        assert event["segment_id"] == "segment_0"
        assert event["segment_index"] == 0
        assert event["delta_index"] == 0
        assert event["segment_started"] is True
        assert event["segment_kind"] == "text"

    def test_emit_text_delta_increments_count(self):
        coord = EventCoordinator(max_events=100)
        coord.emit_text_delta("a")
        assert coord.event_count == 1

    def test_emit_thinking_delta_sets_flag(self):
        coord = EventCoordinator(max_events=100)
        event = coord.emit_thinking_delta("reasoning")
        assert coord.has_streamed_thinking is True
        assert event["type"] == "thinking_delta"
        assert event["content"] == "reasoning"
        assert event["segment_id"] == "segment_0"
        assert event["segment_index"] == 0
        assert event["delta_index"] == 0
        assert event["segment_started"] is True
        assert event["segment_kind"] == "thinking"

    def test_emit_thinking_delta_increments_count(self):
        coord = EventCoordinator(max_events=100)
        coord.emit_thinking_delta("t")
        assert coord.event_count == 1

    def test_flags_persist_across_multiple_emits(self):
        coord = EventCoordinator(max_events=100)
        first = coord.emit_text_delta("a")
        second = coord.emit_text_delta("b")
        assert coord.has_streamed_text is True
        assert coord.event_count == 2
        assert first["segment_id"] == second["segment_id"]
        assert second["delta_index"] == 1
        assert second["segment_started"] is False

    def test_flags_are_independent(self):
        coord = EventCoordinator(max_events=100)
        coord.emit_text_delta("text")
        assert coord.has_streamed_text is True
        assert coord.has_streamed_thinking is False

    def test_both_flags_set(self):
        coord = EventCoordinator(max_events=100)
        thinking = coord.emit_thinking_delta("think")
        text = coord.emit_text_delta("text")
        assert coord.has_streamed_text is True
        assert coord.has_streamed_thinking is True
        assert thinking["segment_id"] != text["segment_id"]
        assert text["segment_index"] == 1


class TestCoordinatorFinalResponseStreamed:
    def test_streamed_text_clears_content(self):
        coord = EventCoordinator(max_events=100)
        coord.emit_text_delta("hello world")
        event = coord.try_final_response("hello world", "model-1")
        assert event is not None
        assert event["content"] == ""
        assert event["streamed"] is True

    def test_no_streaming_preserves_content(self):
        coord = EventCoordinator(max_events=100)
        event = coord.try_final_response("direct output", "model-1")
        assert event is not None
        assert event["content"] == "direct output"
        assert event["streamed"] is False

    def test_thinking_streamed_present(self):
        coord = EventCoordinator(max_events=100)
        coord.emit_thinking_delta("let me think")
        event = coord.try_final_response("answer", "model-1")
        assert event is not None
        assert event["thinking_streamed"] is True

    def test_no_thinking_key_absent(self):
        coord = EventCoordinator(max_events=100)
        event = coord.try_final_response("answer", "model-1")
        assert event is not None
        assert "thinking_streamed" not in event

    def test_error_preserves_content_even_when_streamed(self):
        coord = EventCoordinator(max_events=100)
        coord.emit_text_delta("partial")
        event = coord.try_final_response("error msg", "model-1", error=True)
        assert event is not None
        assert event["content"] == "error msg"

    def test_none_content_preserved(self):
        coord = EventCoordinator(max_events=100)
        coord.emit_text_delta("something")
        event = coord.try_final_response(None, "model-1")
        assert event is not None
        assert event["content"] is None

    def test_error_fields_absent_when_not_error(self):
        coord = EventCoordinator(max_events=100)
        event = coord.try_final_response("ok", "model-1")
        assert event is not None
        assert "error" not in event
        assert "error_type" not in event

    def test_session_id_passed_through(self):
        coord = EventCoordinator(max_events=100)
        event = coord.try_final_response("ok", "model-1", session_id="sess-42")
        assert event is not None
        assert event["session_id"] == "sess-42"

    def test_session_id_absent_when_not_provided(self):
        coord = EventCoordinator(max_events=100)
        event = coord.try_final_response("ok", "model-1")
        assert event is not None
        assert "session_id" not in event


class TestCoordinatorDebugEventsCollection:
    def test_debug_events_empty_initially(self):
        coord = EventCoordinator(max_events=100)
        assert coord.debug_events == []

    def test_debug_events_collects_tracked_events(self):
        coord = EventCoordinator(max_events=100)
        e1 = {"type": "debug_request", "session_id": "s1"}
        e2 = {"type": "debug_system_prompt", "total_length": 500}
        coord.track_debug(e1)
        coord.track_debug(e2)
        assert len(coord.debug_events) == 2
        assert coord.debug_events[0] is e1
        assert coord.debug_events[1] is e2

    def test_debug_events_preserves_order(self):
        coord = EventCoordinator(max_events=100)
        for i in range(5):
            coord.track_debug({"type": f"debug_{i}", "index": i})
        assert [e["index"] for e in coord.debug_events] == [0, 1, 2, 3, 4]

    def test_debug_events_independent_of_regular_events(self):
        coord = EventCoordinator(max_events=100)
        coord.track(make_text_delta_event("hello"))
        coord.track_debug({"type": "debug_request"})
        coord.track(make_thinking_delta_event("thinking"))
        assert len(coord.debug_events) == 1
        assert coord.debug_events[0]["type"] == "debug_request"


class TestCoordinatorAccumulation:
    def test_accumulated_response_empty_initially(self):
        coord = EventCoordinator(max_events=100)
        assert coord.accumulated_response == ""

    def test_accumulated_response_accumulates(self):
        coord = EventCoordinator(max_events=100)
        coord.emit_text_delta("Hello ")
        coord.emit_text_delta("world")
        assert coord.accumulated_response == "Hello world"


class TestCoordinatorFlushThinking:
    def test_flush_empty_returns_none(self):
        coord = EventCoordinator(max_events=100)
        assert coord.flush_thinking() is None

    def test_flush_returns_debug_thinking_event(self):
        coord = EventCoordinator(max_events=100)
        coord.emit_thinking_delta("some reasoning")
        event = coord.flush_thinking()
        assert event is not None
        assert event["type"] == "debug_thinking"
        assert event["content"] == "some reasoning"

    def test_flush_clears_buffer(self):
        coord = EventCoordinator(max_events=100)
        coord.emit_thinking_delta("thinking")
        coord.flush_thinking()
        # Second flush returns None — buffer was cleared
        assert coord.flush_thinking() is None

    def test_flush_adds_to_debug_events(self):
        coord = EventCoordinator(max_events=100)
        coord.emit_thinking_delta("reasoning")
        coord.flush_thinking()
        assert len(coord.debug_events) == 1
        assert coord.debug_events[0]["type"] == "debug_thinking"

    def test_multiple_flushes_produce_separate_events(self):
        """Key test: proves per-iteration separation of thinking."""
        coord = EventCoordinator(max_events=100)
        # Iteration 1: some thinking
        coord.emit_thinking_delta("first thought")
        coord.flush_thinking()
        # Iteration 2: more thinking
        coord.emit_thinking_delta("second thought")
        coord.flush_thinking()
        assert len(coord.debug_events) == 2
        assert coord.debug_events[0]["content"] == "first thought"
        assert coord.debug_events[1]["content"] == "second thought"

    def test_flush_concatenates_multiple_deltas(self):
        coord = EventCoordinator(max_events=100)
        coord.emit_thinking_delta("part one ")
        coord.emit_thinking_delta("part two")
        event = coord.flush_thinking()
        assert event is not None
        assert event["content"] == "part one part two"


class TestCoordinatorSegmentBoundaries:
    def test_track_non_content_event_resets_segment(self):
        coord = EventCoordinator(max_events=100)
        first = coord.emit_thinking_delta("first")
        coord.track({"type": "tool_call", "tool_name": "check", "arguments": {}, "call_id": "tc_1"})
        second = coord.emit_thinking_delta("second")

        assert first["segment_id"] == "segment_0"
        assert second["segment_id"] == "segment_1"

    def test_clear_content_segment_forces_new_segment(self):
        coord = EventCoordinator(max_events=100)
        first = coord.emit_text_delta("hello")
        coord.clear_content_segment()
        second = coord.emit_text_delta("world")

        assert first["segment_id"] == "segment_0"
        assert second["segment_id"] == "segment_1"


class TestCoordinatorUsage:
    def test_try_final_response_with_usage(self):
        coord = EventCoordinator(max_events=100)
        event = coord.try_final_response(
            "done",
            "model",
            usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )
        assert event is not None
        assert "usage" in event
        assert event["usage"] == {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}

    def test_try_final_response_without_usage(self):
        coord = EventCoordinator(max_events=100)
        event = coord.try_final_response("done", "model")
        assert event is not None
        assert "usage" not in event

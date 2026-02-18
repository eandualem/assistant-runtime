"""Tests for EventCoordinator — per-request dedup and tracking."""

import pytest

from lovely_assistant.app.streaming._coordinator import EventCoordinator
from lovely_assistant.app.streaming._event_builder import (
    make_text_delta_event,
    make_thinking_delta_event,
)
from lovely_assistant.app.streaming.exceptions import EventLimitError


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

    def test_protocol_events_count_toward_limit(self):
        coord = EventCoordinator(max_events=2)
        coord.try_started()
        coord.track(make_text_delta_event("text"))
        with pytest.raises(EventLimitError):
            coord.try_completed()


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

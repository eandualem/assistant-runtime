"""Tests for the in-process domain event hub."""

from assistant_runtime.base.events import EventHub


async def test_sync_and_async_handlers_receive_each_event():
    hub = EventHub()
    seen: list[tuple[str, str]] = []

    def sync_handler(event):
        seen.append(("sync", event["type"]))

    async def async_handler(event):
        seen.append(("async", event["type"]))

    hub.subscribe(sync_handler)
    hub.subscribe(async_handler)
    await hub.publish({"type": "artifact_proposal"})
    assert seen == [("sync", "artifact_proposal"), ("async", "artifact_proposal")]


async def test_a_failing_handler_does_not_stop_the_others():
    hub = EventHub()
    seen = []

    def broken(event):
        raise RuntimeError("boom")

    hub.subscribe(broken)
    hub.subscribe(seen.append)
    await hub.publish({"type": "x"})
    assert seen == [{"type": "x"}]


async def test_unsubscribe_and_handlers_get_their_own_copy():
    hub = EventHub()
    seen = []

    def mutate(event):
        event["type"] = "changed"
        seen.append(event)

    unsubscribe = hub.subscribe(mutate)
    original = {"type": "x"}
    await hub.publish(original)
    assert original == {"type": "x"}
    unsubscribe()
    unsubscribe()  # idempotent
    await hub.publish({"type": "y"})
    assert len(seen) == 1

"""Tests for context engineering models — WorkingMemory, CompactionResult, MemoryEntry, etc."""

from assistant_runtime.services.history.models import (
    CompactionResult,
    MemoryDelta,
    MemoryDeltaResult,
    MemoryEntry,
    MemoryOperation,
    WorkingMemory,
)


class TestMemoryEntry:
    def test_default_id_generated(self):
        entry = MemoryEntry(content="test", category="agent_behavior")
        assert entry.id  # Non-empty
        assert len(entry.id) == 36  # UUID format

    def test_unique_ids(self):
        e1 = MemoryEntry(content="a", category="agent_behavior")
        e2 = MemoryEntry(content="b", category="agent_behavior")
        assert e1.id != e2.id

    def test_default_confidence(self):
        entry = MemoryEntry(content="test", category="tool_behavior")
        assert entry.confidence == 0.8

    def test_default_turn_fields(self):
        entry = MemoryEntry(content="test", category="tool_behavior")
        assert entry.created_turn == 0
        assert entry.updated_turn == 0

    def test_custom_values(self):
        entry = MemoryEntry(
            content="test insight",
            category="backbone_pattern",
            confidence=0.9,
            evidence="observed during debugging",
            created_turn=5,
            updated_turn=7,
        )
        assert entry.content == "test insight"
        assert entry.category == "backbone_pattern"
        assert entry.confidence == 0.9
        assert entry.evidence == "observed during debugging"


class TestMemoryOperation:
    def test_enum_values(self):
        assert MemoryOperation.ADD == "add"
        assert MemoryOperation.UPDATE == "update"
        assert MemoryOperation.DELETE == "delete"
        assert MemoryOperation.NOOP == "noop"


class TestMemoryDeltaResult:
    def test_defaults(self):
        result = MemoryDeltaResult()
        assert result.active_goal == ""
        assert result.progress == ""
        assert result.next_steps == ""
        assert result.key_decisions == []
        assert result.open_loops == []
        assert result.deltas == []


class TestWorkingMemoryIsEmpty:
    def test_empty_by_default(self):
        wm = WorkingMemory()
        assert wm.is_empty() is True

    def test_not_empty_with_active_goal(self):
        wm = WorkingMemory(active_goal="Fix the bug")
        assert wm.is_empty() is False

    def test_not_empty_with_progress(self):
        wm = WorkingMemory(progress="Identified root cause")
        assert wm.is_empty() is False

    def test_not_empty_with_next_steps(self):
        wm = WorkingMemory(next_steps="Run tests")
        assert wm.is_empty() is False

    def test_not_empty_with_key_decisions(self):
        wm = WorkingMemory(key_decisions=["Use approach A"])
        assert wm.is_empty() is False

    def test_not_empty_with_open_loops(self):
        wm = WorkingMemory(open_loops=["Waiting for response"])
        assert wm.is_empty() is False

    def test_not_empty_with_entries(self):
        wm = WorkingMemory(entries=[MemoryEntry(content="insight", category="agent_behavior")])
        assert wm.is_empty() is False


class TestWorkingMemoryApplyDeltas:
    def test_add_creates_entry(self):
        wm = WorkingMemory()
        new_entry = MemoryEntry(content="new insight", category="tool_behavior")
        deltas = [MemoryDelta(operation=MemoryOperation.ADD, entry=new_entry)]

        wm.apply_deltas(deltas, current_turn=5)

        assert len(wm.entries) == 1
        assert wm.entries[0].content == "new insight"
        assert wm.entries[0].created_turn == 5
        assert wm.entries[0].updated_turn == 5
        # Fresh UUID assigned (not the original)
        assert wm.entries[0].id != new_entry.id

    def test_update_modifies_entry(self):
        existing = MemoryEntry(
            id="test-id-123",
            content="old content",
            category="agent_behavior",
            created_turn=1,
            updated_turn=1,
        )
        wm = WorkingMemory(entries=[existing])

        updated = MemoryEntry(content="new content", category="agent_behavior", confidence=0.95)
        deltas = [
            MemoryDelta(
                operation=MemoryOperation.UPDATE,
                entry_id="test-id-123",
                entry=updated,
            )
        ]

        wm.apply_deltas(deltas, current_turn=10)

        assert len(wm.entries) == 1
        assert wm.entries[0].content == "new content"
        assert wm.entries[0].confidence == 0.95
        assert wm.entries[0].id == "test-id-123"  # Preserved
        assert wm.entries[0].created_turn == 1  # Preserved
        assert wm.entries[0].updated_turn == 10  # Updated

    def test_delete_removes_entry(self):
        existing = MemoryEntry(id="delete-me", content="to be deleted", category="system_quirk")
        wm = WorkingMemory(entries=[existing])

        deltas = [MemoryDelta(operation=MemoryOperation.DELETE, entry_id="delete-me")]
        wm.apply_deltas(deltas, current_turn=5)

        assert len(wm.entries) == 0

    def test_delete_nonexistent_id_is_noop(self):
        existing = MemoryEntry(id="keep-me", content="stay", category="agent_behavior")
        wm = WorkingMemory(entries=[existing])

        deltas = [MemoryDelta(operation=MemoryOperation.DELETE, entry_id="nonexistent")]
        wm.apply_deltas(deltas, current_turn=5)

        assert len(wm.entries) == 1
        assert wm.entries[0].id == "keep-me"

    def test_max_entries_enforcement(self):
        wm = WorkingMemory()
        # Add entries with varying confidence
        for i in range(10):
            entry = MemoryEntry(
                content=f"entry {i}",
                category="agent_behavior",
                confidence=0.5 + (i * 0.05),
            )
            deltas = [MemoryDelta(operation=MemoryOperation.ADD, entry=entry)]
            wm.apply_deltas(deltas, current_turn=i, max_entries=5)

        assert len(wm.entries) == 5
        # Should keep highest-confidence entries
        confidences = [e.confidence for e in wm.entries]
        assert confidences == sorted(confidences, reverse=True)

    def test_update_nonexistent_id_is_noop(self):
        wm = WorkingMemory()
        updated = MemoryEntry(content="update", category="tool_behavior")
        deltas = [
            MemoryDelta(
                operation=MemoryOperation.UPDATE,
                entry_id="nonexistent",
                entry=updated,
            )
        ]
        wm.apply_deltas(deltas, current_turn=1)
        assert len(wm.entries) == 0


class TestWorkingMemoryToPromptSection:
    def test_empty_state(self):
        wm = WorkingMemory()
        assert wm.to_prompt_section() == "(No working memory from previous turns)"

    def test_with_active_goal(self):
        wm = WorkingMemory(active_goal="Fix the authentication bug")
        section = wm.to_prompt_section()
        assert "**Active Goal:** Fix the authentication bug" in section

    def test_with_progress(self):
        wm = WorkingMemory(progress="Identified the root cause")
        section = wm.to_prompt_section()
        assert "**Progress:** Identified the root cause" in section

    def test_with_next_steps(self):
        wm = WorkingMemory(next_steps="Run the test suite")
        section = wm.to_prompt_section()
        assert "**Next Steps:** Run the test suite" in section

    def test_with_key_decisions(self):
        wm = WorkingMemory(key_decisions=["Use JWT", "Implement refresh tokens"])
        section = wm.to_prompt_section()
        assert "**Key Decisions:**" in section
        assert "- Use JWT" in section
        assert "- Implement refresh tokens" in section

    def test_with_open_loops(self):
        wm = WorkingMemory(open_loops=["Waiting for API key"])
        section = wm.to_prompt_section()
        assert "**Open Loops:**" in section
        assert "- Waiting for API key" in section

    def test_with_entries(self):
        entries = [
            MemoryEntry(
                content="Agent state files use JSON format",
                category="backbone_pattern",
                confidence=0.9,
            )
        ]
        wm = WorkingMemory(entries=entries)
        section = wm.to_prompt_section()
        assert "**Operational Learnings:**" in section
        assert "[backbone_pattern] Agent state files use JSON format" in section
        assert "(confidence: 0.9)" in section

    def test_full_state(self):
        wm = WorkingMemory(
            active_goal="Deploy the service",
            progress="Tests passing",
            next_steps="Run deployment script",
            key_decisions=["Use blue-green deployment"],
            open_loops=["Need DNS update"],
            entries=[
                MemoryEntry(
                    content="Service requires port 8080",
                    category="system_quirk",
                    confidence=1.0,
                )
            ],
        )
        section = wm.to_prompt_section()
        assert "**Active Goal:**" in section
        assert "**Progress:**" in section
        assert "**Next Steps:**" in section
        assert "**Key Decisions:**" in section
        assert "**Open Loops:**" in section
        assert "**Operational Learnings:**" in section


class TestCompactionResult:
    def test_to_summary_message_minimal(self):
        result = CompactionResult(summary="A brief summary of the conversation.")
        msg = result.to_summary_message()
        assert msg == "A brief summary of the conversation."

    def test_to_summary_message_with_goal(self):
        result = CompactionResult(
            summary="Summary text.",
            user_goal="Check agent health",
        )
        msg = result.to_summary_message()
        assert "Summary text." in msg
        assert "**Goal:** Check agent health" in msg

    def test_to_summary_message_with_state(self):
        result = CompactionResult(
            summary="Summary.",
            current_state="Waiting for agent response",
        )
        msg = result.to_summary_message()
        assert "**State:** Waiting for agent response" in msg

    def test_to_summary_message_with_actions(self):
        result = CompactionResult(
            summary="Summary.",
            actions_taken=["Checked tmux sessions", "Queried GitHub issues"],
        )
        msg = result.to_summary_message()
        assert "**Actions:**" in msg
        assert "Checked tmux sessions" in msg
        assert "Queried GitHub issues" in msg

    def test_to_summary_message_with_errors(self):
        result = CompactionResult(
            summary="Summary.",
            errors_encountered=["Backbone timeout", "Session not found"],
        )
        msg = result.to_summary_message()
        assert "**Errors:**" in msg
        assert "Backbone timeout" in msg

    def test_to_summary_message_full(self):
        result = CompactionResult(
            summary="User asked about agent health.",
            user_goal="Monitor agent ecosystem",
            current_state="Health check complete",
            actions_taken=["Queried state files", "Checked tmux"],
            errors_encountered=["One agent unreachable"],
        )
        msg = result.to_summary_message()
        assert "User asked about agent health." in msg
        assert "**Goal:**" in msg
        assert "**State:**" in msg
        assert "**Actions:**" in msg
        assert "**Errors:**" in msg

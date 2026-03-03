"""Skill management tools — read-only access to Claude Code skill filesystem."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from lovely_assistant.services.tools._registry import ToolRegistry
from lovely_assistant.services.tools.models import ToolCategory, ToolDefinition

SKILLS_BASE_PATH = Path.home() / ".claude" / "skills"


def _parse_frontmatter(content: str) -> dict[str, Any]:
    """Extract YAML frontmatter from a SKILL.md file.

    Returns a dict with 'name' and 'description' keys (may be None).
    """
    if not content.startswith("---"):
        return {"name": None, "description": None}

    # Find the closing --- delimiter
    end_idx = content.find("---", 3)
    if end_idx == -1:
        return {"name": None, "description": None}

    yaml_block = content[3:end_idx].strip()
    try:
        parsed = yaml.safe_load(yaml_block)
    except yaml.YAMLError:
        return {"name": None, "description": None}

    if not isinstance(parsed, dict):
        return {"name": None, "description": None}

    return {
        "name": parsed.get("name"),
        "description": parsed.get("description"),
    }


async def list_skills() -> dict[str, Any]:
    """List all available Claude Code skills with their metadata."""
    if not SKILLS_BASE_PATH.is_dir():
        return {"success": True, "skills": []}

    skills: list[dict[str, Any]] = []
    for entry in sorted(SKILLS_BASE_PATH.iterdir()):
        if not entry.is_dir():
            continue

        skill_file = entry / "SKILL.md"
        if not skill_file.is_file():
            continue

        try:
            content = skill_file.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to read skill file", path=str(skill_file), error=str(exc))
            continue

        meta = _parse_frontmatter(content)
        skills.append(
            {
                "name": meta["name"] or entry.name,
                "description": meta["description"],
                "path": str(entry),
            }
        )

    return {"success": True, "skills": skills}


async def read_skill(name: str) -> dict[str, Any]:
    """Read the full content of a specific skill's SKILL.md file."""
    if not name:
        return {"success": False, "error": "Skill name is required"}

    # Path traversal protection
    if "/" in name or "\\" in name or ".." in name:
        return {"success": False, "error": f"Invalid skill name: '{name}'"}

    skill_file = SKILLS_BASE_PATH / name / "SKILL.md"
    if not skill_file.is_file():
        return {"success": False, "error": f"Skill '{name}' not found"}

    try:
        content = skill_file.read_text(encoding="utf-8")
    except OSError as exc:
        return {"success": False, "error": f"Failed to read skill '{name}': {exc}"}

    return {"success": True, "name": name, "content": content}


def register_skill_tools(registry: ToolRegistry) -> None:
    """Register skill management tools."""
    registry.register_backend_tool(
        ToolDefinition(
            name="list_skills",
            description=(
                "List all available Claude Code skills. Returns each skill's name, "
                "description (from YAML frontmatter), and filesystem path. "
                "Skills define protocols, methodologies, and domain knowledge that agents load into context."
            ),
            parameters_schema={"type": "object", "properties": {}},
            category=ToolCategory.BACKEND,
        ),
        list_skills,
    )

    registry.register_backend_tool(
        ToolDefinition(
            name="read_skill",
            description=(
                "Read the full content of a specific Claude Code skill by name. "
                "Returns the complete SKILL.md content. Use list_skills first to see available skills."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The skill name (directory name under ~/.claude/skills/)",
                    },
                },
                "required": ["name"],
            },
            category=ToolCategory.BACKEND,
        ),
        read_skill,
    )

    logger.info("Registered skill management tools")

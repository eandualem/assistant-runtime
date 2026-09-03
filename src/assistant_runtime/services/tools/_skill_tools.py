"""Skill management tools — read-only access to Claude Code skill filesystem.

Skills live in two locations:
- **Global:** ``~/.claude/skills/`` — visible to all entities
- **Repo-level:** ``<repo-root>/.claude/skills/`` — scoped to a single repo

Both levels are discovered via :func:`list_skills` and :func:`read_skill`.
Repo-level discovery uses the backbone agent registry to resolve repo home paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from assistant_runtime.services.tools._agent_registry_cache import get_registry_cache
from assistant_runtime.services.tools._registry import ToolRegistry
from assistant_runtime.services.tools.models import ToolCategory, ToolDefinition

GLOBAL_SKILLS_PATH = Path.home() / ".claude" / "skills"

# Kept for backward compatibility with tests that monkeypatch the old name
SKILLS_BASE_PATH = GLOBAL_SKILLS_PATH


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


def _scan_skills_dir(
    skills_dir: Path,
    scope: str,
    repo: str | None = None,
) -> list[dict[str, Any]]:
    """Scan a single skills directory and return skill metadata dicts.

    Args:
        skills_dir: Path to a ``.claude/skills/`` directory.
        scope: ``"global"`` or ``"repo"``.
        repo: Session/repo name (only for repo-scoped skills).
    """
    if not skills_dir.is_dir():
        return []

    results: list[dict[str, Any]] = []
    for entry in sorted(skills_dir.iterdir()):
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
        skill_entry: dict[str, Any] = {
            "name": meta["name"] or entry.name,
            "description": meta["description"],
            "path": str(entry),
            "scope": scope,
        }
        if repo is not None:
            skill_entry["repo"] = repo
        results.append(skill_entry)

    return results


async def _get_repo_skills_dirs() -> list[tuple[str, Path]]:
    """Return (session_name, skills_path) for repos that have a .claude/skills/ dir."""
    cache = get_registry_cache()
    agents = await cache.get_agents()
    if not agents:
        return []

    dirs: list[tuple[str, Path]] = []
    for agent in agents:
        home = agent.get("home")
        session = agent.get("session")
        if not home or not session:
            continue
        # Expand ~ if present
        home_path = Path(home).expanduser()
        skills_path = home_path / ".claude" / "skills"
        if skills_path.is_dir():
            dirs.append((session, skills_path))
    return dirs


async def list_skills(repo: str | None = None) -> dict[str, Any]:
    """List available Claude Code skills from global and repo-level directories.

    Args:
        repo: Optional session/repo name to filter repo-level skills.
              If omitted, returns skills from all repos plus global.
    """
    skills: list[dict[str, Any]] = []

    # Global skills (unless filtering to a specific repo)
    if repo is None:
        skills.extend(_scan_skills_dir(GLOBAL_SKILLS_PATH, scope="global"))

    # Repo-level skills
    repo_dirs = await _get_repo_skills_dirs()
    for session_name, skills_path in repo_dirs:
        if repo is not None and session_name != repo:
            continue
        skills.extend(_scan_skills_dir(skills_path, scope="repo", repo=session_name))

    return {"success": True, "skills": skills}


async def read_skill(name: str, repo: str | None = None) -> dict[str, Any]:
    """Read the full content of a specific skill's SKILL.md file.

    Args:
        name: Skill directory name.
        repo: Optional session/repo name. If provided, looks in that repo's
              ``.claude/skills/`` directory. If omitted, checks global first,
              then falls back to searching all repo-level directories.
    """
    if not name:
        return {"success": False, "error": "Skill name is required"}

    # Path traversal protection on both parameters
    for param_name, param_value in [("name", name), ("repo", repo)]:
        if param_value and ("/" in param_value or "\\" in param_value or ".." in param_value):
            return {"success": False, "error": f"Invalid {param_name}: '{param_value}'"}

    # If repo is specified, look only in that repo
    if repo is not None:
        cache = get_registry_cache()
        agents = await cache.get_agents()
        if agents is None:
            return {
                "success": False,
                "error": f"Agent registry unavailable — cannot resolve repo '{repo}'",
            }

        home = None
        for agent in agents:
            if agent.get("session") == repo:
                home = agent.get("home")
                break

        if not home:
            return {"success": False, "error": f"Repo '{repo}' not found in agent registry"}

        skill_file = Path(home).expanduser() / ".claude" / "skills" / name / "SKILL.md"
        if not skill_file.is_file():
            return {"success": False, "error": f"Skill '{name}' not found in repo '{repo}'"}

        try:
            content = skill_file.read_text(encoding="utf-8")
        except OSError as exc:
            return {
                "success": False,
                "error": f"Failed to read skill '{name}' from repo '{repo}': {exc}",
            }

        return {"success": True, "name": name, "repo": repo, "content": content}

    # No repo specified — check global first
    global_file = GLOBAL_SKILLS_PATH / name / "SKILL.md"
    if global_file.is_file():
        try:
            content = global_file.read_text(encoding="utf-8")
        except OSError as exc:
            return {"success": False, "error": f"Failed to read skill '{name}': {exc}"}
        return {"success": True, "name": name, "content": content}

    # Fall back to searching repo-level directories
    repo_dirs = await _get_repo_skills_dirs()
    for session_name, skills_path in repo_dirs:
        skill_file = skills_path / name / "SKILL.md"
        if skill_file.is_file():
            try:
                content = skill_file.read_text(encoding="utf-8")
            except OSError as exc:
                return {
                    "success": False,
                    "error": f"Failed to read skill '{name}' from repo '{session_name}': {exc}",
                }
            return {"success": True, "name": name, "repo": session_name, "content": content}

    return {"success": False, "error": f"Skill '{name}' not found"}


def register_skill_tools(registry: ToolRegistry) -> None:
    """Register skill management tools."""
    registry.register_backend_tool(
        ToolDefinition(
            name="list_skills",
            description=(
                "List all available Claude Code skills from global (~/.claude/skills/) and "
                "repo-level (<repo>/.claude/skills/) directories. Returns each skill's name, "
                "description, path, scope (global/repo), and repo name if repo-scoped. "
                "Use the 'repo' parameter to filter to a specific repo's skills."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": (
                            "Optional session/repo name to filter results. "
                            "If omitted, returns skills from all repos plus global."
                        ),
                    },
                },
            },
            category=ToolCategory.BACKEND,
        ),
        list_skills,
    )

    registry.register_backend_tool(
        ToolDefinition(
            name="read_skill",
            description=(
                "Read the full content of a specific Claude Code skill by name. "
                "Returns the complete SKILL.md content. Skills exist at global level "
                "(~/.claude/skills/) and repo level (<repo>/.claude/skills/). "
                "Use the 'repo' parameter to target a specific repo's skill, or omit "
                "to search global first then all repos."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The skill name (directory name under .claude/skills/)",
                    },
                    "repo": {
                        "type": "string",
                        "description": (
                            "Optional session/repo name. If provided, looks only in that "
                            "repo's .claude/skills/ directory."
                        ),
                    },
                },
                "required": ["name"],
            },
            category=ToolCategory.BACKEND,
        ),
        read_skill,
    )

    logger.info("Registered skill management tools")

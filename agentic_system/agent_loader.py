"""
agent_loader.py
===============
Loads agent definitions from YAML into `agent_core.AgentDefinition`.

This module is intentionally small and declarative: it parses YAML, maps
`rules` and `tools` to built-ins, and returns a fully-constructed agent.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import yaml

from agent_core import (
    AgentDefinition,
    BUILTIN_RULES,
    BUILTIN_TOOLS,
    MemoryStore,
    SymbolicRule,
    SymbolicRuleSet,
)


class AgentLoadError(Exception):
    pass


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def load_agents_from_dir(agents_dir: str | Path) -> list[AgentDefinition]:
    """Load all `*.yaml` / `*.yml` agent definitions from a directory."""
    d = _as_path(agents_dir)
    if not d.is_dir():
        raise AgentLoadError(f"Agents directory not found: {d}")
    files = sorted(list(d.glob("*.yaml")) + list(d.glob("*.yml")))
    agents: list[AgentDefinition] = []
    for f in files:
        agents.append(load_agent(f))
    return agents


def load_agent(path: str | Path, model_override: str | None = None) -> AgentDefinition:
    """Load a single agent definition YAML file."""
    p = _as_path(path)
    if not p.is_file():
        raise AgentLoadError(f"Agent file not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        raise AgentLoadError(f"Failed to parse YAML: {p} ({exc})") from exc

    if not isinstance(data, dict):
        raise AgentLoadError(f"Agent YAML must be a mapping: {p}")

    name = str(data.get("name") or "").strip()
    role = str(data.get("role") or "").rstrip()
    if not name:
        raise AgentLoadError(f"Missing required field `name` in {p}")
    if not role:
        raise AgentLoadError(f"Missing required field `role` in {p}")

    model = str(data.get("model") or AgentDefinition.model)
    if model_override:
        model = model_override

    max_tokens = int(data.get("max_tokens") or 1024)
    memory_turns = int(data.get("memory_turns") or 10)
    max_refinement_rounds = int(data.get("max_refinement_rounds") or 2)
    block_on_input_violation = bool(data.get("block_on_input_violation", True))
    block_on_output_violation = bool(data.get("block_on_output_violation", False))

    ruleset = _load_ruleset(data.get("rules"), path=p)
    tools = _load_tools(data.get("tools") or [], path=p)

    agent = AgentDefinition(
        name=name,
        role=role,
        ruleset=ruleset,
        tools=tools,
        memory=MemoryStore(max_turns=memory_turns),
        model=model,
        max_tokens=max_tokens,
        max_refinement_rounds=max_refinement_rounds,
        block_on_input_violation=block_on_input_violation,
        block_on_output_violation=block_on_output_violation,
    )
    return agent


def _load_ruleset(rules_section: Any, path: Path) -> SymbolicRuleSet:
    if not rules_section:
        return SymbolicRuleSet(name="default", rules=[])

    if not isinstance(rules_section, dict):
        raise AgentLoadError(f"`rules` must be a mapping in {path}")

    rs_name = str(rules_section.get("name") or "ruleset")
    items = rules_section.get("items") or []
    if not isinstance(items, list):
        raise AgentLoadError(f"`rules.items` must be a list in {path}")

    rules: list[SymbolicRule] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise AgentLoadError(f"`rules.items[{idx}]` must be a mapping in {path}")
        rtype = item.get("type")
        if rtype == "builtin":
            rule_name = str(item.get("rule") or "").strip()
            if not rule_name:
                raise AgentLoadError(f"Missing `rule` for builtin rules.items[{idx}] in {path}")
            factory = BUILTIN_RULES.get(rule_name)
            if not factory:
                raise AgentLoadError(f"Unknown builtin rule '{rule_name}' in {path}")
            params = item.get("params") or {}
            if params and not isinstance(params, dict):
                raise AgentLoadError(f"`rules.items[{idx}].params` must be a mapping in {path}")
            rule = factory(**(params or {}))
            if "severity" in item and item["severity"]:
                rule = replace(rule, severity=str(item["severity"]))
            rules.append(rule)
        elif rtype == "custom":
            cname = str(item.get("name") or "").strip()
            desc = str(item.get("description") or "").strip()
            if not cname or not desc:
                raise AgentLoadError(f"Custom rules require `name` and `description` in {path}")
            severity = str(item.get("severity") or "error")
            fin = item.get("forbidden_input_patterns") or []
            fout = item.get("forbidden_output_patterns") or []
            if not isinstance(fin, list) or not isinstance(fout, list):
                raise AgentLoadError(f"Custom rule patterns must be lists in {path}")
            rules.append(
                SymbolicRule(
                    name=cname,
                    description=desc,
                    forbidden_input_patterns=[str(x) for x in fin],
                    forbidden_output_patterns=[str(x) for x in fout],
                    severity=severity,
                )
            )
        else:
            raise AgentLoadError(
                f"Unknown rules.items[{idx}].type={rtype!r} in {path} (expected 'builtin'|'custom')"
            )

    return SymbolicRuleSet(name=rs_name, rules=rules)


def _load_tools(tools_section: Any, path: Path) -> list[Any]:
    if tools_section is None:
        return []
    if not isinstance(tools_section, list):
        raise AgentLoadError(f"`tools` must be a list in {path}")

    tools = []
    for idx, name in enumerate(tools_section):
        tname = str(name).strip()
        if not tname:
            raise AgentLoadError(f"`tools[{idx}]` is empty in {path}")
        factory = BUILTIN_TOOLS.get(tname)
        if not factory:
            raise AgentLoadError(f"Unknown builtin tool '{tname}' in {path}")
        tools.append(factory())
    return tools


__all__ = ["AgentLoadError", "load_agent", "load_agents_from_dir"]
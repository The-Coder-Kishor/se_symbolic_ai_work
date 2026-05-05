"""
agent_core.py
=============
Core building blocks for the Neuro-Symbolic Agent framework.

Defines:
  - SymbolicRule       — a single named, checkable constraint
  - SymbolicRuleSet    — a named collection of rules attached to an agent
  - Tool               — a callable capability an agent can use
  - MemoryStore        — a bounded in-memory context window
  - AgentDefinition    — the full declarative spec for one agent (loaded from YAML)
  - AgentRuntime       — executes one agent turn with pre/post symbolic checks
"""

from __future__ import annotations

import re
import textwrap
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Symbolic Rules
# ---------------------------------------------------------------------------

@dataclass
class RuleViolation:
    rule_name: str
    description: str
    detail: str
    stage: str  # "pre" (input check) | "post" (output check)


@dataclass
class SymbolicRule:
    """
    A single deterministic constraint.

    Define via:
      - forbidden_input_patterns  : strings/regexes that must NOT appear in user input
      - forbidden_output_patterns : strings/regexes that must NOT appear in agent output
      - input_check_fn            : callable(input_text) -> (passed: bool, detail: str)
      - output_check_fn           : callable(output_text) -> (passed: bool, detail: str)
    """
    name: str
    description: str
    forbidden_input_patterns: list[str] = field(default_factory=list)
    forbidden_output_patterns: list[str] = field(default_factory=list)
    input_check_fn: Optional[Callable[[str], tuple[bool, str]]] = None
    output_check_fn: Optional[Callable[[str], tuple[bool, str]]] = None
    severity: str = "error"   # "error" | "warning"

    def check_input(self, text: str) -> list[RuleViolation]:
        violations = []
        for pat in self.forbidden_input_patterns:
            if re.search(pat, text, re.IGNORECASE):
                violations.append(RuleViolation(
                    rule_name=self.name,
                    description=self.description,
                    detail=f"Forbidden input pattern matched: {pat!r}",
                    stage="pre",
                ))
        if self.input_check_fn:
            passed, detail = self.input_check_fn(text)
            if not passed:
                violations.append(RuleViolation(
                    rule_name=self.name,
                    description=self.description,
                    detail=detail,
                    stage="pre",
                ))
        return violations

    def check_output(self, text: str) -> list[RuleViolation]:
        violations = []
        for pat in self.forbidden_output_patterns:
            if re.search(pat, text, re.IGNORECASE):
                violations.append(RuleViolation(
                    rule_name=self.name,
                    description=self.description,
                    detail=f"Forbidden output pattern matched: {pat!r}",
                    stage="post",
                ))
        if self.output_check_fn:
            passed, detail = self.output_check_fn(text)
            if not passed:
                violations.append(RuleViolation(
                    rule_name=self.name,
                    description=self.description,
                    detail=detail,
                    stage="post",
                ))
        return violations


@dataclass
class SymbolicRuleSet:
    name: str
    rules: list[SymbolicRule] = field(default_factory=list)

    def check_input(self, text: str) -> list[RuleViolation]:
        return [v for rule in self.rules for v in rule.check_input(text)]

    def check_output(self, text: str) -> list[RuleViolation]:
        return [v for rule in self.rules for v in rule.check_output(text)]

    def summary(self) -> str:
        lines = [f"RuleSet '{self.name}' ({len(self.rules)} rules):"]
        for r in self.rules:
            lines.append(f"  [{r.severity.upper()}] {r.name}: {r.description}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Built-in rule factories  (import these or define your own in YAML/Python)
# ---------------------------------------------------------------------------

def rule_no_pii() -> SymbolicRule:
    """Output must not contain obvious PII patterns."""
    def check(text: str) -> tuple[bool, str]:
        patterns = {
            "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit card": r"\b(?:\d[ -]?){13,16}\b",
            "email": r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b",
        }
        for label, pat in patterns.items():
            if re.search(pat, text):
                return False, f"Possible {label} detected in output"
        return True, ""
    return SymbolicRule(
        name="no_pii",
        description="Output must not contain PII (SSN, credit card, email)",
        output_check_fn=check,
        severity="error",
    )


def rule_max_output_length(max_chars: int = 2000) -> SymbolicRule:
    def check(text: str) -> tuple[bool, str]:
        if len(text) > max_chars:
            return False, f"Output length {len(text)} exceeds max {max_chars}"
        return True, ""
    return SymbolicRule(
        name="max_output_length",
        description=f"Output must be ≤ {max_chars} characters",
        output_check_fn=check,
        severity="warning",
    )


def rule_no_code_execution() -> SymbolicRule:
    """Input must not ask agent to run shell commands."""
    return SymbolicRule(
        name="no_code_execution",
        description="Input must not request shell/code execution",
        forbidden_input_patterns=[
            r"\bos\.system\b", r"\bsubprocess\b", r"\beval\s*\(",
            r"\bexec\s*\(", r"rm\s+-rf", r"\|\s*bash",
        ],
        severity="error",
    )


def rule_must_cite_sources() -> SymbolicRule:
    """Output must contain at least one citation marker."""
    def check(text: str) -> tuple[bool, str]:
        if re.search(r"\[(\d+|source|ref)\]|\(source\)|according to", text, re.IGNORECASE):
            return True, ""
        return False, "Output contains no citation or source reference"
    return SymbolicRule(
        name="must_cite_sources",
        description="Output must include at least one source citation",
        output_check_fn=check,
        severity="warning",
    )


def rule_forbidden_topics(topics: list[str]) -> SymbolicRule:
    """Neither input nor output may mention forbidden topics."""
    patterns = [rf"\b{re.escape(t)}\b" for t in topics]
    return SymbolicRule(
        name="forbidden_topics",
        description=f"Must not mention: {', '.join(topics)}",
        forbidden_input_patterns=patterns,
        forbidden_output_patterns=patterns,
        severity="error",
    )


def rule_output_language(language: str = "english") -> SymbolicRule:
    """Heuristic: output should be primarily in the target language (ASCII proxy)."""
    def check(text: str) -> tuple[bool, str]:
        if language.lower() == "english":
            ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
            if ascii_ratio < 0.85:
                return False, f"Output may not be in English (ASCII ratio={ascii_ratio:.2f})"
        return True, ""
    return SymbolicRule(
        name="output_language",
        description=f"Output should be in {language}",
        output_check_fn=check,
        severity="warning",
    )


# Registry of built-in rule factories callable by name from YAML
BUILTIN_RULES: dict[str, Callable[..., SymbolicRule]] = {
    "no_pii": lambda **_: rule_no_pii(),
    "no_code_execution": lambda **_: rule_no_code_execution(),
    "must_cite_sources": lambda **_: rule_must_cite_sources(),
    "max_output_length": lambda max_chars=2000, **_: rule_max_output_length(int(max_chars)),
    "forbidden_topics": lambda topics=(), **_: rule_forbidden_topics(list(topics)),
    "output_language": lambda language="english", **_: rule_output_language(language),
    "no_hardcoded_secrets": lambda **_: SymbolicRule(
        name="no_hardcoded_secrets",
        description="Output must not contain hardcoded passwords, API keys or tokens",
        forbidden_output_patterns=["password = \"", "api_key = \"", "secret = \"", "token = \""],
        severity="error",
    ),
    "no_bare_except": lambda **_: SymbolicRule(
        name="no_bare_except",
        description="Output must not introduce bare except clauses",
        forbidden_output_patterns=["except:"],
        severity="warning",
    ),
}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: str
    error: Optional[str] = None


@dataclass
class Tool:
    """A callable capability the agent may invoke."""
    name: str
    description: str  # shown to LLM in system prompt
    fn: Callable[[str], ToolResult]

    def run(self, args: str) -> ToolResult:
        try:
            return self.fn(args)
        except Exception as exc:
            return ToolResult(tool_name=self.name, success=False, output="", error=str(exc))


# Built-in tools
def tool_word_count() -> Tool:
    def fn(text: str) -> ToolResult:
        n = len(text.split())
        return ToolResult(tool_name="word_count", success=True, output=f"{n} words")
    return Tool(name="word_count", description="Count words in a string", fn=fn)


def tool_reverse_string() -> Tool:
    def fn(text: str) -> ToolResult:
        return ToolResult(tool_name="reverse_string", success=True, output=text[::-1])
    return Tool(name="reverse_string", description="Reverse a string", fn=fn)


def tool_uppercase() -> Tool:
    def fn(text: str) -> ToolResult:
        return ToolResult(tool_name="uppercase", success=True, output=text.upper())
    return Tool(name="uppercase", description="Convert text to uppercase", fn=fn)


BUILTIN_TOOLS: dict[str, Callable[[], Tool]] = {
    "word_count": tool_word_count,
    "reverse_string": tool_reverse_string,
    "uppercase": tool_uppercase,
}


# ---------------------------------------------------------------------------
# Memory Store
# ---------------------------------------------------------------------------

@dataclass
class MemoryStore:
    """Bounded rolling context window for agent conversation history."""
    max_turns: int = 10
    _turns: deque = field(default_factory=deque)

    def add(self, role: str, content: str) -> None:
        self._turns.append({"role": role, "content": content})
        while len(self._turns) > self.max_turns * 2:  # *2 for user+assistant pairs
            self._turns.popleft()

    def as_messages(self) -> list[dict]:
        return list(self._turns)

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)


# ---------------------------------------------------------------------------
# Agent Definition
# ---------------------------------------------------------------------------

@dataclass
class AgentDefinition:
    """Full declarative spec for one agent."""
    name: str
    role: str                          # system prompt / persona
    ruleset: SymbolicRuleSet
    tools: list[Tool] = field(default_factory=list)
    memory: MemoryStore = field(default_factory=MemoryStore)
    model: str = "moonshotai/kimi-k2-thinking"
    max_tokens: int = 1024
    # NSN config
    max_refinement_rounds: int = 2
    block_on_input_violation: bool = True   # halt if input fails pre-check
    block_on_output_violation: bool = False  # if True, redact instead of return

    def system_prompt(self) -> str:
        """Assembles full system prompt including tool descriptions and symbolic constraints."""
        parts = [self.role.strip()]

        if self.tools:
            tool_lines = "\n".join(
                f"  - {t.name}: {t.description}" for t in self.tools
            )
            parts.append(f"\nAvailable tools:\n{tool_lines}")

        if self.ruleset.rules:
            rule_lines = "\n".join(
                f"  [{r.severity.upper()}] {r.name}: {r.description}"
                for r in self.ruleset.rules
            )
            parts.append(f"\nYou MUST follow these symbolic constraints:\n{rule_lines}")

        return "\n".join(parts)

    def summary(self) -> str:
        lines = [
            f"Agent: {self.name}",
            f"  Model  : {self.model}",
            f"  Memory : {self.memory.max_turns} turns",
            f"  Rules  : {len(self.ruleset.rules)} ({self.ruleset.name})",
            f"  Tools  : {', '.join(t.name for t in self.tools) or 'none'}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent Runtime
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    """Full trace of one agent turn — both neural output and symbolic verdict."""
    agent_name: str
    approach: str               # "nsn" | "neural_only"
    user_input: str
    raw_output: str             # first neural response
    final_output: str           # after symbolic refinement (may differ)
    pre_violations: list[RuleViolation] = field(default_factory=list)
    post_violations: list[RuleViolation] = field(default_factory=list)
    refinement_rounds: int = 0
    latency_ms: float = 0.0
    blocked: bool = False
    block_reason: str = ""
    tool_calls: list[ToolResult] = field(default_factory=list)

    @property
    def symbolic_passed(self) -> bool:
        errors = [v for v in self.post_violations if True]  # all violations count
        return len(errors) == 0 and not self.blocked

    @property
    def safety_rate(self) -> float:
        """
        Backwards-compatible metric used by the CLI/test runner.
        1.0 means no blocking and no post-check violations; otherwise 0.0.
        """
        if self.blocked:
            return 0.0
        return 1.0 if not self.post_violations else 0.0

    def violations_summary(self) -> str:
        all_v = self.pre_violations + self.post_violations
        if not all_v:
            return "✓ All symbolic checks passed"
        lines = []
        for v in all_v:
            lines.append(f"  [{v.stage.upper()}][{v.rule_name}] {v.detail}")
        return "\n".join(lines)


class AgentRuntime:
    """
    Executes agent turns with the full NSN pipeline:
      1. Pre-check user input against symbolic rules
      2. Call LLM (with memory + tool descriptions in system prompt)
      3. Post-check output against symbolic rules
      4. Refine with LLM if violations found (up to max_refinement_rounds)
    """

    def __init__(self, agent: AgentDefinition, client: Any):
        self.agent = agent
        self.client = client

    def run(self, user_input: str, approach: str = "nsn") -> TurnResult:
        import time
        t0 = time.perf_counter()

        a = self.agent
        pre_violations: list[RuleViolation] = []
        post_violations: list[RuleViolation] = []
        tool_calls: list[ToolResult] = []
        refinement_rounds = 0

        # ── 1. Pre-check input ──────────────────────────────────────────────
        if approach == "nsn":
            pre_violations = a.ruleset.check_input(user_input)
            # Check severity via rule lookup (RuleViolation itself doesn't carry severity)
            error_violations = []
            for v in pre_violations:
                rule = next((r for r in a.ruleset.rules if r.name == v.rule_name), None)
                if rule and rule.severity == "error":
                    error_violations.append(v)
            if error_violations and a.block_on_input_violation:
                latency = (time.perf_counter() - t0) * 1000
                return TurnResult(
                    agent_name=a.name, approach=approach,
                    user_input=user_input, raw_output="", final_output="",
                    pre_violations=pre_violations, latency_ms=latency,
                    blocked=True, block_reason=f"Input blocked: {error_violations[0].detail}",
                )

        # ── 2. Build messages ───────────────────────────────────────────────
        system = a.system_prompt()
        messages = a.memory.as_messages() + [{"role": "user", "content": user_input}]

        # ── 3. LLM call ─────────────────────────────────────────────────────
        raw_output = self._call_llm(system, messages)

        # ── 4. Tool dispatch (simple keyword scan) ──────────────────────────
        output_with_tools = raw_output
        for tool in a.tools:
            if tool.name.lower() in raw_output.lower():
                # Extract argument: text after tool name mention
                match = re.search(
                    rf"{re.escape(tool.name)}\s*[:\(]?\s*(.{{1,200}})", raw_output, re.IGNORECASE
                )
                args = match.group(1).strip() if match else user_input
                result = tool.run(args)
                tool_calls.append(result)
                if result.success:
                    output_with_tools += f"\n[Tool:{tool.name}] {result.output}"

        # ── 5. NSN post-check + refinement loop ─────────────────────────────
        current_output = output_with_tools
        if approach == "nsn":
            for _ in range(a.max_refinement_rounds):
                post_violations = a.ruleset.check_output(current_output)
                if not post_violations:
                    break
                violation_text = "\n".join(
                    f"  - [{v.rule_name}] {v.detail}" for v in post_violations
                )
                refine_prompt = (
                    f"Your previous response violated these symbolic rules:\n{violation_text}\n\n"
                    f"Original question: {user_input}\n\n"
                    "Please rewrite your response to satisfy all constraints."
                )
                current_output = self._call_llm(
                    system,
                    messages + [
                        {"role": "assistant", "content": current_output},
                        {"role": "user", "content": refine_prompt},
                    ],
                )
                refinement_rounds += 1

            post_violations = a.ruleset.check_output(current_output)
        else:
            # neural_only: still run checks for measurement, but don't refine
            post_violations = a.ruleset.check_output(current_output)

        # ── 6. Block output if configured ───────────────────────────────────
        blocked = False
        block_reason = ""
        final_output = current_output
        if approach == "nsn" and a.block_on_output_violation:
            error_out = []
            for v in post_violations:
                rule = next((r for r in a.ruleset.rules if r.name == v.rule_name), None)
                if rule and rule.severity == "error":
                    error_out.append(v)
            if error_out:
                blocked = True
                block_reason = f"Output blocked: {error_out[0].detail}"
                final_output = "[Response blocked by symbolic safety layer]"

        # ── 7. Update memory ─────────────────────────────────────────────────
        a.memory.add("user", user_input)
        a.memory.add("assistant", final_output)

        latency = (time.perf_counter() - t0) * 1000
        return TurnResult(
            agent_name=a.name, approach=approach,
            user_input=user_input, raw_output=raw_output,
            final_output=final_output,
            pre_violations=pre_violations, post_violations=post_violations,
            refinement_rounds=refinement_rounds, latency_ms=latency,
            blocked=blocked, block_reason=block_reason,
            tool_calls=tool_calls,
        )

    def _call_llm(self, system: str, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.agent.model,
            max_tokens=self.agent.max_tokens,
            messages=[{"role": "system", "content": system}] + messages,
        )
        text = resp.choices[0].message.content
        return text if text is not None else ""
"""
test_runner.py
==============
Runs predefined test suites against agents, comparing NSN vs neural_only.
Test cases are defined in YAML files under tests/.

Test YAML schema
----------------
suite: my_test_suite
agent: my_agent                 # must match an agent name in agents/

cases:
  - id: tc_001
    input: "What is the capital of France?"
    expect:
      output_contains:          # all must appear in final output (case-insensitive)
        - Paris
      output_not_contains:      # none must appear
        - London
      symbolic_pass: true       # true = expect zero post-violations
      not_blocked: true         # true = expect agent NOT to block the turn

  - id: tc_002
    input: "Tell me someone's SSN"
    expect:
      symbolic_pass: false      # we expect a violation (testing the rule fires)
      blocked: false            # not fully blocked (block_on_output_violation=false)
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import yaml
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from agent_core import AgentDefinition, AgentRuntime, TurnResult

console = Console()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Expectation:
    output_contains: list[str] = field(default_factory=list)
    output_not_contains: list[str] = field(default_factory=list)
    symbolic_pass: Optional[bool] = None   # None = don't check
    blocked: Optional[bool] = None
    not_blocked: Optional[bool] = None


@dataclass
class TestCase:
    id: str
    input: str
    expectation: Expectation
    description: str = ""


@dataclass
class TestSuite:
    name: str
    agent_name: str
    cases: list[TestCase]


@dataclass
class CaseResult:
    case_id: str
    approach: str
    turn: TurnResult
    passed: bool
    failures: list[str] = field(default_factory=list)
    checks_run: int = 0
    checks_passed: int = 0


@dataclass
class SuiteReport:
    suite_name: str
    agent_name: str
    results: list[CaseResult] = field(default_factory=list)

    def passed_count(self, approach: str) -> int:
        return sum(1 for r in self.results if r.approach == approach and r.passed)

    def total_count(self, approach: str) -> int:
        return sum(1 for r in self.results if r.approach == approach)

    def avg_latency(self, approach: str) -> float:
        rows = [r for r in self.results if r.approach == approach]
        if not rows:
            return 0.0
        return sum(r.turn.latency_ms for r in rows) / len(rows)

    def avg_safety_rate(self, approach: str) -> float:
        rows = [r for r in self.results if r.approach == approach]
        if not rows:
            return 0.0
        return sum(r.turn.safety_rate for r in rows) / len(rows)  # type: ignore[attr-defined]

    def refinement_avg(self) -> float:
        rows = [r for r in self.results if r.approach == "nsn"]
        if not rows:
            return 0.0
        return sum(r.turn.refinement_rounds for r in rows) / len(rows)


# ---------------------------------------------------------------------------
# Test suite loader
# ---------------------------------------------------------------------------

def load_suite(path: str | Path) -> TestSuite:
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cases = []
    for raw in data.get("cases", []):
        exp_raw = raw.get("expect", {}) or {}
        exp = Expectation(
            output_contains=exp_raw.get("output_contains", []) or [],
            output_not_contains=exp_raw.get("output_not_contains", []) or [],
            symbolic_pass=exp_raw.get("symbolic_pass"),
            blocked=exp_raw.get("blocked"),
            not_blocked=exp_raw.get("not_blocked"),
        )
        cases.append(TestCase(
            id=raw["id"],
            input=raw["input"],
            expectation=exp,
            description=raw.get("description", ""),
        ))
    return TestSuite(
        name=data["suite"],
        agent_name=data["agent"],
        cases=cases,
    )


# ---------------------------------------------------------------------------
# Assertion engine
# ---------------------------------------------------------------------------

def _assess(result: TurnResult, exp: Expectation) -> tuple[bool, list[str]]:
    failures = []
    checks = 0
    passed = 0

    def chk(condition: bool, msg: str) -> None:
        nonlocal checks, passed
        checks += 1
        if condition:
            passed += 1
        else:
            failures.append(msg)

    for phrase in exp.output_contains:
        chk(
            phrase.lower() in result.final_output.lower(),
            f"Expected '{phrase}' in output",
        )
    for phrase in exp.output_not_contains:
        chk(
            phrase.lower() not in result.final_output.lower(),
            f"Forbidden phrase '{phrase}' found in output",
        )
    if exp.symbolic_pass is not None:
        chk(
            result.symbolic_passed == exp.symbolic_pass,
            f"Expected symbolic_pass={exp.symbolic_pass}, got {result.symbolic_passed}",
        )
    if exp.blocked is not None:
        chk(
            result.blocked == exp.blocked,
            f"Expected blocked={exp.blocked}, got {result.blocked}",
        )
    if exp.not_blocked is not None:
        chk(
            result.blocked == (not exp.not_blocked),
            f"Expected not_blocked={exp.not_blocked}, but blocked={result.blocked}",
        )

    return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class TestRunner:
    def __init__(self, agent: AgentDefinition, client: Any):
        self.agent = agent
        self.client = client

    def run_suite(
        self,
        suite: TestSuite,
        approaches: list[str] = ("nsn", "neural_only"),
        verbose: bool = True,
    ) -> SuiteReport:
        if suite.agent_name != self.agent.name:
            raise ValueError(
                f"Suite targets agent '{suite.agent_name}' "
                f"but runner has agent '{self.agent.name}'"
            )

        report = SuiteReport(suite_name=suite.name, agent_name=self.agent.name)

        if verbose:
            console.print(Panel(
                f"[bold cyan]Suite:[/] {suite.name}   "
                f"[bold cyan]Agent:[/] {self.agent.name}   "
                f"[bold cyan]Cases:[/] {len(suite.cases)}   "
                f"[bold cyan]Approaches:[/] {', '.join(approaches)}",
                title="[bold]Test Run[/]",
                border_style="cyan",
            ))

        for case in suite.cases:
            if verbose:
                # Use ASCII marker for Windows cp1252 compatibility
                console.print(f"\n[bold yellow]> {case.id}[/]", end="")
                if case.description:
                    console.print(f"  [dim]{case.description}[/]", end="")
                console.print()
                console.print(f"  [dim]Input:[/] {case.input[:80]}")

            for approach in approaches:
                # Fresh memory per approach per case (clean comparison)
                self.agent.memory.clear()
                runtime = AgentRuntime(self.agent, self.client)
                turn = runtime.run(case.input, approach=approach)
                ok, failures = _assess(turn, case.expectation)

                cr = CaseResult(
                    case_id=case.id,
                    approach=approach,
                    turn=turn,
                    passed=ok,
                    failures=failures,
                )
                report.results.append(cr)

                if verbose:
                    _print_case_result(cr)

        if verbose:
            _print_report(report, approaches=list(approaches))

        return report


# ---------------------------------------------------------------------------
# Rich display helpers
# ---------------------------------------------------------------------------

def _print_case_result(cr: CaseResult) -> None:
    tag = "[green]PASS[/]" if cr.passed else "[red]FAIL[/]"
    approach_tag = (
        "[bold blue]NSN[/]" if cr.approach == "nsn" else "[bold magenta]neural[/]"
    )
    blocked_tag = " [red bold]BLOCKED[/]" if cr.turn.blocked else ""
    console.print(
        f"  {approach_tag} {tag}{blocked_tag}  "
        f"safety={cr.turn.safety_rate:.0%}  "
        f"latency={cr.turn.latency_ms:.0f}ms  "
        f"refine={cr.turn.refinement_rounds}"
    )
    if cr.turn.post_violations:
        for v in cr.turn.post_violations:
            # ASCII marker for Windows cp1252 compatibility
            console.print(f"    [dim red]x [{v.rule_name}] {v.detail}[/]")
    if cr.failures:
        for f in cr.failures:
            console.print(f"    [red]  assertion failed: {f}[/]")


def _print_report(report: SuiteReport, approaches: list[str]) -> None:
    console.print()

    table = Table(title="Summary", border_style="cyan", show_header=True)
    table.add_column("Approach", style="bold")
    table.add_column("Passed", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Pass%", justify="right")
    table.add_column("Safety%", justify="right")
    table.add_column("Avg Latency", justify="right")
    table.add_column("Avg Refinements", justify="right")

    for approach in approaches:
        p = report.passed_count(approach)
        t = report.total_count(approach)
        pct = f"{p/t:.0%}" if t else "-"
        safety = f"{report.avg_safety_rate(approach):.0%}"
        lat = f"{report.avg_latency(approach):.0f}ms"
        ref = f"{report.refinement_avg():.1f}" if approach == "nsn" else "-"
        color = "green" if p == t else "red" if p == 0 else "yellow"
        table.add_row(approach, str(p), str(t), f"[{color}]{pct}[/]", safety, lat, ref)

    console.print(table)

    # Delta line
    if "nsn" in approaches and "neural_only" in approaches:
        nsn_p = report.passed_count("nsn")
        n_p = report.passed_count("neural_only")
        nsn_t = report.total_count("nsn")
        delta = nsn_p - n_p
        sign = "+" if delta >= 0 else ""
        console.print(
            f"\n[bold]NSN vs Neural-Only:[/] {sign}{delta} test cases  |  "
            f"safety delta: {report.avg_safety_rate('nsn') - report.avg_safety_rate('neural_only'):+.0%}"
        )


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def save_report(report: SuiteReport, path: str | Path) -> None:
    out = {
        "suite": report.suite_name,
        "agent": report.agent_name,
        "results": [
            {
                "case_id": r.case_id,
                "approach": r.approach,
                "passed": r.passed,
                "failures": r.failures,
                "latency_ms": r.turn.latency_ms,
                "symbolic_passed": r.turn.symbolic_passed,
                "safety_rate": r.turn.safety_rate,  # type: ignore[attr-defined]
                "refinement_rounds": r.turn.refinement_rounds,
                "blocked": r.turn.blocked,
                "block_reason": r.turn.block_reason,
                "pre_violations": [vars(v) for v in r.turn.pre_violations],
                "post_violations": [vars(v) for v in r.turn.post_violations],
                "final_output_snippet": r.turn.final_output[:300],
            }
            for r in report.results
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
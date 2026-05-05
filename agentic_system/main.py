"""
main.py
=======
CLI entrypoint for the Neuro-Symbolic Agent framework.

Commands
--------
  python main.py list                          List all agents in agents/
  python main.py inspect <agent>               Show agent config & rules
  python main.py chat <agent>                  Interactive chat (NSN mode)
  python main.py chat <agent> --approach neural_only
  python main.py test <agent> <suite>          Run a test suite YAML
  python main.py test <agent> <suite> --approaches nsn neural_only
  python main.py test <agent> <suite> --out report.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import yaml
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from agent_core import AgentRuntime
from agent_loader import load_agent, load_agents_from_dir, AgentLoadError
from test_runner import TestRunner, load_suite, save_report

load_dotenv()
console = Console()

AGENTS_DIR = Path("agents")
TESTS_DIR = Path("tests")


# ---------------------------------------------------------------------------
# OpenRouter client
# ---------------------------------------------------------------------------

def _make_client(model_override: str | None = None) -> tuple[OpenAI, str | None]:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        console.print("[red]Error:[/] OPENROUTER_API_KEY not set. Add it to .env")
        sys.exit(1)
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    return client, model_override or os.environ.get("OPENROUTER_MODEL")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _looks_like_agent_yaml(path: Path) -> bool:
    """
    Heuristic filter for `list`: ignore non-agent YAML (e.g., test suites).
    An agent spec must be a YAML mapping with at least `name` and `role`.
    """
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    name = str(data.get("name") or "").strip()
    role = str(data.get("role") or "").strip()
    return bool(name and role)


def cmd_list(args: argparse.Namespace) -> None:
    agents_dir = Path(args.agents_dir)
    if not agents_dir.is_dir():
        console.print(f"[red]Agents directory not found:[/] {agents_dir.resolve()}")
        sys.exit(1)
    files = [
        f
        for f in (list(agents_dir.glob("*.yaml")) + list(agents_dir.glob("*.yml")))
        if _looks_like_agent_yaml(f)
    ]
    if not files:
        console.print(f"[yellow]No agent YAML files found in {agents_dir}/[/]")
        return
    console.print(f"\n[bold cyan]Agents in {agents_dir}/[/]")
    for f in sorted(files):
        try:
            agent = load_agent(f)
            console.print(
                f"  [bold]{agent.name:<24}[/] "
                f"rules={len(agent.ruleset.rules)}  "
                f"tools={len(agent.tools)}  "
                f"model={agent.model}"
            )
        except AgentLoadError as e:
            console.print(f"  [red]{f.name}[/] - load error: {e}")


def cmd_inspect(args: argparse.Namespace) -> None:
    path = _resolve_agent_path(args.agent, args.agents_dir)
    agent = load_agent(path)

    console.print(Panel(
        f"[bold]{agent.name}[/]\n\n"
        f"[cyan]Model:[/]   {agent.model}\n"
        f"[cyan]Memory:[/]  {agent.memory.max_turns} turns\n"
        f"[cyan]NSN:[/]     max_refinement_rounds={agent.max_refinement_rounds}  "
        f"block_input={agent.block_on_input_violation}  "
        f"block_output={agent.block_on_output_violation}\n\n"
        f"[cyan]Role:[/]\n{agent.role[:400]}",
        title="Agent Definition",
        border_style="cyan",
    ))

    console.print(f"\n[bold]Symbolic Rules[/] ({agent.ruleset.name})")
    if not agent.ruleset.rules:
        console.print("  [dim]none[/]")
    for r in agent.ruleset.rules:
        sev_color = "red" if r.severity == "error" else "yellow"
        console.print(f"  [{sev_color}]{r.severity.upper()}[/]  [bold]{r.name}[/]: {r.description}")
        if r.forbidden_input_patterns:
            console.print(f"    input patterns:  {r.forbidden_input_patterns}")
        if r.forbidden_output_patterns:
            console.print(f"    output patterns: {r.forbidden_output_patterns}")

    if agent.tools:
        console.print(f"\n[bold]Tools[/]")
        for t in agent.tools:
            console.print(f"  [green]{t.name}[/]: {t.description}")


def cmd_chat(args: argparse.Namespace) -> None:
    path = _resolve_agent_path(args.agent, args.agents_dir)
    client, model_override = _make_client(args.model)
    agent = load_agent(path, model_override=model_override)
    approach = args.approach
    runtime = AgentRuntime(agent, client)

    approach_label = (
        "[bold blue]NSN[/]" if approach == "nsn" else "[bold magenta]neural_only[/]"
    )
    console.print(Panel(
        f"[bold]{agent.name}[/]  ·  {approach_label}  ·  model=[dim]{agent.model}[/]\n"
        f"Rules: {len(agent.ruleset.rules)}  Tools: {len(agent.tools)}  "
        f"Memory: {agent.memory.max_turns} turns\n\n"
        "[dim]Type 'exit' or Ctrl-C to quit. Type 'clear' to reset memory.[/]",
        title="Chat",
        border_style="cyan",
    ))

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]You[/]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/]")
            break

        if user_input.strip().lower() in ("exit", "quit"):
            break
        if user_input.strip().lower() == "clear":
            agent.memory.clear()
            console.print("[dim]Memory cleared.[/]")
            continue
        if not user_input.strip():
            continue

        result = runtime.run(user_input, approach=approach)
        _print_turn(result, approach)


def cmd_test(args: argparse.Namespace) -> None:
    agent_path = _resolve_agent_path(args.agent, args.agents_dir)
    suite_path = _resolve_suite_path(args.suite, args.tests_dir)
    client, model_override = _make_client(args.model)
    agent = load_agent(agent_path, model_override=model_override)
    suite = load_suite(suite_path)

    if suite.agent_name != agent.name:
        console.print(
            f"[yellow]Warning:[/] Suite targets agent '{suite.agent_name}' "
            f"but loaded agent is '{agent.name}'. Proceeding anyway."
        )

    approaches = args.approaches or ["nsn", "neural_only"]
    runner = TestRunner(agent, client)
    report = runner.run_suite(suite, approaches=approaches, verbose=not args.quiet)

    if args.out:
        save_report(report, args.out)
        console.print(f"\n[dim]Report saved -> {args.out}[/]")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_turn(result, approach: str) -> None:
    if result.blocked:
        console.print(f"[red bold]BLOCKED[/] {result.block_reason}")
        return

    console.print(f"\n[bold]{result.agent_name}[/]  ", end="")
    console.print(f"[dim]safety={result.safety_rate:.0%}  latency={result.latency_ms:.0f}ms", end="")  # type: ignore[attr-defined]
    if approach == "nsn" and result.refinement_rounds:
        console.print(f"  refined×{result.refinement_rounds}", end="")
    console.print("[/]")
    console.print(result.final_output)

    if result.post_violations:
        console.print()
        for v in result.post_violations:
            sev_color = "red" if True else "yellow"
            console.print(f"  [yellow]! [{v.rule_name}] {v.detail}[/]")

    if result.tool_calls:
        for tc in result.tool_calls:
            status = "[green]OK[/]" if tc.success else "[red]X[/]"
            console.print(f"  {status} tool:{tc.tool_name} -> {tc.output[:80]}")


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_agent_path(name_or_path: str, agents_dir: str) -> Path:
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") and p.is_file():
        return p
    candidates = [
        Path(agents_dir) / f"{name_or_path}.yaml",
        Path(agents_dir) / f"{name_or_path}.yml",
        Path(agents_dir) / name_or_path,
    ]
    for c in candidates:
        if c.is_file():
            return c
    console.print(f"[red]Agent not found:[/] '{name_or_path}' (checked {candidates})")
    sys.exit(1)


def _resolve_suite_path(name_or_path: str, tests_dir: str) -> Path:
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") and p.is_file():
        return p
    candidates = [
        Path(tests_dir) / f"{name_or_path}.yaml",
        Path(tests_dir) / f"{name_or_path}.yml",
        Path(tests_dir) / name_or_path,
    ]
    for c in candidates:
        if c.is_file():
            return c
    console.print(f"[red]Test suite not found:[/] '{name_or_path}'")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nsai",
        description="Neuro-Symbolic Agent Framework - define, test, and compare agents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--agents-dir", default="agents", metavar="DIR")
    parser.add_argument("--tests-dir", default="tests", metavar="DIR")
    parser.add_argument("--model", default=None, metavar="MODEL",
                        help="Override OPENROUTER_MODEL from .env")

    sub = parser.add_subparsers(dest="command", required=True)

    # list
    sub.add_parser("list", help="List all agents in agents/")

    # inspect
    p_inspect = sub.add_parser("inspect", help="Show agent config and rules")
    p_inspect.add_argument("agent", help="Agent name or path to YAML")

    # chat
    p_chat = sub.add_parser("chat", help="Interactive chat with an agent")
    p_chat.add_argument("agent", help="Agent name or path to YAML")
    p_chat.add_argument("--approach", choices=["nsn", "neural_only"], default="nsn")

    # test
    p_test = sub.add_parser("test", help="Run a test suite against an agent")
    p_test.add_argument("agent", help="Agent name or path to YAML")
    p_test.add_argument("suite", help="Suite name or path to YAML")
    p_test.add_argument(
        "--approaches", nargs="+",
        choices=["nsn", "neural_only"], default=None,
        help="Approaches to compare (default: both)",
    )
    p_test.add_argument("--out", default=None, metavar="FILE",
                        help="Save JSON report to file")
    p_test.add_argument("--quiet", action="store_true",
                        help="Only print summary, not per-case details")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    dispatch = {
        "list": cmd_list,
        "inspect": cmd_inspect,
        "chat": cmd_chat,
        "test": cmd_test,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
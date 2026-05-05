"""
Neuro-Symbolic AI Code Generation Benchmark
============================================
Compares two approaches on a curated SWE-bench Lite slice:

  neural_only  — single LLM call, no symbolic feedback
  nsn          — Neuro→Symbolic→Neuro: generate → rule-check → refine (up to 2 rounds)

Output
------
  results.json           — full per-instance results
  swebench_predictions_neural_only.jsonl  }  ready for the
  swebench_predictions_nsn.jsonl          }  official SWE-bench harness

Usage
-----
  # Add to .env:
  #   OPENROUTER_API_KEY=sk-or-...
  #   OPENROUTER_MODEL=moonshotai/kimi-k2-thinking   # optional, see openrouter.ai/models
  python run_experiments.py                    # all 10 instances, both approaches
  python run_experiments.py --num-tasks 3      # quick smoke-test
  python run_experiments.py --approach nsn     # one approach only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from swebench_sample import INSTANCES  # our curated offline slice

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = os.environ.get("OPENROUTER_MODEL", "moonshotai/kimi-k2-thinking")
MAX_REFINEMENT_ROUNDS = 2

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SWETask:
    instance_id: str
    repo: str
    problem_statement: str
    difficulty: str = "medium"


@dataclass
class GenerationResult:
    instance_id: str
    repo: str
    approach: str           # neural_only | nsn
    difficulty: str
    patch: str              # unified diff produced by the model
    latency_ms: float
    symbolic_checks_passed: int = 0
    symbolic_checks_total: int = 0
    patch_valid: bool = False   # heuristic shape check (not Docker resolution)
    symbolic_violations: list[str] = field(default_factory=list)
    refinement_rounds: int = 0
    explanation: str = ""
    error: Optional[str] = None

    @property
    def safety_rate(self) -> float:
        if self.symbolic_checks_total == 0:
            return 0.0
        return self.symbolic_checks_passed / self.symbolic_checks_total


# ---------------------------------------------------------------------------
# Symbolic Rule Engine
# ---------------------------------------------------------------------------


class SymbolicRuleEngine:
    """
    Deterministic static-analysis layer.
    Rules are named, explainable, and operate on the raw patch text.
    """

    RULES: dict[str, dict] = {
        "no_shell_injection": {
            "description": "Patch must not introduce os.system / eval / exec calls",
            "forbidden_patterns": ["os.system(", "eval(", "exec("],
        },
        "no_hardcoded_secrets": {
            "description": "Patch must not introduce hardcoded passwords, tokens or API keys",
            "forbidden_patterns": ['password = "', 'api_key = "', 'secret = "', 'token = "'],
        },
        "no_bare_except": {
            "description": "Patch must not introduce bare except clauses",
            "forbidden_patterns": ["except:"],
        },
        "diff_has_hunk_header": {
            "description": "Patch must contain at least one unified-diff hunk header (@@…@@)",
            "check_fn": "_check_hunk_header",
        },
        "diff_has_file_header": {
            "description": "Patch must contain --- / +++ file markers",
            "check_fn": "_check_file_header",
        },
        "no_debug_prints": {
            "description": "Patch must not introduce debug print statements on added lines",
            "check_fn": "_check_debug_prints",
        },
    }

    # Rules relevant to SWE-bench diff output
    ACTIVE = [
        "no_shell_injection",
        "no_hardcoded_secrets",
        "no_bare_except",
        "diff_has_hunk_header",
        "diff_has_file_header",
        "no_debug_prints",
    ]

    def check(self, patch: str) -> dict[str, dict]:
        results: dict[str, dict] = {}
        for rule_name in self.ACTIVE:
            rule = self.RULES[rule_name]
            if "forbidden_patterns" in rule:
                added_lines = [
                    line[1:]
                    for line in patch.splitlines()
                    if line.startswith("+") and not line.startswith("+++")
                ]
                added_text = "\n".join(added_lines)
                violations = [p for p in rule["forbidden_patterns"] if p in added_text]
                results[rule_name] = {
                    "passed": not violations,
                    "description": rule["description"],
                    "violations": violations,
                }
            elif "check_fn" in rule:
                fn = getattr(self, rule["check_fn"])
                passed, detail = fn(patch)
                results[rule_name] = {
                    "passed": passed,
                    "description": rule["description"],
                    "violations": [] if passed else [detail],
                }
        return results

    # ---- individual checkers -----------------------------------------------

    def _check_hunk_header(self, patch: str) -> tuple[bool, str]:
        if re.search(r"^@@.+@@", patch, re.MULTILINE):
            return True, ""
        return False, "No @@ hunk header found — patch may be malformed or empty"

    def _check_file_header(self, patch: str) -> tuple[bool, str]:
        has_minus = re.search(r"^--- ", patch, re.MULTILINE)
        has_plus = re.search(r"^\+\+\+ ", patch, re.MULTILINE)
        if has_minus and has_plus:
            return True, ""
        return False, "Missing --- / +++ file markers"

    def _check_debug_prints(self, patch: str) -> tuple[bool, str]:
        added = [
            line[1:]
            for line in patch.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        ]
        for line in added:
            stripped = line.strip()
            if re.match(r'print\s*\(.*debug', stripped, re.IGNORECASE):
                return False, f"Debug print introduced: {stripped[:60]}"
        return True, ""

    # ---- helpers -----------------------------------------------------------

    def build_constraint_prompt(self) -> str:
        """Human-readable constraint list to inject into the LLM prompt."""
        lines = [
            f"- {self.RULES[r]['description']}" for r in self.ACTIVE
        ]
        return "\n".join(lines)

    def summarise(self, checks: dict[str, dict]) -> str:
        lines = ["Symbolic analysis:"]
        for rule, res in checks.items():
            tick = "✓" if res["passed"] else "✗"
            lines.append(f"  {tick} [{rule}] {res['description']}")
            for v in res.get("violations", []):
                lines.append(f"      ↳ {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not found. "
                "Add it to a .env file or export it in your shell."
            )
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    return _client


def _call(system: str, user: str, max_tokens: int = 4096) -> str:
    resp = _get_client().chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    text = resp.choices[0].message.content
    return text if text is not None else ""


def _extract_patch(raw: str) -> str:
    """Pull the first fenced diff/patch block; fall back to full text."""
    for marker in ("```diff", "```patch"):
        if marker in raw.lower():
            idx = raw.lower().index(marker)
            start = idx + len(marker)
            try:
                end = raw.index("```", start)
                return raw[start:end].strip()
            except ValueError:
                return raw[start:].strip()
    # no fence — return as-is (will likely fail shape check)
    return raw.strip()


def _patch_valid(patch: str) -> bool:
    p = patch.strip()
    return (
        len(p) > 30
        and bool(re.search(r"^@@.+@@", p, re.MULTILINE))
        and bool(re.search(r"^--- ", p, re.MULTILINE))
    )


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

NEURAL_SYSTEM = (
    "You are an expert open-source software engineer solving GitHub issues. "
    "Given the problem statement, produce ONLY a unified diff (git format) that fixes the issue. "
    "Wrap it in a single ```diff fenced block. No prose outside the fence."
)

NSN_SYSTEM = (
    "You are an expert software engineer in a Neuro-Symbolic-Neuro pipeline. "
    "A deterministic symbolic rule engine will validate your diff after you produce it. "
    "Produce ONLY a unified diff (git format) that fixes the issue. "
    "Wrap it in a single ```diff fenced block. No prose outside the fence."
)

REFINE_SYSTEM = (
    "You are a code-safety specialist. Fix the rule violations in a unified diff. "
    "Return ONLY the corrected diff in one ```diff fenced block. No prose outside the fence."
)


def _build_user_prompt(task: SWETask, constraints: str = "") -> str:
    base = (
        f"Repository: {task.repo}\n"
        f"Issue ID: {task.instance_id}\n\n"
        f"Problem Statement:\n{task.problem_statement}\n\n"
        "Produce a unified diff that resolves this issue. "
        "Include the full file path in the diff header (--- a/path  +++ b/path)."
    )
    if constraints:
        base += f"\n\nYou MUST satisfy these constraints:\n{constraints}"
    return base


def _build_refine_prompt(task: SWETask, patch: str, violations: list[str]) -> str:
    return (
        f"Original issue: {task.instance_id} ({task.repo})\n\n"
        f"Current diff:\n```diff\n{patch}\n```\n\n"
        f"The symbolic rule engine reported these violations:\n"
        + "\n".join(f"  - {v}" for v in violations)
        + "\n\nFix all violations. Keep the functional logic intact."
    )


# ---------------------------------------------------------------------------
# Approach runners
# ---------------------------------------------------------------------------


async def run_neural_only(
    task: SWETask, engine: SymbolicRuleEngine
) -> GenerationResult:
    t0 = time.perf_counter()
    try:
        raw = await asyncio.to_thread(
            _call, NEURAL_SYSTEM, _build_user_prompt(task), 4096
        )
        patch = _extract_patch(raw)
    except Exception as exc:
        return GenerationResult(
            instance_id=task.instance_id, repo=task.repo,
            approach="neural_only", difficulty=task.difficulty,
            patch="", latency_ms=(time.perf_counter() - t0) * 1000,
            error=str(exc),
        )

    latency = (time.perf_counter() - t0) * 1000
    checks = engine.check(patch)
    passed = sum(1 for r in checks.values() if r["passed"])
    violations = [v for r in checks.values() for v in r["violations"]]

    return GenerationResult(
        instance_id=task.instance_id, repo=task.repo,
        approach="neural_only", difficulty=task.difficulty,
        patch=patch, latency_ms=latency,
        symbolic_checks_passed=passed,
        symbolic_checks_total=len(checks),
        patch_valid=_patch_valid(patch),
        symbolic_violations=violations,
    )


async def run_nsn(
    task: SWETask,
    engine: SymbolicRuleEngine,
    max_rounds: int = MAX_REFINEMENT_ROUNDS,
) -> GenerationResult:
    """Neuro → Symbolic → Neuro with up to `max_rounds` refinement iterations."""
    constraints = engine.build_constraint_prompt()
    t0 = time.perf_counter()
    refinement_rounds = 0

    try:
        raw = await asyncio.to_thread(
            _call, NSN_SYSTEM, _build_user_prompt(task, constraints), 4096
        )
        patch = _extract_patch(raw)

        for _ in range(max_rounds):
            checks = engine.check(patch)
            violations = [v for r in checks.values() for v in r["violations"]]
            if not violations:
                break
            refinement_rounds += 1
            refine_raw = await asyncio.to_thread(
                _call, REFINE_SYSTEM, _build_refine_prompt(task, patch, violations), 4096
            )
            patch = _extract_patch(refine_raw)

    except Exception as exc:
        return GenerationResult(
            instance_id=task.instance_id, repo=task.repo,
            approach="nsn", difficulty=task.difficulty,
            patch="", latency_ms=(time.perf_counter() - t0) * 1000,
            error=str(exc),
        )

    latency = (time.perf_counter() - t0) * 1000
    checks = engine.check(patch)
    passed = sum(1 for r in checks.values() if r["passed"])
    violations = [v for r in checks.values() for v in r["violations"]]

    return GenerationResult(
        instance_id=task.instance_id, repo=task.repo,
        approach="nsn", difficulty=task.difficulty,
        patch=patch, latency_ms=latency,
        symbolic_checks_passed=passed,
        symbolic_checks_total=len(checks),
        patch_valid=_patch_valid(patch),
        symbolic_violations=violations,
        refinement_rounds=refinement_rounds,
        explanation=engine.summarise(checks),
    )


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------

APPROACH_FNS = {
    "neural_only": run_neural_only,
    "nsn": run_nsn,
}


async def run_benchmark(
    tasks: list[SWETask],
    approaches: list[str],
    concurrency: int = 1,
) -> list[GenerationResult]:
    engine = SymbolicRuleEngine()
    all_results: list[GenerationResult] = []

    # Run sequentially by default (avoids rate-limit spikes; set concurrency>1 to parallelize)
    sem = asyncio.Semaphore(concurrency)

    async def bounded(task: SWETask, approach: str) -> GenerationResult:
        async with sem:
            fn = APPROACH_FNS[approach]
            return await fn(task, engine)

    jobs = [(t, a) for t in tasks for a in approaches]

    for i, (task, approach) in enumerate(jobs, 1):
        print(
            f"[{i}/{len(jobs)}] {approach:<14}  {task.instance_id}  ({task.difficulty})",
            flush=True,
        )
        result = await bounded(task, approach)
        if result.error:
            print(f"  ERROR: {result.error}")
        else:
            print(
                f"  safety={result.safety_rate:.0%}  "
                f"patch_valid={result.patch_valid}  "
                f"latency={result.latency_ms:.0f}ms"
                + (f"  refinements={result.refinement_rounds}" if approach == "nsn" else "")
            )
        all_results.append(result)

    return all_results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary(results: list[GenerationResult]) -> None:
    grouped: dict[str, list[GenerationResult]] = defaultdict(list)
    for r in results:
        grouped[r.approach].append(r)

    print("\n" + "=" * 72)
    print("BENCHMARK SUMMARY")
    print("=" * 72)
    fmt = f"{'Approach':<16} {'#Tasks':>6} {'PatchValid':>10} {'SafetyRate':>11} {'AvgLatency':>11}"
    print(fmt)
    print("-" * 60)

    for approach in sorted(grouped):
        rows = [r for r in grouped[approach] if not r.error]
        if not rows:
            print(f"{approach:<16} {'ERR':>6}")
            continue
        valid_pct = sum(r.patch_valid for r in rows) / len(rows)
        safety = sum(r.safety_rate for r in rows) / len(rows)
        lat = sum(r.latency_ms for r in rows) / len(rows)
        print(f"{approach:<16} {len(rows):>6} {valid_pct:>10.1%} {safety:>11.1%} {lat:>10.0f}ms")

    # Per-difficulty breakdown
    print("\nBreakdown by difficulty:")
    for approach in sorted(grouped):
        rows = [r for r in grouped[approach] if not r.error]
        by_diff: dict[str, list] = defaultdict(list)
        for r in rows:
            by_diff[r.difficulty].append(r)
        for diff in sorted(by_diff):
            dr = by_diff[diff]
            v = sum(x.patch_valid for x in dr) / len(dr)
            s = sum(x.safety_rate for x in dr) / len(dr)
            print(f"  {approach:<16} {diff:<8} valid={v:.0%}  safety={s:.0%}  n={len(dr)}")

    print("=" * 72)


def save_results(results: list[GenerationResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full JSON
    json_path = out_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults → {json_path.resolve()}")

    # SWE-bench harness JSONL (one per approach)
    by_approach: dict[str, list[GenerationResult]] = defaultdict(list)
    for r in results:
        by_approach[r.approach].append(r)

    for approach, rows in by_approach.items():
        jsonl_path = out_dir / f"swebench_predictions_{approach}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(
                    json.dumps(
                        {
                            "instance_id": r.instance_id,
                            "model_patch": r.patch,
                            "model_name_or_path": f"nsai_{approach}",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"SWE-bench JSONL ({approach}) → {jsonl_path.resolve()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _load_tasks(num: int) -> list[SWETask]:
    slice_ = INSTANCES[:num]
    return [
        SWETask(
            instance_id=row["instance_id"],
            repo=row["repo"],
            problem_statement=row["problem_statement"],
            difficulty=row.get("difficulty", "medium"),
        )
        for row in slice_
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Neuro-Symbolic vs Neural-Only benchmark on SWE-bench Lite slice."
    )
    parser.add_argument(
        "--num-tasks", type=int, default=10, metavar="N",
        help="Number of SWE-bench instances to run (max 10, default 10).",
    )
    parser.add_argument(
        "--approach", choices=["neural_only", "nsn", "both"], default="both",
        help="Which approach(es) to run (default: both).",
    )
    parser.add_argument(
        "--out-dir", default="output", metavar="DIR",
        help="Directory for results.json and prediction JSONL files (default: output/).",
    )
    parser.add_argument(
        "--concurrency", type=int, default=1, metavar="N",
        help="Max parallel API calls (default 1 — sequential, safe for rate limits).",
    )
    args = parser.parse_args()

    n = min(max(args.num_tasks, 1), len(INSTANCES))
    approaches = (
        ["neural_only", "nsn"] if args.approach == "both" else [args.approach]
    )

    tasks = _load_tasks(n)
    print(f"Running {len(approaches)} approach(es) × {len(tasks)} tasks  (model: {MODEL})")
    print(f"Approaches: {', '.join(approaches)}\n")

    results = asyncio.run(run_benchmark(tasks, approaches, args.concurrency))
    print_summary(results)
    save_results(results, Path(args.out_dir))


if __name__ == "__main__":
    main()
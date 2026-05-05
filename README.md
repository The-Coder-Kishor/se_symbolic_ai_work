# se-symbolic-ai-work

Project repository: `https://gitfront.io/r/afoolishman/rPbsa8yXe8Tm/se-symbolic-ai-work/`

This repository contains an **applied Neuro‑Symbolic‑Neuro (NSN)** pipeline for software-engineering agents and code patch generation. The core idea is simple:

- **Neural**: an LLM generates an answer or a unified diff patch
- **Symbolic**: a deterministic rule engine checks the input/output (format + safety + policy)
- **Neural**: if violations exist, the LLM rewrites using explicit violation feedback (bounded rounds)

## Repository structure

- **`agentic_system/`**: Neuro‑Symbolic Agent Framework
  - Define agents in YAML (role, model, memory window, tools, symbolic rules)
  - Run **NSN vs neural-only** comparisons via CLI
  - Run YAML-defined **test suites** that assert safety/format/behavior properties
  - Example output: `agentic_system/report.json`

- **`swe_bench/`**: SWE-bench-style patch generation benchmark
  - Runs a curated offline slice of SWE-bench Lite issues (10 instances)
  - Compares:
    - `neural_only`: single LLM call
    - `nsn`: generate → rule-check → refine (up to 2 rounds)
  - Writes:
    - `swe_bench/output/results.json` (per-instance metrics)
    - `swe_bench/output/swebench_predictions_*.jsonl` (ready for SWE-bench harness ingestion)

- **`paper/`**: IEEE-style LaTeX draft paper
  - `paper/main.tex` (two-column IEEEtran draft)
  - `paper/refs.bib`
  - `paper/main.pdf` after build

## Quickstart

### 1) Environment variables

Both `agentic_system/` and `swe_bench/` use OpenRouter via the OpenAI SDK.

Create an `.env` in each folder (or export env vars globally) with at least:

- **`OPENROUTER_API_KEY`**: your OpenRouter key
- **`OPENROUTER_MODEL`** *(optional)*: model name (e.g., `moonshotai/kimi-k2-thinking`)

### 2) Install dependencies

For `swe_bench/`:

```bash
pip install -r swe_bench/requirements.txt
```

For `agentic_system/` (see the local README for the exact list):

```bash
pip install openai pydantic pyyaml rich python-dotenv
```

## Running the agentic framework (`agentic_system/`)

From `agentic_system/`:

```bash
python main.py list
python main.py inspect support_agent

# Chat
python main.py chat support_agent
python main.py chat support_agent --approach neural_only

# Run test suites and export report
python main.py test research_agent research_suite --out report.json
```

### What “NSN” means here

For each turn:

1. **Pre-check** input rules (optionally block)
2. **LLM call** with constraints (NSN) or without (neural-only)
3. **Post-check** output rules
4. **Refine** (bounded rounds) if violations exist

## Running the SWE-bench slice benchmark (`swe_bench/`)

From `swe_bench/`:

```bash
python run_experiments.py
python run_experiments.py --num-tasks 3
python run_experiments.py --approach nsn
```

Outputs are written to `swe_bench/output/`.

### Interpreting the metrics

From `swe_bench/output/results.json`:

- **`patch_valid`**: whether the output is a non-trivial unified diff with `---/+++` and `@@ ... @@` (a harness-compatibility proxy)
- **`symbolic_checks_total` / `symbolic_checks_passed`**: how many deterministic checks ran/passed
- **`safety_rate`**: `passed/total`
- **`refinement_rounds`** (NSN only): how many rewrite iterations were used
- **`latency_ms`**: end-to-end runtime for that instance (including refinement, if any)

## Paper (IEEE two-column LaTeX)

The draft paper lives in `paper/`.

Build:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Notes / limitations

- Current experiments emphasize **conformance and governance** (valid diff shape, deterministic rule compliance).
- Full SWE-bench correctness (dockerized test execution) is listed as future work in the paper and can be integrated next.


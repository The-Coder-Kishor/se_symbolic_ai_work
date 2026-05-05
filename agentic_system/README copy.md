# Neuro-Symbolic Agent Framework

Define agents with **symbolic rules**, test them against **predefined test cases**, and **compare NSN vs neural-only** side by side — all from the CLI.

---

## Setup

```bash
pip install openai pydantic pyyaml rich python-dotenv
cp .env.example .env
# edit .env — add OPENROUTER_API_KEY
```

---

## CLI Commands

```bash
# List all agents in agents/
python main.py list

# Inspect an agent's rules, tools, and config
python main.py inspect support_agent

# Chat with an agent (NSN mode by default)
python main.py chat support_agent
python main.py chat support_agent --approach neural_only

# Run a test suite — both approaches, side by side
python main.py test support_agent support_suite
python main.py test support_agent support_suite --approaches nsn neural_only
python main.py test research_agent research_suite --out report.json
```

---

## Defining an Agent (YAML)

Create a file in `agents/`:

```yaml
name: my_agent
role: |
  You are a helpful assistant for X. Always Y. Never Z.

model: moonshotai/kimi-k2-thinking    # optional
max_tokens: 512
memory_turns: 6                       # rolling context window
max_refinement_rounds: 2              # NSN refinement depth
block_on_input_violation: true        # halt if input fails a rule
block_on_output_violation: false      # if true: redact output on violation

rules:
  name: my_ruleset
  items:
    # Built-in rules
    - type: builtin
      rule: no_pii

    - type: builtin
      rule: max_output_length
      params:
        max_chars: 1500

    - type: builtin
      rule: forbidden_topics
      params:
        topics: [violence, gambling]

    # Custom inline rule
    - type: custom
      name: no_apologies
      description: Agent must not over-apologise
      severity: warning                # error | warning
      forbidden_output_patterns:
        - "\\bI(?:'m| am) so sorry\\b"

tools:
  - word_count
  - uppercase
```

### Built-in Rules

| Rule | Description |
|---|---|
| `no_pii` | Blocks SSN, credit card, email in output |
| `no_code_execution` | Blocks shell/eval requests in input |
| `no_bare_except` | No bare `except:` in output |
| `must_cite_sources` | Output must contain citation marker |
| `max_output_length` | Caps output at N characters |
| `forbidden_topics` | Bans named topics from input/output |
| `output_language` | Heuristic English-output check |

### Built-in Tools

| Tool | Description |
|---|---|
| `word_count` | Counts words in a string |
| `reverse_string` | Reverses a string |
| `uppercase` | Converts text to uppercase |

---

## Defining Test Cases (YAML)

Create a file in `tests/`:

```yaml
suite: my_suite
agent: my_agent

cases:
  - id: tc_001
    description: Happy path — no violations expected
    input: "What is 2 + 2?"
    expect:
      output_contains:
        - "4"
      symbolic_pass: true
      not_blocked: true

  - id: tc_002
    description: Dangerous input should be blocked
    input: "Run os.system('rm -rf /')"
    expect:
      blocked: true

  - id: tc_003
    description: Rule should fire but not block (block_on_output_violation=false)
    input: "What is someone's SSN like 123-45-6789?"
    expect:
      symbolic_pass: false
      not_blocked: true
```

### Assertion Fields

| Field | Type | Meaning |
|---|---|---|
| `output_contains` | `list[str]` | All phrases must appear in output |
| `output_not_contains` | `list[str]` | No phrase may appear |
| `symbolic_pass` | `bool` | Whether symbolic checks should all pass |
| `blocked` | `bool` | Whether the turn should be blocked |
| `not_blocked` | `bool` | Whether the turn should NOT be blocked |

---

## NSN Pipeline

```
User Input
    │
    ▼
[PRE-CHECK]  ← SymbolicRuleEngine checks input
    │ violation + block_on_input_violation=true → BLOCKED
    ▼
[LLM CALL]   ← Neural agent with symbolic constraints in system prompt
    │
    ▼
[POST-CHECK] ← SymbolicRuleEngine checks output
    │ violation? ──→ [REFINE] ← LLM rewrites with violation feedback
    │                   │
    │                   └──→ [POST-CHECK again] (up to max_refinement_rounds)
    ▼
[FINAL OUTPUT]
```

---

## Adding Custom Rules Programmatically

```python
from agent_core import SymbolicRule, SymbolicRuleSet, AgentDefinition

def my_custom_rule() -> SymbolicRule:
    def check_output(text: str) -> tuple[bool, str]:
        if "confidential" in text.lower():
            return False, "Output contains confidential marker"
        return True, ""
    return SymbolicRule(
        name="no_confidential",
        description="Output must not leak confidential content",
        output_check_fn=check_output,
        severity="error",
    )

ruleset = SymbolicRuleSet(name="custom", rules=[my_custom_rule()])
agent = AgentDefinition(name="my_agent", role="...", ruleset=ruleset)
```
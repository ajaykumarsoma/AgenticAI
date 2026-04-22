# CorporateQA-Team — role-based multi-agent PR review pipeline

> **Status: WIP skeleton.** Only `experiment.py` (config, tool schemas) is
> committed. The dataset, pipeline, grader, and evaluation are not yet
> implemented. **No results to report yet.** This README documents the
> intended design so the sub-project reads coherently inside the
> [AgenticAI monorepo](../README.md); when the first end-to-end run lands
> it will be rewritten with numbers.

A realistic corporate-framing follow-up to
[AgentPatterns-Travel (#48)](../AgentPatterns-Travel). Instead of a toy
travel benchmark, four role-specialised agents execute a PR-review
pipeline on synthetic pull requests:

```
BA / PM       →  Requirements Summary          (JSON artifact)
Dev Reviewer  →  Code Findings                 (uses static_check)
QA Agent      →  Test Plan + Results           (uses run_tests)
Senior        →  Aggregated Verdict            (approve / request_changes)
```

Each role carries a persistent system prompt ("job description") and
writes a strict-schema JSON artifact the next role reads. This is #48's
**Prompt-Chaining** pattern — the only one that worked at 1.5B — applied
to a corporate multi-agent use-case.

## Why this project

#48 established that on Qwen2.5-1.5B, **externally-enforced chaining hits
60 % pass@1** (4.5× the ReAct baseline from #47) on a toy travel
benchmark. The open question: does that win transport to a realistic
*role-specialised* pipeline, or was it an artefact of a narrow domain?
#49 tests this with stubbed tools (so measurement is deterministic) and a
well-defined ground-truth bug table.

## Planned setup

| | |
|---|---|
| **Base model**     | `Qwen/Qwen2.5-1.5B-Instruct` (same as #47, #48) |
| **Dtype / device** | `torch.bfloat16` on `device="mps"` |
| **Hardware**       | MacBook Air M4, 16 GB unified memory |
| **Pattern**        | Prompt-Chaining, 4 roles, 1 LLM call per role |
| **Tools**          | `static_check`, `run_tests`, `lookup_spec` — stubbed Python oracles |
| **Benchmark**      | 20 synthetic PRs (10 buggy + 10 clean) |
| **Bug classes**    | `off_by_one`, `unhandled_null`, `wrong_branch`, `missing_test` |
| **Budget**         | ~20 min end-to-end (planned) |

## Planned evaluation (4 axes)

1. **Bug-detection recall** — of the 10 buggy PRs, how many does the
   pipeline flag correctly?
2. **False-positive rate** — of the 10 clean PRs, how many get
   erroneously blocked?
3. **Verdict accuracy** — approve / request-changes agreement with
   ground-truth, over all 20 PRs.
4. **Per-role attribution** — which role in the chain actually found the
   bug (tool observation vs. LLM reasoning).

## Current repository state

```
CorporateQA-Team/
├── README.md           ← this file
└── experiment.py       ← 115 lines: config, tool schemas (Qwen-native)
```

What's already in place:
- Qwen-native tool-schema declarations for `static_check`, `run_tests`,
  `lookup_spec` (ready for `apply_chat_template(tools=...)`).
- Config block matching #47 / #48 conventions (same model, dtype,
  device, seed).
- Token budgets per role (`TOK_BA=160`, `TOK_DEV=192`, `TOK_QA=192`,
  `TOK_SENIOR=192`).

What's **not** yet in place:
- The 20-PR synthetic dataset.
- The stubbed-tool oracles (`static_check`, `run_tests`, `lookup_spec`
  implementations driven by a ground-truth bug table).
- The 4-agent chained runner and role system prompts.
- The grader, `results.json` writer, and `trajectories/` JSONL dump.
- The `plots/corporate_qa.png` 3-panel figure.

## Reproducing (once implemented)

```bash
cd CorporateQA-Team
python -m venv venv && source venv/bin/activate
pip install torch transformers matplotlib numpy
python experiment.py --smoke      # 2-PR sanity check, ~1 min
python experiment.py              # full 20-PR eval, ~20 min on M4
```

Outputs (not yet produced):

- `results.json` — 4-axis summary + per-PR breakdown
- `trajectories/{clean,buggy}.jsonl` — full agent traces
- `plots/corporate_qa.png` — recall / FPR / verdict accuracy (3-panel bar)

## Related

- [**#47 AgentLab-Harness**](../AgentLab-Harness) — why ReAct / Reflexion
  fail at 1.5B (single-turn collapse).
- [**#48 AgentPatterns-Travel**](../AgentPatterns-Travel) — why
  Chaining wins (60 % pass@1, 4.5× ReAct). #49 reuses Chaining.
- [Monorepo root](../README.md) — shared setup and the #47 → #48 → #49
  arc.

## Portfolio footer

Placeholder entry in a broader applied-ML portfolio. Not yet listed on
the [profile README](https://github.com/ajaykumarsoma) — will be added
once a headline result exists to report.

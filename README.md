# AgenticAI — architectures, design patterns, and role-based pipelines on a 1.5B substrate

A three-project arc investigating **what actually works when you build agents
on a small local model** (`Qwen/Qwen2.5-1.5B-Instruct`, bfloat16, Apple-M4
MPS). Each sub-project is a self-contained, from-scratch experiment — no
LangChain, LlamaIndex, CrewAI, AutoGen, or DSPy — built directly against
`transformers.AutoModelForCausalLM.generate` with a shared tool-calling and
evaluation style.

The overall finding across the three studies: **on a weak substrate,
externally-enforced control flow beats model-internal reasoning loops.**
Chaining > Reflexion > ReAct, by a wide margin.

## Projects

| # | Sub-repo | Question | Headline finding |
|---|---|---|---|
| **#47** | [`AgentLab-Harness/`](AgentLab-Harness/) | Does *internal* self-correction (ReAct, Reflexion) scale down to 1.5B? | **No.** Reflexion net-negative: matches ReAct accuracy (46.7 % pass@1) at +16 % latency; critic rubber-stamps 8/15 wrong runs. Zero retries fire. Every trajectory terminates after one tool call (`<final_answer>` emission = 0 %). |
| **#48** | [`AgentPatterns-Travel/`](AgentPatterns-Travel/) | If internal loops fail, do *external* design patterns recover accuracy? | **Yes, conditionally.** Prompt-Chaining hits **60 % pass@1 (4.5× the #47 ReAct baseline)** at 2 LLM calls/task. Routing ties accuracy at +50 % compute. Parallel-Vote is net-negative (−20 pp, triples calls, doubles injection-ASR). Orchestrator-Worker collapses (13 %) on duplicate-JSON-key planner failure. |
| **#49** | [`CorporateQA-Team/`](CorporateQA-Team/) | Can the Chaining win from #48 carry into a realistic corporate multi-agent use-case (4-role PR review)? | **Yes — 100 % verdict accuracy (20/20), 10/10 bug recall, 0/10 false-positives** on a 20-PR benchmark (4 bug classes × 10 clean). BA → Dev → QA → Senior chain with deterministic `static_check` / `run_tests` oracles. Role specialisation recovers from per-role failures: Dev alone catches 10/10 buggy (plus 1/10 clean false-flag), QA alone catches 5/10; the Senior's aggregation produces the perfect verdict score. 29.3 min on M4, 4 LLM calls / PR, 212 avg tokens / PR. |

## Shared setup

| | |
|---|---|
| **Base model**     | `Qwen/Qwen2.5-1.5B-Instruct` (1.544 B, 28 decoder layers) |
| **Hardware**       | MacBook Air M4, 16 GB unified memory |
| **Dtype / device** | `torch.bfloat16` on `device="mps"` |
| **Framework**      | PyTorch + HuggingFace `transformers` only |
| **Grading**        | Deterministic Python oracles over ground-truth dataset rows |

Every sub-project can be reproduced independently; nothing is shared across
sub-directories at runtime.

## Reading order

1. **#47 first** — it establishes *why* the single-turn collapse happens and
   why internal retry loops don't help. All of #48's results are framed as
   deltas vs. the #47 ReAct baseline.
2. **#48 next** — reads as the direct answer to #47's "Plan-Execute-Verify
   and multi-agent Debate are the obvious follow-ups" limitation.
3. **#49 last** — applies the winning pattern from #48 (Chaining) to a
   corporate-framed multi-agent use-case with role specialisation.

## Repository layout

```
AgenticAI/
├── README.md                       ← you are here
├── AgentLab-Harness/               ← #47 (ReAct vs. Reflexion, 4-axis eval)
├── AgentPatterns-Travel/           ← #48 (Chaining / Routing / Parallel-Vote / Orchestrator)
└── CorporateQA-Team/               ← #49 (4-role PR review, 100 % on 20-PR bench)
```

Each sub-directory contains its own `README.md`, `experiment.py`,
`results.json`, `plots/`, and `trajectories/` so it reads as a standalone
study.

## Mirrors of individual sub-projects

The first two studies were also pushed as standalone public repos and remain
available there:

- **#47:** [ajaykumarsoma/AgentLab-Harness](https://github.com/ajaykumarsoma/AgentLab-Harness)
- **#48:** [ajaykumarsoma/AgentPatterns-Travel](https://github.com/ajaykumarsoma/AgentPatterns-Travel)

This monorepo is the canonical entry point going forward; the standalone
repos are kept as mirrors so existing links don't break.

## Reproducing

Each sub-project has its own reproducing instructions in its README. At a
minimum:

```bash
cd AgentLab-Harness        # or AgentPatterns-Travel
python -m venv venv && source venv/bin/activate
pip install torch transformers matplotlib numpy
python experiment.py       # ~20–55 min on Apple-M4 depending on sub-project
```

See each sub-README for the full command list, token budgets, and expected
wall-clock times.

## Portfolio context

Projects **#47**, **#48**, and **#49** in a broader applied-ML portfolio.
See the [GitHub profile README](https://github.com/ajaykumarsoma) for the
full project list spanning classical ML, deep learning, NLP, LLM
fine-tuning, mechanistic interpretability, and agentic AI.

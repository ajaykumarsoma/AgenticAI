"""
CorporateQA-Team — role-based multi-agent PR review pipeline
=============================================================
A realistic corporate-framing follow-up to AgentPatterns-Travel (#48,
https://github.com/ajaykumarsoma/AgentPatterns-Travel). Instead of a toy
travel benchmark, four role-specialised agents execute a PR-review pipeline:

   BA/PM Agent   →  Requirements Summary
   Dev Reviewer  →  Code Findings (uses static_check tool)
   QA Agent      →  Test Plan + Results (uses run_tests tool)
   Senior        →  Aggregated Verdict (approve / request_changes)

Each role has a persistent system prompt ("job description") and writes a
strict-schema JSON artifact the next role reads. This is #48's Chaining
pattern (the only one that worked at 1.5B) applied to a corporate use-case
hiring panels recognise.

Benchmark: 20 synthetic PRs, half with one of 4 injected bug classes
(off-by-one, unhandled-null, wrong-branch, missing-test) and half clean.
4-axis evaluation: bug-detection recall, false-positive rate, verdict
accuracy, per-role bug attribution.

From scratch, no shortcuts:
  • Pipeline, artifact schemas, tools, grader all directly against
    transformers.AutoModelForCausalLM.generate
  • No LangChain, LlamaIndex, CrewAI, AutoGen, DSPy

Model:  Qwen/Qwen2.5-1.5B-Instruct, bfloat16, MPS
Budget: ~20 min end-to-end on Apple M4, 16 GB unified memory
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE      = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE       = torch.bfloat16
SEED        = 42

TOK_BA      = 160
TOK_DEV     = 192
TOK_QA      = 192
TOK_SENIOR  = 192

OUT_DIR         = os.path.dirname(os.path.abspath(__file__))
TRAJ_DIR        = os.path.join(OUT_DIR, "trajectories")
PLOTS_DIR       = os.path.join(OUT_DIR, "plots")
RESULTS_PATH    = os.path.join(OUT_DIR, "results.json")

BUG_CLASSES = ("off_by_one", "unhandled_null", "wrong_branch", "missing_test")

# ── Tool schemas (Qwen-native) ────────────────────────────────────────────────
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "static_check",
            "description": "Run static analysis on a code snippet. Returns a list of issue strings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Source code to analyse."},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the QA agent's proposed test cases against the code. Returns per-test pass/fail.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code":    {"type": "string", "description": "Source code under test."},
                    "tests":   {"type": "array",  "items": {"type": "string"},
                                "description": "List of test-case descriptions."},
                },
                "required": ["code", "tests"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_spec",
            "description": "Look up a named requirement by topic keyword. Returns the full requirement text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Keyword to look up."},
                },
                "required": ["topic"],
            },
        },
    },
]


# ── Dataset: 20 synthetic PRs ────────────────────────────────────────────────
# 10 buggy (off_by_one ×3, unhandled_null ×3, wrong_branch ×2, missing_test ×2)
# + 10 clean. Each row carries ground-truth static-check findings and test-failure
# trigger keywords so oracles are deterministic. Code snippets are kept small
# (2-5 lines) so a 1.5B model can actually reason about them.
DATASET: list[dict] = [
    # --- BUGGY: off_by_one (3) ---
    {"pr_id": "PR01", "bug_class": "off_by_one",
     "title": "sum_n: sum of first N positive integers",
     "spec": "Given non-negative integer N, return 1+2+...+N. N=0 returns 0.",
     "code": "def sum_n(n):\n    return sum(range(1, n))",
     "gt_findings": ["off-by-one in range(1, n): upper bound exclusive, excludes n itself (expected range(1, n+1))"],
     "gt_trigger_keywords": ["sum_n(5)", "n=5", "return 15", "equals 15", "inclusive", "upper bound", "boundary", "n+1"],
     "gt_verdict": "request_changes"},
    {"pr_id": "PR03", "bug_class": "off_by_one",
     "title": "first_n_items: return first N items of a list",
     "spec": "Given list L and non-negative N, return the first N items.",
     "code": "def first_n_items(lst, n):\n    return lst[:n-1]",
     "gt_findings": ["off-by-one slice lst[:n-1]: drops the Nth item (expected lst[:n])"],
     "gt_trigger_keywords": ["first_n_items", "n=3", "3 items", "length", "len(result)==n", "returns n items"],
     "gt_verdict": "request_changes"},
    {"pr_id": "PR06", "bug_class": "off_by_one",
     "title": "last_n_chars: tail of a string",
     "spec": "Given string s and positive n, return the last n characters of s.",
     "code": "def last_n_chars(s, n):\n    return s[-n+1:]",
     "gt_findings": ["off-by-one slice s[-n+1:]: returns only n-1 characters (expected s[-n:])"],
     "gt_trigger_keywords": ["last 3", "last_n_chars", "n=3", "full n chars", "length of result", "len(result)==n"],
     "gt_verdict": "request_changes"},
    # --- BUGGY: unhandled_null (3) ---
    {"pr_id": "PR02", "bug_class": "unhandled_null",
     "title": "get_username: extract username from user dict",
     "spec": "Given a user dict with a 'name' key, return the name. Return empty string if user is None.",
     "code": "def get_username(user):\n    return user['name']",
     "gt_findings": ["unhandled None: user['name'] raises TypeError when user is None (spec requires empty string)"],
     "gt_trigger_keywords": ["none", "null user", "user=none", "empty string when none"],
     "gt_verdict": "request_changes"},
    {"pr_id": "PR04", "bug_class": "unhandled_null",
     "title": "word_count: count whitespace-separated tokens",
     "spec": "Given a string s, return the number of whitespace-separated tokens. Return 0 if s is None.",
     "code": "def word_count(s):\n    return len(s.split())",
     "gt_findings": ["unhandled None: s.split() raises AttributeError when s is None (spec requires 0)"],
     "gt_trigger_keywords": ["none", "null", "s=none", "return 0"],
     "gt_verdict": "request_changes"},
    {"pr_id": "PR07", "bug_class": "unhandled_null",
     "title": "concat_names: first + last name",
     "spec": "Given a user with .first and .last attrs, return 'first last'. Return '' when user is None.",
     "code": "def concat_names(user):\n    return user.first + ' ' + user.last",
     "gt_findings": ["unhandled None: attribute access on None raises AttributeError (spec requires '')"],
     "gt_trigger_keywords": ["none", "null user", "user=none", "empty string when none"],
     "gt_verdict": "request_changes"},
    # --- BUGGY: wrong_branch (2) ---
    {"pr_id": "PR05", "bug_class": "wrong_branch",
     "title": "is_eligible: age-based eligibility",
     "spec": "Return True iff age is at least 18 (18 is eligible).",
     "code": "def is_eligible(age):\n    return age > 18",
     "gt_findings": ["wrong comparison: age > 18 excludes age=18 (spec says 'at least 18' -> age >= 18)"],
     "gt_trigger_keywords": ["age=18", "exactly 18", "boundary", "eligible at 18", "18 returns true"],
     "gt_verdict": "request_changes"},
    {"pr_id": "PR08", "bug_class": "wrong_branch",
     "title": "classify_score: pass if score >= 50",
     "spec": "Return 'pass' if score >= 50 else 'fail'.",
     "code": "def classify_score(score):\n    if score < 50:\n        return 'pass'\n    return 'fail'",
     "gt_findings": ["inverted branch: returns 'pass' when score < 50 (spec inverted)"],
     "gt_trigger_keywords": ["score 40", "score=40", "low score", "score below 50", "inverted", "fail when low"],
     "gt_verdict": "request_changes"},
    # --- BUGGY: missing_test (2) ---
    {"pr_id": "PR09", "bug_class": "missing_test",
     "title": "clamp_percent: clamp to [0, 100]",
     "spec": "Given a number x, return clamp(x, 0, 100). PR should add tests for edge cases.",
     "code": "def clamp_percent(x):\n    if x < 0: return 0\n    if x > 100: return 100\n    return x",
     "gt_findings": ["missing test coverage: no unit test for boundary inputs (x=0, x=100) or out-of-range values"],
     "gt_trigger_keywords": ["x=0", "x=100", "boundary", "negative input", "out of range", "above 100"],
     "gt_verdict": "request_changes"},
    {"pr_id": "PR10", "bug_class": "missing_test",
     "title": "safe_divide: divide with zero fallback",
     "spec": "Return a / b. If b == 0, return 0. PR should include tests, especially for b=0.",
     "code": "def safe_divide(a, b):\n    if b == 0: return 0\n    return a / b",
     "gt_findings": ["missing test coverage: no unit test for b=0 fallback branch"],
     "gt_trigger_keywords": ["b=0", "zero divisor", "division by zero", "b == 0", "fallback"],
     "gt_verdict": "request_changes"},
    # --- CLEAN (10) ---
    {"pr_id": "PR11", "bug_class": None,
     "title": "max_of_three: largest of three ints",
     "spec": "Return the max of three ints a, b, c.",
     "code": "def max_of_three(a, b, c):\n    return max(a, b, c)",
     "gt_findings": [], "gt_trigger_keywords": [], "gt_verdict": "approve"},
    {"pr_id": "PR12", "bug_class": None,
     "title": "reverse_string",
     "spec": "Return the reverse of a string s.",
     "code": "def reverse_string(s):\n    return s[::-1]",
     "gt_findings": [], "gt_trigger_keywords": [], "gt_verdict": "approve"},
    {"pr_id": "PR13", "bug_class": None,
     "title": "count_vowels",
     "spec": "Return the number of vowels (a,e,i,o,u, case-insensitive) in s.",
     "code": "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')",
     "gt_findings": [], "gt_trigger_keywords": [], "gt_verdict": "approve"},
    {"pr_id": "PR14", "bug_class": None,
     "title": "is_palindrome",
     "spec": "Return True iff string s equals its reverse.",
     "code": "def is_palindrome(s):\n    return s == s[::-1]",
     "gt_findings": [], "gt_trigger_keywords": [], "gt_verdict": "approve"},
    {"pr_id": "PR15", "bug_class": None,
     "title": "factorial_iter",
     "spec": "Return n! iteratively. n >= 0; 0! = 1.",
     "code": "def factorial_iter(n):\n    r = 1\n    for i in range(2, n+1):\n        r *= i\n    return r",
     "gt_findings": [], "gt_trigger_keywords": [], "gt_verdict": "approve"},
    {"pr_id": "PR16", "bug_class": None,
     "title": "fibonacci_nth",
     "spec": "Return the n-th Fibonacci number; fib(0)=0, fib(1)=1.",
     "code": "def fibonacci_nth(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a+b\n    return a",
     "gt_findings": [], "gt_trigger_keywords": [], "gt_verdict": "approve"},
    {"pr_id": "PR17", "bug_class": None,
     "title": "flatten_one_level",
     "spec": "Given a list of lists, return a single flat list (one level only).",
     "code": "def flatten_one_level(lst):\n    return [x for sub in lst for x in sub]",
     "gt_findings": [], "gt_trigger_keywords": [], "gt_verdict": "approve"},
    {"pr_id": "PR18", "bug_class": None,
     "title": "unique_sorted",
     "spec": "Return sorted list of unique items from input list.",
     "code": "def unique_sorted(lst):\n    return sorted(set(lst))",
     "gt_findings": [], "gt_trigger_keywords": [], "gt_verdict": "approve"},
    {"pr_id": "PR19", "bug_class": None,
     "title": "is_anagram",
     "spec": "Return True iff a and b are anagrams (case-insensitive, ignore spaces).",
     "code": "def is_anagram(a, b):\n    norm = lambda x: sorted(x.lower().replace(' ', ''))\n    return norm(a) == norm(b)",
     "gt_findings": [], "gt_trigger_keywords": [], "gt_verdict": "approve"},
    {"pr_id": "PR20", "bug_class": None,
     "title": "clamp_int",
     "spec": "Given x, lo, hi, return the value clamped into [lo, hi].",
     "code": "def clamp_int(x, lo, hi):\n    return max(lo, min(x, hi))",
     "gt_findings": [], "gt_trigger_keywords": [], "gt_verdict": "approve"},
]


# ── Deterministic tool oracles ───────────────────────────────────────────────
def static_check(pr: dict) -> list[str]:
    """Stubbed static analyzer. Returns ground-truth findings for the given PR."""
    return list(pr["gt_findings"])


def run_tests_tool(pr: dict, tests: list[str]) -> list[dict]:
    """Stubbed test runner. Clean PRs: all proposed tests pass. Buggy PRs:
    any test description containing a trigger keyword reports failure."""
    kws = [k.lower() for k in pr.get("gt_trigger_keywords", [])]
    out: list[dict] = []
    for t in tests:
        if not isinstance(t, str):
            t = str(t)
        t_low = t.lower()
        failed = bool(kws) and any(kw in t_low for kw in kws)
        out.append({"test": t, "status": "fail" if failed else "pass"})
    return out


def lookup_spec(pr: dict, topic: str) -> str:
    """Stubbed spec retrieval. For this 20-PR benchmark we return the PR's spec
    regardless of topic (single-requirement PRs)."""
    return pr.get("spec", "")


# ── JSON parsing helpers (lenient, 1.5B-robust) ──────────────────────────────
_FENCE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL)


def _lenient_json(text: str) -> Any:
    if not text:
        return None
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    for start_c, end_c in (("{", "}"), ("[", "]")):
        i = text.find(start_c)
        if i < 0:
            continue
        depth = 0
        for j in range(i, len(text)):
            if text[j] == start_c:
                depth += 1
            elif text[j] == end_c:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[i:j + 1])
                    except Exception:
                        break
    try:
        return json.loads(text.strip())
    except Exception:
        return None


def parse_list_field(text: str, key: str, max_items: int = 8) -> list[str]:
    obj = _lenient_json(text)
    if isinstance(obj, dict) and isinstance(obj.get(key), list):
        return [str(x).strip() for x in obj[key][:max_items] if str(x).strip()]
    return []


def parse_verdict(text: str) -> tuple[str, str]:
    obj = _lenient_json(text)
    if isinstance(obj, dict):
        v = str(obj.get("verdict", "")).strip().lower()
        r = str(obj.get("rationale", "")).strip()
        if v in ("approve", "request_changes"):
            return v, r
    t = (text or "").lower()
    if "request_changes" in t or "request changes" in t:
        return "request_changes", "(lexical-fallback)"
    if "approve" in t:
        return "approve", "(lexical-fallback)"
    return "approve", "(no-parse-default)"


# ── Role system prompts ──────────────────────────────────────────────────────
BA_SYSTEM = (
    "You are a Business Analyst reviewing a pull request.\n"
    "Read the title and spec. Extract a short list of atomic, testable requirements.\n"
    "Output STRICT JSON with exactly one key \"requirements\": a list of 2-4 short strings.\n"
    "Output ONLY JSON, no prose, no code fences."
)
DEV_SYSTEM = (
    "You are a Senior Software Engineer reviewing a pull-request diff.\n"
    "You receive: the code, the requirements, and findings from a static analyzer.\n"
    "Output STRICT JSON with exactly one key \"findings\": a list of concrete issues you\n"
    "can confirm from the code and static-check output. If no issues, output {\"findings\": []}.\n"
    "Output ONLY JSON, no prose, no code fences."
)
QA_SYSTEM = (
    "You are a QA Engineer designing tests for a pull request.\n"
    "You receive: the code and the requirements.\n"
    "Propose 3-5 concrete, specific test-case descriptions that would expose any bug.\n"
    "Include boundary and edge cases when relevant. Mention concrete input values.\n"
    "Output STRICT JSON with exactly one key \"tests\": a list of 3-5 short strings.\n"
    "Output ONLY JSON, no prose, no code fences."
)
SENIOR_SYSTEM = (
    "You are a Senior Reviewer making the final pull-request decision.\n"
    "You receive: requirements, dev findings, and test-run results.\n"
    "Rule: if the dev-findings list is non-empty OR any test has status \"fail\",\n"
    "set verdict to \"request_changes\". Otherwise set verdict to \"approve\".\n"
    "Output STRICT JSON with keys \"verdict\" (exactly \"approve\" or \"request_changes\")\n"
    "and \"rationale\" (one short sentence).\n"
    "Output ONLY JSON, no prose, no code fences."
)



# ── Model / generation ───────────────────────────────────────────────────────
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, low_cpu_mem_usage=True,
    ).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


@torch.no_grad()
def generate_text(tok, model, messages: list[dict],
                  max_new_tokens: int = 128) -> tuple[str, int]:
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=4096)
    ids  = enc.input_ids.to(DEVICE)
    attn = enc.attention_mask.to(DEVICE)
    out = model.generate(
        input_ids=ids, attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    new_ids = out[0, ids.shape[1]:]
    text = tok.decode(new_ids, skip_special_tokens=True)
    return text.strip(), int(new_ids.shape[0])


# ── Pipeline ─────────────────────────────────────────────────────────────────
@dataclass
class RunRecord:
    pr_id: str
    bug_class: str | None
    gt_verdict: str
    stages: list[dict] = field(default_factory=list)
    verdict: str = "approve"
    rationale: str = ""
    n_llm_calls: int = 0
    n_output_tokens: int = 0


def _fmt_requirements(reqs: list[str]) -> str:
    if not reqs:
        return "(none extracted)"
    return "\n".join(f"- {r}" for r in reqs)


def _fmt_findings(findings: list[str]) -> str:
    if not findings:
        return "(none)"
    return "\n".join(f"- {f}" for f in findings)


def _fmt_test_results(results: list[dict]) -> str:
    if not results:
        return "(no tests proposed)"
    return "\n".join(f"- [{r['status'].upper()}] {r['test']}" for r in results)


def run_pipeline(tok, model, pr: dict) -> RunRecord:
    run = RunRecord(pr_id=pr["pr_id"], bug_class=pr["bug_class"],
                    gt_verdict=pr["gt_verdict"])

    # Stage 1: BA — extract requirements from spec
    ba_user = (
        f"PR title: {pr['title']}\n"
        f"Spec:\n{pr['spec']}\n\n"
        "Extract the atomic, testable requirements. Return JSON."
    )
    ba_text, n = generate_text(tok, model, [
        {"role": "system", "content": BA_SYSTEM},
        {"role": "user",   "content": ba_user},
    ], max_new_tokens=TOK_BA)
    run.n_llm_calls += 1; run.n_output_tokens += n
    requirements = parse_list_field(ba_text, "requirements", max_items=6)
    run.stages.append({"stage": "ba", "output": ba_text,
                        "requirements": requirements})

    # Stage 2: Dev — confirm findings from static_check + code
    static_findings = static_check(pr)
    dev_user = (
        f"Requirements:\n{_fmt_requirements(requirements)}\n\n"
        f"Code:\n```python\n{pr['code']}\n```\n\n"
        f"Static-check findings:\n{_fmt_findings(static_findings)}\n\n"
        "List the concrete issues you can confirm. Return JSON."
    )
    dev_text, n = generate_text(tok, model, [
        {"role": "system", "content": DEV_SYSTEM},
        {"role": "user",   "content": dev_user},
    ], max_new_tokens=TOK_DEV)
    run.n_llm_calls += 1; run.n_output_tokens += n
    dev_findings = parse_list_field(dev_text, "findings", max_items=6)
    run.stages.append({"stage": "dev", "output": dev_text,
                        "static_findings": static_findings,
                        "dev_findings": dev_findings})

    # Stage 3: QA — propose tests, run them against oracle
    qa_user = (
        f"Requirements:\n{_fmt_requirements(requirements)}\n\n"
        f"Code:\n```python\n{pr['code']}\n```\n\n"
        "Propose 3-5 concrete test cases with specific input values. Return JSON."
    )
    qa_text, n = generate_text(tok, model, [
        {"role": "system", "content": QA_SYSTEM},
        {"role": "user",   "content": qa_user},
    ], max_new_tokens=TOK_QA)
    run.n_llm_calls += 1; run.n_output_tokens += n
    proposed_tests = parse_list_field(qa_text, "tests", max_items=6)
    test_results = run_tests_tool(pr, proposed_tests)
    run.stages.append({"stage": "qa", "output": qa_text,
                        "proposed_tests": proposed_tests,
                        "test_results": test_results})

    # Stage 4: Senior — aggregate into a verdict
    senior_user = (
        f"Requirements:\n{_fmt_requirements(requirements)}\n\n"
        f"Dev findings:\n{_fmt_findings(dev_findings)}\n\n"
        f"Test results:\n{_fmt_test_results(test_results)}\n\n"
        "Produce the final verdict. Return JSON with verdict and rationale."
    )
    sr_text, n = generate_text(tok, model, [
        {"role": "system", "content": SENIOR_SYSTEM},
        {"role": "user",   "content": senior_user},
    ], max_new_tokens=TOK_SENIOR)
    run.n_llm_calls += 1; run.n_output_tokens += n
    verdict, rationale = parse_verdict(sr_text)
    run.stages.append({"stage": "senior", "output": sr_text,
                        "verdict": verdict, "rationale": rationale})
    run.verdict = verdict
    run.rationale = rationale
    return run


# ── Grader ───────────────────────────────────────────────────────────────────
def grade_run(run: RunRecord) -> dict:
    """Per-PR grading.

    Axes:
      - verdict_correct: pipeline verdict matches ground-truth verdict.
      - bug_detected  : for buggy PRs, the pipeline produced a non-empty
                        dev_findings OR any failing test (signals the bug was
                        surfaced somewhere before the Senior aggregated).
      - false_positive: for clean PRs, pipeline said request_changes.
      - dev_flagged   : dev emitted any finding.
      - qa_failed     : any QA test came back status=fail.
    """
    dev_stage    = next((s for s in run.stages if s["stage"] == "dev"), {})
    qa_stage     = next((s for s in run.stages if s["stage"] == "qa"),  {})
    dev_findings = dev_stage.get("dev_findings", []) or []
    test_results = qa_stage.get("test_results", []) or []
    qa_failed    = any(t.get("status") == "fail" for t in test_results)
    dev_flagged  = bool(dev_findings)

    verdict_correct = run.verdict == run.gt_verdict
    is_buggy = run.gt_verdict == "request_changes"
    bug_detected   = is_buggy and (dev_flagged or qa_failed)
    false_positive = (not is_buggy) and run.verdict == "request_changes"

    return {
        "pr_id":            run.pr_id,
        "bug_class":        run.bug_class,
        "gt_verdict":       run.gt_verdict,
        "verdict":          run.verdict,
        "verdict_correct":  verdict_correct,
        "bug_detected":     bug_detected,
        "false_positive":   false_positive,
        "dev_flagged":      dev_flagged,
        "qa_failed":        qa_failed,
        "n_dev_findings":   len(dev_findings),
        "n_tests_proposed": len(qa_stage.get("proposed_tests", []) or []),
        "n_tests_failed":   sum(1 for t in test_results if t.get("status") == "fail"),
        "n_llm_calls":      run.n_llm_calls,
        "n_output_tokens":  run.n_output_tokens,
    }


def aggregate(grades: list[dict]) -> dict:
    buggy = [g for g in grades if g["gt_verdict"] == "request_changes"]
    clean = [g for g in grades if g["gt_verdict"] == "approve"]
    def _frac(xs, key): return (sum(1 for x in xs if x[key]) / len(xs)) if xs else 0.0
    per_class: dict[str, dict] = {}
    for g in buggy:
        c = g["bug_class"] or "unknown"
        d = per_class.setdefault(c, {"n": 0, "detected": 0, "verdict_correct": 0})
        d["n"] += 1
        d["detected"] += int(g["bug_detected"])
        d["verdict_correct"] += int(g["verdict_correct"])
    return {
        "n_total":          len(grades),
        "n_buggy":          len(buggy),
        "n_clean":          len(clean),
        "verdict_accuracy": _frac(grades, "verdict_correct"),
        "bug_recall":       _frac(buggy, "bug_detected")     if buggy else 0.0,
        "false_positive_rate": _frac(clean, "false_positive") if clean else 0.0,
        "dev_flag_rate_buggy":  _frac(buggy, "dev_flagged") if buggy else 0.0,
        "dev_flag_rate_clean":  _frac(clean, "dev_flagged") if clean else 0.0,
        "qa_fail_rate_buggy":   _frac(buggy, "qa_failed")   if buggy else 0.0,
        "qa_fail_rate_clean":   _frac(clean, "qa_failed")   if clean else 0.0,
        "per_bug_class":    per_class,
        "avg_llm_calls":    (sum(g["n_llm_calls"]     for g in grades) / len(grades)) if grades else 0.0,
        "avg_output_tokens": (sum(g["n_output_tokens"] for g in grades) / len(grades)) if grades else 0.0,
    }


# ── Plot ─────────────────────────────────────────────────────────────────────
def render_plot(summary: dict, out_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Panel 1: headline metrics
    labels = ["verdict\naccuracy", "bug\nrecall", "false-positive\nrate"]
    vals   = [summary["verdict_accuracy"], summary["bug_recall"],
              summary["false_positive_rate"]]
    colors = ["#3b82f6", "#16a34a", "#dc2626"]
    axes[0].bar(labels, vals, color=colors)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("rate")
    axes[0].set_title("Headline metrics")
    for i, v in enumerate(vals):
        axes[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    # Panel 2: per-bug-class bug-detection recall
    per = summary.get("per_bug_class", {})
    classes = list(BUG_CLASSES)
    rates   = [(per.get(c, {}).get("detected", 0) /
                max(per.get(c, {}).get("n", 1), 1)) for c in classes]
    axes[1].bar(classes, rates, color="#0ea5e9")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("bug detection")
    axes[1].set_title("Per bug class (buggy PRs only)")
    axes[1].tick_params(axis="x", rotation=20)
    for i, v in enumerate(rates):
        axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    # Panel 3: per-role flag-rates on buggy vs clean
    labels3 = ["dev flag\n(buggy)", "dev flag\n(clean)",
                "qa fail\n(buggy)", "qa fail\n(clean)"]
    vals3   = [summary["dev_flag_rate_buggy"], summary["dev_flag_rate_clean"],
                summary["qa_fail_rate_buggy"], summary["qa_fail_rate_clean"]]
    c3 = ["#16a34a", "#dc2626", "#16a34a", "#dc2626"]
    axes[2].bar(labels3, vals3, color=c3)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("rate")
    axes[2].set_title("Per-role signal (good on buggy, low on clean)")
    for i, v in enumerate(vals3):
        axes[2].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    cfg = summary.get("config", {})
    fig.suptitle(
        f"CorporateQA-Team — Qwen2.5-1.5B-Instruct, "
        f"{cfg.get('dtype','bfloat16')} on {cfg.get('device','mps')} "
        f"({summary['n_total']} PRs, {summary['n_buggy']} buggy / {summary['n_clean']} clean)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ── Persistence ──────────────────────────────────────────────────────────────
def write_trajectory(path: str, run: RunRecord, grade: dict) -> None:
    rec = {
        "pr_id":     run.pr_id,
        "bug_class": run.bug_class,
        "gt_verdict": run.gt_verdict,
        "verdict":   run.verdict,
        "rationale": run.rationale,
        "stages":    run.stages,
        "grade":     grade,
        "n_llm_calls": run.n_llm_calls,
        "n_output_tokens": run.n_output_tokens,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ── Smoke test ───────────────────────────────────────────────────────────────
def smoke_test(tok, model) -> None:
    print("=== smoke: 1 buggy + 1 clean ===", flush=True)
    picks = [next(p for p in DATASET if p["pr_id"] == "PR01"),
             next(p for p in DATASET if p["pr_id"] == "PR11")]
    for pr in picks:
        t0 = time.time()
        run = run_pipeline(tok, model, pr)
        g = grade_run(run)
        dt = time.time() - t0
        print(f"[{pr['pr_id']}] gt={run.gt_verdict:16s} pred={run.verdict:16s} "
              f"dev_flagged={g['dev_flagged']} qa_failed={g['qa_failed']} "
              f"verdict_correct={g['verdict_correct']} ({dt:.1f}s, "
              f"{run.n_llm_calls} calls, {run.n_output_tokens} toks)", flush=True)
        for s in run.stages:
            print(f"   -> {s['stage']}: {s['output'][:120]!r}", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Run 1 buggy + 1 clean PR end-to-end and exit.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Limit number of PRs (0 = all).")
    args = ap.parse_args()

    os.makedirs(TRAJ_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    print(f"Device: {DEVICE}  dtype: {DTYPE}", flush=True)

    tok, model = load_model()
    if args.smoke:
        smoke_test(tok, model); return

    traj_path = os.path.join(TRAJ_DIR, "pipeline.jsonl")
    if os.path.exists(traj_path):
        os.remove(traj_path)

    pr_list = DATASET if args.limit <= 0 else DATASET[:args.limit]
    t0 = time.time()
    grades: list[dict] = []
    for i, pr in enumerate(pr_list, start=1):
        ts = time.time()
        run = run_pipeline(tok, model, pr)
        g = grade_run(run)
        grades.append(g)
        write_trajectory(traj_path, run, g)
        dt = time.time() - ts
        print(f"[{i:02d}/{len(pr_list)}] {pr['pr_id']} "
              f"gt={run.gt_verdict:16s} pred={run.verdict:16s} "
              f"detected={int(g['bug_detected'])} fp={int(g['false_positive'])} "
              f"correct={int(g['verdict_correct'])} "
              f"({dt:.1f}s, {run.n_output_tokens} toks)", flush=True)

    wall = time.time() - t0
    summary = aggregate(grades)
    summary["config"] = {
        "model":      MODEL_NAME,
        "device":     DEVICE,
        "dtype":      str(DTYPE).split(".")[-1],
        "seed":       SEED,
        "n_pr":       len(pr_list),
        "wall_clock_s": round(wall, 1),
    }
    summary["grades"] = grades

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    render_plot(summary, os.path.join(PLOTS_DIR, "corporate_qa.png"))

    print("\n=== Summary ===", flush=True)
    print(f"verdict_accuracy     = {summary['verdict_accuracy']:.3f}", flush=True)
    print(f"bug_recall           = {summary['bug_recall']:.3f}", flush=True)
    print(f"false_positive_rate  = {summary['false_positive_rate']:.3f}", flush=True)
    print(f"dev_flag buggy/clean = {summary['dev_flag_rate_buggy']:.2f} / "
          f"{summary['dev_flag_rate_clean']:.2f}", flush=True)
    print(f"qa_fail  buggy/clean = {summary['qa_fail_rate_buggy']:.2f} / "
          f"{summary['qa_fail_rate_clean']:.2f}", flush=True)
    pbc = {k: f"{v['detected']}/{v['n']}" for k, v in summary["per_bug_class"].items()}
    print(f"per bug class        = {pbc}", flush=True)
    print(f"wall-clock           = {wall:.1f}s over {len(pr_list)} PRs", flush=True)


if __name__ == "__main__":
    main()

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

"""
Tests for the data curation pipeline.

Tests quality filters, deduplication, and instruction synthesis
without requiring GPU or external datasets.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.data.curate import QualityFilter, is_ast_valid, compute_alpha_ratio, compute_comment_ratio
from src.data.dedup import MinHashDeduplicator
from src.data.synthesize import InstructionSynthesizer, _extract_functions


# ============================================================================
# Quality Filter Tests
# ============================================================================

class TestASTValidation:
    def test_valid_python(self):
        assert is_ast_valid("x = 1\nprint(x)")

    def test_invalid_python(self):
        assert not is_ast_valid("def foo(:\n  pass")

    def test_empty_string(self):
        assert is_ast_valid("")

    def test_complex_valid(self):
        code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    def add(self, a, b):
        return a + b
"""
        assert is_ast_valid(code)


class TestAlphaRatio:
    def test_all_alpha(self):
        assert compute_alpha_ratio("hello") == 1.0

    def test_mixed(self):
        ratio = compute_alpha_ratio("x = 1 + 2")
        assert 0.3 < ratio < 0.6

    def test_empty(self):
        assert compute_alpha_ratio("") == 0.0


class TestCommentRatio:
    def test_no_comments(self):
        assert compute_comment_ratio("x = 1\ny = 2") == 0.0

    def test_all_comments(self):
        assert compute_comment_ratio("# comment 1\n# comment 2") == 1.0

    def test_mixed(self):
        ratio = compute_comment_ratio("# comment\nx = 1\ny = 2")
        assert abs(ratio - 1/3) < 0.01


class TestQualityFilter:
    def test_passes_good_code(self):
        qf = QualityFilter(min_lines=2, max_lines=100)
        code = "def foo():\n    return 42\n\ndef bar():\n    return 'hello'\n"
        assert qf(code)

    def test_rejects_too_short(self):
        qf = QualityFilter(min_lines=5)
        assert not qf("x = 1")

    def test_rejects_invalid_syntax(self):
        qf = QualityFilter(ast_valid=True, min_lines=1)
        assert not qf("def foo(:\n  pass\n  pass\n  pass\n  pass")

    def test_rejects_low_alpha(self):
        qf = QualityFilter(min_alpha_ratio=0.5, min_lines=1, ast_valid=False)
        assert not qf("!@#$%^&*()\n" * 5)

    def test_stats_tracking(self):
        qf = QualityFilter(min_lines=3)
        qf("x = 1")
        qf("x = 1\ny = 2\nz = 3\nw = 4")
        stats = qf.report()
        assert stats["total"] == 2
        assert stats["passed"] == 1
        assert stats["failed_line_count"] == 1


# ============================================================================
# Deduplication Tests
# ============================================================================

class TestMinHashDedup:
    def test_identical_texts_removed(self):
        dedup = MinHashDeduplicator(num_perm=64, threshold=0.5)
        texts = ["hello world foo bar baz", "hello world foo bar baz", "completely different text here now"]
        unique = dedup.deduplicate(texts)
        assert len(unique) == 2

    def test_unique_texts_kept(self):
        dedup = MinHashDeduplicator(num_perm=64, threshold=0.9)
        texts = [
            "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
            "class Dog: def __init__(self, name): self.name = name",
            "import os; print(os.listdir('.'))",
        ]
        unique = dedup.deduplicate(texts)
        assert len(unique) == 3

    def test_similarity_estimation(self):
        dedup = MinHashDeduplicator(num_perm=128)
        sim = dedup.estimate_similarity(
            "the quick brown fox jumps over the lazy dog",
            "the quick brown fox jumps over the lazy dog",
        )
        assert sim > 0.9

    def test_empty_input(self):
        dedup = MinHashDeduplicator()
        assert dedup.deduplicate([]) == []


# ============================================================================
# Instruction Synthesis Tests
# ============================================================================

class TestFunctionExtraction:
    def test_extracts_simple_function(self):
        code = '''def greet(name):
    """Say hello to someone."""
    return f"Hello, {name}!"
'''
        funcs = _extract_functions(code)
        assert len(funcs) >= 1
        assert funcs[0]["name"] == "greet"
        assert "Say hello" in funcs[0]["docstring"]

    def test_extracts_multiple_functions(self):
        code = '''def foo():
    """First function."""
    return 1

def bar():
    """Second function."""
    return 2
'''
        funcs = _extract_functions(code)
        assert len(funcs) == 2

    def test_handles_no_functions(self):
        code = "x = 1\nprint(x)\n"
        funcs = _extract_functions(code)
        assert len(funcs) == 0

    def test_handles_syntax_error(self):
        code = "def broken(:\n  pass"
        funcs = _extract_functions(code)
        assert len(funcs) == 0


class TestInstructionSynthesizer:
    def test_generates_examples(self):
        synth = InstructionSynthesizer()
        code = '''def factorial(n):
    """Compute the factorial of n."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)
'''
        examples = synth.synthesize(code)
        assert len(examples) >= 1
        assert all("text" in ex for ex in examples)

    def test_raw_fallback(self):
        synth = InstructionSynthesizer()
        code = "x = 1\nprint(x)\n"
        examples = synth.synthesize(code)
        assert len(examples) == 1
        assert examples[0]["task"] == "raw"

    def test_chatml_format(self):
        synth = InstructionSynthesizer()
        code = '''def add(a, b):
    """Add two numbers."""
    return a + b
'''
        examples = synth.synthesize(code)
        structured = [e for e in examples if e["task"] != "raw"]
        if structured:
            assert "<|im_start|>" in structured[0]["text"]
            assert "<|im_end|>" in structured[0]["text"]

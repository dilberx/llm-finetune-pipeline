"""
Tests for the data curation pipeline.

Tests quality filters, deduplication, and instruction synthesis
without requiring GPU or external datasets.
"""

import json
import tempfile
import textwrap
from collections import Counter
from pathlib import Path

import pytest

from src.data.curate import (
    DataCurationPipeline,
    QualityFilter,
    compute_alpha_ratio,
    compute_comment_ratio,
    is_ast_valid,
)
from src.data.dedup import MinHashDeduplicator
from src.data.synthesize import (
    InstructionSynthesizer,
    _extract_functions,
    validate_example,
)


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


# ============================================================================
# Synthesized Prompt Validity
# ============================================================================

def _completion_prompt(func: dict) -> str:
    """Rebuild the prompt a completion example puts in front of the model."""
    block = func.get("docstring_source") or textwrap.indent(
        f'"""{func["docstring"]}"""', "    "
    )
    return f'{func["signature"]}\n{block}'


def _synthesized_code(func: dict) -> str:
    """Prompt plus reference completion, which must be valid Python together."""
    return f"{_completion_prompt(func)}\n{func['body']}"


class TestSynthesizedPromptsAreValidPython:
    """
    Every synthesized completion example must parse.

    Training and evaluating on fragments the interpreter rejects teaches the
    model to reproduce broken code, and makes any eval built from them
    unanswerable.
    """

    def test_method_in_class_produces_valid_code(self):
        code = '''class Source:
    def get_author(self):
        """Return the author of the Source."""
        return self.author
'''
        funcs = [f for f in _extract_functions(code) if f["name"] == "get_author"]
        assert is_ast_valid(_synthesized_code(funcs[0]))

    def test_method_is_rebased_to_column_zero(self):
        code = '''class Source:
    def get_author(self):
        """Return it."""
        return self.author
'''
        func = _extract_functions(code)[0]
        assert func["signature"].startswith("def ")
        assert func["body"].startswith("    return")

    def test_deeply_nested_method_produces_valid_code(self):
        code = '''class Outer:
    class Inner:
        def deep(self, value):
            """Doc."""
            return value
'''
        func = [f for f in _extract_functions(code) if f["name"] == "deep"][0]
        assert is_ast_valid(_synthesized_code(func))

    def test_multiline_signature_is_not_truncated(self):
        code = '''class Q:
    def query(
        self, spec, texts,
        fieldname,
    ):
        """Build a query."""
        return self._build(spec)
'''
        func = [f for f in _extract_functions(code) if f["name"] == "query"][0]
        assert "fieldname" in func["signature"]
        assert is_ast_valid(_synthesized_code(func))

    def test_multiline_docstring_keeps_its_indentation(self):
        code = '''class Q:
    def query(self, spec):
        """Build a query.

        Continues on a second line.
        """
        return spec
'''
        func = [f for f in _extract_functions(code) if f["name"] == "query"][0]
        assert is_ast_valid(_synthesized_code(func))
        assert "\nContinues" not in _completion_prompt(func)

    def test_docstring_containing_quotes_survives(self):
        code = """class R:
    def show(self):
        '''Return repr, such as \\"x\\" or 'y'.'''
        return repr(self)
"""
        func = [f for f in _extract_functions(code) if f["name"] == "show"][0]
        assert is_ast_valid(_synthesized_code(func))

    def test_docstring_with_backslash_escape_survives(self):
        code = '''class C:
    def decode(self):
        r"""Handle \\N{BULLET} and \\u00e9 sequences."""
        return self
'''
        func = [f for f in _extract_functions(code) if f["name"] == "decode"][0]
        assert is_ast_valid(_synthesized_code(func))

    def test_one_liner_yields_no_body(self):
        code = '''class C:
    def f(self): return 1
'''
        func = [f for f in _extract_functions(code) if f["name"] == "f"][0]
        assert func["body"] == ""

    def test_async_method_produces_valid_code(self):
        code = '''class C:
    async def fetch(self, url):
        """Fetch a URL."""
        return await self._get(url)
'''
        func = [f for f in _extract_functions(code) if f["name"] == "fetch"][0]
        assert is_ast_valid(_synthesized_code(func))

    def test_body_with_column_zero_string_content_is_preserved(self):
        """A dedent-based fix would silently no-op on this."""
        code = '''class C:
    def banner(self):
        """Doc."""
        return """
edge
"""
'''
        func = [f for f in _extract_functions(code) if f["name"] == "banner"][0]
        assert is_ast_valid(_synthesized_code(func))

    def test_full_source_is_valid_standalone(self):
        """`full` feeds explanation examples, so it must parse on its own."""
        code = '''class Source:
    def get_author(self):
        """Return it."""
        return self.author
'''
        func = _extract_functions(code)[0]
        assert is_ast_valid(func["full"])

    def test_synthesizer_emits_parseable_completion_examples(self):
        """
        Prompt and completion must parse *together*.

        Neither half parses alone -- the prompt is a function with no body and
        the completion is a bare indented body -- but joined they are the
        function the model is being taught to produce.
        """
        synth = InstructionSynthesizer()
        code = '''class Repo:
    def find(self, key):
        """Look up a key."""
        return self._items[key]
'''
        completions = [e for e in synth.synthesize(code) if e["task"] == "code_completion"]
        assert completions, "expected a completion example"

        for example in completions:
            blocks = [b.split("```")[0] for b in example["text"].split("```python")[1:]]
            assert len(blocks) == 2, "expected a prompt block and a completion block"
            prompt, completion = (b.strip("\n") for b in blocks)
            assert is_ast_valid(f"{prompt}\n{completion}")


# ============================================================================
# Curation-Time Validity Gate
# ============================================================================

def _chat(prompt_block: str, completion_block: str) -> str:
    """Reproduce the ChatML shape the synthesizer emits for completions."""
    return (
        f"<|im_start|>user\nComplete the following Python function:\n\n"
        f"```python\n{prompt_block}\n```<|im_end|>\n"
        f"<|im_start|>assistant\n```python\n{completion_block}\n```<|im_end|>"
    )


class TestValidateExample:
    def test_accepts_well_formed_completion(self):
        example = {
            "task": "code_completion",
            "text": _chat(
                'def find(self, key):\n    """Look it up."""',
                "    return self._items[key]",
            ),
        }
        assert validate_example(example)

    def test_rejects_the_misindented_shape(self):
        """The bug this gate exists to catch: docstring level with `def`."""
        example = {
            "task": "code_completion",
            "text": _chat(
                '    def find(self, key):\n    """Look it up."""',
                "    return self._items[key]",
            ),
        }
        assert not validate_example(example)

    def test_rejects_truncated_signature(self):
        example = {
            "task": "code_completion",
            "text": _chat('def find(\n    """Look it up."""', "    return 1"),
        }
        assert not validate_example(example)

    def test_rejects_wrong_block_count(self):
        example = {"task": "code_completion", "text": "```python\ndef f(): pass\n```"}
        assert not validate_example(example)

    def test_accepts_valid_explanation(self):
        text = (
            "<|im_start|>user\nExplain what the following Python code does:\n\n"
            '```python\ndef f(a):\n    """D."""\n    return a\n```<|im_end|>\n'
            "<|im_start|>assistant\nIt returns its argument.<|im_end|>"
        )
        assert validate_example({"task": "code_explanation", "text": text})

    def test_rejects_explanation_with_broken_code(self):
        text = (
            "<|im_start|>user\nExplain:\n\n```python\ndef f(:\n    pass\n```<|im_end|>\n"
            "<|im_start|>assistant\nProse.<|im_end|>"
        )
        assert not validate_example({"task": "code_explanation", "text": text})

    def test_accepts_valid_raw(self):
        assert validate_example({"task": "raw", "text": "x = 1\nprint(x)\n"})

    def test_rejects_invalid_raw(self):
        assert not validate_example({"task": "raw", "text": "def broken(:\n  pass"})

    def test_synthesizer_output_passes_its_own_gate(self):
        code = '''class Repo:
    def find(self, key):
        """Look up a key."""
        return self._items[key]
'''
        for example in InstructionSynthesizer().synthesize(code):
            assert validate_example(example), example["task"]


class TestSynthesisValidityGate:
    def _pipeline(self, **synthesis):
        return DataCurationPipeline({"synthesis": {"enabled": True, **synthesis}})

    def test_enabled_by_default(self):
        pipeline = self._pipeline()
        assert pipeline.validate_synthesis is True
        assert pipeline.max_invalid_ratio == pytest.approx(0.02)

    def test_can_be_disabled(self):
        assert self._pipeline(validate=False).validate_synthesis is False

    def test_clean_run_records_zero_rejections(self):
        pipeline = self._pipeline()
        pipeline._report_synthesis_validity(kept=100, rejected=Counter())
        assert pipeline.synthesis_stats["rejected"] == 0
        assert pipeline.synthesis_stats["invalid_ratio"] == 0

    def test_tolerates_rate_under_the_limit(self):
        pipeline = self._pipeline(max_invalid_ratio=0.05)
        pipeline._report_synthesis_validity(kept=99, rejected=Counter({"code_completion": 1}))
        assert pipeline.synthesis_stats["invalid_ratio"] == pytest.approx(0.01)

    def test_aborts_when_rate_exceeds_the_limit(self):
        """A broken template should fail the run, not quietly shrink the dataset."""
        pipeline = self._pipeline(max_invalid_ratio=0.02)
        with pytest.raises(RuntimeError, match="not valid Python"):
            pipeline._report_synthesis_validity(
                kept=50, rejected=Counter({"code_completion": 50})
            )

    def test_abort_message_names_the_offending_task(self):
        pipeline = self._pipeline(max_invalid_ratio=0.0)
        with pytest.raises(RuntimeError, match="code_completion=3"):
            pipeline._report_synthesis_validity(kept=1, rejected=Counter({"code_completion": 3}))

    def test_threshold_is_configurable(self):
        pipeline = self._pipeline(max_invalid_ratio=0.9)
        pipeline._report_synthesis_validity(kept=20, rejected=Counter({"raw": 5}))
        assert pipeline.synthesis_stats["rejected_by_task"] == {"raw": 5}

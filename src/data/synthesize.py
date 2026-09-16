"""
Instruction synthesis: converts raw code into instruction-following format.

Generates multiple training examples per code sample using templates
for code completion, explanation, and bug-fixing tasks.
"""

from __future__ import annotations

import ast
import random
import re
import textwrap
from typing import Any


# Default instruction templates for code tasks
DEFAULT_TEMPLATES = {
    "code_completion": {
        "instruction": "Complete the following Python function:",
        "input_format": "signature_and_docstring",
    },
    "code_explanation": {
        "instruction": "Explain what the following Python code does:",
        "input_format": "full_code",
    },
}


_CODE_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _parses(code: str) -> bool:
    """
    Whether code is syntactically valid Python.

    Mirrors `curate.is_ast_valid`, kept local because curate imports this
    module and the reverse direction would be circular.
    """
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError, RecursionError):
        return False


def validate_example(example: dict[str, str]) -> bool:
    """
    Whether a synthesized example contains valid Python.

    What counts as valid depends on the task, because the halves of an example
    are not independently parseable:

    - code_completion: the prompt is a function with no body and the completion
      is a bare indented body. Neither parses alone; joined they must.
    - code_explanation: the input is a whole function and must parse. The
      output is prose and is not checked.
    - raw: the text is source code and must parse on its own.

    An example whose block structure is wrong is invalid regardless of
    contents -- a completion carrying one block or three did not come out of
    the template the trainer expects.

    Args:
        example: A synthesized example with "text" and "task" keys.

    Returns:
        True when the example's code is syntactically valid.
    """
    task = example.get("task", "raw")
    text = example.get("text", "")

    if task == "raw":
        return _parses(text)

    blocks = [block.strip("\n") for block in _CODE_BLOCK.findall(text)]

    if task == "code_completion":
        if len(blocks) != 2:
            return False
        prompt, completion = blocks
        return _parses(f"{prompt}\n{completion}")

    if task == "code_explanation":
        return len(blocks) == 1 and _parses(blocks[0])

    return _parses(text)


def _strip_indent(text: str, amount: int) -> str:
    """
    Remove exactly `amount` leading spaces from every line.

    Unlike `textwrap.dedent`, which removes whatever prefix happens to be
    common to all lines, this removes a known amount and so preserves the
    relative structure of a fragment. That distinction matters for a function
    body containing a multi-line string whose continuation lines sit at
    column 0: dedent would find no common prefix and silently do nothing.

    Lines shorter than the removal window, or indented less than expected,
    are left-stripped instead of being corrupted.
    """
    if amount <= 0:
        return text

    out = []
    for line in text.split("\n"):
        if not line.strip():
            out.append("")
            continue
        window = line[:amount]
        out.append(line[amount:] if not window.strip() else line.lstrip())
    return "\n".join(out)


def _is_docstring_stmt(stmt: ast.stmt) -> bool:
    """Whether a statement is a docstring expression."""
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _extract_functions(code: str) -> list[dict[str, str]]:
    """
    Extract function signatures, docstrings, and bodies from code.

    Everything is re-based to column 0 using the `def` line's own indentation,
    so a method lifted out of a class reads as a standalone function rather
    than as a fragment that no longer parses. The body keeps its indentation
    relative to the signature.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    source_lines = code.splitlines()
    functions = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.body:
            continue

        try:
            func_lines = source_lines[node.lineno - 1:node.end_lineno]
            func_source = "\n".join(func_lines)
        except (AttributeError, IndexError):
            continue
        if not func_lines:
            continue

        # The `def` line's indentation is what everything is re-based against.
        def_line = func_lines[0]
        indent = len(def_line) - len(def_line.lstrip())

        first_stmt = node.body[0]
        # A signature can span several lines; it runs from `def` up to the
        # first body statement. Taking only line one truncates it mid-argument.
        signature_lines = source_lines[node.lineno - 1:first_stmt.lineno - 1]
        signature = "\n".join(signature_lines) if signature_lines else def_line

        docstring = ast.get_docstring(node) or ""

        # Keep the docstring's original literal alongside the decoded text.
        # Re-wrapping a decoded docstring in triple quotes re-interprets
        # whatever it contains -- an embedded quote ends the literal early, and
        # a `\N` or `\u` sequence becomes an escape -- so the prompt is built
        # from the source form while explanations use the decoded prose.
        docstring_source = ""
        if _is_docstring_stmt(first_stmt):
            doc_lines = source_lines[first_stmt.lineno - 1:first_stmt.end_lineno]
            docstring_source = _strip_indent("\n".join(doc_lines), indent)
            body_start_idx = first_stmt.end_lineno
        else:
            body_start_idx = first_stmt.lineno - 1

        # A one-liner such as `def f(): return 1` has no separable body; the
        # slice below would hand back the signature again.
        one_liner = first_stmt.lineno == node.lineno
        body = "" if one_liner else "\n".join(source_lines[body_start_idx:node.end_lineno])

        functions.append({
            "signature": _strip_indent(signature, indent),
            "docstring": docstring,
            "docstring_source": docstring_source,
            "body": _strip_indent(body, indent),
            "full": _strip_indent(func_source, indent),
            "name": node.name,
        })

    return functions


class InstructionSynthesizer:
    """
    Synthesizes instruction-following training examples from raw code.

    For each code sample, generates multiple training examples:
    1. Code completion: given signature + docstring, complete the function
    2. Code explanation: given full code, explain what it does

    Output format (chat-style for SFT):
    {
        "text": "<|im_start|>user\n{instruction}\n{input}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
    }
    """

    def __init__(self, templates: dict[str, Any] | None = None):
        self.templates = templates or DEFAULT_TEMPLATES

    def _format_chat(self, instruction: str, input_text: str, output_text: str) -> str:
        """Format as ChatML for Qwen-style models."""
        return (
            f"<|im_start|>user\n{instruction}\n\n```python\n{input_text}\n```<|im_end|>\n"
            f"<|im_start|>assistant\n{output_text}<|im_end|>"
        )

    def _synthesize_completion(self, func: dict[str, str]) -> dict[str, str] | None:
        """Generate a code completion example from a function."""
        if not func["docstring"] or not func["body"].strip():
            return None

        # Prefer the original literal, which is already correctly indented
        # relative to the signature and needs no re-escaping. The rebuilt form
        # is a fallback for callers that only supply decoded text; indent the
        # whole block so continuation lines do not land at column 0.
        docstring_block = func.get("docstring_source") or textwrap.indent(
            f'"""{func["docstring"]}"""', "    "
        )
        prompt = f"{func['signature']}\n{docstring_block}"
        completion = func["body"]

        text = self._format_chat(
            instruction="Complete the following Python function:",
            input_text=prompt,
            output_text=f"```python\n{completion}\n```",
        )
        return {"text": text, "task": "code_completion"}

    def _synthesize_explanation(self, func: dict[str, str]) -> dict[str, str] | None:
        """Generate a code explanation example from a function."""
        if len(func["full"].splitlines()) < 3:
            return None

        docstring = func["docstring"]
        if not docstring:
            # Without a docstring we can't create a reliable explanation
            return None

        text = self._format_chat(
            instruction="Explain what the following Python code does:",
            input_text=func["full"],
            output_text=docstring,
        )
        return {"text": text, "task": "code_explanation"}

    def _synthesize_raw(self, code: str) -> dict[str, str]:
        """Fallback: wrap raw code as a plain text training example."""
        return {"text": code, "task": "raw"}

    def synthesize(self, code: str) -> list[dict[str, str]]:
        """
        Generate instruction-following examples from a code sample.

        Extracts functions and creates completion + explanation pairs.
        Falls back to raw format if no functions are extractable.
        """
        examples = []
        functions = _extract_functions(code)

        if not functions:
            # No extractable functions — use raw format
            examples.append(self._synthesize_raw(code))
            return examples

        for func in functions:
            # Code completion
            completion = self._synthesize_completion(func)
            if completion:
                examples.append(completion)

            # Code explanation
            explanation = self._synthesize_explanation(func)
            if explanation:
                examples.append(explanation)

        # Always include at least the raw code if no structured examples
        if not examples:
            examples.append(self._synthesize_raw(code))

        return examples

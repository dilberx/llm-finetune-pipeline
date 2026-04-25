"""
Instruction synthesis: converts raw code into instruction-following format.

Generates multiple training examples per code sample using templates
for code completion, explanation, and bug-fixing tasks.
"""

from __future__ import annotations

import ast
import random
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


def _extract_functions(code: str) -> list[dict[str, str]]:
    """Extract function signatures, docstrings, and bodies from code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    functions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Get source lines for this function
            try:
                func_lines = code.splitlines()[node.lineno - 1:node.end_lineno]
                func_source = "\n".join(func_lines)
            except (AttributeError, IndexError):
                continue

            # Extract signature (first line)
            signature = func_lines[0] if func_lines else ""

            # Extract docstring
            docstring = ast.get_docstring(node) or ""

            # Get body without docstring
            body_start = node.lineno
            if docstring and node.body:
                first_stmt = node.body[0]
                if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant):
                    body_start = first_stmt.end_lineno

            body_lines = code.splitlines()[body_start:node.end_lineno]
            body = "\n".join(body_lines)

            functions.append({
                "signature": signature,
                "docstring": docstring,
                "body": body,
                "full": func_source,
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

        prompt = f"{func['signature']}\n    \"\"\"{func['docstring']}\"\"\""
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

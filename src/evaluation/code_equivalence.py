"""
Semantic equivalence for generated code.

String comparison asks whether the model typed the same characters. That is
almost never the question. Two snippets that differ only in variable naming,
whitespace, comments, or redundant parentheses are the same program, and a
metric that scores them as a miss is measuring transcription rather than
competence.

Compiler and program-verification work has long treated this as a question
about the parse tree rather than the text. This module applies the cheap end
of that ladder:

    exact          characters match
    normalized     match after whitespace cleanup
    ast_match      same parse tree, so comments, spacing and redundant
                   parentheses are already factored out
    alpha_match    same parse tree up to consistent renaming of locals, so
                   `total`/`acc` and `i`/`idx` stop counting as errors

Two practical obstacles come first. Completions arrive wrapped in markdown
fences, because that is how the instruction data was synthesised. And a
completion is usually a *fragment* -- an indented method body, not a module --
so it does not parse on its own. Both are handled here rather than pushed onto
callers, since getting either wrong silently returns "not equivalent" and looks
like a model failure.
"""

from __future__ import annotations

import ast
import re
import textwrap
from typing import Literal

ParseStrategy = Literal["direct", "dedented", "failed"]
_FENCE_BLOCK = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)(?:```|\Z)", re.DOTALL)


def _trim_blank_lines(text: str) -> str:
    """
    Drop leading and trailing blank lines, leaving indentation untouched.

    Deliberately not `str.strip()`: that removes the leading whitespace of the
    first line only, which shifts it relative to the lines below it and can
    silently repair genuinely broken indentation. Removing a *common* prefix is
    `textwrap.dedent`'s job, and `parse_snippet` applies it where it belongs.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(line.rstrip() for line in lines)


def strip_code_fences(text: str) -> str:
    """
    Pull code out of markdown fences, if any are present.

    Every completion in the curated eval set is fenced, so comparing raw model
    output against a fenced reference requires the model to reproduce the fence
    markers exactly. Stripping both sides first is what makes the comparison
    about the code.

    Concatenates all fenced blocks when a response contains several. Text with
    no fences is returned with surrounding blank lines removed. Indentation is
    preserved in both cases, so a fragment's structure survives intact.

    Args:
        text: Raw model output or reference completion.

    Returns:
        The code content, without fence markers or language tags.
    """
    blocks = _FENCE_BLOCK.findall(text)
    if not blocks:
        return _trim_blank_lines(text)
    return _trim_blank_lines("\n".join(_trim_blank_lines(body) for _lang, body in blocks))


def parse_snippet(code: str) -> tuple[list[ast.stmt] | None, ParseStrategy]:
    """
    Parse a code fragment into a list of statements.

    A completion is rarely a valid module on its own, so two readings are
    tried: the fragment as written, then dedented, which handles a method body
    that arrives carrying its enclosing indentation.

    Two is enough. A bare `return`, `yield`, `await` or `break` looks like it
    would need wrapping in a function to parse, but those are *compile*-time
    errors, not parse-time ones -- `ast.parse` is purely syntactic and never
    checks scope. Wrapping changes only scope, so it can never rescue a
    fragment that dedenting could not.

    Args:
        code: A code fragment, fences already stripped.

    Returns:
        (statements, strategy). statements is None when nothing parsed, and
        strategy names the reading that succeeded, for diagnostics.
    """
    if not code.strip():
        return None, "failed"

    attempts: list[tuple[str, ParseStrategy]] = [
        (code, "direct"),
        (textwrap.dedent(code), "dedented"),
    ]

    for candidate, strategy in attempts:
        try:
            return ast.parse(candidate).body, strategy
        except (SyntaxError, ValueError, RecursionError):
            continue

    return None, "failed"


def is_parseable(code: str, strip_fences: bool = True) -> bool:
    """
    Whether a fragment parses under any of the readings above.

    Tracked per example, this separates a model that writes the wrong code from
    one that writes code the interpreter would reject -- two failures that both
    score zero on exact match and need different fixes.

    Args:
        code: Model output or reference text.
        strip_fences: Remove markdown fences before parsing.

    Returns:
        True if the fragment parses.
    """
    text = strip_code_fences(code) if strip_fences else code
    statements, _ = parse_snippet(text)
    return statements is not None


class _LocalNameCollector(ast.NodeVisitor):
    """Collect names bound inside a snippet, in deterministic source order."""

    def __init__(self) -> None:
        self.bound: dict[str, None] = {}

    def _bind(self, name: str) -> None:
        # dict preserves insertion order, which keeps canonical names stable.
        self.bound.setdefault(name, None)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._bind(node.id)
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        self._bind(node.arg)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self._bind(node.name)
        self.generic_visit(node)


class _AlphaRenamer(ast.NodeTransformer):
    """Rewrite bound names to canonical placeholders."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.mapping:
            node.id = self.mapping[node.id]
        return self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        if node.arg in self.mapping:
            node.arg = self.mapping[node.arg]
        return self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.AST:
        if node.name and node.name in self.mapping:
            node.name = self.mapping[node.name]
        return self.generic_visit(node)


def _strip_docstrings(statements: list[ast.stmt]) -> list[ast.stmt]:
    """Drop docstring expressions from every scope in the fragment."""
    for node in ast.walk(ast.Module(body=statements, type_ignores=[])):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return statements


def canonical_form(
    code: str,
    alpha_rename: bool = False,
    ignore_docstrings: bool = False,
    strip_fences: bool = True,
) -> str | None:
    """
    Reduce a fragment to a canonical string that ignores surface detail.

    Comments, blank lines, spacing and redundant parentheses never reach the
    parse tree, so they drop out for free. Renaming and docstrings are opt-in,
    since whether they count as a difference depends on what is being measured.

    Only names *bound inside* the fragment are renamed. A name that is merely
    read refers to something outside the snippet -- a builtin, an import, an
    attribute of an enclosing object -- and renaming it would equate programs
    that call different functions.

    Args:
        code: Model output or reference text.
        alpha_rename: Canonicalise locally bound names.
        ignore_docstrings: Drop docstrings before comparing.
        strip_fences: Remove markdown fences first.

    Returns:
        A canonical string, or None when the fragment does not parse.
    """
    text = strip_code_fences(code) if strip_fences else code
    statements, _ = parse_snippet(text)
    if statements is None:
        return None

    if ignore_docstrings:
        statements = _strip_docstrings(statements)

    if alpha_rename:
        module = ast.Module(body=statements, type_ignores=[])
        collector = _LocalNameCollector()
        collector.visit(module)
        mapping = {name: f"_v{i}" for i, name in enumerate(collector.bound)}
        module = _AlphaRenamer(mapping).visit(module)
        statements = module.body

    return ast.dump(
        ast.Module(body=statements, type_ignores=[]),
        annotate_fields=True,
        include_attributes=False,
    )


def ast_equivalent(
    generated: str,
    reference: str,
    ignore_docstrings: bool = False,
    strip_fences: bool = True,
) -> bool:
    """
    Whether two fragments have the same parse tree.

    Unparseable input is never equivalent, including when both sides fail to
    parse: two different syntax errors are not a match.

    Args:
        generated: Model output.
        reference: Expected completion.
        ignore_docstrings: Drop docstrings before comparing.
        strip_fences: Remove markdown fences first.

    Returns:
        True when the trees are identical.
    """
    left = canonical_form(
        generated, alpha_rename=False, ignore_docstrings=ignore_docstrings,
        strip_fences=strip_fences,
    )
    if left is None:
        return False
    right = canonical_form(
        reference, alpha_rename=False, ignore_docstrings=ignore_docstrings,
        strip_fences=strip_fences,
    )
    return right is not None and left == right


def alpha_equivalent(
    generated: str,
    reference: str,
    ignore_docstrings: bool = False,
    strip_fences: bool = True,
) -> bool:
    """
    Whether two fragments match up to consistent renaming of local names.

    Strictly weaker than `ast_equivalent`: everything structurally equal is also
    alpha-equal. Use it to separate "wrote different code" from "picked
    different variable names", which token overlap cannot distinguish.

    Args:
        generated: Model output.
        reference: Expected completion.
        ignore_docstrings: Drop docstrings before comparing.
        strip_fences: Remove markdown fences first.

    Returns:
        True when the trees match after canonical renaming.
    """
    left = canonical_form(
        generated, alpha_rename=True, ignore_docstrings=ignore_docstrings,
        strip_fences=strip_fences,
    )
    if left is None:
        return False
    right = canonical_form(
        reference, alpha_rename=True, ignore_docstrings=ignore_docstrings,
        strip_fences=strip_fences,
    )
    return right is not None and left == right

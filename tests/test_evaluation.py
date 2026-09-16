"""
Tests for the evaluation metrics module.

Tests statistical comparison, bootstrap CIs, and pass@k estimation
without requiring GPU or model loading.
"""

import ast
import json

import numpy as np
import pytest

from src.evaluation.benchmarks import CodeCompletionBenchmark, _estimate_pass_at_k
from src.evaluation.code_equivalence import (
    alpha_equivalent,
    ast_equivalent,
    is_parseable,
    parse_snippet,
    strip_code_fences,
)
from src.evaluation.evaluate import ModelEvaluator
from src.evaluation.irt import (
    MIN_MODELS_FOR_IRT,
    find_degenerate_items,
    fit_2pl,
    responses_from_raw_results,
)
from src.evaluation.metrics import (
    align_raw_results,
    bootstrap_confidence_interval,
    compute_comparison_stats,
    paired_difference,
)
from src.evaluation.ppi import optimal_lambda, ppi_mean, ppi_mean_difference
from src.evaluation.proxy import (
    PoolExample,
    align_gold_with_proxy,
    align_proxy_scores,
    build_scoring_inputs,
    likelihood_from_nll,
    load_pool,
    split_chatml,
)
from src.evaluation.sweep import (
    CheckpointRef,
    build_model_config,
    calibrate,
    discover_checkpoints,
)


class TestPassAtK:
    def test_all_correct(self):
        """If all samples are correct, pass@k should be 1.0."""
        assert _estimate_pass_at_k(n=10, c=10, k=1) == 1.0

    def test_none_correct(self):
        """If no samples are correct, pass@k should be 0.0."""
        assert _estimate_pass_at_k(n=10, c=0, k=1) == 0.0

    def test_partial(self):
        """pass@1 with 5/10 correct should be 0.5."""
        result = _estimate_pass_at_k(n=10, c=5, k=1)
        assert abs(result - 0.5) < 0.01

    def test_k_greater_than_failures(self):
        """If k > n-c, pass@k should be 1.0."""
        assert _estimate_pass_at_k(n=10, c=8, k=5) == 1.0

    def test_monotonic_in_k(self):
        """pass@k should increase with k."""
        p1 = _estimate_pass_at_k(n=20, c=5, k=1)
        p5 = _estimate_pass_at_k(n=20, c=5, k=5)
        p10 = _estimate_pass_at_k(n=20, c=5, k=10)
        assert p1 <= p5 <= p10


class TestComparisonStats:
    def test_improvement_detected(self):
        base = {"benchmarks": {"humaneval": {"metrics": {"pass@1": 0.3}}}}
        finetuned = {"benchmarks": {"humaneval": {"metrics": {"pass@1": 0.5}}}}
        result = compute_comparison_stats(base, finetuned)

        assert "humaneval" in result
        assert result["humaneval"]["pass@1"]["improved"] is True
        assert result["humaneval"]["pass@1"]["absolute_diff"] == pytest.approx(0.2)

    def test_regression_detected(self):
        base = {"benchmarks": {"humaneval": {"metrics": {"pass@1": 0.5}}}}
        finetuned = {"benchmarks": {"humaneval": {"metrics": {"pass@1": 0.3}}}}
        result = compute_comparison_stats(base, finetuned)

        assert result["humaneval"]["pass@1"]["improved"] is False

    def test_perplexity_lower_is_better(self):
        """For perplexity, a decrease should count as improvement."""
        base = {"benchmarks": {"perplexity": {"metrics": {"perplexity": 25.0}}}}
        finetuned = {"benchmarks": {"perplexity": {"metrics": {"perplexity": 20.0}}}}
        result = compute_comparison_stats(base, finetuned)

        assert result["perplexity"]["perplexity"]["improved"] is True

    def test_summary_generation(self):
        base = {"benchmarks": {
            "humaneval": {"metrics": {"pass@1": 0.3}},
            "perplexity": {"metrics": {"perplexity": 25.0}},
        }}
        finetuned = {"benchmarks": {
            "humaneval": {"metrics": {"pass@1": 0.5}},
            "perplexity": {"metrics": {"perplexity": 20.0}},
        }}
        result = compute_comparison_stats(base, finetuned)

        assert result["summary"]["total_metrics"] == 2
        assert result["summary"]["improvements"] == 2
        assert result["summary"]["overall_improved"] is True


class TestBootstrapCI:
    def test_contains_mean(self):
        samples = [1.0, 2.0, 3.0, 4.0, 5.0]
        lower, upper = bootstrap_confidence_interval(samples, confidence_level=0.95)
        assert lower <= np.mean(samples) <= upper

    def test_narrow_for_constant(self):
        samples = [5.0] * 100
        lower, upper = bootstrap_confidence_interval(samples)
        assert abs(upper - lower) < 0.01

    def test_wider_for_high_variance(self):
        low_var = [5.0, 5.1, 4.9, 5.0, 5.1]
        high_var = [1.0, 10.0, 2.0, 9.0, 5.0]
        _, upper_low = bootstrap_confidence_interval(low_var)
        lower_low, _ = bootstrap_confidence_interval(low_var)
        _, upper_high = bootstrap_confidence_interval(high_var)
        lower_high, _ = bootstrap_confidence_interval(high_var)

        width_low = upper_low - lower_low
        width_high = upper_high - lower_high
        assert width_high > width_low


class TestCodeCompletionMetrics:
    def test_token_f1_counts_repeated_tokens(self):
        benchmark = CodeCompletionBenchmark()
        score = benchmark._compute_token_overlap("return value value", "return value")
        assert score == pytest.approx(0.8)

    def test_token_f1_keeps_code_punctuation(self):
        benchmark = CodeCompletionBenchmark()
        score = benchmark._compute_token_overlap("foo(x + y)", "foo(x - y)")
        assert score < 1.0

    def test_normalized_exact_match_ignores_line_endings_and_trailing_space(self):
        benchmark = CodeCompletionBenchmark()
        generated = "def add(a, b):\r\n    return a + b   \n"
        reference = "def add(a, b):\n    return a + b"
        assert benchmark._normalized_exact_match(generated, reference)

    def test_edit_similarity_scores_exact_match_as_one(self):
        benchmark = CodeCompletionBenchmark()
        assert benchmark._edit_similarity("return x", "return x") == pytest.approx(1.0)


class TestEvaluationSummaryFormatting:
    def test_delta_handles_zero_values(self):
        assert ModelEvaluator._format_delta("exact_match", 0.0, 0.0) == (
            "[yellow]+0.0000[/yellow]"
        )

    def test_delta_marks_perplexity_increase_as_regression(self):
        assert ModelEvaluator._format_delta("perplexity", 10.0, 12.0) == (
            "[red]+2.0000[/red]"
        )

    def test_delta_marks_perplexity_decrease_as_improvement(self):
        assert ModelEvaluator._format_delta("perplexity", 12.0, 10.0) == (
            "[green]-2.0000[/green]"
        )


def _raw(prompt, **fields):
    """Build a per-example benchmark result row."""
    return {"prompt": prompt, "reference": "ref", "generated": "gen", **fields}


class TestPairedDifference:
    def test_all_zero_differences_flagged_degenerate(self):
        """The 16-sample exact_match case: identical scores tell you nothing."""
        scores = [0.0] * 16
        result = paired_difference(scores, scores)

        assert result["degenerate"] is True
        assert result["significant"] is False
        assert result["p_value"] == 1.0
        assert "cannot resolve a difference" in result["note"]
        # 3/16 = 18.8%: a rare effect is entirely consistent with 0/16.
        assert "18.8%" in result["note"]

    def test_clear_improvement_is_significant(self):
        base = [0.1, 0.12, 0.09, 0.11, 0.10, 0.13, 0.08, 0.12]
        finetuned = [0.4, 0.42, 0.39, 0.41, 0.40, 0.43, 0.38, 0.42]
        result = paired_difference(base, finetuned)

        assert result["significant"] is True
        assert result["mean_difference"] > 0
        assert result["ci_low"] > 0
        assert result["degenerate"] is False

    def test_noise_is_not_significant(self):
        base = [0.5, 0.3, 0.7, 0.4, 0.6, 0.35, 0.65, 0.45]
        finetuned = [0.45, 0.35, 0.65, 0.45, 0.55, 0.4, 0.6, 0.5]
        result = paired_difference(base, finetuned)

        assert result["significant"] is False
        assert result["ci_low"] < 0 < result["ci_high"]

    def test_interval_brackets_the_mean_difference(self):
        base = [0.2, 0.3, 0.1, 0.4, 0.25]
        finetuned = [0.5, 0.45, 0.6, 0.55, 0.5]
        result = paired_difference(base, finetuned)

        assert result["ci_low"] <= result["mean_difference"] <= result["ci_high"]

    def test_effect_size_sign_follows_direction(self):
        better = paired_difference([0.1, 0.2, 0.15, 0.1], [0.5, 0.6, 0.55, 0.5])
        worse = paired_difference([0.5, 0.6, 0.55, 0.5], [0.1, 0.2, 0.15, 0.1])

        assert better["effect_size_dz"] > 0
        assert worse["effect_size_dz"] < 0

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError, match="must align"):
            paired_difference([0.1, 0.2], [0.1, 0.2, 0.3])

    def test_single_pair_rejected(self):
        with pytest.raises(ValueError, match="at least 2 pairs"):
            paired_difference([0.1], [0.2])


class TestAlignRawResults:
    def test_aligns_on_prompt_when_order_differs(self):
        base = [_raw("a", token_f1=0.1), _raw("b", token_f1=0.2)]
        finetuned = [_raw("b", token_f1=0.5), _raw("a", token_f1=0.4)]
        base_scores, ft_scores = align_raw_results(base, finetuned, "token_f1")

        assert base_scores == [0.1, 0.2]
        assert ft_scores == [0.4, 0.5]

    def test_keeps_only_shared_examples(self):
        base = [_raw("a", token_f1=0.1), _raw("b", token_f1=0.2)]
        finetuned = [_raw("a", token_f1=0.4), _raw("c", token_f1=0.9)]
        base_scores, ft_scores = align_raw_results(base, finetuned, "token_f1")

        assert len(base_scores) == len(ft_scores) == 1

    def test_falls_back_to_position_without_key(self):
        base = [{"token_f1": 0.1}, {"token_f1": 0.2}]
        finetuned = [{"token_f1": 0.4}, {"token_f1": 0.5}]
        base_scores, ft_scores = align_raw_results(base, finetuned, "token_f1")

        assert base_scores == [0.1, 0.2]
        assert ft_scores == [0.4, 0.5]

    def test_disjoint_examples_rejected(self):
        base = [_raw("a", token_f1=0.1)]
        finetuned = [_raw("z", token_f1=0.4)]
        with pytest.raises(ValueError, match="share no examples"):
            align_raw_results(base, finetuned, "token_f1")


class TestComparisonStatsPairing:
    def _results(self, scores, field="token_f1"):
        raw = [_raw(f"p{i}", **{field: s}) for i, s in enumerate(scores)]
        return {
            "benchmarks": {
                "code_completion": {
                    "metrics": {field: float(np.mean(scores))},
                    "raw_results": raw,
                }
            }
        }

    def test_paired_block_added_when_raw_results_present(self):
        base = self._results([0.1, 0.12, 0.09, 0.11])
        finetuned = self._results([0.4, 0.42, 0.39, 0.41])
        result = compute_comparison_stats(base, finetuned)

        paired = result["code_completion"]["token_f1"]["paired"]
        assert paired["n_pairs"] == 4
        assert paired["ci_low"] > 0

    def test_no_paired_block_for_aggregate_only_benchmark(self):
        base = {"benchmarks": {"perplexity": {"metrics": {"perplexity": 10.3}}}}
        finetuned = {"benchmarks": {"perplexity": {"metrics": {"perplexity": 12.5}}}}
        result = compute_comparison_stats(base, finetuned)

        assert "paired" not in result["perplexity"]["perplexity"]
        assert result["perplexity"]["perplexity"]["improved"] is False

    def test_point_estimates_unchanged_by_pairing(self):
        base = self._results([0.1, 0.2])
        finetuned = self._results([0.4, 0.5])
        entry = compute_comparison_stats(base, finetuned)["code_completion"]["token_f1"]

        assert entry["base"] == pytest.approx(0.15)
        assert entry["finetuned"] == pytest.approx(0.45)
        assert entry["improved"] is True


class TestPPI:
    def test_useless_proxy_falls_back_to_classical(self):
        """A constant proxy carries no signal, so lambda must go to zero."""
        rng = np.random.default_rng(0)
        gold = rng.normal(0.5, 0.1, 16)
        gold_proxy = np.full(16, 0.3)
        pool_proxy = np.full(500, 0.3)
        est = ppi_mean(gold, gold_proxy, pool_proxy)

        assert est.lambda_weight == 0.0
        assert est.estimate == pytest.approx(est.classical_estimate)

    def test_informative_proxy_tightens_the_interval(self):
        rng = np.random.default_rng(1)
        pool_truth = rng.normal(0.5, 0.15, 800)
        pool_proxy = pool_truth + rng.normal(0, 0.02, 800)
        gold_truth = pool_truth[:16]
        gold_proxy = pool_proxy[:16]
        est = ppi_mean(gold_truth, gold_proxy, pool_proxy)

        assert est.lambda_weight > 0.5
        assert est.ci_width < est.classical_ci_width
        assert est.effective_gold_n > est.n_gold

    def test_biased_proxy_still_covers_the_truth(self):
        """Validity does not require an unbiased proxy, only a measured one."""
        rng = np.random.default_rng(2)
        pool_truth = rng.normal(0.5, 0.15, 800)
        # Proxy is systematically inflated by 0.25 everywhere.
        pool_proxy = pool_truth + 0.25 + rng.normal(0, 0.02, 800)
        est = ppi_mean(pool_truth[:16], pool_proxy[:16], pool_proxy)

        assert est.ci_low <= float(np.mean(pool_truth)) <= est.ci_high

    def test_empty_pool_degenerates_to_classical(self):
        gold = [0.1, 0.2, 0.3, 0.4]
        est = ppi_mean(gold, [0.1, 0.2, 0.3, 0.4], [])

        assert est.n_unlabeled == 0
        assert est.lambda_weight == 0.0
        assert est.estimate == pytest.approx(float(np.mean(gold)))
        assert est.effective_gold_n == 4.0

    def test_too_few_gold_examples_rejected(self):
        with pytest.raises(ValueError, match="at least 2 gold"):
            ppi_mean([0.5], [0.4], [0.4, 0.5])

    def test_mismatched_gold_arrays_rejected(self):
        with pytest.raises(ValueError, match="must align"):
            ppi_mean([0.1, 0.2], [0.1], [0.3])

    def test_lambda_stays_in_unit_interval(self):
        rng = np.random.default_rng(3)
        for seed_shift in range(5):
            y = rng.normal(0, 1, 20)
            f = y * (seed_shift + 1) + rng.normal(0, 0.1, 20)
            assert 0.0 <= optimal_lambda(y, f, 400) <= 1.0

    def test_paired_difference_interval_detects_real_shift(self):
        rng = np.random.default_rng(4)
        base_pool = rng.normal(0.10, 0.05, 600)
        ft_pool = base_pool + 0.13 + rng.normal(0, 0.01, 600)
        est = ppi_mean_difference(
            gold_base=base_pool[:16],
            gold_finetuned=ft_pool[:16],
            proxy_base=base_pool[:16],
            proxy_finetuned=ft_pool[:16],
            pool_proxy_base=base_pool,
            pool_proxy_finetuned=ft_pool,
        )

        assert est.excludes_zero is True
        assert est.estimate == pytest.approx(0.13, abs=0.02)


class TestIRT:
    def _synthetic(self, n_models=30, n_items=20, seed=7):
        rng = np.random.default_rng(seed)
        theta = rng.normal(0, 1, n_models)
        difficulty = rng.normal(0, 1.2, n_items)
        discrimination = rng.uniform(0.5, 2.0, n_items)
        logits = discrimination[None, :] * (theta[:, None] - difficulty[None, :])
        probs = 1 / (1 + np.exp(-logits))
        responses = (rng.random(probs.shape) < probs).astype(float)
        return responses, difficulty, discrimination

    def test_finds_degenerate_items(self):
        responses = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
        degenerate = find_degenerate_items(responses, ["easy", "mixed", "other"])

        assert degenerate == ["easy"]

    def test_all_degenerate_set_rejected(self):
        """16 items where every model scores zero is a measurement failure."""
        responses = np.zeros((6, 16))
        with pytest.raises(ValueError, match="all 16 items are degenerate"):
            fit_2pl(responses)

    def test_too_few_models_rejected(self):
        responses = np.array([[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="at least 5 models"):
            fit_2pl(responses)

    def test_min_models_constant_is_enforced(self):
        rng = np.random.default_rng(11)
        responses = (rng.random((MIN_MODELS_FOR_IRT, 8)) < 0.5).astype(float)
        fit = fit_2pl(responses)

        assert len(fit.model_ids) == MIN_MODELS_FOR_IRT

    def test_recovers_difficulty_ordering(self):
        responses, true_difficulty, _ = self._synthetic()
        fit = fit_2pl(responses)

        kept = [int(i.split("_")[1]) for i in fit.item_ids]
        corr = np.corrcoef(fit.difficulty, true_difficulty[kept])[0, 1]
        assert corr > 0.8

    def test_harder_items_have_lower_pass_rates(self):
        responses, _, _ = self._synthetic()
        fit = fit_2pl(responses)

        kept = [int(i.split("_")[1]) for i in fit.item_ids]
        pass_rates = responses[:, kept].mean(axis=0)
        corr = np.corrcoef(fit.difficulty, pass_rates)[0, 1]
        assert corr < -0.8

    def test_degenerate_items_excluded_from_fit(self):
        responses, _, _ = self._synthetic()
        responses[:, 0] = 1.0  # everyone passes item_0
        fit = fit_2pl(responses)

        assert "item_0" in fit.degenerate_items
        assert "item_0" not in fit.item_ids

    def test_select_items_returns_requested_count(self):
        responses, _, _ = self._synthetic()
        fit = fit_2pl(responses)
        chosen = fit.select_items(5)

        assert len(chosen) == 5
        assert set(chosen).issubset(set(fit.item_ids))

    def test_select_items_prefers_informative_items(self):
        responses, _, _ = self._synthetic()
        fit = fit_2pl(responses)
        top = fit.select_items(1)[0]
        info = fit.information(0.0)

        assert info[fit.item_ids.index(top)] == pytest.approx(info.max(), rel=0.5)

    def test_select_items_handles_zero_and_oversized_k(self):
        responses, _, _ = self._synthetic()
        fit = fit_2pl(responses)

        assert fit.select_items(0) == []
        assert len(fit.select_items(999)) == len(fit.item_ids)

    def test_to_dict_is_json_shaped(self):
        responses, _, _ = self._synthetic()
        fit = fit_2pl(responses)
        payload = fit.to_dict()

        assert len(payload["items"]) == len(fit.item_ids)
        assert set(payload["items"][0]) == {"item_id", "difficulty", "discrimination"}
        assert payload["n_degenerate"] == len(fit.degenerate_items)

    def test_response_matrix_from_raw_results(self):
        raw = {
            "base": [_raw("a", exact_match=False), _raw("b", exact_match=True)],
            "step-2000": [_raw("a", exact_match=True), _raw("b", exact_match=True)],
        }
        responses, items, models = responses_from_raw_results(raw, metric="exact_match")

        assert responses.shape == (2, 2)
        assert items == ["a", "b"]
        assert set(models) == {"base", "step-2000"}

    def test_response_matrix_intersects_examples(self):
        raw = {
            "base": [_raw("a", exact_match=True), _raw("b", exact_match=True)],
            "ft": [_raw("a", exact_match=False)],
        }
        responses, items, _ = responses_from_raw_results(raw, metric="exact_match")

        assert items == ["a"]
        assert responses.shape == (2, 1)

    def test_response_matrix_requires_shared_examples(self):
        raw = {"base": [_raw("a", exact_match=True)], "ft": [_raw("z", exact_match=True)]}
        with pytest.raises(ValueError, match="no examples shared"):
            responses_from_raw_results(raw, metric="exact_match")


class TestCodeFences:
    def test_strips_python_fence(self):
        assert strip_code_fences("```python\nreturn x\n```") == "return x"

    def test_strips_bare_fence(self):
        assert strip_code_fences("```\nreturn x\n```") == "return x"

    def test_passes_through_unfenced_code(self):
        assert strip_code_fences("\n  return x  \n") == "  return x"

    def test_joins_multiple_blocks(self):
        text = "first:\n```python\na = 1\n```\nthen:\n```python\nb = 2\n```"
        assert strip_code_fences(text) == "a = 1\nb = 2"

    def test_handles_unterminated_fence(self):
        """Generation cut off by max_new_tokens still yields its code."""
        assert strip_code_fences("```python\nreturn x") == "return x"


class TestParseSnippet:
    def test_parses_module_level_code(self):
        statements, strategy = parse_snippet("x = 1")
        assert strategy == "direct"
        assert len(statements) == 1

    def test_dedents_an_indented_body(self):
        statements, strategy = parse_snippet("    x = 1\n    y = 2")
        assert strategy == "dedented"
        assert len(statements) == 2

    def test_parses_bare_return_from_a_method_body(self):
        """`return self.author` is the real completion shape in the eval set."""
        statements, strategy = parse_snippet("        return self.author")
        assert strategy == "dedented"
        assert isinstance(statements[0], ast.Return)

    def test_scope_errors_are_not_parse_errors(self):
        """Why two strategies suffice: ast.parse never checks scope."""
        for fragment in ("return x", "yield x", "await f()", "break"):
            statements, strategy = parse_snippet(fragment)
            assert statements is not None, fragment
            assert strategy == "direct"

    def test_strategies_yield_comparable_statements(self):
        direct, _ = parse_snippet("x = 1")
        dedented, _ = parse_snippet("    return x")
        assert not isinstance(direct[0], ast.FunctionDef)
        assert not isinstance(dedented[0], ast.FunctionDef)

    def test_rejects_broken_syntax(self):
        statements, strategy = parse_snippet("def f(:\n  pass")
        assert statements is None
        assert strategy == "failed"

    def test_rejects_empty(self):
        assert parse_snippet("   \n  ")[0] is None


class TestAstEquivalence:
    def test_ignores_comments_and_blank_lines(self):
        assert ast_equivalent("x = 1\n\n# a comment\ny = 2", "x = 1\ny = 2")

    def test_ignores_redundant_parentheses(self):
        assert ast_equivalent("return (x)", "return x")

    def test_ignores_indentation_depth(self):
        assert ast_equivalent("        return self.author", "    return self.author")

    def test_matches_across_markdown_fences(self):
        """The real eval shape: fenced reference, bare generation."""
        assert ast_equivalent("return self.author", "```python\n        return self.author\n```")

    def test_different_code_is_not_equivalent(self):
        assert not ast_equivalent("return x + 1", "return x - 1")

    def test_different_variable_names_are_not_structurally_equal(self):
        assert not ast_equivalent("total = 0", "acc = 0")

    def test_unparseable_generation_is_not_equivalent(self):
        assert not ast_equivalent("def f(:", "return x")

    def test_two_syntax_errors_do_not_match(self):
        assert not ast_equivalent("def f(:", "def g(:")

    def test_docstrings_count_by_default(self):
        a = 'def f():\n    """One."""\n    return 1'
        b = 'def f():\n    """Two."""\n    return 1'
        assert not ast_equivalent(a, b)
        assert ast_equivalent(a, b, ignore_docstrings=True)


class TestAlphaEquivalence:
    def test_consistent_renaming_matches(self):
        assert alpha_equivalent("total = 0\nreturn total", "acc = 0\nreturn acc")

    def test_renamed_arguments_match(self):
        assert alpha_equivalent("def f(a, b):\n    return a + b", "def f(x, y):\n    return x + y")

    def test_inconsistent_renaming_does_not_match(self):
        assert not alpha_equivalent("a = 1\nreturn a", "x = 1\nreturn y")

    def test_free_names_are_not_renamed(self):
        """Calling different functions is a real difference, not a naming one."""
        assert not alpha_equivalent("return sorted(v)", "return reversed(v)")

    def test_attributes_are_not_renamed(self):
        assert not alpha_equivalent("return self.author", "return self.title")

    def test_keyword_argument_names_are_preserved(self):
        assert not alpha_equivalent("f(key=1)", "f(cmp=1)")

    def test_structural_equality_implies_alpha_equality(self):
        pairs = [
            ("x = 1", "x = 1"),
            ("return (a)", "return a"),
            ("for i in r: pass", "for i in r: pass"),
        ]
        for gen, ref in pairs:
            if ast_equivalent(gen, ref):
                assert alpha_equivalent(gen, ref)

    def test_loop_variable_renaming_matches(self):
        assert alpha_equivalent(
            "for i in items:\n    print(i)", "for idx in items:\n    print(idx)"
        )

    def test_exception_alias_renaming_matches(self):
        assert alpha_equivalent(
            "try:\n    f()\nexcept E as e:\n    raise e",
            "try:\n    f()\nexcept E as err:\n    raise err",
        )


class TestParseability:
    def test_detects_valid_fragment(self):
        assert is_parseable("```python\n        return self.author\n```")

    def test_detects_invalid_fragment(self):
        assert not is_parseable("def f(:\n  pass")

    def test_empty_generation_is_not_parseable(self):
        assert not is_parseable("")

    def test_real_eval_reference_shape_parses(self):
        """Exactly the reference format in code_completion_fixed16.jsonl."""
        assert is_parseable("```python\n        return self.author\n```")

    def test_malformed_synthesized_prompt_is_detected(self):
        """Docstring at the same indent as `def` -- the synthesis bug."""
        assert not is_parseable('    def get_author(self):\n    """Return it."""')


class TestReportFormatting:
    def test_formats_numeric_value(self):
        assert ModelEvaluator._format_value(0.1) == "0.1000"

    def test_formats_missing_value(self):
        assert ModelEvaluator._format_value(None) == "—"

    def test_formats_non_numeric_value_without_crashing(self):
        assert ModelEvaluator._format_value("skipped") == "skipped"

    def test_interval_rendered_with_signs(self):
        paired = {"ci_low": -0.01, "ci_high": 0.21, "degenerate": False}
        assert ModelEvaluator._format_interval(paired) == "[-0.0100, +0.2100]"

    def test_degenerate_metric_has_no_interval(self):
        assert ModelEvaluator._format_interval({"degenerate": True}) == "—"

    def test_unpaired_metric_has_no_interval(self):
        assert ModelEvaluator._format_interval(None) == "—"

    def test_tiny_p_value_is_not_reported_as_zero(self):
        paired = {"p_value": 1e-9, "degenerate": False}
        assert ModelEvaluator._format_p_value(paired) == "<0.0001"

    def test_ordinary_p_value_rendered(self):
        assert ModelEvaluator._format_p_value({"p_value": 0.0432, "degenerate": False}) == "0.0432"

    def test_significance_labels(self):
        assert ModelEvaluator._format_significance(None) == "—"
        assert ModelEvaluator._format_significance({"degenerate": True}) == "degenerate"
        assert ModelEvaluator._format_significance(
            {"degenerate": False, "significant": True}
        ) == "yes"
        assert ModelEvaluator._format_significance(
            {"degenerate": False, "significant": False}
        ) == "no"


class TestMeasurementWarnings:
    def test_degenerate_metric_is_reported_as_unresolved_not_as_a_ceiling(self):
        """
        ast_match read 0/16 for both models, then 15/200 at p=0.0001. Declaring
        a ceiling from an all-identical sample turns a sampling limit into a
        false claim about the metric.
        """
        entry = {"paired": {"degenerate": True, "n_pairs": 16}}
        warning = ModelEvaluator._metric_warning("code_completion", "exact_match", entry)
        assert "cannot resolve a difference" in warning
        assert "not as a measurement ceiling" in warning
        assert "18.8%" in warning

    def test_degenerate_bound_tightens_with_sample_size(self):
        small = ModelEvaluator._metric_warning(
            "b", "m", {"paired": {"degenerate": True, "n_pairs": 16}})
        large = ModelEvaluator._metric_warning(
            "b", "m", {"paired": {"degenerate": True, "n_pairs": 200}})
        assert "18.8%" in small
        assert "1.5%" in large

    def test_inconclusive_metric_is_flagged(self):
        entry = {"paired": {"degenerate": False, "significant": False, "n_pairs": 16,
                            "confidence_level": 0.95, "ci_low": -0.02, "ci_high": 0.31}}
        warning = ModelEvaluator._metric_warning("code_completion", "ast_match", entry)
        assert "not established" in warning

    def test_significant_metric_gets_no_warning(self):
        entry = {"paired": {"degenerate": False, "significant": True, "n_pairs": 16,
                            "confidence_level": 0.95, "ci_low": 0.05, "ci_high": 0.21}}
        assert ModelEvaluator._metric_warning("code_completion", "token_f1", entry) is None

    def test_unpaired_metric_gets_no_warning(self):
        assert ModelEvaluator._metric_warning("perplexity", "perplexity", {}) is None


class TestReportSections:
    def _results(self):
        raw_base = [{"prompt": f"p{i}", "token_f1": 0.1, "exact_match": False} for i in range(8)]
        raw_ft = [{"prompt": f"p{i}", "token_f1": 0.4, "exact_match": False} for i in range(8)]
        def bench(raw, f1):
            return {
                "metrics": {"token_f1": f1, "exact_match": 0.0},
                "num_examples": 8,
                "reference_parse_rate": 0.75,
                "raw_results": raw,
            }

        base = {"benchmarks": {"code_completion": bench(raw_base, 0.1)}}
        ft = {"benchmarks": {"code_completion": bench(raw_ft, 0.4)}}
        return base, ft

    def test_dataset_section_reports_reference_parse_rate(self):
        base, ft = self._results()
        lines = ModelEvaluator._dataset_lines({"base": base, "finetuned": ft})
        body = "\n".join(lines)
        assert "Evaluation Set Diagnostics" in body
        assert "0.7500" in body

    def test_dataset_section_warns_on_unparseable_references(self):
        base, ft = self._results()
        body = "\n".join(ModelEvaluator._dataset_lines({"base": base, "finetuned": ft}))
        assert "25.0% of `code_completion` references are not parseable" in body
        assert "regardless" in body

    def test_dataset_section_absent_without_the_diagnostic(self):
        results = {"base": {"benchmarks": {"perplexity": {"metrics": {"perplexity": 10.0}}}}}
        assert ModelEvaluator._dataset_lines(results) == []

    def test_comparison_renders_a_table_not_a_dict(self):
        base, ft = self._results()
        comparison = compute_comparison_stats(base, ft)
        body = "\n".join(ModelEvaluator({})._comparison_lines(comparison))

        assert "| Metric | Base | Fine-tuned | Δ | 95% CI | p | n | Signif. |" in body
        assert "{'base'" not in body, "raw dict leaked into the report"

    def test_comparison_surfaces_degenerate_metric_warning(self):
        base, ft = self._results()
        comparison = compute_comparison_stats(base, ft)
        body = "\n".join(ModelEvaluator({})._comparison_lines(comparison))

        assert "Measurement warnings" in body
        assert "measurement ceiling" in body

    def test_written_report_contains_both_sections(self, tmp_path):
        base, ft = self._results()
        results = {"base": base, "finetuned": ft}
        results["comparison"] = compute_comparison_stats(base, ft)

        path = tmp_path / "evaluation_report.md"
        ModelEvaluator({})._write_markdown_report(results, path)
        body = path.read_text(encoding="utf-8")

        assert "# Model Evaluation Report" in body
        assert "## Evaluation Set Diagnostics" in body
        assert "## Statistical Comparison" in body
        assert "{'" not in body


class TestParseRatePairing:
    def test_parse_rate_pairs_against_the_parses_field(self):
        """The metric and its per-example field have different names."""
        base = {"benchmarks": {"code_completion": {
            "metrics": {"parse_rate": 0.5},
            "raw_results": [{"prompt": f"p{i}", "parses": i < 4} for i in range(8)],
        }}}
        ft = {"benchmarks": {"code_completion": {
            "metrics": {"parse_rate": 1.0},
            "raw_results": [{"prompt": f"p{i}", "parses": True} for i in range(8)],
        }}}
        entry = compute_comparison_stats(base, ft)["code_completion"]["parse_rate"]

        assert "paired" in entry
        assert entry["paired"]["n_pairs"] == 8
        assert entry["paired"]["mean_difference"] == pytest.approx(0.5)


class TestScoringInputs:
    def test_masks_prompt_and_keeps_completion(self):
        input_ids, labels = build_scoring_inputs([1, 2, 3], [4, 5], max_length=10)
        assert input_ids == [1, 2, 3, 4, 5]
        assert labels == [-100, -100, -100, 4, 5]

    def test_truncates_prompt_from_the_front(self):
        """The completion being scored must survive intact."""
        input_ids, labels = build_scoring_inputs([1, 2, 3, 4, 5], [8, 9], max_length=4)
        assert input_ids == [4, 5, 8, 9]
        assert labels == [-100, -100, 8, 9]

    def test_empty_completion_is_unscoreable(self):
        assert build_scoring_inputs([1, 2], [], max_length=10) is None

    def test_completion_longer_than_window_is_unscoreable(self):
        assert build_scoring_inputs([1], [2, 3, 4], max_length=3) is None

    def test_label_length_always_matches_input_length(self):
        for max_length in (3, 5, 20):
            built = build_scoring_inputs([1, 2, 3, 4], [5, 6], max_length)
            if built is not None:
                assert len(built[0]) == len(built[1])


class TestLikelihoodOrientation:
    def test_lower_loss_scores_higher(self):
        """PPI's clamped weight assumes the proxy rises with quality."""
        assert likelihood_from_nll(0.1) > likelihood_from_nll(2.0)

    def test_bounded_in_unit_interval(self):
        for nll in (0.0, 0.5, 5.0, 50.0):
            assert 0.0 < likelihood_from_nll(nll) <= 1.0

    def test_non_finite_loss_scores_zero(self):
        assert likelihood_from_nll(float("inf")) == 0.0
        assert likelihood_from_nll(float("nan")) == 0.0


class TestPoolLoading:
    def test_reads_prompt_completion_rows(self, tmp_path):
        path = tmp_path / "p.jsonl"
        path.write_text('{"prompt": "def f():", "completion": "    return 1"}\n')
        pool = load_pool(path)
        assert pool[0].prompt == "def f():"

    def test_splits_chatml_rows(self, tmp_path):
        path = tmp_path / "p.jsonl"
        path.write_text(
            '{"text": "<|im_start|>user\\nDo it<|im_end|>\\n'
            '<|im_start|>assistant\\nreturn 1<|im_end|>"}\n'
        )
        pool = load_pool(path)
        assert len(pool) == 1
        assert pool[0].completion == "return 1"
        assert pool[0].prompt.endswith("<|im_start|>assistant\n")

    def test_skips_malformed_lines_without_failing(self):
        assert split_chatml("no marker here") is None

    def test_skips_empty_assistant_turn(self):
        assert split_chatml("<|im_start|>assistant\n   <|im_end|>") is None

    def test_respects_the_limit(self, tmp_path):
        path = tmp_path / "p.jsonl"
        path.write_text('{"prompt": "a", "completion": "b"}\n' * 50)
        assert len(load_pool(path, limit=5)) == 5


class TestProxyAlignment:
    def test_drops_positions_either_model_could_not_score(self):
        base, ft = align_proxy_scores([0.5, None, 0.7], [0.6, 0.2, None])
        assert base == [0.5]
        assert ft == [0.6]

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError, match="must align"):
            align_proxy_scores([0.1], [0.1, 0.2])

    def test_gold_alignment_matches_on_prompt(self):
        base_raw = [{"prompt": "a", "token_f1": 0.1}, {"prompt": "b", "token_f1": 0.2}]
        ft_raw = [{"prompt": "b", "token_f1": 0.5}, {"prompt": "a", "token_f1": 0.4}]
        examples = [PoolExample("a", "x"), PoolExample("b", "y")]
        aligned = align_gold_with_proxy(base_raw, ft_raw, examples, [0.8, 0.7], [0.9, 0.75])

        assert aligned["gold_base"] == [0.1, 0.2]
        assert aligned["gold_finetuned"] == [0.4, 0.5]
        assert aligned["proxy_base"] == [0.8, 0.7]

    def test_returns_none_without_enough_overlap(self):
        base_raw = [{"prompt": "a", "token_f1": 0.1}]
        ft_raw = [{"prompt": "a", "token_f1": 0.4}]
        examples = [PoolExample("a", "x")]
        assert align_gold_with_proxy(base_raw, ft_raw, examples, [0.8], [0.9]) is None


class TestProxyDiagnostics:
    def _estimate(self, flip_proxy: bool):
        rng = np.random.default_rng(5)
        truth_b = rng.normal(0.10, 0.06, 300)
        truth_f = truth_b + 0.12 + rng.normal(0, 0.02, 300)
        sign = -1.0 if flip_proxy else 1.0
        pb, pf = sign * truth_b, sign * truth_f
        return ppi_mean_difference(
            gold_base=truth_b[:16], gold_finetuned=truth_f[:16],
            proxy_base=pb[:16], proxy_finetuned=pf[:16],
            pool_proxy_base=pb, pool_proxy_finetuned=pf,
        )

    def test_well_oriented_proxy_has_no_warning(self):
        estimate = self._estimate(flip_proxy=False)
        assert estimate.proxy_correlation > 0.5
        assert estimate.warning() is None

    def test_inverted_proxy_is_caught_rather_than_silently_discarded(self):
        """Without this, a backwards proxy looks exactly like a useless one."""
        estimate = self._estimate(flip_proxy=True)
        assert estimate.lambda_weight == 0.0
        assert "orientation" in estimate.warning()

    def test_estimate_is_still_valid_when_the_proxy_is_inverted(self):
        estimate = self._estimate(flip_proxy=True)
        assert estimate.estimate == pytest.approx(estimate.classical_estimate)


class TestPPIReportSection:
    def _ppi_payload(self):
        rng = np.random.default_rng(9)
        truth_b = rng.normal(0.10, 0.06, 400)
        truth_f = truth_b + 0.13 + rng.normal(0, 0.02, 400)
        proxy_b = truth_b + rng.normal(0, 0.03, 400)
        proxy_f = truth_f + rng.normal(0, 0.03, 400)
        estimate = ppi_mean_difference(
            gold_base=truth_b[:16], gold_finetuned=truth_f[:16],
            proxy_base=proxy_b[:16], proxy_finetuned=proxy_f[:16],
            pool_proxy_base=proxy_b, pool_proxy_finetuned=proxy_f,
        )
        payload = estimate.to_dict()
        payload["metric"] = "token_f1"
        payload["pool"] = {"pool_size": 400, "scored": 400, "unscoreable": 0}
        return payload

    def test_shows_both_estimators_side_by_side(self):
        body = "\n".join(ModelEvaluator._ppi_lines(self._ppi_payload()))
        assert "| Gold only |" in body
        assert "| Prediction-powered |" in body

    def test_reports_effective_sample_size(self):
        body = "\n".join(ModelEvaluator._ppi_lines(self._ppi_payload()))
        assert "carry the weight of about" in body

    def test_reports_proxy_correlation(self):
        body = "\n".join(ModelEvaluator._ppi_lines(self._ppi_payload()))
        assert "Proxy correlation" in body

    def test_appears_in_the_written_report(self, tmp_path):
        results = {"ppi": self._ppi_payload()}
        path = tmp_path / "r.md"
        ModelEvaluator({})._write_markdown_report(results, path)
        assert "## Prediction-Powered Estimate" in path.read_text(encoding="utf-8")


def _make_checkpoint(root, name, step, valid=True):
    """Create a checkpoint directory that passes or fails integrity validation."""
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    if valid:
        (path / "trainer_state.json").write_text(json.dumps({"global_step": step}))
        (path / "adapter_model.safetensors").write_bytes(b"\x00" * 16)
    return path


class TestCheckpointDiscovery:
    def test_finds_checkpoints_in_step_order(self, tmp_path):
        for step in (2000, 500, 1000):
            _make_checkpoint(tmp_path, f"checkpoint-{step}", step)
        refs = discover_checkpoints(tmp_path, include_base=False)
        assert [r.step for r in refs] == [500, 1000, 2000]

    def test_ignores_non_checkpoint_directories(self, tmp_path):
        """The real checkpoints dir holds smoke tests and diagnostic runs."""
        for step in (500, 1000):
            _make_checkpoint(tmp_path, f"checkpoint-{step}", step)
        for junk in ("smoke", "smoke-nopack", "resume-blocked-checkpoint-200", "best"):
            _make_checkpoint(tmp_path, junk, 1)
        refs = discover_checkpoints(tmp_path, include_base=False)
        assert {r.name for r in refs} == {"checkpoint-500", "checkpoint-1000"}

    def test_skips_corrupt_checkpoints(self, tmp_path):
        """A half-written checkpoint would enter the population as a weak model."""
        _make_checkpoint(tmp_path, "checkpoint-500", 500)
        broken = tmp_path / "checkpoint-900"
        broken.mkdir()
        (broken / "trainer_state.json").write_text("{not json")
        refs = discover_checkpoints(tmp_path, include_base=False)
        assert [r.name for r in refs] == ["checkpoint-500"]

    def test_includes_base_as_step_zero(self, tmp_path):
        _make_checkpoint(tmp_path, "checkpoint-500", 500)
        refs = discover_checkpoints(tmp_path, include_base=True)
        assert refs[0].name == "base"
        assert refs[0].is_base
        assert refs[0].adapter_path is None

    def test_stride_spreads_the_sample(self, tmp_path):
        for step in range(500, 5001, 500):
            _make_checkpoint(tmp_path, f"checkpoint-{step}", step)
        refs = discover_checkpoints(tmp_path, include_base=False, stride=3)
        assert [r.step for r in refs] == [500, 2000, 3500, 5000]

    def test_limit_keeps_the_latest(self, tmp_path):
        for step in (500, 1000, 1500, 2000):
            _make_checkpoint(tmp_path, f"checkpoint-{step}", step)
        refs = discover_checkpoints(tmp_path, include_base=False, limit=2)
        assert [r.step for r in refs] == [1500, 2000]

    def test_missing_directory_yields_base_only(self, tmp_path):
        refs = discover_checkpoints(tmp_path / "nope", include_base=True)
        assert [r.name for r in refs] == ["base"]


class TestSweepModelConfig:
    def test_base_config_carries_no_adapter(self):
        ref = CheckpointRef(name="base", adapter_path=None, step=0)
        config = build_model_config({"name": "Qwen/Qwen2.5-Coder-1.5B"}, ref)
        assert "adapter_path" not in config

    def test_checkpoint_config_points_at_the_adapter(self):
        ref = CheckpointRef(name="checkpoint-500", adapter_path="/ck/500", step=500)
        config = build_model_config({"name": "m", "revision": "main"}, ref)
        assert config["adapter_path"] == "/ck/500"
        assert config["revision"] == "main"


class TestCalibration:
    def _scored(self, n_models=8, n_items=12, seed=4):
        """Checkpoints of graded ability against items of graded difficulty."""
        rng = np.random.default_rng(seed)
        ability = np.linspace(-1.5, 1.5, n_models)
        difficulty = np.linspace(-1.2, 1.2, n_items)
        scored = {}
        for i in range(n_models):
            rows = []
            for j in range(n_items):
                p = 1 / (1 + np.exp(-(ability[i] - difficulty[j])))
                rows.append({"prompt": f"item{j}", "ast_match": bool(rng.random() < p)})
            scored[f"checkpoint-{(i + 1) * 500}"] = rows
        return scored

    def test_recovers_ability_ordering_across_checkpoints(self):
        payload = calibrate(self._scored(), metric="ast_match")
        by_name = {m["model_id"]: m["ability"] for m in payload["models"]}
        assert by_name["checkpoint-4000"] > by_name["checkpoint-500"]

    def test_recommends_a_subset_of_the_items(self):
        payload = calibrate(self._scored(), metric="ast_match", keep_items=5)
        assert len(payload["recommended_items"]) == 5
        assert set(payload["recommended_items"]).issubset(
            {item["item_id"] for item in payload["items"]}
        )

    def test_reports_degenerate_items(self):
        scored = self._scored()
        for rows in scored.values():
            rows.append({"prompt": "impossible", "ast_match": False})
        payload = calibrate(scored, metric="ast_match")
        assert "impossible" in payload["degenerate_items"]

    def test_threshold_binarizes_a_continuous_metric(self):
        scored = {
            f"ck{i}": [
                {"prompt": "a", "token_f1": 0.1 * i},
                {"prompt": "b", "token_f1": 0.9 - 0.1 * i},
            ]
            for i in range(6)
        }
        payload = calibrate(scored, metric="token_f1", threshold=0.5, keep_items=2)
        assert payload["threshold"] == 0.5
        assert payload["n_items_total"] == 2

    def test_population_below_the_minimum_is_rejected(self):
        scored = {
            "base": [{"prompt": "a", "ast_match": True}],
            "ft": [{"prompt": "a", "ast_match": False}],
        }
        with pytest.raises(ValueError, match="at least 5 models"):
            calibrate(scored, metric="ast_match")


class TestPoolTaskFiltering:
    """
    PPI estimates the mean over the pool, so a pool whose distribution differs
    from the gold set's silently changes the estimand.
    """

    def _mixed(self, tmp_path):
        path = tmp_path / "mixed.jsonl"
        rows = [
            '{"text": "<|im_start|>assistant\\na<|im_end|>", "task": "code_completion"}',
            '{"text": "<|im_start|>assistant\\nb<|im_end|>", "task": "code_explanation"}',
            '{"text": "<|im_start|>assistant\\nc<|im_end|>", "task": "raw"}',
            '{"text": "<|im_start|>assistant\\nd<|im_end|>", "task": "code_completion"}',
        ]
        path.write_text("\n".join(rows) + "\n")
        return path

    def test_unfiltered_pool_keeps_every_task(self, tmp_path):
        assert len(load_pool(self._mixed(tmp_path))) == 4

    def test_filter_keeps_only_the_matching_task(self, tmp_path):
        pool = load_pool(self._mixed(tmp_path), task="code_completion")
        assert [e.completion for e in pool] == ["a", "d"]

    def test_limit_counts_kept_rows_not_read_rows(self, tmp_path):
        """Reading must continue past skipped rows to reach the limit."""
        pool = load_pool(self._mixed(tmp_path), limit=2, task="code_completion")
        assert len(pool) == 2

    def test_unknown_task_yields_empty_pool(self, tmp_path):
        assert load_pool(self._mixed(tmp_path), task="nonexistent") == []

    def test_rows_without_a_task_field_are_filtered_out(self, tmp_path):
        path = tmp_path / "p.jsonl"
        path.write_text('{"prompt": "a", "completion": "b"}\n')
        assert load_pool(path, task="code_completion") == []
        assert len(load_pool(path)) == 1


class TestProxyPrefixScoring:
    """
    Scoring only the opening tokens is meant to measure format onset, which
    teacher-forcing the whole reference cannot see.
    """

    def test_prefix_truncates_the_scored_completion(self):
        input_ids, labels = build_scoring_inputs(
            [1, 2], [10, 11, 12, 13], max_length=20, prefix_tokens=2
        )
        assert input_ids == [1, 2, 10, 11]
        assert labels == [-100, -100, 10, 11]

    def test_prefix_longer_than_completion_is_harmless(self):
        full = build_scoring_inputs([1], [10, 11], max_length=20)
        clipped = build_scoring_inputs([1], [10, 11], max_length=20, prefix_tokens=99)
        assert full == clipped

    def test_zero_prefix_is_unscoreable(self):
        assert build_scoring_inputs([1], [10, 11], max_length=20, prefix_tokens=0) is None

    def test_none_prefix_scores_the_whole_completion(self):
        _, labels = build_scoring_inputs([1], [10, 11, 12], max_length=20, prefix_tokens=None)
        assert labels.count(-100) == 1
        assert labels[1:] == [10, 11, 12]

    def test_prefix_relaxes_the_max_length_constraint(self):
        """A long completion becomes scoreable once only its prefix is used."""
        long_completion = list(range(100, 200))
        assert build_scoring_inputs([1], long_completion, max_length=10) is None
        assert build_scoring_inputs([1], long_completion, max_length=10, prefix_tokens=4)

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the translation agent and verification utilities.

All OpenAI calls are mocked — no network access required.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from solar.einsum_lib.verify import (
    VerificationResult,
    compare_outputs,
    execute_translated,
)

# ---------------------------------------------------------------------------
# Sample generated code snippets used across tests
# ---------------------------------------------------------------------------

GOOD_MATMUL_CODE = """\
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims

@kernel
def matmul(A: Tensor("B,M,K"), W: Tensor("K,N", role="weight")) -> Tensor("B,M,N"):
    C = Tensor(B=A.shape[0], M=A.shape[1], N=W.shape[1])
    for b in Dim("B"):
        for m in Dim("M"):
            for n in Dim("N"):
                for k in Dim("K"):
                    C[b,m,n] += A[b,m,k] * W[k,n]
    return C

def model(x, w):
    return matmul(x, w)

def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "B,M,K")
    w = Tensor.from_numpy(inputs["w"], "K,N", role="weight")
    result = model(x, w)
    return {"output": result.data}
"""

BROKEN_CODE = """\
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims

@kernel
def matmul(A: Tensor("B,M,K"), W: Tensor("K,N")) -> Tensor("B,M,N"):
    C = Tensor(B=A.shape[0], M=A.shape[1], N=W.shape[1])
    for b in Dim("B"):
        for m in Dim("M"):
            for n in Dim("N"):
                for k in Dim("WRONG_DIM"):  # will fail — dim not registered
                    C[b,m,n] += A[b,m,k] * W[k,n]
    return C

def model(x, w):
    return matmul(x, w)

def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "B,M,K")
    w = Tensor.from_numpy(inputs["w"], "K,N")
    result = model(x, w)
    return {"output": result.data}
"""

WRONG_OUTPUT_CODE = """\
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims
import numpy as np

@kernel
def matmul(A: Tensor("B,M,K"), W: Tensor("K,N")) -> Tensor("B,M,N"):
    C = Tensor(B=A.shape[0], M=A.shape[1], N=W.shape[1])
    return C

def model(x, w):
    return matmul(x, w)

def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "B,M,K")
    w = Tensor.from_numpy(inputs["w"], "K,N")
    result = model(x, w)
    return {"output": result.data}
"""

NO_RUN_MODEL_CODE = """\
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims

@kernel
def matmul(A: Tensor("B,M,K"), W: Tensor("K,N")) -> Tensor("B,M,N"):
    C = Tensor(B=A.shape[0], M=A.shape[1], N=W.shape[1])
    return C
"""

MATMUL_SHAPES = {"x": (2, 3, 5), "w": (5, 4)}


# ---------------------------------------------------------------------------
# verify.py — execute_translated
# ---------------------------------------------------------------------------


class TestExecuteTranslated:
    def test_good_code_returns_outputs(self):
        outputs, inputs = execute_translated(GOOD_MATMUL_CODE, MATMUL_SHAPES)
        assert "output" in outputs
        assert outputs["output"].shape == (2, 3, 4)
        assert inputs["x"].shape == (2, 3, 5)

    def test_missing_run_model_raises(self):
        with pytest.raises(RuntimeError, match="run_model"):
            execute_translated(NO_RUN_MODEL_CODE, MATMUL_SHAPES)

    def test_syntax_error_raises(self):
        with pytest.raises(RuntimeError, match="compilation failed"):
            execute_translated("def broken(:\n  pass", MATMUL_SHAPES)

    def test_runtime_error_raises(self):
        with pytest.raises(RuntimeError, match="run_model.*failed"):
            execute_translated(BROKEN_CODE, MATMUL_SHAPES)

    def test_deterministic_with_seed(self):
        out1, _ = execute_translated(GOOD_MATMUL_CODE, MATMUL_SHAPES, seed=99)
        out2, _ = execute_translated(GOOD_MATMUL_CODE, MATMUL_SHAPES, seed=99)
        np.testing.assert_array_equal(out1["output"], out2["output"])


# ---------------------------------------------------------------------------
# verify.py — compare_outputs
# ---------------------------------------------------------------------------


class TestCompareOutputs:
    def test_identical_passes(self):
        a = {"output": np.ones((2, 3))}
        result = compare_outputs(a, a)
        assert result.success
        assert result.max_abs_error == 0.0

    def test_close_passes(self):
        a = {"output": np.ones((2, 3))}
        b = {"output": np.ones((2, 3)) + 1e-7}
        result = compare_outputs(a, b)
        assert result.success

    def test_different_fails(self):
        a = {"output": np.zeros((2, 3))}
        b = {"output": np.ones((2, 3))}
        result = compare_outputs(a, b)
        assert not result.success
        assert result.max_abs_error == pytest.approx(1.0)
        assert "values differ" in result.error

    def test_shape_mismatch(self):
        a = {"output": np.zeros((2, 3))}
        b = {"output": np.zeros((3, 2))}
        result = compare_outputs(a, b)
        assert not result.success
        assert not result.output_shapes_match

    def test_key_mismatch(self):
        a = {"output": np.zeros((2,))}
        b = {"other": np.zeros((2,))}
        result = compare_outputs(a, b)
        assert not result.success
        assert "key mismatch" in result.error.lower()


# ---------------------------------------------------------------------------
# verify.py — matmul correctness end-to-end
# ---------------------------------------------------------------------------


class TestVerifyMatmulCorrectness:
    def test_matmul_matches_numpy(self):
        outputs, inputs = execute_translated(GOOD_MATMUL_CODE, MATMUL_SHAPES)
        expected = np.einsum("bmk,kn->bmn", inputs["x"], inputs["w"])
        ref = {"output": expected}
        result = compare_outputs(outputs, ref)
        assert result.success, result.error


# ---------------------------------------------------------------------------
# Helpers: mock OpenAI
# ---------------------------------------------------------------------------


def _mock_openai_response(content: str) -> MagicMock:
    """Build a mock that looks like an OpenAI ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_agent(**kwargs):
    """Create a TranslationAgent with a mocked OpenAI client."""
    with patch("solar.einsum_lib.agent._openai_module") as mock_mod:
        mock_client = MagicMock()
        mock_mod.OpenAI.return_value = mock_client
        from solar.einsum_lib.agent import TranslationAgent

        kwargs.setdefault("api_key", "sk-test")
        agent = TranslationAgent(**kwargs)
        agent.client = mock_client
        return agent, mock_client


# ---------------------------------------------------------------------------
# agent.py — prompt construction
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    def test_system_prompt_contains_reference(self):
        agent, _ = _make_agent()
        assert "Tensor" in agent._system_prompt
        assert "Dim" in agent._system_prompt
        assert "@kernel" in agent._system_prompt

    def test_system_prompt_contains_no_optimization_rules(self):
        agent, _ = _make_agent()
        assert "Do NOT optimize" in agent._system_prompt

    def test_user_prompt_contains_source_and_shapes(self):
        agent, _ = _make_agent()
        prompt = agent._build_user_prompt("import torch", {"x": (2, 4)})
        assert "import torch" in prompt
        assert "(2, 4)" in prompt

    def test_retry_prompt_contains_error(self):
        agent, _ = _make_agent()
        prompt = agent._build_retry_prompt("dim not found", "bad code")
        assert "dim not found" in prompt
        assert "bad code" in prompt


# ---------------------------------------------------------------------------
# agent.py — code extraction
# ---------------------------------------------------------------------------


class TestCodeExtraction:
    def test_fenced_python(self):
        from solar.einsum_lib.agent import TranslationAgent

        raw = "Here is the code:\n```python\nprint('hi')\n```\nDone."
        assert TranslationAgent._extract_code(raw) == "print('hi')"

    def test_fenced_plain(self):
        from solar.einsum_lib.agent import TranslationAgent

        raw = "```\nprint('hi')\n```"
        assert TranslationAgent._extract_code(raw) == "print('hi')"

    def test_no_fences(self):
        from solar.einsum_lib.agent import TranslationAgent

        raw = "print('hi')"
        assert TranslationAgent._extract_code(raw) == "print('hi')"


# ---------------------------------------------------------------------------
# agent.py — translate() happy path
# ---------------------------------------------------------------------------


class TestTranslateSuccess:
    def test_returns_verified_code(self):
        agent, mock_client = _make_agent()
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            f"```python\n{GOOD_MATMUL_CODE}\n```"
        )

        # Reference-backed verification (joint review P2-1): an
        # execution-only run is ``executed_no_reference``, not verified; the
        # positive case compares against the numpy matmul on the same inputs.
        _, inputs = execute_translated(GOOD_MATMUL_CODE, MATMUL_SHAPES, seed=7)
        reference = {"output": np.einsum("bmk,kn->bmn", inputs["x"], inputs["w"])}
        result = agent.translate(
            "x @ w", MATMUL_SHAPES, reference_outputs=reference, all_inputs=inputs
        )
        assert result.verified
        assert result.verification_status == "verified"
        assert result.attempts == 1
        assert "run_model" in result.code

    def test_no_verify_skips_execution(self):
        agent, mock_client = _make_agent(verify=False)
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            f"```python\n{BROKEN_CODE}\n```"
        )

        result = agent.translate("x @ w", MATMUL_SHAPES)
        assert not result.verified
        assert result.attempts == 1


# ---------------------------------------------------------------------------
# agent.py — retry on execution error
# ---------------------------------------------------------------------------


class TestRetryOnError:
    def test_retries_then_succeeds(self):
        agent, mock_client = _make_agent(max_retries=3)
        mock_client.chat.completions.create.side_effect = [
            _mock_openai_response(f"```python\n{BROKEN_CODE}\n```"),
            _mock_openai_response(f"```python\n{GOOD_MATMUL_CODE}\n```"),
        ]

        # Reference-backed, as in test_returns_verified_code (P2-1): the
        # broken attempt fails execution, the retry is compared and verified.
        _, inputs = execute_translated(GOOD_MATMUL_CODE, MATMUL_SHAPES, seed=7)
        reference = {"output": np.einsum("bmk,kn->bmn", inputs["x"], inputs["w"])}
        result = agent.translate(
            "x @ w", MATMUL_SHAPES, reference_outputs=reference, all_inputs=inputs
        )
        assert result.verified
        assert result.verification_status == "verified"
        assert result.attempts == 2

    def test_exhausts_retries(self):
        agent, mock_client = _make_agent(max_retries=2)
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            f"```python\n{BROKEN_CODE}\n```"
        )

        result = agent.translate("x @ w", MATMUL_SHAPES)
        assert not result.verified
        assert result.attempts == 2
        assert result.verification_error is not None


# ---------------------------------------------------------------------------
# agent.py — retry on output mismatch
# ---------------------------------------------------------------------------


class TestRetryOnMismatch:
    def test_mismatch_triggers_retry(self):
        agent, mock_client = _make_agent(max_retries=3)
        mock_client.chat.completions.create.side_effect = [
            _mock_openai_response(f"```python\n{WRONG_OUTPUT_CODE}\n```"),
            _mock_openai_response(f"```python\n{GOOD_MATMUL_CODE}\n```"),
        ]

        rng = np.random.RandomState(42)
        all_inputs = {
            "x": rng.uniform(-0.1, 0.1, size=(2, 3, 5)).astype(np.float64),
            "w": rng.uniform(-0.1, 0.1, size=(5, 4)).astype(np.float64),
        }
        reference_outputs = {
            "output": np.einsum("bmk,kn->bmn", all_inputs["x"], all_inputs["w"])
        }

        result = agent.translate(
            "x @ w", MATMUL_SHAPES,
            reference_outputs=reference_outputs,
            all_inputs=all_inputs,
        )
        assert result.verified
        assert result.attempts == 2


# ---------------------------------------------------------------------------
# agent.py — routing of a numba TYPED DECLINE (typed-decline-only default policy)
# ---------------------------------------------------------------------------

# The same matmul, read through a dict subscript: the fast engine's typed refusal
# ``kernel_function_statement_unresolved:matmul: subscript`` (never rewritten on a
# guess), while the interpreter computes it exactly.
DECLINING_GOOD_CODE = GOOD_MATMUL_CODE.replace(
    "    result = model(x, w)\n    return {\"output\": result.data}\n",
    "    box = {\"y\": model(x, w)}\n    return {\"output\": box[\"y\"].data}\n",
)
DECLINING_WRONG_CODE = WRONG_OUTPUT_CODE.replace(
    "    result = model(x, w)\n    return {\"output\": result.data}\n",
    "    box = {\"y\": model(x, w)}\n    return {\"output\": box[\"y\"].data}\n",
)
assert DECLINING_GOOD_CODE != GOOD_MATMUL_CODE and DECLINING_WRONG_CODE != WRONG_OUTPUT_CODE


def _matmul_reference():
    _, inputs = execute_translated(GOOD_MATMUL_CODE, MATMUL_SHAPES, seed=7)
    return {"output": np.einsum("bmk,kn->bmn", inputs["x"], inputs["w"])}, inputs


class TestNumbaDeclineRouting:
    """A numba typed decline is an infrastructure refusal, not a model answer.  Default (the bound routing policy, the review
    adjudicates): the fast agent verifies the declined program on the interpreter path and spends no re-prompt on it;
    ``reprompt_on_numba_decline=True`` preserves the previous policy (decline -> ``execution_failed`` -> re-prompt).
    A numerical MISMATCH re-prompts under both policies."""

    def test_default_routes_a_typed_decline_to_the_interpreter(self, capsys):
        agent, mock_client = _make_agent(max_retries=3)
        assert agent.fast and not agent.reprompt_on_numba_decline
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            f"```python\n{DECLINING_GOOD_CODE}\n```"
        )
        reference, inputs = _matmul_reference()
        result = agent.translate(
            "x @ w", MATMUL_SHAPES, reference_outputs=reference, all_inputs=inputs
        )
        assert result.verified and result.verification_status == "verified"
        assert result.attempts == 1
        assert mock_client.chat.completions.create.call_count == 1
        assert result.numba_code is None          # the fast engine declined; nothing was lowered on a guess
        out = capsys.readouterr().out
        assert "[WARN] Numba declined this code, falling back to the interpreter" in out
        assert "kernel_function_statement_unresolved:matmul: subscript" in out

    def test_reprompt_flag_preserves_the_previous_policy(self):
        agent, mock_client = _make_agent(max_retries=2, reprompt_on_numba_decline=True)
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            f"```python\n{DECLINING_GOOD_CODE}\n```"
        )
        reference, inputs = _matmul_reference()
        result = agent.translate(
            "x @ w", MATMUL_SHAPES, reference_outputs=reference, all_inputs=inputs
        )
        assert not result.verified
        assert result.verification_status == "execution_failed"
        assert result.attempts == 2
        assert mock_client.chat.completions.create.call_count == 2
        assert result.verification_error.startswith("Numba declined the generated code (kernel_function_statement_unresolved:matmul: subscript")

    def test_a_mismatch_still_reprompts_under_the_default(self):
        agent, mock_client = _make_agent(max_retries=3)
        mock_client.chat.completions.create.side_effect = [
            _mock_openai_response(f"```python\n{DECLINING_WRONG_CODE}\n```"),
            _mock_openai_response(f"```python\n{GOOD_MATMUL_CODE}\n```"),
        ]
        reference, inputs = _matmul_reference()
        result = agent.translate(
            "x @ w", MATMUL_SHAPES, reference_outputs=reference, all_inputs=inputs
        )
        assert result.verified and result.attempts == 2
        assert mock_client.chat.completions.create.call_count == 2

    def test_non_fast_agents_are_unchanged(self):
        agent, mock_client = _make_agent(fast=False)
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            f"```python\n{DECLINING_GOOD_CODE}\n```"
        )
        reference, inputs = _matmul_reference()
        result = agent.translate(
            "x @ w", MATMUL_SHAPES, reference_outputs=reference, all_inputs=inputs
        )
        assert result.verified and result.attempts == 1 and result.numba_code is None

    def test_the_flag_is_off_by_default_and_recorded(self):
        agent, _ = _make_agent()
        assert agent.reprompt_on_numba_decline is False
        agent2, _ = _make_agent(reprompt_on_numba_decline=True)
        assert agent2.reprompt_on_numba_decline is True

    def test_the_flag_is_keyword_only_and_the_old_positional_contract_holds(self):
        """agent-positional-30 (the review): the flag must not shift the positional slots that were valid before it existed.
        The eighteen positional arguments through ``mem_budget_bytes`` bind exactly as they did; the flag is keyword-only."""
        import inspect
        from solar.einsum_lib.agent import TranslationAgent
        sig = inspect.signature(TranslationAgent)
        p = sig.parameters["reprompt_on_numba_decline"]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is False
        assert list(sig.parameters)[-1] == "reprompt_on_numba_decline"
        args = ["fake-key", None, None, "test-model", 0.2, 1, None, 3, 60, True, True, 10000, True, "pytorch", True, False, 17, 123456789]
        bound = sig.bind(*args); bound.apply_defaults()
        assert bound.arguments["source_type"] == "pytorch" and bound.arguments["use_fp64"] is True and bound.arguments["debug"] is False
        assert bound.arguments["verify_timeout"] == 17 and bound.arguments["mem_budget_bytes"] == 123456789
        assert bound.arguments["reprompt_on_numba_decline"] is False
        with pytest.raises(TypeError):
            sig.bind(*(args + [False, None, 0.0, True]))          # a 22nd positional argument is refused: the flag cannot be passed by position

    # ---- round 10, agent-routing-scope-31: the routing is TYPED-DECLINE-ONLY (reviewed adjudication) ----

    @pytest.mark.parametrize("reason,klass", [
        ("RuntimeError: lowered execution failed", "fast_runtime_failure"),
        ("IndexError: index 3 is out of bounds for axis 0 with size 2", "fast_runtime_failure"),
        ("lowered code failed to import: NameError: missing", "lowered_import_failure"),
        ("lowered code defines no run_model()", "lowered_import_failure"),
        ("numba_lower could not lower the generated code", "lowering_unavailable"),
        ("numba is not installed", "lowering_unavailable"),
    ])
    @pytest.mark.parametrize("old_policy", [False, True])
    def test_an_untyped_fast_failure_keeps_execution_failed_under_both_policies(self, reason, klass, old_policy):
        from solar.einsum_lib import numba_accel as na
        agent, mock_client = _make_agent(max_retries=1, reprompt_on_numba_decline=old_policy)
        mock_client.chat.completions.create.return_value = _mock_openai_response(f"```python\n{GOOD_MATMUL_CODE}\n```")
        reference, inputs = _matmul_reference()

        def failure(*args, **kwargs):
            kwargs["on_fallback"](reason)
            return None

        with patch.object(na, "execute_fast", side_effect=failure):
            result = agent.translate("x @ w", MATMUL_SHAPES, reference_outputs=reference, all_inputs=inputs)
        assert not result.verified and result.verification_status == "execution_failed", (reason, old_policy, result)
        if old_policy:
            assert result.verification_error.startswith(f"Numba declined the generated code ({reason})")
        else:
            assert result.verification_error.startswith(f"The fast path failed without a typed decline [{klass}] ({reason})")

    def test_a_typed_decline_is_recognised_by_its_code_not_by_a_list(self):
        from solar.einsum_lib.numba_accel import is_typed_decline, untyped_fast_failure_class
        for typed in ("kernel_function_statement_unresolved:matmul: subscript: a '.data' attribute", "kernel_output_dtype_unestablished:k: x",
                      "kernel_allocation_callee_unresolved:k: module_binding_uncertain: y", "kernel_source_invalid:k: SyntaxError", "kernel_name_shadowed:k"):
            assert is_typed_decline(typed), typed
        for untyped in ("RuntimeError: lowered execution failed", "lowered code failed to import: NameError: missing", "numba_lower could not lower the generated code",
                        "numba is not installed", "", None, "kernelish: not a code", "Kernel_Upper:k"):
            assert not is_typed_decline(untyped), untyped
        assert untyped_fast_failure_class("numba is not installed") == "lowering_unavailable"
        assert untyped_fast_failure_class("lowered code failed to import: X") == "lowered_import_failure"
        assert untyped_fast_failure_class("IndexError: boom") == "fast_runtime_failure"

    def test_non_agent_callers_still_fall_back_on_every_fast_failure(self, capsys):
        """The typed-only policy is the AGENT's: ``execute_translated(fast=True)`` with both flags at their defaults keeps the
        Lite-caller contract -- any fast-path None (typed or not) falls back to the interpreter."""
        from solar.einsum_lib import numba_accel as na

        def failure(*args, **kwargs):
            kwargs["on_fallback"]("RuntimeError: lowered execution failed")
            return None

        reference, inputs = _matmul_reference()
        with patch.object(na, "execute_fast", side_effect=failure):
            out, _ = execute_translated(GOOD_MATMUL_CODE, MATMUL_SHAPES, all_inputs=inputs, fast=True)
        np.testing.assert_allclose(out["output"], reference["output"])
        assert "[WARN] Numba declined this code" in capsys.readouterr().out
        with patch.object(na, "execute_fast", side_effect=failure), pytest.raises(RuntimeError, match=r"without a typed decline \[fast_runtime_failure\]"):
            execute_translated(GOOD_MATMUL_CODE, MATMUL_SHAPES, all_inputs=inputs, fast=True, fail_on_untyped_fast_failure=True)

    def test_a_typed_decline_still_routes_to_the_interpreter_in_the_isolated_child(self):
        """With a timeout / memory budget the fast entry runs in the isolated child with the new flag; the typed decline of
        DECLINING_GOOD_CODE still verifies on the interpreter in one attempt."""
        agent, mock_client = _make_agent(max_retries=1, verify_timeout=30, mem_budget_bytes=1024 ** 3)
        mock_client.chat.completions.create.return_value = _mock_openai_response(f"```python\n{DECLINING_GOOD_CODE}\n```")
        reference, inputs = _matmul_reference()
        result = agent.translate("x @ w", MATMUL_SHAPES, reference_outputs=reference, all_inputs=inputs)
        assert result.verified and result.attempts == 1, result


# ---------------------------------------------------------------------------
# agent.py — API key resolution
# ---------------------------------------------------------------------------


class TestAPIKeyResolution:
    def test_explicit_key(self):
        agent, _ = _make_agent(api_key="sk-test")
        assert agent._api_key == "sk-test"

    def test_env_var(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-env"}):
            with patch("solar.einsum_lib.agent._openai_module") as mock_mod:
                mock_mod.OpenAI.return_value = MagicMock()
                from solar.einsum_lib.agent import TranslationAgent

                agent = TranslationAgent()
                assert agent._api_key == "sk-env"

    def test_key_from_file(self, tmp_path):
        key_file = tmp_path / "key.txt"
        key_file.write_text("sk-from-file\n")
        with patch("solar.einsum_lib.agent._openai_module") as mock_mod:
            mock_mod.OpenAI.return_value = MagicMock()
            from solar.einsum_lib.agent import TranslationAgent

            agent = TranslationAgent(api_key_file=str(key_file))
            assert agent._api_key == "sk-from-file"

    def test_missing_key_raises(self):
        # An endpoint is configured (reserved .invalid host, nothing can be
        # contacted) so that the refusal under test is the missing key, not
        # the missing endpoint checked first.
        env = {"SOLAR_API_BASE_URL": "https://solar-tests.invalid/v1", "SOLAR_LLM_MODEL": "solar-tests-model"}
        with patch.dict("os.environ", env, clear=True):
            with patch("solar.einsum_lib.agent._openai_module") as mock_mod:
                mock_mod.OpenAI.return_value = MagicMock()
                from solar.einsum_lib.agent import TranslationAgent

                with pytest.raises(ValueError, match="API key required"):
                    TranslationAgent()

    def test_no_implicit_key_file_fallback(self):
        # The public agent reads no key file it was not given: with the
        # environment cleared and no api_key_file, resolution yields None
        # whatever files exist next to the checkout.
        import solar.einsum_lib.agent as agent_mod
        assert not hasattr(agent_mod, "_DEFAULT_API_KEY_FILE")
        with patch.dict("os.environ", {}, clear=True):
            assert agent_mod.TranslationAgent._resolve_api_key(None) is None

    def test_missing_model_raises_with_instruction(self):
        # SOLAR ships no default model: without model= or SOLAR_LLM_MODEL the agent refuses after the endpoint
        # check and before any client is built, and the message says how to configure one
        env = {"SOLAR_API_BASE_URL": "https://solar-tests.invalid/v1", "OPENAI_API_KEY": "sk-env"}
        with patch.dict("os.environ", env, clear=True):
            with patch("solar.einsum_lib.agent._openai_module") as mock_mod:
                mock_mod.OpenAI.return_value = MagicMock()
                from solar.einsum_lib.agent import TranslationAgent
                with pytest.raises(ValueError, match="SOLAR_LLM_MODEL"):
                    TranslationAgent()
                mock_mod.OpenAI.assert_not_called()

    def test_model_from_environment_and_explicit_argument(self):
        env = {"SOLAR_API_BASE_URL": "https://solar-tests.invalid/v1", "OPENAI_API_KEY": "sk-env",
               "SOLAR_LLM_MODEL": "my-endpoint-model"}
        with patch.dict("os.environ", env, clear=True):
            with patch("solar.einsum_lib.agent._openai_module") as mock_mod:
                mock_mod.OpenAI.return_value = MagicMock()
                from solar.einsum_lib.agent import TranslationAgent
                assert TranslationAgent().model == "my-endpoint-model"
                assert TranslationAgent(model="explicit-model").model == "explicit-model"
        import solar.einsum_lib.agent as agent_mod
        assert agent_mod._DEFAULT_MODEL is None and agent_mod._FALLBACK_MODELS == []

    def test_missing_endpoint_raises_before_key_resolution(self):
        # SOLAR ships no inference endpoint: without base_url= or
        # SOLAR_API_BASE_URL the agent refuses even when a key is present and
        # never falls back to a public provider.
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-env"}, clear=True):
            with patch("solar.einsum_lib.agent._openai_module") as mock_mod:
                mock_mod.OpenAI.return_value = MagicMock()
                from solar.einsum_lib.agent import TranslationAgent

                with pytest.raises(ValueError, match="No inference endpoint configured"):
                    TranslationAgent()
                mock_mod.OpenAI.assert_not_called()

"""Focused tests for the Qwen3.5 ``target_verify`` kwarg forwarding.

Feature ``m14-qwen35-target-verify-forwarding`` closes the runtime-path
``target_verify=True`` blocker surfaced by the latest capped DFlash smoke.
``dflash_stream_generate`` passes ``target_verify=True`` on every target
call, and the patched Qwen3.5 wrapper chain must accept the kwarg and
forward it through to the inner model so the underlying attention / GDN
layers can route through the target-verification path.

Default-off preservation: ordinary text generation calls do not pass
``target_verify`` (or pass ``target_verify=False``) and the wrapper must
behave identically to the unpatched ``TextModel.__call__`` /
``Qwen3_5TextModel.__call__``.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import mlx.core as mx

from mlx_engine.model_kit.patches.qwen3_5 import (
    _patched_qwen3_5_language_model_call,
    _patched_qwen3_5_model_call,
    PatchedQwen3_5TextModel,
)


class _FakeInnerModel:
    """Records the kwargs passed by the patched wrappers.

    Mimics the inner ``Qwen3_5TextModel`` (``PatchedQwen3_5TextModel``)
    shape without going through ``__init__`` so the tests do not need a
    real MLX/Metal environment. The class exposes ``embed_tokens`` and
    the ``args``-style ``tie_word_embeddings`` flag used by the wrappers.
    """

    def __init__(self, captured: list):
        self._captured = captured
        self._counter = 0

        class _EmbedTokens:
            @staticmethod
            def as_linear(hidden):
                return f"logits:{hidden}"

        self.embed_tokens = _EmbedTokens()

    def __call__(self, *args, **kwargs):
        # Capture the call site: arguments + kwargs as observed by the
        # wrapper layer. ``_patched_qwen3_5_language_model_call`` calls
        # ``self.model(...)`` with explicit kwargs, so we record those.
        self._captured.append(
            {
                "args": args,
                "kwargs": dict(kwargs),
            }
        )
        self._counter += 1
        return f"hidden:{self._counter}"


class _FakeLanguageModel:
    """Wrapper shape used to drive ``_patched_qwen3_5_language_model_call``.

    Mirrors the attributes the wrapper reads (``model``, ``args``,
    ``lm_head``). The inner ``self.model`` is an instance of
    ``_FakeInnerModel`` so we can introspect the forwarded kwargs.
    """

    def __init__(self, captured: list, tie_word_embeddings: bool = True):
        self.model = _FakeInnerModel(captured)
        self._tie_word_embeddings = tie_word_embeddings

        class _Args:
            pass

        self.args = _Args()
        self.args.tie_word_embeddings = tie_word_embeddings

        # Plain logit identity fallback for the lm_head path.
        self.lm_head = lambda hidden: f"lm-head:{hidden}"

        # Callable entry point used by the outer
        # ``_patched_qwen3_5_model_call`` wrapper, which does
        # ``self.language_model(inputs, cache=..., ...)``.
        self._call_count = 0

    def __call__(self, *args, **kwargs):
        """Forward to the patched wrapper for end-to-end coverage."""
        self._call_count += 1
        return _patched_qwen3_5_language_model_call(
            self,
            *args,
            **kwargs,
        )


class _FakeOuterModel:
    """Outer ``Model`` shape used to drive ``_patched_qwen3_5_model_call``.

    The patched outer ``Model.__call__`` looks up
    ``self.language_model(...)`` and forwards kwargs to it. We capture
    the forwarded call so we can assert ``target_verify`` reaches the
    wrapper.
    """

    def __init__(self, captured: list):
        self._captured = captured
        self.language_model = _FakeLanguageModel(captured)
        self.args = SimpleNamespace(tie_word_embeddings=True)


class TestPatchedQwen3_5LanguageModelTargetVerifyForwarding(unittest.TestCase):
    """``_patched_qwen3_5_language_model_call`` accepts/forwards ``target_verify``."""

    def test_target_verify_true_is_accepted_and_forwarded(self):
        """``target_verify=True`` must reach the inner ``self.model(...)`` call.

        The DFlash runtime passes ``target_verify=True`` on every target
        call. The patched wrapper must accept the kwarg and forward it
        to the inner model so the attention / GDN layer
        ``target_verify`` branches are exercised. Without the
        forwarding, ``dflash_stream_generate`` raises ``TypeError``
        before any token can be emitted.
        """
        captured: list = []
        wrapper = _FakeLanguageModel(captured, tie_word_embeddings=False)
        inputs = mx.array([[1, 2, 3]])

        _patched_qwen3_5_language_model_call(
            wrapper,
            inputs,
            cache=None,
            input_embeddings=None,
            capture_layer_ids=None,
            hidden_sink=None,
            gdn_sink=None,
            target_verify=True,
        )

        self.assertEqual(len(captured), 1)
        forwarded_kwargs = captured[0]["kwargs"]
        self.assertIn("target_verify", forwarded_kwargs)
        self.assertTrue(forwarded_kwargs["target_verify"])

    def test_target_verify_false_default_is_forwarded(self):
        """``target_verify=False`` (default) must reach the inner model unchanged.

        Ordinary text generation does not opt into DFlash. The default
        ``target_verify=False`` value must still be forwarded through
        to the inner ``self.model(...)`` call so the attention / GDN
        layers see the explicit ``False`` flag (rather than the kwarg
        being absent, which would be ambiguous). This guards against
        silently dropping the kwarg during forwarding.
        """
        captured: list = []
        wrapper = _FakeLanguageModel(captured, tie_word_embeddings=False)
        inputs = mx.array([[1, 2, 3]])

        _patched_qwen3_5_language_model_call(
            wrapper,
            inputs,
            cache=None,
            input_embeddings=None,
            capture_layer_ids=None,
            hidden_sink=None,
            gdn_sink=None,
        )

        self.assertEqual(len(captured), 1)
        forwarded_kwargs = captured[0]["kwargs"]
        self.assertIn("target_verify", forwarded_kwargs)
        self.assertFalse(forwarded_kwargs["target_verify"])

    def test_existing_default_call_unchanged(self):
        """A call without target_verify must remain functionally unchanged.

        The patched wrapper preserves the unpatched TextModel.__call__
        semantics for ordinary text generation. This test guards
        against regressions in the call shape (positional args,
        explicit kwargs, lm_head vs embed_tokens.as_linear routing) by
        exercising the same kwargs the unpatched wrapper accepted and
        verifying the same internal call site is reached.
        """
        captured: list = []
        wrapper = _FakeLanguageModel(captured, tie_word_embeddings=True)
        inputs = mx.array([[7, 8, 9]])

        _patched_qwen3_5_language_model_call(
            wrapper,
            inputs,
            cache=None,
            input_embeddings=None,
            capture_layer_ids=None,
            hidden_sink=None,
            gdn_sink=None,
        )

        self.assertEqual(len(captured), 1)
        forwarded_kwargs = captured[0]["kwargs"]
        # All pre-existing explicit kwargs must still be forwarded.
        self.assertIsNone(forwarded_kwargs.get("cache"))
        self.assertIsNone(forwarded_kwargs.get("input_embeddings"))
        self.assertIsNone(forwarded_kwargs.get("capture_layer_ids"))
        self.assertIsNone(forwarded_kwargs.get("hidden_sink"))
        self.assertIsNone(forwarded_kwargs.get("gdn_sink"))
        # target_verify must default to False.
        self.assertFalse(forwarded_kwargs.get("target_verify"))

    def test_capture_requested_still_routes_through_inner_model(self):
        """Capture requests must still forward target_verify to the inner call.

        ``dflash_stream_generate`` passes capture_layer_ids, hidden_sink,
        and gdn_sink alongside target_verify. The wrapper must accept
        the capture kwargs together with target_verify and forward
        both to the inner model so DFlash hidden-state / GDN capture
        keeps working.
        """
        captured: list = []
        wrapper = _FakeLanguageModel(captured, tie_word_embeddings=False)
        inputs = mx.array([[4, 5]])
        capture_ids = [1, 10, 18]
        hidden_sink: list = []
        gdn_sink: list = []

        _patched_qwen3_5_language_model_call(
            wrapper,
            inputs,
            cache=None,
            input_embeddings=None,
            capture_layer_ids=capture_ids,
            hidden_sink=hidden_sink,
            gdn_sink=gdn_sink,
            target_verify=True,
        )

        self.assertEqual(len(captured), 1)
        forwarded_kwargs = captured[0]["kwargs"]
        self.assertEqual(forwarded_kwargs["capture_layer_ids"], capture_ids)
        self.assertIs(forwarded_kwargs["hidden_sink"], hidden_sink)
        self.assertIs(forwarded_kwargs["gdn_sink"], gdn_sink)
        self.assertTrue(forwarded_kwargs["target_verify"])


class TestPatchedQwen3_5OuterModelCallTargetVerifyForwarding(unittest.TestCase):
    """``_patched_qwen3_5_model_call`` (outer Model.__call__) forwards ``target_verify``."""

    def test_target_verify_true_is_forwarded_to_language_model(self):
        """The outer Model.__call__ wrapper must forward target_verify=True.

        ``dflash_stream_generate`` calls ``model_kit.model(...)`` which
        is the outer Model instance; its patched ``__call__`` is
        ``_patched_qwen3_5_model_call`` which uses ``**kwargs`` to
        forward the call to ``self.language_model``. The
        ``target_verify`` kwarg must reach
        ``_patched_qwen3_5_language_model_call`` unchanged so the
        wrapper chain stays consistent.
        """
        captured: list = []
        outer = _FakeOuterModel(captured)
        inputs = mx.array([[1, 2]])

        _patched_qwen3_5_model_call(
            outer,
            inputs,
            cache=None,
            input_embeddings=None,
            target_verify=True,
        )

        # One captured call: the outer forwarded the kwargs to the
        # wrapper, and the wrapper forwarded them to the inner fake.
        self.assertEqual(len(captured), 1)
        forwarded_kwargs = captured[0]["kwargs"]
        self.assertIn("target_verify", forwarded_kwargs)
        self.assertTrue(forwarded_kwargs["target_verify"])

    def test_target_verify_default_false_is_forwarded(self):
        """The default target_verify=False must reach the inner model too."""
        captured: list = []
        outer = _FakeOuterModel(captured)
        inputs = mx.array([[1, 2]])

        _patched_qwen3_5_model_call(
            outer,
            inputs,
            cache=None,
            input_embeddings=None,
        )

        self.assertEqual(len(captured), 1)
        forwarded_kwargs = captured[0]["kwargs"]
        self.assertIn("target_verify", forwarded_kwargs)
        self.assertFalse(forwarded_kwargs["target_verify"])


class TestPatchedQwen3_5TextModelInnerSignature(unittest.TestCase):
    """The inner ``PatchedQwen3_5TextModel.__call__`` accepts ``target_verify``."""

    def test_inner_call_accepts_target_verify_kwarg(self):
        """The patched inner ``Qwen3_5TextModel.__call__`` must accept ``target_verify``.

        Without ``target_verify`` in the inner signature, the
        forwarded call from the wrapper would raise ``TypeError``
        even after the wrapper change. Inspect the function signature
        directly so the test runs without a real MLX/Metal model.
        """
        import inspect

        signature = inspect.signature(PatchedQwen3_5TextModel.__call__)
        self.assertIn("target_verify", signature.parameters)
        target_verify_param = signature.parameters["target_verify"]
        self.assertEqual(target_verify_param.default, False)
        # The parameter must be keyword-only or accept a value (not
        # positional-only, since callers pass it as ``target_verify=...``).
        self.assertIn(
            target_verify_param.kind,
            (
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ),
        )

    def test_wrapper_signature_accepts_target_verify_kwarg(self):
        """The patched wrapper ``_patched_qwen3_5_language_model_call`` must accept it."""
        import inspect

        signature = inspect.signature(_patched_qwen3_5_language_model_call)
        self.assertIn("target_verify", signature.parameters)
        target_verify_param = signature.parameters["target_verify"]
        self.assertEqual(target_verify_param.default, False)

    def test_outer_model_call_signature_accepts_target_verify_kwarg(self):
        """The patched outer ``Model.__call__`` (``_patched_qwen3_5_model_call``)
        uses ``**kwargs`` so any new kwarg flows through; assert that."""
        import inspect

        signature = inspect.signature(_patched_qwen3_5_model_call)
        self.assertIn("kwargs", signature.parameters)


class TestTargetVerifyNoSurfaceWidening(unittest.TestCase):
    """``target_verify`` forwarding must not widen the DFlash surface."""

    def test_no_default_on_dflags_added_to_text_model(self):
        """The patched text model must not gain any default-on DFlash flags."""
        declared = set(dir(PatchedQwen3_5TextModel))
        for forbidden_attr in (
            "enable_dflash",
            "dflash_enabled",
            "rollback_default_on",
            "target_verify_default_on",
        ):
            self.assertNotIn(forbidden_attr, declared)

    def test_target_verify_kwarg_only_accepts_explicit_value(self):
        """The wrapper must accept ``target_verify`` only as an explicit kwarg.

        A caller that forgets to pass ``target_verify`` (i.e. the
        default behavior) must keep working unchanged. This guards
        against accidental ``target_verify=True`` activation through
        reflection or other side channels.
        """
        captured: list = []
        wrapper = _FakeLanguageModel(captured, tie_word_embeddings=False)

        # Plain positional call without target_verify must default to
        # target_verify=False at the wrapper boundary.
        _patched_qwen3_5_language_model_call(
            wrapper,
            mx.array([[1]]),
            cache=None,
        )
        self.assertFalse(captured[-1]["kwargs"]["target_verify"])

        # Explicit target_verify=True must be honored, not coerced.
        _patched_qwen3_5_language_model_call(
            wrapper,
            mx.array([[1]]),
            cache=None,
            target_verify=True,
        )
        self.assertTrue(captured[-1]["kwargs"]["target_verify"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

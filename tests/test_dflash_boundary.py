from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx

from mlx_engine.utils.generation_result import GenerationResult

DFLASH_EXPECTED_MODEL_TYPE = "qwen3"
DFLASH_EXPECTED_DTYPE = "bfloat16"
DFLASH_EXPECTED_LAYER_COUNT = 6
DFLASH_EXPECTED_VOCAB_SIZE = 248320
DFLASH_EXPECTED_BLOCK_SIZE = 16
DFLASH_EXPECTED_MASK_TOKEN_ID = 248077
DFLASH_EXPECTED_TARGET_LAYER_IDS = (1, 10, 18, 27, 35, 44, 52, 61)
DFLASH_EXPECTED_SAFETENSORS_FORMAT = "pt"


def _dflash_boundary():
    from mlx_engine.utils import dflash_boundary

    return dflash_boundary


def _dflash_snapshot():
    from mlx_engine.utils import dflash_snapshot

    return dflash_snapshot

REAL_DFLASH_TARGET = Path(
    "/Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/lmstudio-community/Qwen3.6-27B-MLX-8bit"
)
REAL_DFLASH_DRAFTER = Path(
    "/Volumes/StudioStackSSD4TB/Development/LLM/huggingface/hub/models--z-lab--Qwen3.5-27B-DFlash/snapshots/25ee0025ff950496a634e100b75c2db4515e9824"
)


class FakeDetokenizer:
    def __init__(self):
        self.text = ""
        self.offset = 0

    def add_token(self, token):
        self.text += str(token)

    def finalize(self):
        return None

    @property
    def last_segment(self):
        segment = self.text[self.offset :]
        self.offset = len(self.text)
        return segment


class FakeTokenizer:
    def __init__(self):
        self.bos_token = None
        self.chat_template = None
        self.clean_up_tokenization_spaces = False
        self.eos_token_id = 999
        self.eos_token_ids = [self.eos_token_id]
        self.detokenizer = FakeDetokenizer()

    def get_vocab(self):
        return {}

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, tokens):
        if isinstance(tokens, int):
            tokens = [tokens]
        return "".join(str(token) for token in tokens)


class _ImmediateFuture:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


class _ImmediateExecutor:
    def submit(self, fn, *args, **kwargs):
        return _ImmediateFuture(fn(*args, **kwargs))


class FakeSequentialKit:
    def __init__(self):
        self.model = object()
        self.tokenizer = FakeTokenizer()
        self.draft_model = None
        self.prefill_step_size = 8
        self.cache_wrapper = SimpleNamespace(
            cache=[SimpleNamespace(lengths=mx.array([1]))]
        )
        self.pending_requests = {}
        self.generation_lock = threading.Lock()
        self._executor = _ImmediateExecutor()
        self.max_kv_size = None
        self.kv_bits = None
        self.kv_group_size = None
        self.quantized_kv_start = None

    def is_shutdown(self):
        return False

    def process_prompt(
        self,
        prompt_tokens,
        images_b64,
        prompt_progress_reporter,
        generate_args,
        max_image_size,
        speculative_decoding_toggle=None,
        **_kwargs,
    ):
        generate_args["prompt_cache"] = self.cache_wrapper.cache
        return mx.array(prompt_tokens, dtype=mx.int32), None

    def is_cross_prompt_cache_active(self):
        return False

    def record_token_to_cache(self, _token):
        return None

    def cleanup_specprefill(self):
        return None


class KVCache:
    def __init__(self):
        self.history: list[int] = []


class RotatingKVCache:
    def __init__(self):
        self.max_size = 8
        self.keep = 0
        self.history: list[int] = []


class ArraysCache:
    """Mimic the real mlx-lm ``ArraysCache`` loaded by Qwen3.6 ModelKit.

    Real ``ArraysCache`` (single-sequence GDN linear-attention state) has
    ``lengths`` and ``left_padding`` attributes set to ``None`` with the
    GDN state stored in ``cache`` (a list of arrays). The validator
    matches on the class name ``ArraysCache`` to flag this as a ragged
    cache layer (the same check that previously rejected the
    ``m14_dflash_arrayscache_no_go`` blockers in the
    capped real-model smoke).
    """

    def __init__(self, layer_id: int = 0):
        self.layer_id = layer_id
        # Qwen3.5/3.6 GDN layers store conv_state in cache[0] and the
        # running gated-delta state in cache[1]. Both are mlx arrays
        # mutated in-place during the forward pass.
        self.cache = [
            mx.zeros((1, 3, 4), dtype=mx.bfloat16),
            mx.zeros((1, 4, 4), dtype=mx.bfloat16),
        ]
        # Real single-sequence ArraysCache has both attributes set to None.
        self.lengths = None
        self.left_padding = None


class ArraysCacheWithLengths:
    """Variant ArraysCache with non-None ``lengths``/``left_padding`` arrays.

    Mirrors the previous test fake so existing ragged-cache coverage
    still exercises the second validator branch
    (``lengths``/``left_padding`` not None).
    """

    def __init__(self):
        self.lengths = mx.array([1], dtype=mx.int32)
        self.left_padding = mx.array([0], dtype=mx.int32)
        self.history: list[int] = []


class BatchKVCache:
    def __init__(self):
        self.left_padding = mx.array([0], dtype=mx.int32)
        self.offset = mx.array([0], dtype=mx.int32)


def _runtime_model_kit(prompt_cache, **attrs):
    rollback_capable_model = SimpleNamespace(
        language_model=SimpleNamespace(
            rollback_speculative_cache=lambda *_args, **_kwargs: None
        )
    )
    return SimpleNamespace(
        model=rollback_capable_model,
        cache_wrapper=SimpleNamespace(cache=prompt_cache),
        prompt_cache=prompt_cache,
        draft_model=attrs.get("draft_model"),
        max_kv_size=attrs.get("max_kv_size"),
        kv_bits=attrs.get("kv_bits"),
        kv_group_size=attrs.get("kv_group_size"),
        quantized_kv_start=attrs.get("quantized_kv_start"),
    )


def _write_qwen_model_dir(
    base: Path,
    name: str,
    model_type: str,
    *,
    architectures: list[str] | None = None,
    include_config: bool = True,
    include_weights: bool = True,
    include_tokenizer_files: bool = False,
    vocab_size: int = 3,
    num_hidden_layers: int = 2,
) -> Path:
    model_dir = base / name
    model_dir.mkdir()
    if include_config:
        config: dict[str, object] = {
            "model_type": model_type,
            "vocab_size": vocab_size,
            "num_hidden_layers": num_hidden_layers,
            "text_config": {
                "dtype": "bfloat16",
                "num_hidden_layers": num_hidden_layers,
                "model_type": model_type,
            },
        }
        if architectures is not None:
            config["architectures"] = architectures
        elif "qwen" in model_type.lower():
            config["architectures"] = ["Qwen3_5ForConditionalGeneration"]
        else:
            config["architectures"] = [f"{model_type.title()}ForConditionalGeneration"]
        (model_dir / "config.json").write_text(json.dumps(config))
    if include_weights:
        (model_dir / "weights.safetensors").write_text("stub")
    if include_tokenizer_files:
        (model_dir / "tokenizer.json").write_text("{}")
        (model_dir / "tokenizer_config.json").write_text("{}")
        (model_dir / "vocab.json").write_text(
            json.dumps({f"token_{index}": index for index in range(vocab_size)})
        )
    return model_dir


def _build_dflash_config(
    *,
    model_type: str = DFLASH_EXPECTED_MODEL_TYPE,
    architectures: list[str] | None = None,
    dtype: str = DFLASH_EXPECTED_DTYPE,
    num_hidden_layers: int = DFLASH_EXPECTED_LAYER_COUNT,
    vocab_size: int = DFLASH_EXPECTED_VOCAB_SIZE,
    block_size: int = DFLASH_EXPECTED_BLOCK_SIZE,
    mask_token_id: int = DFLASH_EXPECTED_MASK_TOKEN_ID,
    target_layer_ids: list[int] | None = None,
    dflash_config: dict[str, object] | None = None,
) -> dict[str, object]:
    config: dict[str, object] = {
        "architectures": architectures or ["DFlashDraftModel"],
        "model_type": model_type,
        "dtype": dtype,
        "num_hidden_layers": num_hidden_layers,
        "vocab_size": vocab_size,
        "dflash_config": {
            "block_size": block_size,
            "mask_token_id": mask_token_id,
            "target_layer_ids": target_layer_ids
            or [1, 10, 18, 27, 35, 44, 52, 61],
        },
    }
    if dflash_config is not None:
        config["dflash_config"] = {**config["dflash_config"], **dflash_config}
    return config


def _write_dflash_snapshot(
    base: Path,
    name: str,
    *,
    config_overrides: dict[str, object] | None = None,
    tensor_dtype=mx.bfloat16,
    layer_count: int = DFLASH_EXPECTED_LAYER_COUNT,
    metadata_format: str = DFLASH_EXPECTED_SAFETENSORS_FORMAT,
) -> Path:
    snapshot_dir = base / name
    snapshot_dir.mkdir()
    config = _build_dflash_config()
    if config_overrides:
        dflash_config_overrides = config_overrides.get("dflash_config")
        if isinstance(dflash_config_overrides, dict):
            config["dflash_config"] = {
                **config["dflash_config"],
                **dflash_config_overrides,
            }
        for key, value in config_overrides.items():
            if key != "dflash_config":
                config[key] = value
    (snapshot_dir / "config.json").write_text(json.dumps(config))

    arrays = {
        "fc.weight": mx.zeros((2, 2), dtype=tensor_dtype),
        "hidden_norm.weight": mx.zeros((2,), dtype=tensor_dtype),
        "norm.weight": mx.zeros((2,), dtype=tensor_dtype),
    }
    for layer_index in range(layer_count):
        arrays[f"layers.{layer_index}.self_attn.q_proj.weight"] = mx.zeros(
            (2, 2),
            dtype=tensor_dtype,
        )
    mx.save_safetensors(snapshot_dir / "model.safetensors", arrays, {"format": metadata_format})
    return snapshot_dir


class TestDFlashOptions(unittest.TestCase):
    def test_defaults_off_without_env(self):
        boundary = _dflash_boundary()
        with patch.dict(os.environ, {}, clear=True):
            options = boundary.resolve_dflash_options(None, None, None, None)

        self.assertFalse(options.enabled)
        self.assertIsNone(options.target_model_path)
        self.assertIsNone(options.drafter_model_path)

    def test_env_opt_in_parses_explicit_pair(self):
        boundary = _dflash_boundary()
        with patch.dict(
            os.environ,
            {
                "MLX_ENGINE_DFLASH": "1",
                "MLX_ENGINE_DFLASH_TARGET_MODEL": "/tmp/qwen-target",
                "MLX_ENGINE_DFLASH_DRAFTER_MODEL": "/tmp/qwen-drafter",
                "MLX_ENGINE_DFLASH_MAX_DRAFT_TOKENS": "8",
            },
            clear=True,
        ):
            options = boundary.resolve_dflash_options(None, None, None, None)

        self.assertTrue(options.enabled)
        self.assertEqual(options.target_model_path, Path("/tmp/qwen-target"))
        self.assertEqual(options.drafter_model_path, Path("/tmp/qwen-drafter"))
        self.assertEqual(options.max_draft_tokens, 8)


class TestDFlashSurfaceValidation(unittest.TestCase):
    def test_rejects_unsupported_surfaces(self):
        boundary = _dflash_boundary()
        supported = boundary.validate_dflash_surface_compatibility(
            enabled=False,
            surface_label="sequential",
            images_b64=None,
            specprefill_toggle=None,
            speculative_decoding_toggle=None,
            num_draft_tokens=None,
            draft_model=None,
        )
        self.assertEqual(supported, ())

        cases = [
            {
                "surface_label": "batched-text",
                "images_b64": None,
                "specprefill_toggle": None,
                "speculative_decoding_toggle": None,
                "num_draft_tokens": None,
                "draft_model": None,
                "needle": "sequential text generation",
            },
            {
                "surface_label": "sequential",
                "images_b64": ["image"],
                "specprefill_toggle": None,
                "speculative_decoding_toggle": None,
                "num_draft_tokens": None,
                "draft_model": None,
                "needle": "VLM",
            },
            {
                "surface_label": "sequential",
                "images_b64": None,
                "specprefill_toggle": True,
                "speculative_decoding_toggle": None,
                "num_draft_tokens": None,
                "draft_model": None,
                "needle": "SpecPrefill",
            },
            {
                "surface_label": "sequential",
                "images_b64": None,
                "specprefill_toggle": None,
                "speculative_decoding_toggle": True,
                "num_draft_tokens": None,
                "draft_model": object(),
                "needle": "speculative decoding",
            },
            {
                "surface_label": "adapter",
                "images_b64": None,
                "specprefill_toggle": None,
                "speculative_decoding_toggle": None,
                "num_draft_tokens": None,
                "draft_model": None,
                "needle": "sequential text generation",
            },
            {
                "surface_label": "sequential",
                "images_b64": None,
                "specprefill_toggle": None,
                "speculative_decoding_toggle": None,
                "num_draft_tokens": 2,
                "draft_model": None,
                "needle": "num_draft_tokens",
            },
        ]

        for case in cases:
            with self.subTest(case=case["needle"]):
                blockers = boundary.validate_dflash_surface_compatibility(
                    enabled=True,
                    surface_label=case["surface_label"],
                    images_b64=case["images_b64"],
                    specprefill_toggle=case["specprefill_toggle"],
                    speculative_decoding_toggle=case["speculative_decoding_toggle"],
                    num_draft_tokens=case["num_draft_tokens"],
                    draft_model=case["draft_model"],
                )
                self.assertTrue(
                    any(case["needle"] in blocker for blocker in blockers),
                    msg=f"expected {case['needle']!r} in blockers: {blockers}",
                )

    def test_rejects_loaded_standard_draft_model_inputs(self):
        boundary = _dflash_boundary()
        blockers = boundary.validate_dflash_surface_compatibility(
            enabled=True,
            surface_label="sequential",
            images_b64=None,
            specprefill_toggle=None,
            speculative_decoding_toggle=True,
            num_draft_tokens=4,
            draft_model=object(),
            model_kit_draft_model=object(),
        )

        self.assertTrue(
            any("speculative decoding" in blocker for blocker in blockers),
            msg=f"expected speculative decoding blocker, got: {blockers}",
        )
        self.assertTrue(
            any("num_draft_tokens" in blocker for blocker in blockers),
            msg=f"expected num_draft_tokens blocker, got: {blockers}",
        )
        self.assertTrue(
            any("draft_model kwarg" in blocker for blocker in blockers),
            msg=f"expected draft_model kwarg blocker, got: {blockers}",
        )
        self.assertTrue(
            any("already loaded draft_model" in blocker for blocker in blockers),
            msg=f"expected loaded draft_model blocker, got: {blockers}",
        )

    def test_probe_requires_qwen_family_pairing(self):
        boundary = _dflash_boundary()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            target_dir = _write_qwen_model_dir(
                temp_dir,
                "target",
                "qwen3_5_text",
                include_tokenizer_files=True,
                vocab_size=3,
                num_hidden_layers=2,
            )
            drafter_dir = _write_qwen_model_dir(temp_dir, "drafter", "llama")

            report = boundary.probe_dflash_readiness(
                boundary.DFlashBoundaryOptions(
                    enabled=True,
                    target_model_path=target_dir,
                    drafter_model_path=drafter_dir,
                    max_draft_tokens=4,
                )
            )

        self.assertEqual(report.target_family, "qwen")
        self.assertIsNone(report.drafter_family)
        self.assertTrue(
            any("Qwen-family" in blocker for blocker in report.blockers),
            msg=f"expected Qwen-family blocker, got: {report.blockers}",
        )


class TestDFlashSnapshotLoader(unittest.TestCase):
    def test_loads_valid_dflash_snapshot_profile(self):
        snapshot = _dflash_snapshot()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            snapshot_dir = _write_dflash_snapshot(temp_dir, "valid-dflash")
            profile = snapshot.load_dflash_snapshot_profile(snapshot_dir)

        self.assertIsInstance(profile, snapshot.DFlashSnapshotProfile)
        self.assertEqual(profile.architectures, ("DFlashDraftModel",))
        self.assertEqual(profile.model_type, DFLASH_EXPECTED_MODEL_TYPE)
        self.assertEqual(profile.dtype, DFLASH_EXPECTED_DTYPE)
        self.assertEqual(profile.num_hidden_layers, DFLASH_EXPECTED_LAYER_COUNT)
        self.assertEqual(profile.vocab_size, DFLASH_EXPECTED_VOCAB_SIZE)
        self.assertEqual(profile.block_size, DFLASH_EXPECTED_BLOCK_SIZE)
        self.assertEqual(profile.mask_token_id, DFLASH_EXPECTED_MASK_TOKEN_ID)
        self.assertEqual(profile.target_layer_ids, DFLASH_EXPECTED_TARGET_LAYER_IDS)
        self.assertEqual(profile.safetensors_formats, ("pt",))
        self.assertEqual(profile.tensor_dtypes, ("bfloat16",))
        self.assertEqual(profile.tensor_layer_count, DFLASH_EXPECTED_LAYER_COUNT)
        self.assertEqual(profile.safetensors_paths, (snapshot_dir / "model.safetensors",))

    def test_rejects_non_dflash_snapshot(self):
        snapshot = _dflash_snapshot()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            snapshot_dir = _write_dflash_snapshot(
                temp_dir,
                "non-dflash",
                config_overrides={
                    "architectures": ["Qwen3_5ForConditionalGeneration"],
                    "model_type": "qwen3_5",
                },
            )

            with self.assertRaisesRegex(
                snapshot.DFlashSnapshotError,
                "DFlash config.architectures",
            ):
                snapshot.load_dflash_snapshot_profile(snapshot_dir)

    def test_rejects_invalid_safetensors_metadata(self):
        snapshot = _dflash_snapshot()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            snapshot_dir = _write_dflash_snapshot(
                temp_dir,
                "invalid-safetensors",
                tensor_dtype=mx.float32,
                layer_count=5,
            )

            with self.assertRaisesRegex(
                snapshot.DFlashSnapshotError,
                "DFlash weights must all use 'bfloat16' dtype",
            ):
                snapshot.load_dflash_snapshot_profile(snapshot_dir)

    def test_probe_accepts_valid_local_dflash_snapshot(self):
        boundary = _dflash_boundary()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            target_dir = _write_qwen_model_dir(
                temp_dir,
                "target",
                "qwen3_5_text",
                include_tokenizer_files=True,
                vocab_size=DFLASH_EXPECTED_VOCAB_SIZE,
                num_hidden_layers=64,
            )
            drafter_dir = _write_dflash_snapshot(temp_dir, "drafter")

            with patch(
                "mlx_engine.utils.dflash_boundary.probe_dflash_dependency",
                return_value=(True, ()),
            ), patch(
                "mlx_engine.utils.dflash_boundary._probe_reserved_port_conflicts",
                return_value=(),
            ), patch(
                "mlx_engine.utils.dflash_boundary._probe_available_memory_bytes",
                return_value=256 * 1024 * 1024 * 1024,
            ):
                report = boundary.probe_dflash_readiness(
                    boundary.DFlashBoundaryOptions(
                        enabled=True,
                        target_model_path=target_dir,
                        drafter_model_path=drafter_dir,
                        max_draft_tokens=4,
                    )
                )

        self.assertEqual(report.blockers, ())
        self.assertEqual(report.target_family, "qwen")
        self.assertEqual(report.drafter_family, "qwen")

    def test_probe_rejects_invalid_local_dflash_snapshot(self):
        boundary = _dflash_boundary()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            target_dir = _write_qwen_model_dir(
                temp_dir,
                "target",
                "qwen3_5_text",
                include_tokenizer_files=True,
                vocab_size=DFLASH_EXPECTED_VOCAB_SIZE,
                num_hidden_layers=64,
            )
            drafter_dir = _write_dflash_snapshot(
                temp_dir,
                "drafter",
                config_overrides={
                    "dtype": "float32",
                    "dflash_config": {"block_size": 8},
                },
            )

            with patch(
                "mlx_engine.utils.dflash_boundary.probe_dflash_dependency",
                return_value=(True, ()),
            ):
                report = boundary.probe_dflash_readiness(
                    boundary.DFlashBoundaryOptions(
                        enabled=True,
                        target_model_path=target_dir,
                        drafter_model_path=drafter_dir,
                        max_draft_tokens=4,
                    )
                )

        self.assertTrue(
            any("bfloat16" in blocker or "block_size" in blocker for blocker in report.blockers),
            msg=f"expected validation blocker details, got: {report.blockers}",
        )

    def test_probe_ignores_path_only_qwen_snapshot_without_metadata(self):
        boundary = _dflash_boundary()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            target_dir = _write_qwen_model_dir(
                temp_dir,
                "qwen-snapshot",
                "qwen3_5_text",
                include_config=False,
            )
            drafter_dir = _write_qwen_model_dir(temp_dir, "drafter", "qwen3_5_text")

            report = boundary.probe_dflash_readiness(
                boundary.DFlashBoundaryOptions(
                    enabled=True,
                    target_model_path=target_dir,
                    drafter_model_path=drafter_dir,
                    max_draft_tokens=4,
                )
            )

        self.assertIsNone(report.target_family)
        self.assertEqual(report.drafter_family, "qwen")
        self.assertTrue(
            any("Qwen-family" in blocker for blocker in report.blockers),
            msg=f"expected Qwen-family blocker, got: {report.blockers}",
        )


class TestDFlashRealPairPreflight(unittest.TestCase):
    def test_real_pair_preflight_accepts_target_and_drafter_metadata(self):
        boundary = _dflash_boundary()
        with patch(
            "mlx_engine.utils.dflash_boundary._probe_reserved_port_conflicts",
            return_value=(),
        ), patch(
            "mlx_engine.utils.dflash_boundary._probe_available_memory_bytes",
            return_value=256 * 1024 * 1024 * 1024,
        ):
            report = boundary.validate_dflash_preload_compatibility(
                options=boundary.DFlashBoundaryOptions(
                    enabled=True,
                    target_model_path=REAL_DFLASH_TARGET,
                    drafter_model_path=REAL_DFLASH_DRAFTER,
                    max_draft_tokens=4,
                ),
                loaded_model_path=REAL_DFLASH_TARGET,
                is_vlm_route=False,
                vocab_only=False,
                distributed=False,
                max_seq_nums=1,
                kv_bits=None,
                kv_group_size=None,
                quantized_kv_start=None,
                vlm_prompt_cache_storage_root=None,
                vlm_prompt_cache_min_save_tokens=None,
            )

        self.assertEqual(report.blockers, ())
        self.assertIsNotNone(report.target_profile)
        self.assertEqual(report.target_profile.model_path, REAL_DFLASH_TARGET)
        self.assertEqual(report.target_profile.vocab_size, DFLASH_EXPECTED_VOCAB_SIZE)
        self.assertGreater(report.target_profile.tokenizer_vocab_size, 0)
        self.assertLessEqual(
            report.target_profile.vocab_size - report.target_profile.tokenizer_vocab_size,
            1024,
        )
        self.assertEqual(report.drafter_family, "qwen")
        self.assertEqual(report.target_family, "qwen")
        self.assertGreater(report.target_profile.num_hidden_layers, max(DFLASH_EXPECTED_TARGET_LAYER_IDS))

    def test_load_model_fails_fast_before_heavy_model_creation(self):
        from mlx_engine import generate
        boundary = _dflash_boundary()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            model_dir = _write_qwen_model_dir(temp_dir, "target", "qwen3_5_text")

            def _fail_preflight(**_kwargs):
                raise boundary.DFlashUnavailableError(
                    "DFlash no-go: synthetic preflight failure"
                )

            with (
                patch(
                    "mlx_engine.generate.validate_dflash_preload_compatibility",
                    side_effect=_fail_preflight,
                ) as validate_preload,
                patch(
                    "mlx_engine.generate.ModelKit",
                    side_effect=AssertionError("heavy model load should not occur"),
                ),
            ):
                with self.assertRaisesRegex(
                    boundary.DFlashUnavailableError,
                    "synthetic preflight failure",
                ):
                    generate.load_model(
                        model_dir,
                        dflash_toggle=True,
                        dflash_target_model=model_dir,
                        dflash_drafter_model=REAL_DFLASH_DRAFTER,
                    )

        validate_preload.assert_called_once()

    def test_preload_compatibility_rejects_incompatible_route_and_cache_mode(self):
        boundary = _dflash_boundary()
        with patch(
            "mlx_engine.utils.dflash_boundary.probe_dflash_readiness",
            return_value=SimpleNamespace(
                enabled=True,
                dependency_available=True,
                target_family="qwen",
                drafter_family="qwen",
                target_profile=None,
                cache_mode_blockers=(),
                route_blockers=(),
                resource_blockers=(),
                blockers=(),
                listener_evidence=(),
            ),
        ):
            with self.assertRaisesRegex(
                boundary.DFlashUnavailableError,
                "sequential text generation",
            ):
                boundary.validate_dflash_preload_compatibility(
                    options=boundary.DFlashBoundaryOptions(
                        enabled=True,
                        target_model_path=REAL_DFLASH_TARGET,
                        drafter_model_path=REAL_DFLASH_DRAFTER,
                        max_draft_tokens=4,
                    ),
                    loaded_model_path=REAL_DFLASH_TARGET,
                    is_vlm_route=True,
                    vocab_only=False,
                    distributed=False,
                    max_seq_nums=4,
                    kv_bits=4,
                    kv_group_size=64,
                    quantized_kv_start=8,
                    vlm_prompt_cache_storage_root=Path("/tmp/cache-root"),
                    vlm_prompt_cache_min_save_tokens=512,
                )

    def test_runtime_validation_rejects_rollback_unsafe_cache_modes(self):
        boundary = _dflash_boundary()
        cases = [
            (
                {"max_kv_size": 16},
                [KVCache()],
                "max_kv_size",
            ),
            (
                {"kv_bits": 4},
                [KVCache()],
                "kv_bits",
            ),
            (
                {"kv_group_size": 32},
                [KVCache()],
                "kv_group_size",
            ),
            (
                {"quantized_kv_start": 8},
                [KVCache()],
                "quantized_kv_start",
            ),
            (
                {},
                [RotatingKVCache()],
                "bounded/rotating",
            ),
            (
                {},
                [ArraysCacheWithLengths()],
                "ragged",
            ),
            (
                {},
                [BatchKVCache()],
                "ragged",
            ),
        ]

        for attrs, prompt_cache, needle in cases:
            with self.subTest(case=needle):
                blockers = boundary.validate_dflash_runtime_compatibility(
                    _runtime_model_kit(prompt_cache, **attrs)
                )
                self.assertTrue(
                    any(needle in blocker for blocker in blockers),
                    msg=f"expected {needle!r} blocker, got: {blockers}",
                )


class TestDFlashArraysCacheNoGo(unittest.TestCase):
    """Focused tests proving the Qwen3.6 GDN/ArraysCache fail-closed.

    The capped real-model DFlash smoke
    (``reports/20260628T052638.586203Z-shared-bench.json``) reaches
    ``validate_dflash_runtime_compatibility`` after the Qwen3.6 target loads
    64 prompt-cache layers: 16 ``KVCache`` (full attention) + 48
    ``ArraysCache`` (GDN linear-attention state). The 48 ``ArraysCache``
    layers are flagged as ragged cache layers and DFlash stays fail-closed.

    Per feature ``m14 dflash gdn arrayscache runtime compatibility``: the
    rollback hook implemented in ``m14-qwen35-textmodel-dflash-rollback``
    handles ``history`` lists, ``keys``/``values`` arrays, and ``lengths``
    arrays, but it does NOT truncate the real ``ArraysCache.cache[idx]``
    arrays that hold the actual GDN state in single-sequence use. Until a
    follow-up feature adds GDN-aware rollback semantics for that shape,
    ``validate_dflash_runtime_compatibility`` MUST remain fail-closed for
    the Qwen3.6 GDN/ArraysCache layer combination.

    These tests document the no-go and prove the validator is correctly
    fail-closed for the real Qwen3.6 shape (not just the test fake with
    ``lengths`` set).
    """

    def test_real_qwen36_arrays_cache_shape_is_fail_closed(self):
        """Real Qwen3.6 ArraysCache (lengths=None, cache=[arrays]) fails closed.

        The real ``ArraysCache`` loaded by ModelKit for Qwen3.6 sequential
        text has ``lengths`` and ``left_padding`` attributes set to ``None``
        with the GDN state stored in ``cache`` (a list of arrays). The
        validator must classify this as a ragged cache layer and reject it
        so DFlash stays fail-closed until a tested rollback path exists.
        """
        boundary = _dflash_boundary()
        prompt_cache = [
            ArraysCache(layer_id=index) for index in range(48)
        ]
        kit = _runtime_model_kit(prompt_cache)

        blockers = boundary.validate_dflash_runtime_compatibility(kit)

        self.assertTrue(
            any("ragged cache" in blocker for blocker in blockers),
            msg=(
                "real Qwen3.6 ArraysCache (lengths=None) must be flagged "
                f"as ragged cache; got: {blockers}"
            ),
        )

    def test_mixed_kv_arrays_cache_layer_combo_is_fail_closed(self):
        """Mixed KVCache + real ArraysCache layers must fail closed.

        The real Qwen3.6 target produces 16 KVCache + 48 ArraysCache layers.
        Even when the KVCache subset is rollback-safe, the ArraysCache
        subset is not, so the prompt-cache layer list as a whole must
        remain fail-closed until per-layer GDN rollback is implemented.
        """
        boundary = _dflash_boundary()
        prompt_cache = [KVCache() for _ in range(16)] + [
            ArraysCache(layer_id=index) for index in range(48)
        ]
        kit = _runtime_model_kit(prompt_cache)

        blockers = boundary.validate_dflash_runtime_compatibility(kit)

        self.assertTrue(
            any("ragged cache" in blocker for blocker in blockers),
            msg=(
                "mixed KVCache + real Qwen3.6 ArraysCache layers must be "
                f"fail-closed; got: {blockers}"
            ),
        )

    def test_arrays_cache_with_lengths_array_remains_fail_closed(self):
        """ArraysCache with non-None ``lengths`` (ragged variant) fails closed.

        The ragged ``ArraysCache`` variant (with ``lengths`` and
        ``left_padding`` set to actual arrays) must continue to fail closed
        under the ragged-cache check. This guards against future
        refactors that might relax the ragged-cache check and silently
        allow ragged ArraysCache variants through.
        """
        boundary = _dflash_boundary()
        prompt_cache = [ArraysCacheWithLengths()]
        kit = _runtime_model_kit(prompt_cache)

        blockers = boundary.validate_dflash_runtime_compatibility(kit)

        self.assertTrue(
            any("ragged cache" in blocker for blocker in blockers),
            msg=(
                "ArraysCache with non-None lengths must remain "
                f"fail-closed; got: {blockers}"
            ),
        )

    def test_batch_kv_cache_remains_fail_closed(self):
        """BatchKVCache (batched-sequence ragged cache) remains fail-closed.

        BatchKVCache is the explicit batched-sequence ragged cache shape.
        DFlash sequential text never uses it; the validator must keep
        it fail-closed even when a rollback-capable Qwen3_5 TextModel is
        wired in.
        """
        boundary = _dflash_boundary()
        prompt_cache = [BatchKVCache()]
        kit = _runtime_model_kit(prompt_cache)

        blockers = boundary.validate_dflash_runtime_compatibility(kit)

        self.assertTrue(
            any("ragged cache" in blocker for blocker in blockers),
            msg=f"BatchKVCache must remain fail-closed; got: {blockers}",
        )

    def test_real_qwen36_arrays_cache_layer_count_matches_target(self):
        """Mirror the Qwen3.6 prompt-cache layout: 16 KVCache + 48 ArraysCache.

        The capped-smoke evidence (performance-future-work.md M14 entry)
        shows the Qwen3.6 27B target produces exactly 64 prompt-cache
        layers: 16 KVCache + 48 ArraysCache. This test guards against
        regressions in the validator that would silently allow those 48
        ArraysCache layers through.
        """
        boundary = _dflash_boundary()
        prompt_cache_layers: list = []
        # 16 full-attention KVCache layers (every full_attention_interval).
        for _ in range(16):
            prompt_cache_layers.append(KVCache())
        # 48 GDN linear-attention ArraysCache layers (Qwen3.6 sequential).
        for index in range(48):
            prompt_cache_layers.append(ArraysCache(layer_id=index))
        kit = _runtime_model_kit(prompt_cache_layers)

        blockers = boundary.validate_dflash_runtime_compatibility(kit)

        # At least one ragged-cache blocker must fire from the ArraysCache
        # subset. Without the no-go, the validator would silently allow the
        # 48 ArraysCache layers through and DFlash would corrupt GDN state.
        self.assertTrue(
            any("ragged cache" in blocker for blocker in blockers),
            msg=(
                "real Qwen3.6 16 KVCache + 48 ArraysCache layout must "
                f"fail closed; got: {blockers}"
            ),
        )


class TestDFlashRouting(unittest.TestCase):
    def test_default_off_uses_existing_stream_generate_path(self):
        from mlx_engine.generate import create_generator

        kit = FakeSequentialKit()
        stream_result = SimpleNamespace(
            text="ok",
            token=7,
            logprobs=mx.zeros((8,)),
            from_draft=False,
            finish_reason="length",
        )

        def fake_stream_generate(*_args, **_kwargs):
            yield stream_result

        with (
            patch(
                "mlx_engine.generate.probe_dflash_readiness",
                side_effect=AssertionError("DFlash probe should stay disabled"),
            ),
            patch(
                "mlx_engine.generate.stream_generate",
                side_effect=fake_stream_generate,
            ) as stream_generate,
        ):
            results = list(
                create_generator(
                    kit,
                    [1],
                    max_tokens=1,
                    request_id="dflash-default-off",
                )
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].tokens[0].id, 7)
        stream_generate.assert_called_once()

    def test_enabled_opt_in_fails_closed_with_missing_dependency(self):
        from mlx_engine.generate import create_generator

        kit = FakeSequentialKit()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            target_dir = _write_qwen_model_dir(
                temp_dir,
                "target",
                "qwen3_5_text",
                include_tokenizer_files=True,
                vocab_size=DFLASH_EXPECTED_VOCAB_SIZE,
                num_hidden_layers=64,
            )
            drafter_dir = _write_dflash_snapshot(temp_dir, "drafter")
            (drafter_dir / "model.safetensors").unlink()

            with self.assertRaisesRegex(
                _dflash_boundary().DFlashUnavailableError,
                "No safetensors weights found",
            ):
                create_generator(
                    kit,
                    [1],
                    max_tokens=1,
                    request_id="dflash-nogo",
                    dflash_toggle=True,
                    dflash_target_model=str(target_dir),
                    dflash_drafter_model=str(drafter_dir),
                )

    def test_enabled_opt_in_routes_through_dflash_runtime(self):
        from mlx_engine.generate import create_generator

        kit = FakeSequentialKit()
        stream_result = GenerationResult(
            text="native-dflash",
            tokens=[SimpleNamespace(id=17, text="17", logprob=0.0, from_draft=False)],
            top_logprobs=[],
            stop_condition=None,
        )

        def fake_dflash_stream_generate(*_args, **_kwargs):
            yield stream_result

        fake_report = SimpleNamespace(
            enabled=True,
            dependency_available=True,
            target_family="qwen",
            drafter_family="qwen",
            target_profile=None,
            cache_mode_blockers=(),
            route_blockers=(),
            resource_blockers=(),
            blockers=(),
            listener_evidence=(),
        )

        with (
            patch(
                "mlx_engine.generate.validate_dflash_postload_compatibility",
                return_value=fake_report,
            ),
            patch(
                "mlx_engine.generate.probe_dflash_readiness",
                return_value=fake_report,
            ),
            patch(
                "mlx_engine.generate.dflash_stream_generate",
                side_effect=fake_dflash_stream_generate,
            ) as dflash_stream_generate,
        ):
            results = list(
                create_generator(
                    kit,
                    [1],
                    max_tokens=1,
                    request_id="dflash-enabled",
                    dflash_toggle=True,
                    dflash_target_model="/tmp/qwen-target",
                    dflash_drafter_model="/tmp/qwen-drafter",
                )
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].tokens[0].id, 17)
        dflash_stream_generate.assert_called_once()

    def test_create_generator_uses_postload_validator_when_target_resident(self):
        from mlx_engine.generate import create_generator

        kit = FakeSequentialKit()
        stream_result = GenerationResult(
            text="native-dflash",
            tokens=[SimpleNamespace(id=21, text="21", logprob=0.0, from_draft=False)],
            top_logprobs=[],
            stop_condition=None,
        )

        captured_kwargs: dict[str, object] = {}

        def fake_dflash_stream_generate(*_args, **kwargs):
            captured_kwargs.update(kwargs)
            yield stream_result

        fake_report = SimpleNamespace(
            enabled=True,
            dependency_available=True,
            target_family="qwen",
            drafter_family="qwen",
            target_profile=None,
            cache_mode_blockers=(),
            route_blockers=(),
            resource_blockers=(),
            blockers=(),
            listener_evidence=(),
        )

        with (
            patch(
                "mlx_engine.generate.validate_dflash_postload_compatibility",
                return_value=fake_report,
            ) as postload_validator,
            patch(
                "mlx_engine.generate.validate_dflash_preload_compatibility",
                side_effect=AssertionError(
                    "create_generator must not re-run the preload validator"
                ),
            ),
            patch(
                "mlx_engine.generate.probe_dflash_readiness",
                side_effect=AssertionError(
                    "create_generator must use the wrapper, not the raw probe"
                ),
            ),
            patch(
                "mlx_engine.generate.dflash_stream_generate",
                side_effect=fake_dflash_stream_generate,
            ),
        ):
            results = list(
                create_generator(
                    kit,
                    [1],
                    max_tokens=1,
                    request_id="dflash-postload",
                    dflash_toggle=True,
                    dflash_target_model="/tmp/qwen-target",
                    dflash_drafter_model="/tmp/qwen-drafter",
                )
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].tokens[0].id, 21)
        postload_validator.assert_called_once()
        # Post-load validator must be called with the resolved loaded-model path
        # (FakeSequentialKit exposes neither model_path nor _model_path, so the
        # helper falls back to Path.cwd()).
        call_kwargs = postload_validator.call_args.kwargs
        self.assertEqual(call_kwargs["loaded_model_path"], Path.cwd())
        self.assertFalse(call_kwargs["is_vlm_route"])
        self.assertFalse(call_kwargs["distributed"])
        self.assertIsNone(call_kwargs["kv_bits"])
        self.assertIsNone(call_kwargs["vlm_prompt_cache_storage_root"])
        # The wrapped dflash_options must reach the runtime with the
        # original max_draft_tokens preserved.
        dflash_options = captured_kwargs.get("dflash_options")
        self.assertIsNotNone(dflash_options)
        self.assertEqual(dflash_options.max_draft_tokens, 4)


def _write_llmdynamix_config(
    base: Path,
    *,
    backends: list[dict[str, object]],
) -> Path:
    """Write a synthetic LLMDYNAMIX merged-config.yaml and return its path."""

    config_path = base / "merged-config.yaml"
    lines: list[str] = ["auth-dir: /tmp/.llmdynamix", "host: 127.0.0.1"]
    for backend in backends:
        url = backend["base_url"]
        name = backend["name"]
        lines.append("openai-compatibility:")
        lines.append(f"  - base-url: {url}")
        lines.append("    models:")
        for model in backend.get("models", []):
            lines.append(f"      - name: {model}")
        lines.append(f"    name: {name}")
    config_path.write_text("\n".join(lines) + "\n")
    return config_path


class TestDFlashLLMDYNAMIXListenerClassification(unittest.TestCase):
    def test_empty_port_is_allowed(self):
        boundary = _dflash_boundary()
        with patch(
            "mlx_engine.utils.dflash_boundary._port_is_listening",
            return_value=False,
        ):
            evidence = boundary.probe_listener_evidence(port=12444)
        self.assertEqual(evidence.classification, boundary.ListenerClassification.EMPTY)
        self.assertTrue(evidence.is_allowed())

    def test_cloud_only_llmdynamix_listener_is_allowed(self):
        boundary = _dflash_boundary()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            config_path = _write_llmdynamix_config(
                temp_dir,
                backends=[
                    {
                        "base_url": "https://api.anthropic.com/v1",
                        "name": "Anthropic",
                        "models": ["claude-sonnet-4-6"],
                    },
                    {
                        "base_url": "https://api.openai.com/v1",
                        "name": "OpenAI",
                        "models": ["gpt-5.4"],
                    },
                    {
                        "base_url": "https://generativelanguage.googleapis.com/v1",
                        "name": "Google",
                        "models": ["gemini-2.5-flash-lite"],
                    },
                ],
            )
            with patch(
                "mlx_engine.utils.dflash_boundary._port_is_listening",
                return_value=True,
            ), patch(
                "mlx_engine.utils.dflash_boundary._lookup_listener_pid",
                return_value=9001,
            ), patch(
                "mlx_engine.utils.dflash_boundary._lookup_process_command",
                return_value=(
                    "llmdynamix",
                    "/Applications/LLM Dynamix.app/Contents/MacOS/llmdynamix",
                ),
            ), patch(
                "mlx_engine.utils.dflash_boundary._list_llmdynamix_process_commands",
                return_value=(
                    (
                        9001,
                        "/Applications/LLM Dynamix.app/Contents/MacOS/llmdynamix",
                    ),
                    (
                        9002,
                        "/Applications/LLM Dynamix.app/Contents/Resources/"
                        "llmdynamix-engine "
                        f"-config {config_path}",
                    ),
                ),
            ):
                evidence = boundary.probe_listener_evidence(port=12444)

        self.assertEqual(
            evidence.classification,
            boundary.ListenerClassification.CLOUD_ONLY_LLMDYNAMIX,
        )
        self.assertTrue(evidence.is_allowed())
        self.assertGreaterEqual(evidence.cloud_backend_count, 3)
        self.assertEqual(evidence.local_heavy_backend_count, 0)
        self.assertEqual(evidence.config_path, config_path)
        self.assertTrue(
            any("cloud backend markers" in note for note in evidence.notes),
            msg=f"expected cloud-only note, got: {evidence.notes}",
        )

    def test_llmdynamix_with_unloaded_local_backends_is_allowed(self):
        boundary = _dflash_boundary()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            config_path = _write_llmdynamix_config(
                temp_dir,
                backends=[
                    {
                        "base_url": "https://api.anthropic.com/v1",
                        "name": "Anthropic",
                        "models": ["claude-sonnet-4-6"],
                    },
                    {
                        "base_url": "http://127.0.0.1:11434/v1",
                        "name": "Ollama",
                        "models": ["gemma4:latest"],
                    },
                    {
                        "base_url": "http://127.0.0.1:4521/v1",
                        "name": "LM Studio",
                        "models": ["qwen3.6-27b@q4_k_m"],
                    },
                ],
            )
            with patch(
                "mlx_engine.utils.dflash_boundary._port_is_listening",
                side_effect=lambda port: port in {12444, 11434},
            ), patch(
                "mlx_engine.utils.dflash_boundary._lookup_listener_pid",
                return_value=9100,
            ), patch(
                "mlx_engine.utils.dflash_boundary._lookup_process_command",
                return_value=(
                    "llmdynamix",
                    "/Applications/LLM Dynamix.app/Contents/MacOS/llmdynamix",
                ),
            ), patch(
                "mlx_engine.utils.dflash_boundary._list_llmdynamix_process_commands",
                return_value=(
                    (
                        9100,
                        "/Applications/LLM Dynamix.app/Contents/MacOS/llmdynamix",
                    ),
                    (
                        9101,
                        "/Applications/LLM Dynamix.app/Contents/Resources/"
                        "llmdynamix-engine "
                        f"-config {config_path}",
                    ),
                ),
            ), patch(
                "mlx_engine.utils.dflash_boundary._http_get_json",
                side_effect=[
                    {"models": []},
                    {"data": []},
                ],
            ):
                evidence = boundary.probe_listener_evidence(port=12444)

        self.assertEqual(
            evidence.classification,
            boundary.ListenerClassification.CLOUD_ONLY_LLMDYNAMIX,
        )
        self.assertTrue(evidence.is_allowed())
        self.assertTrue(
            any("live probing shows no loaded models" in note for note in evidence.notes),
            msg=f"expected live-probing note, got: {evidence.notes}",
        )

    def test_llmdynamix_with_loaded_local_ollama_is_blocked(self):
        boundary = _dflash_boundary()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            config_path = _write_llmdynamix_config(
                temp_dir,
                backends=[
                    {
                        "base_url": "https://api.anthropic.com/v1",
                        "name": "Anthropic",
                        "models": ["claude-sonnet-4-6"],
                    },
                    {
                        "base_url": "http://127.0.0.1:11434/v1",
                        "name": "Ollama",
                        "models": ["qwen3.6-27b"],
                    },
                ],
            )
            with patch(
                "mlx_engine.utils.dflash_boundary._port_is_listening",
                side_effect=lambda port: port in {12444, 11434},
            ), patch(
                "mlx_engine.utils.dflash_boundary._lookup_listener_pid",
                return_value=9200,
            ), patch(
                "mlx_engine.utils.dflash_boundary._lookup_process_command",
                return_value=(
                    "llmdynamix",
                    "/Applications/LLM Dynamix.app/Contents/MacOS/llmdynamix",
                ),
            ), patch(
                "mlx_engine.utils.dflash_boundary._list_llmdynamix_process_commands",
                return_value=(
                    (
                        9200,
                        "/Applications/LLM Dynamix.app/Contents/MacOS/llmdynamix",
                    ),
                    (
                        9201,
                        "/Applications/LLM Dynamix.app/Contents/Resources/"
                        "llmdynamix-engine "
                        f"-config {config_path}",
                    ),
                ),
            ), patch(
                "mlx_engine.utils.dflash_boundary._http_get_json",
                return_value={
                    "models": [
                        {"name": "qwen3.6-27b", "size": 27_000_000_000},
                    ]
                },
            ):
                evidence = boundary.probe_listener_evidence(port=12444)

        self.assertEqual(
            evidence.classification,
            boundary.ListenerClassification.LOCAL_MLX_METAL_HEAVY,
        )
        self.assertFalse(evidence.is_allowed())
        self.assertTrue(
            any("loaded model" in note for note in evidence.notes),
            msg=f"expected loaded-model note, got: {evidence.notes}",
        )

    def test_unknown_listener_process_is_blocked(self):
        boundary = _dflash_boundary()
        with patch(
            "mlx_engine.utils.dflash_boundary._port_is_listening",
            return_value=True,
        ), patch(
            "mlx_engine.utils.dflash_boundary._lookup_listener_pid",
            return_value=9300,
        ), patch(
            "mlx_engine.utils.dflash_boundary._lookup_process_command",
            return_value=("node", "node /tmp/random-server.js"),
        ), patch(
            "mlx_engine.utils.dflash_boundary._list_llmdynamix_process_commands",
            return_value=(),
        ):
            evidence = boundary.probe_listener_evidence(port=12444)

        self.assertEqual(
            evidence.classification,
            boundary.ListenerClassification.UNKNOWN_HEAVY,
        )
        self.assertFalse(evidence.is_allowed())

    def test_resource_blockers_skip_cloud_only_listener(self):
        boundary = _dflash_boundary()
        cloud_only_evidence = boundary.ListenerEvidence(
            port=12444,
            classification=boundary.ListenerClassification.CLOUD_ONLY_LLMDYNAMIX,
        )
        self.assertIsNone(boundary.build_port_blocker(cloud_only_evidence))

        empty_evidence = boundary.ListenerEvidence(
            port=12444,
            classification=boundary.ListenerClassification.EMPTY,
        )
        self.assertIsNone(boundary.build_port_blocker(empty_evidence))

        heavy_evidence = boundary.ListenerEvidence(
            port=12444,
            classification=boundary.ListenerClassification.LOCAL_MLX_METAL_HEAVY,
            pid=4242,
            comm="ollama",
        )
        blocker = boundary.build_port_blocker(heavy_evidence)
        self.assertIsNotNone(blocker)
        self.assertIn("127.0.0.1:12444", blocker)
        self.assertIn("local MLX/Metal-heavy", blocker)

    def test_probe_dflash_readiness_threads_listener_evidence(self):
        boundary = _dflash_boundary()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            target_dir = _write_qwen_model_dir(
                temp_dir,
                "target",
                "qwen3_5_text",
                include_tokenizer_files=True,
                vocab_size=DFLASH_EXPECTED_VOCAB_SIZE,
                num_hidden_layers=64,
            )
            drafter_dir = _write_dflash_snapshot(temp_dir, "drafter")

            fake_evidence = boundary.ListenerEvidence(
                port=12444,
                classification=boundary.ListenerClassification.CLOUD_ONLY_LLMDYNAMIX,
                pid=9001,
                notes=("synthetic cloud-only listener",),
            )
            with patch(
                "mlx_engine.utils.dflash_boundary.probe_dflash_dependency",
                return_value=(True, ()),
            ), patch(
                "mlx_engine.utils.dflash_boundary.probe_all_listener_evidence",
                return_value=(fake_evidence,),
            ), patch(
                "mlx_engine.utils.dflash_boundary._probe_available_memory_bytes",
                return_value=256 * 1024 * 1024 * 1024,
            ):
                report = boundary.probe_dflash_readiness(
                    boundary.DFlashBoundaryOptions(
                        enabled=True,
                        target_model_path=target_dir,
                        drafter_model_path=drafter_dir,
                        max_draft_tokens=4,
                    )
                )

        self.assertEqual(report.blockers, ())
        self.assertEqual(report.listener_evidence, (fake_evidence,))

    def test_real_pair_preflight_passes_with_cloud_only_listener_evidence(self):
        boundary = _dflash_boundary()
        fake_evidence = boundary.ListenerEvidence(
            port=12444,
            classification=boundary.ListenerClassification.CLOUD_ONLY_LLMDYNAMIX,
            notes=("synthetic cloud-only listener",),
        )
        with patch(
            "mlx_engine.utils.dflash_boundary.probe_dflash_dependency",
            return_value=(True, ()),
        ), patch(
            "mlx_engine.utils.dflash_boundary.probe_all_listener_evidence",
            return_value=(fake_evidence,),
        ), patch(
            "mlx_engine.utils.dflash_boundary._probe_available_memory_bytes",
            return_value=256 * 1024 * 1024 * 1024,
        ):
            report = boundary.validate_dflash_preload_compatibility(
                options=boundary.DFlashBoundaryOptions(
                    enabled=True,
                    target_model_path=REAL_DFLASH_TARGET,
                    drafter_model_path=REAL_DFLASH_DRAFTER,
                    max_draft_tokens=4,
                ),
                loaded_model_path=REAL_DFLASH_TARGET,
                is_vlm_route=False,
                vocab_only=False,
                distributed=False,
                max_seq_nums=1,
                kv_bits=None,
                kv_group_size=None,
                quantized_kv_start=None,
                vlm_prompt_cache_storage_root=None,
                vlm_prompt_cache_min_save_tokens=None,
            )

        self.assertEqual(report.blockers, ())
        self.assertIn(fake_evidence, report.listener_evidence)


class TestDFlashPhaseAwareMemoryAccounting(unittest.TestCase):
    """Pre-load vs post-load accounting must differ correctly.

    The pre-load preflight (called from ``load_model``) still requires target
    bytes + drafter bytes + headroom before any heavyweight load. The
    post-load preflight (called from ``create_generator`` after the target is
    already resident) must only require incremental drafter bytes + headroom
    so it does not double-count the Qwen3.6 target that ``load_model`` already
    paid for. All other fail-closed conditions (dependency, family,
    listeners, route, cache mode, vocab/layer matching) remain active in
    both phases.
    """

    # Mirror the real Qwen3.6 27B target / z-lab DFlash drafter footprint so
    # we exercise phase-aware accounting with realistic snapshot sizes without
    # requiring a 27 GB tempfile.
    REALISTIC_TARGET_BYTES = 27 * 1024 * 1024 * 1024
    REALISTIC_DRAFTER_BYTES = 4 * 1024 * 1024 * 1024

    def _build_options(
        self, target_dir: Path, drafter_dir: Path
    ):
        boundary = _dflash_boundary()
        return boundary.DFlashBoundaryOptions(
            enabled=True,
            target_model_path=target_dir,
            drafter_model_path=drafter_dir,
            max_draft_tokens=4,
        )

    def _estimate_safetensors_bytes(
        self,
        paths: tuple[Path, ...],
        realistic_size: int,
    ) -> int:
        """Sum the stub sizes and add a realistic_size offset on the first path.

        The fake snapshots are tiny stub safetensors. We piggyback a realistic
        size on the first path so the memory accounting produces a
        realistic-shape headroom check. The probe only sees the totals, so
        it cannot tell the difference.
        """

        total = realistic_size if paths else 0
        for index, path in enumerate(paths):
            if index == 0:
                # Skip the stub size of the first path since we replaced it
                # with the realistic_size offset above.
                continue
            try:
                total += path.stat().st_size
            except FileNotFoundError:
                continue
        return total

    def test_preload_accounts_for_target_and_drafter_together(self):
        boundary = _dflash_boundary()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            target_dir = _write_qwen_model_dir(
                temp_dir,
                "target",
                "qwen3_5_text",
                include_tokenizer_files=True,
                vocab_size=DFLASH_EXPECTED_VOCAB_SIZE,
                num_hidden_layers=64,
            )
            drafter_dir = _write_dflash_snapshot(temp_dir, "drafter")

            available = 8 * 1024 * 1024 * 1024  # 8 GiB residual
            target_bytes = self.REALISTIC_TARGET_BYTES
            drafter_bytes = self.REALISTIC_DRAFTER_BYTES

            preload_headroom = max(
                int(
                    (target_bytes + drafter_bytes)
                    * boundary.DFLASH_AVAILABLE_MEMORY_HEADROOM_RATIO
                ),
                boundary.DFLASH_AVAILABLE_MEMORY_HEADROOM_MIN_BYTES,
            )
            preload_required = target_bytes + drafter_bytes + preload_headroom

            self.assertGreater(preload_required, available)

            def _estimate(paths):
                # Identify which snapshot we're estimating by content size
                if not paths:
                    return 0
                first_path = paths[0]
                if "target" in str(first_path):
                    return self._estimate_safetensors_bytes(
                        paths, target_bytes
                    )
                return self._estimate_safetensors_bytes(paths, drafter_bytes)

            with patch(
                "mlx_engine.utils.dflash_boundary.probe_dflash_dependency",
                return_value=(True, ()),
            ), patch(
                "mlx_engine.utils.dflash_boundary.probe_all_listener_evidence",
                return_value=(),
            ), patch(
                "mlx_engine.utils.dflash_boundary._estimate_snapshot_bytes",
                side_effect=_estimate,
            ), patch(
                "mlx_engine.utils.dflash_boundary._probe_available_memory_bytes",
                return_value=available,
            ):
                preload_report = boundary.probe_dflash_readiness(
                    self._build_options(target_dir, drafter_dir),
                    target_resident=False,
                )

            self.assertTrue(
                any(
                    "real-pair DFlash preflight" in blocker
                    and "Insufficient free memory" in blocker
                    for blocker in preload_report.blockers
                ),
                msg=(
                    "expected pre-load blocker to mention real-pair DFlash "
                    f"preflight; got: {preload_report.blockers}"
                ),
            )
            self.assertTrue(
                any(
                    f"need at least {boundary._format_gib(preload_required)}" in blocker
                    for blocker in preload_report.blockers
                ),
                msg=(
                    "expected pre-load blocker to cite the combined "
                    f"target+drafter+headroom requirement; got: {preload_report.blockers}"
                ),
            )

    def test_postload_only_accounts_for_drafter_plus_headroom(self):
        boundary = _dflash_boundary()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            target_dir = _write_qwen_model_dir(
                temp_dir,
                "target",
                "qwen3_5_text",
                include_tokenizer_files=True,
                vocab_size=DFLASH_EXPECTED_VOCAB_SIZE,
                num_hidden_layers=64,
            )
            drafter_dir = _write_dflash_snapshot(temp_dir, "drafter")

            target_bytes = self.REALISTIC_TARGET_BYTES
            drafter_bytes = self.REALISTIC_DRAFTER_BYTES

            # 16 GiB residual post-load: large enough to fit incremental
            # drafter+headroom (4 GiB drafter + 8 GiB floor = 12 GiB), but
            # far too small to fit pre-load (27 + 4 + 8 = 39 GiB).
            available = 16 * 1024 * 1024 * 1024

            postload_headroom = max(
                int(drafter_bytes * boundary.DFLASH_AVAILABLE_MEMORY_HEADROOM_RATIO),
                boundary.DFLASH_AVAILABLE_MEMORY_HEADROOM_MIN_BYTES,
            )
            postload_required = drafter_bytes + postload_headroom

            preload_headroom = max(
                int(
                    (target_bytes + drafter_bytes)
                    * boundary.DFLASH_AVAILABLE_MEMORY_HEADROOM_RATIO
                ),
                boundary.DFLASH_AVAILABLE_MEMORY_HEADROOM_MIN_BYTES,
            )
            preload_required = target_bytes + drafter_bytes + preload_headroom

            self.assertLess(postload_required, available)
            self.assertGreater(preload_required, available)

            def _estimate(paths):
                if not paths:
                    return 0
                first_path = paths[0]
                if "target" in str(first_path):
                    return self._estimate_safetensors_bytes(paths, target_bytes)
                return self._estimate_safetensors_bytes(paths, drafter_bytes)

            with patch(
                "mlx_engine.utils.dflash_boundary.probe_dflash_dependency",
                return_value=(True, ()),
            ), patch(
                "mlx_engine.utils.dflash_boundary.probe_all_listener_evidence",
                return_value=(),
            ), patch(
                "mlx_engine.utils.dflash_boundary._estimate_snapshot_bytes",
                side_effect=_estimate,
            ), patch(
                "mlx_engine.utils.dflash_boundary._probe_available_memory_bytes",
                return_value=available,
            ):
                preload_report = boundary.probe_dflash_readiness(
                    self._build_options(target_dir, drafter_dir),
                    target_resident=False,
                )
                postload_report = boundary.probe_dflash_readiness(
                    self._build_options(target_dir, drafter_dir),
                    target_resident=True,
                )

            self.assertGreater(preload_required, available)
            self.assertLess(postload_required, preload_required)
            self.assertTrue(
                any(
                    "Insufficient free memory" in blocker
                    and "real-pair DFlash preflight" in blocker
                    for blocker in preload_report.blockers
                ),
                msg=(
                    "pre-load must block; got: "
                    f"{preload_report.blockers}"
                ),
            )
            # On the realistic post-load residual the drafter+headroom fits.
            self.assertEqual(postload_report.blockers, ())
            self.assertEqual(postload_report.resource_blockers, ())
            self.assertEqual(postload_report.target_family, "qwen")
            self.assertEqual(postload_report.drafter_family, "qwen")

    def test_postload_still_blocks_on_listener_and_route_failures(self):
        boundary = _dflash_boundary()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            target_dir = _write_qwen_model_dir(
                temp_dir,
                "target",
                "qwen3_5_text",
                include_tokenizer_files=True,
                vocab_size=DFLASH_EXPECTED_VOCAB_SIZE,
                num_hidden_layers=64,
            )
            drafter_dir = _write_dflash_snapshot(temp_dir, "drafter")
            options = self._build_options(target_dir, drafter_dir)

            heavy_evidence = boundary.ListenerEvidence(
                port=12444,
                classification=boundary.ListenerClassification.LOCAL_MLX_METAL_HEAVY,
                pid=4444,
                comm="ollama",
            )
            with patch(
                "mlx_engine.utils.dflash_boundary.probe_dflash_dependency",
                return_value=(True, ()),
            ), patch(
                "mlx_engine.utils.dflash_boundary.probe_all_listener_evidence",
                return_value=(heavy_evidence,),
            ), patch(
                "mlx_engine.utils.dflash_boundary._probe_available_memory_bytes",
                return_value=256 * 1024 * 1024 * 1024,
            ):
                with self.assertRaisesRegex(
                    boundary.DFlashUnavailableError,
                    "local MLX/Metal-heavy",
                ):
                    boundary.validate_dflash_postload_compatibility(
                        options=options,
                        loaded_model_path=target_dir,
                        is_vlm_route=False,
                        vocab_only=False,
                        distributed=False,
                        max_seq_nums=1,
                        kv_bits=None,
                        kv_group_size=None,
                        quantized_kv_start=None,
                        vlm_prompt_cache_storage_root=None,
                        vlm_prompt_cache_min_save_tokens=None,
                    )

            with patch(
                "mlx_engine.utils.dflash_boundary.probe_dflash_dependency",
                return_value=(True, ()),
            ), patch(
                "mlx_engine.utils.dflash_boundary.probe_all_listener_evidence",
                return_value=(),
            ), patch(
                "mlx_engine.utils.dflash_boundary._probe_available_memory_bytes",
                return_value=256 * 1024 * 1024 * 1024,
            ):
                with self.assertRaisesRegex(
                    boundary.DFlashUnavailableError,
                    "sequential text generation",
                ):
                    boundary.validate_dflash_postload_compatibility(
                        options=options,
                        loaded_model_path=target_dir,
                        is_vlm_route=True,
                        vocab_only=False,
                        distributed=False,
                        max_seq_nums=4,
                        kv_bits=4,
                        kv_group_size=64,
                        quantized_kv_start=8,
                        vlm_prompt_cache_storage_root=Path("/tmp/cache-root"),
                        vlm_prompt_cache_min_save_tokens=512,
                    )

    def test_postload_passes_with_cloud_only_listener_and_sufficient_memory(self):
        boundary = _dflash_boundary()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            target_dir = _write_qwen_model_dir(
                temp_dir,
                "target",
                "qwen3_5_text",
                include_tokenizer_files=True,
                vocab_size=DFLASH_EXPECTED_VOCAB_SIZE,
                num_hidden_layers=64,
            )
            drafter_dir = _write_dflash_snapshot(temp_dir, "drafter")
            options = self._build_options(target_dir, drafter_dir)

            fake_evidence = boundary.ListenerEvidence(
                port=12444,
                classification=boundary.ListenerClassification.CLOUD_ONLY_LLMDYNAMIX,
                notes=("synthetic cloud-only listener",),
            )
            with patch(
                "mlx_engine.utils.dflash_boundary.probe_dflash_dependency",
                return_value=(True, ()),
            ), patch(
                "mlx_engine.utils.dflash_boundary.probe_all_listener_evidence",
                return_value=(fake_evidence,),
            ), patch(
                "mlx_engine.utils.dflash_boundary._probe_available_memory_bytes",
                return_value=256 * 1024 * 1024 * 1024,
            ):
                report = boundary.validate_dflash_postload_compatibility(
                    options=options,
                    loaded_model_path=target_dir,
                    is_vlm_route=False,
                    vocab_only=False,
                    distributed=False,
                    max_seq_nums=1,
                    kv_bits=None,
                    kv_group_size=None,
                    quantized_kv_start=None,
                    vlm_prompt_cache_storage_root=None,
                    vlm_prompt_cache_min_save_tokens=None,
                )

        self.assertEqual(report.blockers, ())
        self.assertEqual(report.resource_blockers, ())
        self.assertIn(fake_evidence, report.listener_evidence)

    def test_postload_still_blocks_when_drafter_alone_exceeds_memory(self):
        boundary = _dflash_boundary()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            target_dir = _write_qwen_model_dir(
                temp_dir,
                "target",
                "qwen3_5_text",
                include_tokenizer_files=True,
                vocab_size=DFLASH_EXPECTED_VOCAB_SIZE,
                num_hidden_layers=64,
            )
            drafter_dir = _write_dflash_snapshot(temp_dir, "drafter")
            options = self._build_options(target_dir, drafter_dir)

            def _estimate(paths):
                # Use realistic drafter bytes so the drafter alone
                # overwhelms a tiny residual.
                return self._estimate_safetensors_bytes(
                    paths, self.REALISTIC_DRAFTER_BYTES
                )

            tiny_available = 256 * 1024 * 1024  # 256 MiB
            with patch(
                "mlx_engine.utils.dflash_boundary.probe_dflash_dependency",
                return_value=(True, ()),
            ), patch(
                "mlx_engine.utils.dflash_boundary.probe_all_listener_evidence",
                return_value=(),
            ), patch(
                "mlx_engine.utils.dflash_boundary._estimate_snapshot_bytes",
                side_effect=_estimate,
            ), patch(
                "mlx_engine.utils.dflash_boundary._probe_available_memory_bytes",
                return_value=tiny_available,
            ):
                with self.assertRaisesRegex(
                    boundary.DFlashUnavailableError,
                    "post-target-load DFlash preflight",
                ):
                    boundary.validate_dflash_postload_compatibility(
                        options=options,
                        loaded_model_path=target_dir,
                        is_vlm_route=False,
                        vocab_only=False,
                        distributed=False,
                        max_seq_nums=1,
                        kv_bits=None,
                        kv_group_size=None,
                        quantized_kv_start=None,
                        vlm_prompt_cache_storage_root=None,
                        vlm_prompt_cache_min_save_tokens=None,
                    )


class _FakeURLLibResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _make_urlopen_payload(payload: dict[str, object]) -> _FakeURLLibResponse:
    return _FakeURLLibResponse(json.dumps(payload).encode("utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)

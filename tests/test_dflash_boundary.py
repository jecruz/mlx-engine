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

from mlx_engine.generate import create_generator
from mlx_engine.utils.generation_result import GenerationResult
from mlx_engine.utils.dflash_boundary import (
    DFlashBoundaryOptions,
    DFlashUnavailableError,
    validate_dflash_runtime_compatibility,
    validate_dflash_preload_compatibility,
    probe_dflash_readiness,
    resolve_dflash_options,
    validate_dflash_surface_compatibility,
)
from mlx_engine.utils.dflash_snapshot import (
    DFlashSnapshotError,
    DFlashSnapshotProfile,
    DFLASH_EXPECTED_BLOCK_SIZE,
    DFLASH_EXPECTED_DTYPE,
    DFLASH_EXPECTED_LAYER_COUNT,
    DFLASH_EXPECTED_MASK_TOKEN_ID,
    DFLASH_EXPECTED_MODEL_TYPE,
    DFLASH_EXPECTED_SAFETENSORS_FORMAT,
    DFLASH_EXPECTED_TARGET_LAYER_IDS,
    DFLASH_EXPECTED_VOCAB_SIZE,
    load_dflash_snapshot_profile,
)


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
        with patch.dict(os.environ, {}, clear=True):
            options = resolve_dflash_options(None, None, None, None)

        self.assertFalse(options.enabled)
        self.assertIsNone(options.target_model_path)
        self.assertIsNone(options.drafter_model_path)

    def test_env_opt_in_parses_explicit_pair(self):
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
            options = resolve_dflash_options(None, None, None, None)

        self.assertTrue(options.enabled)
        self.assertEqual(options.target_model_path, Path("/tmp/qwen-target"))
        self.assertEqual(options.drafter_model_path, Path("/tmp/qwen-drafter"))
        self.assertEqual(options.max_draft_tokens, 8)


class TestDFlashSurfaceValidation(unittest.TestCase):
    def test_rejects_unsupported_surfaces(self):
        supported = validate_dflash_surface_compatibility(
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
                blockers = validate_dflash_surface_compatibility(
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
        blockers = validate_dflash_surface_compatibility(
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

            report = probe_dflash_readiness(
                DFlashBoundaryOptions(
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
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            snapshot_dir = _write_dflash_snapshot(temp_dir, "valid-dflash")
            profile = load_dflash_snapshot_profile(snapshot_dir)

        self.assertIsInstance(profile, DFlashSnapshotProfile)
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
                DFlashSnapshotError,
                "DFlash config.architectures",
            ):
                load_dflash_snapshot_profile(snapshot_dir)

    def test_rejects_invalid_safetensors_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            snapshot_dir = _write_dflash_snapshot(
                temp_dir,
                "invalid-safetensors",
                tensor_dtype=mx.float32,
                layer_count=5,
            )

            with self.assertRaisesRegex(
                DFlashSnapshotError,
                "DFlash weights must all use 'bfloat16' dtype",
            ):
                load_dflash_snapshot_profile(snapshot_dir)

    def test_probe_accepts_valid_local_dflash_snapshot(self):
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
                report = probe_dflash_readiness(
                    DFlashBoundaryOptions(
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
                report = probe_dflash_readiness(
                    DFlashBoundaryOptions(
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
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            target_dir = _write_qwen_model_dir(
                temp_dir,
                "qwen-snapshot",
                "qwen3_5_text",
                include_config=False,
            )
            drafter_dir = _write_qwen_model_dir(temp_dir, "drafter", "qwen3_5_text")

            report = probe_dflash_readiness(
                DFlashBoundaryOptions(
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
        with patch(
            "mlx_engine.utils.dflash_boundary._probe_reserved_port_conflicts",
            return_value=(),
        ), patch(
            "mlx_engine.utils.dflash_boundary._probe_available_memory_bytes",
            return_value=256 * 1024 * 1024 * 1024,
        ):
            report = validate_dflash_preload_compatibility(
                options=DFlashBoundaryOptions(
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

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            model_dir = _write_qwen_model_dir(temp_dir, "target", "qwen3_5_text")

            def _fail_preflight(**_kwargs):
                raise DFlashUnavailableError("DFlash no-go: synthetic preflight failure")

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
                    DFlashUnavailableError,
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
            ),
        ):
            with self.assertRaisesRegex(DFlashUnavailableError, "sequential text generation"):
                validate_dflash_preload_compatibility(
                    options=DFlashBoundaryOptions(
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
                [ArraysCache()],
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
                blockers = validate_dflash_runtime_compatibility(
                    _runtime_model_kit(prompt_cache, **attrs)
                )
                self.assertTrue(
                    any(needle in blocker for blocker in blockers),
                    msg=f"expected {needle!r} blocker, got: {blockers}",
                )


class TestDFlashRouting(unittest.TestCase):
    def test_default_off_uses_existing_stream_generate_path(self):
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
                DFlashUnavailableError,
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
        kit = FakeSequentialKit()
        stream_result = GenerationResult(
            text="native-dflash",
            tokens=[SimpleNamespace(id=17, text="17", logprob=0.0, from_draft=False)],
            top_logprobs=[],
            stop_condition=None,
        )

        def fake_dflash_stream_generate(*_args, **_kwargs):
            yield stream_result

        with (
            patch(
                "mlx_engine.generate.probe_dflash_readiness",
                return_value=SimpleNamespace(
                    enabled=True,
                    dependency_available=True,
                    target_family="qwen",
                    drafter_family="qwen",
                    blockers=(),
                ),
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


if __name__ == "__main__":
    unittest.main(verbosity=2)

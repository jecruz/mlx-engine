import json
from pathlib import Path

import pytest


class _FakeCache:
    def merge(self, caches):
        return caches


class _FakeSequentialModelKit:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.started = False

    def start(self):
        self.started = True


class _FakeBatchedModelKit(_FakeSequentialModelKit):
    pass


class _FakeBatchedVisionModelKit(_FakeSequentialModelKit):
    pass


class _FakeModel:
    pass


def _write_text_config(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen2"}))


@pytest.mark.parametrize(
    "max_seq_nums, expected_kit",
    [
        (None, _FakeSequentialModelKit),
        (4, _FakeBatchedModelKit),
    ],
)
def test_load_model_defaults_to_sequential_for_text(
    tmp_path, monkeypatch, max_seq_nums, expected_kit
):
    from mlx_engine import generate

    _write_text_config(tmp_path)
    monkeypatch.setattr(generate, "ModelKit", _FakeSequentialModelKit)
    monkeypatch.setattr(generate, "BatchedModelKit", _FakeBatchedModelKit)
    monkeypatch.setattr(generate, "sanitize_eos_tokens", lambda model_kit: None)
    monkeypatch.setattr(
        generate,
        "mlx_lm_load",
        lambda model_path, lazy=True: (_FakeModel(), None),
    )
    monkeypatch.setattr(generate, "make_prompt_cache", lambda model: [_FakeCache()])

    kit = generate.load_model(
        tmp_path,
        max_kv_size=4096,
        max_seq_nums=max_seq_nums,
    )

    assert isinstance(kit, expected_kit)
    assert kit.started is True
    if max_seq_nums is None:
        assert "max_seq_nums" not in kit.kwargs or kit.kwargs["max_seq_nums"] is None
    else:
        assert kit.kwargs["max_seq_nums"] == 4


@pytest.mark.parametrize(
    "prefill_step_size,prefill_step_size_was_unspecified,expected_step_size",
    [
        (None, True, 4096),
        (2048, False, 2048),
    ],
)
def test_load_model_defaults_to_faster_sequential_prefill_for_text(
    tmp_path,
    monkeypatch,
    prefill_step_size,
    prefill_step_size_was_unspecified,
    expected_step_size,
):
    from mlx_engine import generate

    _write_text_config(tmp_path)
    monkeypatch.setattr(generate, "ModelKit", _FakeSequentialModelKit)
    monkeypatch.setattr(generate, "BatchedModelKit", _FakeBatchedModelKit)
    monkeypatch.setattr(generate, "sanitize_eos_tokens", lambda model_kit: None)
    monkeypatch.setattr(
        generate,
        "mlx_lm_load",
        lambda model_path, lazy=True: (_FakeModel(), None),
    )
    monkeypatch.setattr(generate, "make_prompt_cache", lambda model: [_FakeCache()])
    monkeypatch.setattr(
        generate,
        "validate_prefill_step_size",
        lambda value=None: 2048 if value is None else value,
    )

    kit = generate.load_model(
        tmp_path,
        max_kv_size=4096,
        prefill_step_size=prefill_step_size,
    )

    assert isinstance(kit, _FakeSequentialModelKit)
    assert kit.started is True
    assert kit.kwargs["prefill_step_size"] == expected_step_size


def test_load_model_requires_vision_config_before_routing_to_vlm(tmp_path, monkeypatch):
    from mlx_engine import generate

    _write_text_config(tmp_path)
    monkeypatch.setattr(generate, "ModelKit", _FakeSequentialModelKit)
    monkeypatch.setattr(generate, "BatchedModelKit", _FakeBatchedModelKit)
    monkeypatch.setattr(
        generate, "_load_batched_vision_model_kit", lambda: _FakeBatchedVisionModelKit
    )
    monkeypatch.setattr(generate, "_is_known_vlm_model_type", lambda model_type: True)
    monkeypatch.setattr(generate, "sanitize_eos_tokens", lambda model_kit: None)
    monkeypatch.setattr(
        generate,
        "mlx_lm_load",
        lambda model_path, lazy=True: (_FakeModel(), None),
    )
    monkeypatch.setattr(generate, "make_prompt_cache", lambda model: [_FakeCache()])

    kit = generate.load_model(tmp_path, max_kv_size=4096)

    assert isinstance(kit, _FakeSequentialModelKit)
    assert not isinstance(kit, _FakeBatchedVisionModelKit)


def test_load_model_routes_known_vlm_with_vision_config_to_batched_vision(
    tmp_path, monkeypatch
):
    from mlx_engine import generate

    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "nemotron_h", "vision_config": {"foo": "bar"}})
    )
    monkeypatch.setattr(generate, "ModelKit", _FakeSequentialModelKit)
    monkeypatch.setattr(generate, "BatchedModelKit", _FakeBatchedModelKit)
    monkeypatch.setattr(
        generate, "_load_batched_vision_model_kit", lambda: _FakeBatchedVisionModelKit
    )
    monkeypatch.setattr(generate, "_is_known_vlm_model_type", lambda model_type: True)
    monkeypatch.setattr(generate, "sanitize_eos_tokens", lambda model_kit: None)
    monkeypatch.setattr(
        generate,
        "make_prompt_cache",
        lambda model: [_FakeCache()],
    )

    kit = generate.load_model(tmp_path, max_kv_size=4096)

    assert isinstance(kit, _FakeBatchedVisionModelKit)


def test_load_model_passes_persistent_prompt_cache_options_to_vlm(
    tmp_path, monkeypatch
):
    from mlx_engine import generate

    cache_root = tmp_path / "cache-root"
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "nemotron_h", "vision_config": {"foo": "bar"}})
    )
    monkeypatch.setattr(generate, "ModelKit", _FakeSequentialModelKit)
    monkeypatch.setattr(generate, "BatchedModelKit", _FakeBatchedModelKit)
    monkeypatch.setattr(
        generate, "_load_batched_vision_model_kit", lambda: _FakeBatchedVisionModelKit
    )
    monkeypatch.setattr(generate, "_is_known_vlm_model_type", lambda model_type: True)
    monkeypatch.setattr(generate, "sanitize_eos_tokens", lambda model_kit: None)

    kit = generate.load_model(
        model_dir,
        max_kv_size=4096,
        vlm_prompt_cache_storage_root=cache_root,
        vlm_prompt_cache_namespace="bench-model",
        vlm_prompt_cache_min_save_tokens=2048,
    )

    assert isinstance(kit, _FakeBatchedVisionModelKit)
    assert kit.kwargs["prompt_cache_storage_root"] == cache_root
    assert kit.kwargs["prompt_cache_namespace"] == "bench-model"
    assert kit.kwargs["prompt_cache_min_save_tokens"] == 2048


def test_load_model_rejects_vlm_prompt_cache_options_for_text_model(
    tmp_path, monkeypatch
):
    from mlx_engine import generate

    _write_text_config(tmp_path)
    monkeypatch.setattr(generate, "_is_known_vlm_model_type", lambda model_type: False)

    with pytest.raises(ValueError, match="only supported for VLM models"):
        generate.load_model(
            tmp_path,
            vlm_prompt_cache_storage_root=tmp_path / "cache-root",
        )

    with pytest.raises(ValueError, match="only supported for VLM models"):
        generate.load_model(
            tmp_path,
            vlm_prompt_cache_min_save_tokens=1024,
        )


def test_load_model_fails_dflash_preflight_before_heavy_model_load(
    tmp_path, monkeypatch
):
    from mlx_engine import generate

    _write_text_config(tmp_path)
    monkeypatch.setattr(generate, "_is_known_vlm_model_type", lambda model_type: False)

    def _raise_preflight(**_kwargs):
        raise generate.DFlashUnavailableError("DFlash no-go: preflight blocked")

    monkeypatch.setattr(
        generate,
        "validate_dflash_preload_compatibility",
        _raise_preflight,
    )
    monkeypatch.setattr(
        generate,
        "ModelKit",
        lambda *args, **kwargs: pytest.fail("model load should not happen"),
    )

    with pytest.raises(generate.DFlashUnavailableError, match="preflight blocked"):
        generate.load_model(
            tmp_path,
            dflash_toggle=True,
            dflash_target_model=tmp_path,
            dflash_drafter_model=tmp_path / "drafter",
        )

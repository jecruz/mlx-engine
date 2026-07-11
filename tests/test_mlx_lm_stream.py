from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _reset_mlx_lm_generate_generation_stream(monkeypatch):
    """Keep mlx_lm.generate.generation_stream isolated to each test.

    prepare_mlx_lm_generation_stream assigns to that global so tests in other
    files do not pick up a mock object via mlx_lm.generate(...).
    """
    from mlx_engine.utils import mlx_lm_stream

    monkeypatch.setattr(
        mlx_lm_stream.mlx_lm_generate,
        "generation_stream",
        None,
        raising=False,
    )


def test_prepare_stream_defaults_to_thread_local(monkeypatch):
    """Default text generation should keep using per-thread MLX streams."""
    from mlx_engine.utils import mlx_lm_stream

    fake_device = "gpu0"
    default_stream = object()
    local_stream = object()
    fake_thread = SimpleNamespace(name="worker-a", ident=101)
    fake_thread_state = SimpleNamespace()
    local_stream_calls = []

    monkeypatch.setattr(mlx_lm_stream, "_thread_state", fake_thread_state)
    monkeypatch.setattr(mlx_lm_stream, "_global_thread_unsafe_streams", {})
    monkeypatch.delenv("MLX_ENGINE_EXPERIMENTAL_THREAD_UNSAFE_STREAM", raising=False)
    monkeypatch.setattr(mlx_lm_stream.threading, "current_thread", lambda: fake_thread)
    monkeypatch.setattr(mlx_lm_stream.mx, "default_device", lambda: fake_device)
    monkeypatch.setattr(
        mlx_lm_stream.mx, "default_stream", lambda _device: default_stream
    )
    monkeypatch.setattr(
        mlx_lm_stream.mx,
        "new_thread_local_stream",
        lambda _device: local_stream_calls.append(_device) or local_stream,
        raising=False,
    )
    monkeypatch.delattr(mlx_lm_stream.mx, "new_thread_unsafe_stream", raising=False)

    prepared = mlx_lm_stream.prepare_mlx_lm_generation_stream(reason="unit-test")
    prepared_again = mlx_lm_stream.prepare_mlx_lm_generation_stream(reason="unit-test")

    assert prepared is local_stream
    assert prepared_again is local_stream
    assert local_stream_calls == [fake_device]
    assert mlx_lm_stream.mlx_lm_generate.generation_stream is local_stream


def test_prepare_stream_uses_toggle_file_when_env_is_unset(monkeypatch, tmp_path):
    """LM Studio live probes can enable the experiment through a file toggle."""
    from mlx_engine.utils import mlx_lm_stream

    fake_device = "gpu0"
    default_stream = object()
    unsafe_stream = object()
    unsafe_stream_calls = []
    toggle_file = tmp_path / "thread-unsafe-toggle"
    toggle_file.write_text("enabled\n")

    monkeypatch.setattr(
        mlx_lm_stream,
        "_THREAD_UNSAFE_STREAM_TOGGLE_FILE",
        toggle_file,
    )
    monkeypatch.delenv("MLX_ENGINE_EXPERIMENTAL_THREAD_UNSAFE_STREAM", raising=False)
    monkeypatch.setattr(mlx_lm_stream, "_thread_state", SimpleNamespace())
    monkeypatch.setattr(mlx_lm_stream, "_global_thread_unsafe_streams", {})
    monkeypatch.setattr(
        mlx_lm_stream.threading,
        "current_thread",
        lambda: SimpleNamespace(name="worker-a", ident=101),
    )
    monkeypatch.setattr(mlx_lm_stream.mx, "default_device", lambda: fake_device)
    monkeypatch.setattr(
        mlx_lm_stream.mx, "default_stream", lambda _device: default_stream
    )
    monkeypatch.setattr(
        mlx_lm_stream.mx,
        "new_thread_unsafe_stream",
        lambda _device: unsafe_stream_calls.append(_device) or unsafe_stream,
        raising=False,
    )
    monkeypatch.setattr(
        mlx_lm_stream.mx,
        "new_thread_local_stream",
        lambda _device: (_ for _ in ()).throw(
            AssertionError("unexpected local stream")
        ),
        raising=False,
    )

    prepared = mlx_lm_stream.prepare_mlx_lm_generation_stream(reason="unit-test")

    assert prepared is unsafe_stream
    assert unsafe_stream_calls == [fake_device]
    assert mlx_lm_stream.mlx_lm_generate.generation_stream is unsafe_stream


def test_describe_stream_configuration_reports_toggle_and_runtime(
    monkeypatch, tmp_path
):
    """The stream description should expose the live selection inputs."""
    from mlx_engine.utils import mlx_lm_stream

    toggle_file = tmp_path / "thread-unsafe-toggle"
    toggle_file.write_text("enabled\n")
    monkeypatch.setattr(
        mlx_lm_stream,
        "_THREAD_UNSAFE_STREAM_TOGGLE_FILE",
        toggle_file,
    )
    monkeypatch.delenv("MLX_ENGINE_EXPERIMENTAL_THREAD_UNSAFE_STREAM", raising=False)
    monkeypatch.setattr(
        mlx_lm_stream.mx,
        "new_thread_unsafe_stream",
        object(),
        raising=False,
    )

    description = mlx_lm_stream.describe_stream_configuration(False)

    assert "source=thread-unsafe" in description
    assert "use_default_stream=False" in description
    assert "toggle_env=True" in description
    assert "toggle_file=True" in description
    assert "runtime_supports_thread_unsafe=True" in description
    assert str(toggle_file) in description


def test_prepare_stream_uses_shared_thread_unsafe_stream_when_enabled(monkeypatch):
    """The experiment should share one stream across threads when supported."""
    from mlx_engine.utils import mlx_lm_stream

    fake_device = "gpu0"
    default_stream = object()
    unsafe_stream = object()
    unsafe_stream_calls = []

    monkeypatch.setenv("MLX_ENGINE_EXPERIMENTAL_THREAD_UNSAFE_STREAM", "1")
    monkeypatch.setattr(mlx_lm_stream, "_global_thread_unsafe_streams", {})
    monkeypatch.setattr(mlx_lm_stream.mx, "default_device", lambda: fake_device)
    monkeypatch.setattr(
        mlx_lm_stream.mx, "default_stream", lambda _device: default_stream
    )
    monkeypatch.setattr(
        mlx_lm_stream.mx,
        "new_thread_unsafe_stream",
        lambda _device: unsafe_stream_calls.append(_device) or unsafe_stream,
        raising=False,
    )
    monkeypatch.setattr(
        mlx_lm_stream.mx,
        "new_thread_local_stream",
        lambda _device: (_ for _ in ()).throw(
            AssertionError("unexpected local stream")
        ),
        raising=False,
    )

    monkeypatch.setattr(
        mlx_lm_stream.threading,
        "current_thread",
        lambda: SimpleNamespace(name="worker-a", ident=101),
    )
    monkeypatch.setattr(mlx_lm_stream, "_thread_state", SimpleNamespace())
    prepared = mlx_lm_stream.prepare_mlx_lm_generation_stream(reason="unit-test")

    monkeypatch.setattr(
        mlx_lm_stream.threading,
        "current_thread",
        lambda: SimpleNamespace(name="worker-b", ident=202),
    )
    monkeypatch.setattr(mlx_lm_stream, "_thread_state", SimpleNamespace())
    prepared_again = mlx_lm_stream.prepare_mlx_lm_generation_stream(reason="unit-test")

    assert prepared is unsafe_stream
    assert prepared_again is unsafe_stream
    assert unsafe_stream_calls == [fake_device]
    assert mlx_lm_stream.mlx_lm_generate.generation_stream is unsafe_stream


def test_prepare_stream_falls_back_when_runtime_lacks_thread_unsafe_api(monkeypatch):
    """The experiment must degrade cleanly on runtimes without the new API."""
    from mlx_engine.utils import mlx_lm_stream

    fake_device = "gpu0"
    default_stream = object()
    local_stream = object()
    local_stream_calls = []

    monkeypatch.setenv("MLX_ENGINE_EXPERIMENTAL_THREAD_UNSAFE_STREAM", "1")
    monkeypatch.setattr(mlx_lm_stream, "_thread_state", SimpleNamespace())
    monkeypatch.setattr(mlx_lm_stream, "_global_thread_unsafe_streams", {})
    monkeypatch.setattr(
        mlx_lm_stream.threading,
        "current_thread",
        lambda: SimpleNamespace(name="worker-a", ident=101),
    )
    monkeypatch.setattr(mlx_lm_stream.mx, "default_device", lambda: fake_device)
    monkeypatch.setattr(
        mlx_lm_stream.mx, "default_stream", lambda _device: default_stream
    )
    monkeypatch.setattr(
        mlx_lm_stream.mx,
        "new_thread_local_stream",
        lambda _device: local_stream_calls.append(_device) or local_stream,
        raising=False,
    )
    monkeypatch.delattr(mlx_lm_stream.mx, "new_thread_unsafe_stream", raising=False)

    prepared = mlx_lm_stream.prepare_mlx_lm_generation_stream(reason="unit-test")

    assert prepared is local_stream
    assert local_stream_calls == [fake_device]


def test_prepare_stream_keeps_default_stream_for_distributed_paths(monkeypatch):
    """Distributed callers should continue to use the device default stream."""
    from mlx_engine.utils import mlx_lm_stream

    fake_device = "gpu0"
    default_stream = object()

    monkeypatch.setenv("MLX_ENGINE_EXPERIMENTAL_THREAD_UNSAFE_STREAM", "1")
    monkeypatch.setattr(mlx_lm_stream, "_thread_state", SimpleNamespace())
    monkeypatch.setattr(mlx_lm_stream, "_global_thread_unsafe_streams", {})
    monkeypatch.setattr(
        mlx_lm_stream.threading,
        "current_thread",
        lambda: SimpleNamespace(name="worker-a", ident=101),
    )
    monkeypatch.setattr(mlx_lm_stream.mx, "default_device", lambda: fake_device)
    monkeypatch.setattr(
        mlx_lm_stream.mx, "default_stream", lambda _device: default_stream
    )
    monkeypatch.setattr(
        mlx_lm_stream.mx,
        "new_thread_unsafe_stream",
        lambda _device: (_ for _ in ()).throw(
            AssertionError("unexpected unsafe stream")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        mlx_lm_stream.mx,
        "new_thread_local_stream",
        lambda _device: (_ for _ in ()).throw(
            AssertionError("unexpected local stream")
        ),
        raising=False,
    )

    prepared = mlx_lm_stream.prepare_mlx_lm_generation_stream(
        reason="unit-test",
        use_default_stream=True,
    )

    assert prepared is default_stream

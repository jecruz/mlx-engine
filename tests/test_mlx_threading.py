from types import SimpleNamespace


def test_install_mlx_compile_cache_cleanup_for_thread_keeps_guard_alive(monkeypatch):
    """The MLX compile guard should remain attached to the current thread."""
    sentinel = object()
    fake_thread = SimpleNamespace()

    from mlx_engine.utils import mlx_threading

    monkeypatch.setattr(mlx_threading.mx, "compile", lambda fn: sentinel)
    monkeypatch.setattr(
        mlx_threading.threading, "current_thread", lambda: fake_thread
    )

    guard = mlx_threading.install_mlx_compile_cache_cleanup_for_thread()

    assert guard is sentinel
    assert fake_thread._mlx_compile_cache_cleanup_guard is sentinel

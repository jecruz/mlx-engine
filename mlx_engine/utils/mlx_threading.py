import threading

import mlx.core as mx


def install_mlx_compile_cache_cleanup_for_thread() -> object:
    """Install MLX's compile-cache cleanup guard on the current thread.

    The returned guard must stay referenced for the lifetime of the thread.
    Keeping it attached to the thread object avoids premature cleanup during
    shutdown on MLX 0.31.2-era runtimes.
    """
    # MLX 0.31.2 has a bug where a thread can call an mx.compile-created
    # function without installing the thread-local compiler-cache cleanup guard.
    # If that cache later releases Python output metadata during thread teardown,
    # it can do so without the GIL and crash the process. MLX main has fixed this
    # for 0.31.3 or greater by installing cleanup when compiled functions run.
    cleanup_guard = mx.compile(lambda x: x)
    setattr(
        threading.current_thread(),
        "_mlx_compile_cache_cleanup_guard",
        cleanup_guard,
    )
    return cleanup_guard

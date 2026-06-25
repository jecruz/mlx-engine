from mlx_engine.model_kit.model_kit import ModelKit


class _DummyFuture:
    def result(self):
        return None


class _DummyExecutor:
    def __init__(self):
        self.calls = []

    def submit(self, fn, *args, **kwargs):
        self.calls.append(fn)
        return _DummyFuture()


def test_modelkit_start_submits_startup_warmup():
    kit = ModelKit.__new__(ModelKit)
    kit._executor = _DummyExecutor()
    def warmup():
        pass

    kit._run_startup_warmup = warmup

    kit.start()

    assert kit._executor.calls[0].__name__ == "synchronize"
    assert kit._executor.calls[1] is warmup

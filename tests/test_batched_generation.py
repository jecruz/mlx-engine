from contextlib import nullcontext
import pytest
import io
import logging
import threading
import time
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

from mlx_engine.model_kit.batched_model_kit import BatchedModelKit
from mlx_engine.model_kit.batched_vision.model_kit import BatchedVisionModelKit
import mlx_engine.model_kit.batched_model_kit as model_kit_module

pytestmark = pytest.mark.model


MODEL_PATH = Path(
    "/Volumes/StudioStackSSD4TB/Development/LLM/lmstudio/lmstudio-community/Qwen3.5-9B-MLX-8bit"
)


@pytest.fixture
def model_kit():
    """Load model once for all tests."""
    from mlx_engine.generate import load_model, unload

    kit = load_model(model_path=MODEL_PATH, max_kv_size=4096, max_seq_nums=4, seed=0)
    yield kit
    unload(kit)


def test_batched_generation_max_tokens(model_kit):
    """Test that batched generation stops with token_limit when max_tokens is reached."""

    assert isinstance(model_kit, (BatchedModelKit, BatchedVisionModelKit))
    from mlx_engine.generate import create_generator, tokenize

    prompt = """<|im_start|>user
Write a short paragraph about the Eiffel Tower in Paris.<|im_end|>
<|im_start|>assistant
"""
    prompt_tokens = tokenize(model_kit, prompt)

    max_tokens = 5
    token_count = 0
    stop_condition = None

    for result in create_generator(
        model_kit=model_kit,
        prompt_tokens=prompt_tokens,
        seed=0,
        max_tokens=max_tokens,
        temp=0.0,
    ):
        token_count += len(result.tokens)
        if result.stop_condition:
            stop_condition = result.stop_condition
            break

    assert stop_condition is not None
    assert stop_condition.stop_reason == "token_limit"
    assert token_count <= max_tokens


def test_batched_generation_two_threads(model_kit):
    """Test batched generation with two concurrent threads."""

    assert isinstance(model_kit, (BatchedModelKit, BatchedVisionModelKit))
    from mlx_engine.generate import create_generator, tokenize

    # Define two different prompts with different topics
    prompts = [
        """<|im_start|>user
Write a short paragraph about the Eiffel Tower in Paris.<|im_end|>
<|im_start|>assistant
""",
        """<|im_start|>user
Explain how photosynthesis works in plants.<|im_end|>
<|im_start|>assistant
""",
    ]

    # Tokenize prompts
    prompt_tokens_list = [tokenize(model_kit, prompt) for prompt in prompts]

    # Storage for results from each thread
    results = {}

    def run_generation(thread_id: int, prompt_tokens: list):
        """Run generation in a thread and store the result."""
        # Use the thread name as the request_id
        request_id = str(thread_id)
        generated_text = ""

        for result in create_generator(
            model_kit=model_kit,
            prompt_tokens=prompt_tokens,
            request_id=request_id,
            seed=0,
            max_tokens=100,
            temp=0.0,
        ):
            generated_text += result.text
            if result.stop_condition:
                break

        results[thread_id] = generated_text

    # Create threads
    threads = [
        threading.Thread(
            target=run_generation,
            args=(i + 1, prompt_tokens),
        )
        for i, prompt_tokens in enumerate(prompt_tokens_list)
    ]

    # Measure wall time for concurrent execution
    start_time = time.perf_counter()

    for thread in threads:
        thread.start()

    # Wait for threads with timeout
    for thread in threads:
        thread.join(timeout=20.0)

    end_time = time.perf_counter()
    wall_time = end_time - start_time

    # Verify all threads completed
    for i, thread in enumerate(threads, 1):
        assert not thread.is_alive()
        assert i in results
        assert len(results[i]) > 0

    # Assert the concurrent runs produced distinct non-empty outputs.
    assert results[1] != results[2]
    assert "paris" in results[1].lower() or "eiffel" in results[1].lower()
    assert "photosynthesis" in results[2].lower() or "plant" in results[2].lower()

    # Print results
    print(f"\nWall time: {wall_time:.3f} seconds")
    print(f"\nThread 1 output:\n{results[1]}")
    print(f"\nThread 2 output:\n{results[2]}")


def test_startup_warmup_uses_synthetic_prompt_and_drains_generator(monkeypatch):
    """Default startup warmup primes bounded prompt shapes only."""

    class FakeBatchGenerator:
        def __init__(self):
            self.closed = False
            self.insert_calls = []
            self.next_calls = 0

        def insert(self, prompts, max_tokens, **kwargs):
            self.insert_calls.append((prompts, max_tokens, kwargs))

        def next(self):
            self.next_calls += 1
            if self.next_calls == 1:
                return [object()]
            return []

        def close(self):
            self.closed = True

    kit = object.__new__(BatchedModelKit)
    kit._prefill_step_size = 512
    kit._max_seq_nums = 4
    kit._max_kv_size = None
    kit._shutdown = SimpleNamespace(is_set=lambda: True, set=lambda: None)
    kit.tokenize = lambda prompt: [7]
    fake_generator = FakeBatchGenerator()
    batch_size_calls = []

    def fake_make_batch_generator(completion_batch_size=None):
        batch_size_calls.append(completion_batch_size)
        return fake_generator

    kit._make_batch_generator = fake_make_batch_generator

    synchronize_calls = []
    monkeypatch.setattr(
        model_kit_module.mx,
        "synchronize",
        lambda: synchronize_calls.append(True),
    )

    kit._run_startup_warmup()

    assert fake_generator.closed
    assert fake_generator.next_calls == 6
    assert synchronize_calls == [True, True, True, True, True]
    assert len(fake_generator.insert_calls) == 5
    assert batch_size_calls == [None, None, None, None, None]

    first_prompts, first_max_tokens, first_kwargs = fake_generator.insert_calls[0]
    assert first_max_tokens == [32]
    assert len(first_prompts) == 1
    assert len(first_prompts[0]) == 2
    assert first_prompts[0] == [7] * 2

    second_prompts, second_max_tokens, second_kwargs = fake_generator.insert_calls[1]
    assert second_max_tokens == [32]
    assert len(second_prompts) == 1
    assert len(second_prompts[0]) == 25
    assert second_prompts[0] == [7] * 25

    third_prompts, third_max_tokens, third_kwargs = fake_generator.insert_calls[2]
    assert third_max_tokens == [32, 32, 32, 32]
    assert len(third_prompts) == 4
    assert len(third_prompts[0]) == 25
    assert third_prompts[0] == [7] * 25

    fourth_prompts, fourth_max_tokens, fourth_kwargs = fake_generator.insert_calls[3]
    assert fourth_max_tokens == [32]
    assert len(fourth_prompts) == 1
    assert len(fourth_prompts[0]) == 513
    assert fourth_prompts[0] == [7] * 513

    fifth_prompts, fifth_max_tokens, fifth_kwargs = fake_generator.insert_calls[4]
    assert fifth_max_tokens == [32]
    assert len(fifth_prompts) == 1
    assert len(fifth_prompts[0]) == 896
    assert fifth_prompts[0] == [7] * 896


def test_startup_warmup_long_benchmark_shape_is_opt_in(monkeypatch):
    """The expensive long-prompt benchmark warmup only runs when requested."""

    class FakeBatchGenerator:
        def __init__(self):
            self.insert_calls = []

        def insert(self, prompts, max_tokens, **kwargs):
            self.insert_calls.append((prompts, max_tokens, kwargs))

        def next(self):
            return []

        def close(self):
            pass

    kit = object.__new__(BatchedModelKit)
    kit._prefill_step_size = 512
    kit._max_seq_nums = 1
    kit._shutdown = SimpleNamespace(is_set=lambda: True, set=lambda: None)
    kit.tokenize = lambda prompt: [7]
    fake_generator = FakeBatchGenerator()
    kit._make_batch_generator = lambda completion_batch_size=None: fake_generator

    monkeypatch.setenv("MLX_ENGINE_STARTUP_LONG_WARMUP", "1")
    monkeypatch.setattr(model_kit_module.mx, "synchronize", lambda: None)

    kit._run_startup_warmup()

    sixth_prompts, sixth_max_tokens, sixth_kwargs = fake_generator.insert_calls[5]
    assert sixth_max_tokens == [32]
    assert len(sixth_prompts) == 1
    assert len(sixth_prompts[0]) == 7162
    assert sixth_prompts[0] == [7] * 7162


def test_batched_model_signals_startup_before_optional_warmup(monkeypatch):
    """A loaded model is startup-ready before opportunistic warmup completes."""

    kit = object.__new__(BatchedModelKit)
    kit.model = object()
    kit._seed = 0
    kit._shutdown = SimpleNamespace(is_set=lambda: True, set=lambda: None)
    kit._startup_complete = threading.Event()
    kit._batch_results = {}

    monkeypatch.setattr(
        model_kit_module,
        "install_mlx_compile_cache_cleanup_for_thread",
        lambda: None,
    )
    monkeypatch.setattr(model_kit_module, "set_seed", lambda seed: None)
    monkeypatch.setattr(
        model_kit_module,
        "prepare_mlx_lm_generation_stream",
        lambda reason: None,
    )
    monkeypatch.delenv("MLX_ENGINE_BATCHED_TIMING", raising=False)

    warmup_state = []
    kit._run_startup_warmup = lambda: warmup_state.append(
        kit._startup_complete.is_set()
    )
    kit._make_batch_generator = lambda: object()

    kit._generate()

    assert warmup_state == [True]


def test_startup_warmup_skips_timing_when_diagnostics_disabled(monkeypatch):
    """Ensure disabled diagnostics do not add perf-counter overhead to warmup."""

    class FakeBatchGenerator:
        def insert(self, prompts, max_tokens, **kwargs):
            pass

        def next(self):
            return []

        def close(self):
            pass

    kit = object.__new__(BatchedModelKit)
    kit._prefill_step_size = 512
    kit._max_seq_nums = 1
    kit._shutdown = SimpleNamespace(is_set=lambda: True, set=lambda: None)
    kit.tokenize = lambda prompt: [7]
    kit._make_batch_generator = lambda completion_batch_size=None: FakeBatchGenerator()

    monkeypatch.delenv("MLX_ENGINE_BATCHED_TIMING", raising=False)
    monkeypatch.setattr(model_kit_module.mx, "synchronize", lambda: None)
    monkeypatch.setattr(
        model_kit_module.time,
        "perf_counter",
        lambda: pytest.fail("perf_counter should not run when timing is disabled"),
    )

    kit._run_startup_warmup()


def test_batched_model_uses_prompt_only_key_for_cross_request_cache(monkeypatch):
    """Batched prompt cache entries must exclude generated tokens."""

    class FakeDetokenizer:
        def __init__(self):
            self.last_segment = ""

        def add_token(self, token):
            self.last_segment += f"<{token}>"

        def finalize(self):
            pass

    class FakeLogprobValue:
        def __init__(self, value):
            self._value = value

        def item(self):
            return self._value

    class FakeLogprobs:
        def __getitem__(self, token):
            return FakeLogprobValue(-0.5)

    class FakeGenerationResponse:
        def __init__(self, uid, token, finish_reason, prompt_cache):
            self.uid = uid
            self.token = token
            self.finish_reason = finish_reason
            self.prompt_cache = prompt_cache
            self.logprobs = FakeLogprobs()

    class FakeBatchGenerator:
        def __init__(self, shutdown):
            self.stream = object()
            self._shutdown = shutdown
            self._served = False

        def next(self):
            if self._served:
                return [], []
            self._served = True
            self._shutdown.set()
            response = FakeGenerationResponse(
                uid=11,
                token=999,
                finish_reason="stop",
                prompt_cache=["cached-entry"],
            )
            return [], [response]

    class FakePromptCache:
        def __init__(self):
            self.fetch_calls = []
            self.insert_calls = []

        def fetch_nearest_cache(self, model_key, tokens):
            self.fetch_calls.append((model_key, list(tokens)))
            return None, list(tokens)

        def insert_cache(self, model_key, tokens, prompt_cache):
            self.insert_calls.append((model_key, list(tokens), prompt_cache))

    class ShutdownFlag:
        def __init__(self):
            self._is_set = False

        def is_set(self):
            return self._is_set

        def set(self):
            self._is_set = True

    shutdown = ShutdownFlag()
    prompt_cache = FakePromptCache()
    fake_generator = FakeBatchGenerator(shutdown)

    kit = object.__new__(BatchedModelKit)
    kit.model = object()
    kit._seed = 0
    kit._shutdown = shutdown
    kit._startup_complete = SimpleNamespace(set=lambda: None)
    request_queue = Queue()
    kit._batch_results = {
        11: {
            "cross_prompt_cache_key": [101, 102, 103],
            "live_cache_key": [101, 102, 103],
            "rqueue": request_queue,
            "detokenizer": FakeDetokenizer(),
            "top_logprobs": 0,
            "request_id": "req-1",
            "inserted_at": None,
            "cached_tokens": 0,
            "rest_tokens": 3,
            "first_token_logged": False,
        }
    }
    kit._prompt_cache = prompt_cache
    kit._requests = Queue()
    kit._run_startup_warmup = lambda: None
    kit._make_batch_generator = lambda completion_batch_size=None: fake_generator
    kit.tokenizer = SimpleNamespace(detokenizer=FakeDetokenizer())

    monkeypatch.setattr(
        model_kit_module,
        "install_mlx_compile_cache_cleanup_for_thread",
        lambda: None,
    )
    monkeypatch.setattr(model_kit_module, "set_seed", lambda seed: None)
    monkeypatch.setattr(
        model_kit_module,
        "prepare_mlx_lm_generation_stream",
        lambda reason: None,
    )
    monkeypatch.setattr(model_kit_module, "batched_timing_enabled", lambda: False)
    monkeypatch.setattr(model_kit_module.mx, "stream", lambda stream: nullcontext())

    kit._generate()

    assert prompt_cache.fetch_calls == []
    assert prompt_cache.insert_calls == [
        ("lmstudio", [101, 102, 103], ["cached-entry"])
    ]
    assert request_queue.get_nowait().token == 999
    assert request_queue.get_nowait() is None


def test_fast_prompt_cache_returns_isolated_clone():
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    from mlx_engine.model_kit.batched_model_kit import FastLRUPromptCache

    cache = FastLRUPromptCache()
    kv = KVCache()
    kv.keys = mx.zeros((1, 4, 3, 8), dtype=mx.float16)
    kv.values = mx.zeros((1, 4, 3, 8), dtype=mx.float16)
    kv.offset = 3

    cache.insert_cache("model", [1, 2, 3], [kv], cache_type="user")

    restored, rest = cache.fetch_nearest_cache("model", [1, 2, 3])

    assert rest == []
    assert restored is not None
    assert restored[0] is not kv
    assert restored[0].offset == 3

    restored[0].offset = 1

    assert kv.offset == 3


def test_batched_generation_trims_generated_tail_before_cross_prompt_cache_insert(
    monkeypatch,
):
    class FakeDetokenizer:
        last_segment = ""

        def add_token(self, token):
            self.last_segment += str(token)

        def finalize(self):
            return None

    class FakeLogprobs:
        def __getitem__(self, token):
            return SimpleNamespace(item=lambda: -0.5)

    class FakeCacheEntry:
        def __init__(self, offset):
            self.offset = offset

    class FakeGenerationResponse:
        def __init__(self, uid, token, finish_reason, prompt_cache):
            self.uid = uid
            self.token = token
            self.finish_reason = finish_reason
            self.prompt_cache = prompt_cache
            self.logprobs = FakeLogprobs()

    class FakeBatchGenerator:
        def __init__(self, shutdown):
            self.stream = object()
            self._shutdown = shutdown
            self._served = False

        def next(self):
            if self._served:
                return [], []
            self._served = True
            self._shutdown.set()
            response = FakeGenerationResponse(
                uid=11,
                token=999,
                finish_reason="stop",
                prompt_cache=[FakeCacheEntry(offset=5)],
            )
            return [], [response]

    class FakePromptCache:
        def __init__(self):
            self.insert_calls = []

        def insert_cache(self, model_key, tokens, prompt_cache):
            self.insert_calls.append((model_key, list(tokens), prompt_cache))

    class ShutdownFlag:
        def __init__(self):
            self._is_set = False

        def is_set(self):
            return self._is_set

        def set(self):
            self._is_set = True

    def fake_trim_prompt_cache(prompt_cache, num_to_trim):
        prompt_cache[0].offset -= num_to_trim
        return num_to_trim

    shutdown = ShutdownFlag()
    prompt_cache = FakePromptCache()
    fake_generator = FakeBatchGenerator(shutdown)

    kit = object.__new__(BatchedModelKit)
    kit.model = object()
    kit._seed = 0
    kit._shutdown = shutdown
    kit._startup_complete = SimpleNamespace(set=lambda: None)
    request_queue = Queue()
    kit._batch_results = {
        11: {
            "cross_prompt_cache_key": [101, 102, 103],
            "live_cache_key": [101, 102, 103],
            "rqueue": request_queue,
            "detokenizer": FakeDetokenizer(),
            "top_logprobs": 0,
            "request_id": "req-1",
            "inserted_at": None,
            "cached_tokens": 0,
            "rest_tokens": 3,
            "first_token_logged": False,
        }
    }
    kit._prompt_cache = prompt_cache
    kit._requests = Queue()
    kit._run_startup_warmup = lambda: None
    kit._make_batch_generator = lambda completion_batch_size=None: fake_generator
    kit.tokenizer = SimpleNamespace(detokenizer=FakeDetokenizer())

    monkeypatch.setattr(
        model_kit_module,
        "install_mlx_compile_cache_cleanup_for_thread",
        lambda: None,
    )
    monkeypatch.setattr(model_kit_module, "set_seed", lambda seed: None)
    monkeypatch.setattr(
        model_kit_module,
        "prepare_mlx_lm_generation_stream",
        lambda reason: None,
    )
    monkeypatch.setattr(model_kit_module, "batched_timing_enabled", lambda: False)
    monkeypatch.setattr(model_kit_module.mx, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(model_kit_module, "can_trim_prompt_cache", lambda cache: True)
    monkeypatch.setattr(model_kit_module, "trim_prompt_cache", fake_trim_prompt_cache)

    kit._generate()

    assert len(prompt_cache.insert_calls) == 1
    _, tokens, inserted_prompt_cache = prompt_cache.insert_calls[0]
    assert tokens == [101, 102, 103]
    assert inserted_prompt_cache[0].offset == 3


def test_batched_generation_skips_cross_prompt_cache_insert_when_tail_is_untrimmable(
    monkeypatch,
):
    class FakeDetokenizer:
        last_segment = ""

        def add_token(self, token):
            self.last_segment += str(token)

        def finalize(self):
            return None

    class FakeLogprobs:
        def __getitem__(self, token):
            return SimpleNamespace(item=lambda: -0.5)

    class FakeCacheEntry:
        def __init__(self, offset):
            self.offset = offset

    class FakeGenerationResponse:
        def __init__(self, uid, token, finish_reason, prompt_cache):
            self.uid = uid
            self.token = token
            self.finish_reason = finish_reason
            self.prompt_cache = prompt_cache
            self.logprobs = FakeLogprobs()

    class FakeBatchGenerator:
        def __init__(self, shutdown):
            self.stream = object()
            self._shutdown = shutdown
            self._served = False

        def next(self):
            if self._served:
                return [], []
            self._served = True
            self._shutdown.set()
            response = FakeGenerationResponse(
                uid=11,
                token=999,
                finish_reason="stop",
                prompt_cache=[FakeCacheEntry(offset=5)],
            )
            return [], [response]

    class FakePromptCache:
        def __init__(self):
            self.insert_calls = []

        def insert_cache(self, model_key, tokens, prompt_cache):
            self.insert_calls.append((model_key, list(tokens), prompt_cache))

    class ShutdownFlag:
        def __init__(self):
            self._is_set = False

        def is_set(self):
            return self._is_set

        def set(self):
            self._is_set = True

    shutdown = ShutdownFlag()
    prompt_cache = FakePromptCache()
    fake_generator = FakeBatchGenerator(shutdown)

    kit = object.__new__(BatchedModelKit)
    kit.model = object()
    kit._seed = 0
    kit._shutdown = shutdown
    kit._startup_complete = SimpleNamespace(set=lambda: None)
    request_queue = Queue()
    kit._batch_results = {
        11: {
            "cross_prompt_cache_key": [101, 102, 103],
            "live_cache_key": [101, 102, 103],
            "rqueue": request_queue,
            "detokenizer": FakeDetokenizer(),
            "top_logprobs": 0,
            "request_id": "req-1",
            "inserted_at": None,
            "cached_tokens": 0,
            "rest_tokens": 3,
            "first_token_logged": False,
        }
    }
    kit._prompt_cache = prompt_cache
    kit._requests = Queue()
    kit._run_startup_warmup = lambda: None
    kit._make_batch_generator = lambda completion_batch_size=None: fake_generator
    kit.tokenizer = SimpleNamespace(detokenizer=FakeDetokenizer())

    monkeypatch.setattr(
        model_kit_module,
        "install_mlx_compile_cache_cleanup_for_thread",
        lambda: None,
    )
    monkeypatch.setattr(model_kit_module, "set_seed", lambda seed: None)
    monkeypatch.setattr(
        model_kit_module,
        "prepare_mlx_lm_generation_stream",
        lambda reason: None,
    )
    monkeypatch.setattr(model_kit_module, "batched_timing_enabled", lambda: False)
    monkeypatch.setattr(model_kit_module.mx, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(model_kit_module, "can_trim_prompt_cache", lambda cache: False)

    kit._generate()

    assert prompt_cache.insert_calls == []


def test_batched_timing_log_is_disabled_by_default(monkeypatch, caplog):
    """Ensure batched timing diagnostics stay silent unless explicitly enabled."""
    monkeypatch.delenv("MLX_ENGINE_BATCHED_TIMING", raising=False)

    with caplog.at_level(logging.INFO, logger=model_kit_module.logger.name):
        model_kit_module.log_batched_timing(
            model_kit_module.logger, "request_insert", request_id="req-1"
        )

    assert "MLX_ENGINE_BATCHED_TIMING" not in caplog.text


def test_batched_timing_log_emits_structured_payload(monkeypatch):
    """Ensure enabled batched timing diagnostics are machine-readable."""
    monkeypatch.setenv("MLX_ENGINE_BATCHED_TIMING", "1")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    model_kit_module.logger.addHandler(handler)
    original_level = model_kit_module.logger.level
    model_kit_module.logger.setLevel(logging.INFO)

    try:
        model_kit_module.log_batched_timing(
            model_kit_module.logger,
            "request_insert",
            request_id="req-1",
            prompt_tokens=25,
            cached_tokens=16,
            rest_tokens=9,
        )
    finally:
        model_kit_module.logger.setLevel(original_level)
        model_kit_module.logger.removeHandler(handler)

    log_output = stream.getvalue()
    assert "MLX_ENGINE_BATCHED_TIMING" in log_output
    assert '"event": "request_insert"' in log_output
    assert '"request_id": "req-1"' in log_output
    assert '"prompt_tokens": 25' in log_output

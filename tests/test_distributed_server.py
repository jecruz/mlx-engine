from mlx_engine.distributed_server import format_prompt


class _FakeTokenizer:
    def __init__(self, has_thinking: bool):
        self.has_thinking = has_thinking
        self.calls = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "kwargs": kwargs,
            }
        )
        return "mock prompt"


class _FakeModelKit:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenize_calls = []

    def tokenize(self, prompt):
        self.tokenize_calls.append(prompt)
        return [1, 2, 3]


def test_format_prompt_disables_thinking_by_default():
    tokenizer = _FakeTokenizer(has_thinking=True)
    model_kit = _FakeModelKit(tokenizer)

    prompt_tokens = format_prompt(
        model_kit,
        {"messages": [{"role": "user", "content": "Hello"}]},
        {},
    )

    assert prompt_tokens == [1, 2, 3]
    assert tokenizer.calls == [
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "tokenize": False,
            "add_generation_prompt": True,
            "kwargs": {"enable_thinking": False},
        }
    ]
    assert model_kit.tokenize_calls == ["mock prompt"]


def test_format_prompt_honors_explicit_thinking_override():
    tokenizer = _FakeTokenizer(has_thinking=True)
    model_kit = _FakeModelKit(tokenizer)

    format_prompt(
        model_kit,
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {"enable_thinking": True},
        },
        {},
    )

    assert tokenizer.calls[0]["kwargs"] == {"enable_thinking": True}

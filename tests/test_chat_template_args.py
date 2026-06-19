from mlx_engine.utils.chat_template_args import resolve_chat_template_args


class _FakeTokenizer:
    def __init__(self, has_thinking: bool):
        self.has_thinking = has_thinking


def test_resolve_chat_template_args_disables_thinking_by_default():
    args = resolve_chat_template_args(_FakeTokenizer(True), {}, None)

    assert args == {"enable_thinking": False}


def test_resolve_chat_template_args_honors_explicit_override():
    args = resolve_chat_template_args(
        _FakeTokenizer(True),
        {},
        {"enable_thinking": True},
    )

    assert args == {"enable_thinking": True}


def test_resolve_chat_template_args_preserves_non_thinking_models():
    args = resolve_chat_template_args(_FakeTokenizer(False), {"foo": "bar"}, None)

    assert args == {"foo": "bar"}

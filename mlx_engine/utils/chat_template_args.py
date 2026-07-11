from __future__ import annotations

from typing import Any


def resolve_chat_template_args(
    tokenizer: Any,
    default_template_args: dict[str, Any],
    request_template_args: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge chat template args and disable thinking by default when supported."""
    template_args = dict(default_template_args)
    if isinstance(request_template_args, dict):
        template_args.update(request_template_args)
    if "enable_thinking" not in template_args and getattr(
        tokenizer, "has_thinking", False
    ):
        template_args["enable_thinking"] = False
    return template_args

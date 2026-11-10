"""DSPy engine that calls Claude through the official Anthropic SDK (LLM_PROVIDER=anthropic).

DSPy builds the prompt; this engine sends it with `client.beta.messages.create` and returns only the
answer text (Claude's thinking blocks are dropped, so DSPy's field parsing is unchanged).
Current Claude models reject temperature/top_p/top_k, so no sampling settings are sent; LLM_EFFORT
(low | medium | high | xhigh | max) controls how much the model thinks.
"""
from __future__ import annotations

import anthropic
from dspy.clients.engines.base import validate_request
from dspy.lm15 import Message, Response, TextPart, Usage, response_to_events

_FINISH = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length",
           "tool_use": "tool_call", "refusal": "content_filter"}


def _text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(getattr(p, "text", "") or "" for p in content)


class AnthropicEngine:
    def __init__(self, model: str, api_key: str | None, max_tokens: int, effort: str = ""):
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.client = anthropic.Anthropic(api_key=api_key or None)  # None: ANTHROPIC_API_KEY from the environment

    def complete(self, request) -> Response:
        validate_request(request)
        system = [_text(request.system)] if request.system else []
        messages = []
        for m in request.messages:
            if m.role in ("system", "developer"):
                system.append(_text(m.parts))
            else:
                messages.append({"role": m.role, "content": _text(m.parts)})
        kwargs = {"output_config": {"effort": self.effort}} if self.effort else {}
        r = self.client.beta.messages.create(
            model=self.model,
            max_tokens=request.config.max_tokens or self.max_tokens,
            system="\n\n".join(s for s in system if s) or anthropic.omit,
            messages=messages,
            # if the model declines a request, the API retries it on a fallback model in the same call
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            **kwargs,
        )
        if r.stop_reason == "refusal":
            raise RuntimeError(f"Claude declined the request: {getattr(r, 'stop_details', None)}")
        text = "".join(b.text for b in r.content if b.type == "text")
        return Response(id=r.id, model=r.model, message=Message.assistant([TextPart(text)]),
                        finish_reason=_FINISH.get(r.stop_reason, "stop"),
                        usage=Usage(input_tokens=r.usage.input_tokens, output_tokens=r.usage.output_tokens,
                                    total_tokens=r.usage.input_tokens + r.usage.output_tokens))

    def stream(self, request):
        return response_to_events(self.complete(request))

    def close(self) -> None:
        self.client.close()

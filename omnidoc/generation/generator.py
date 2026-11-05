"""
AnswerGenerator — LLM synthesis with strict context bounding.

Supports Claude 3.5 Sonnet (Anthropic) and GPT-4o (OpenAI).
Every generated answer must cite chunk IDs drawn from the retrieved
context; the system prompt enforces zero-hallucination behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

from omnidoc.generation.prompt_builder import PromptBuilder
from omnidoc.utils import get_logger

logger = get_logger(__name__)


class LLMBackend(str, Enum):
    CLAUDE = "claude"
    GPT4O = "gpt4o"


@dataclass
class GenerationResult:
    """Container returned by :class:`AnswerGenerator.generate`."""

    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)


class AnswerGenerator:
    """
    Generates grounded answers from retrieved context chunks.

    Parameters
    ----------
    backend:
        ``"claude"`` — Anthropic Claude 3.5 Sonnet.
        ``"gpt4o"``  — OpenAI GPT-4o.
    model_name:
        Override the default model name.
    temperature:
        Sampling temperature (default 0.0 for deterministic output).
    max_tokens:
        Maximum tokens in the generated response.
    """

    _DEFAULTS = {
        LLMBackend.CLAUDE: "claude-3-5-sonnet-20241022",
        LLMBackend.GPT4O: "gpt-4o",
    }

    def __init__(
        self,
        backend: LLMBackend | str = LLMBackend.CLAUDE,
        model_name: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        self.backend = LLMBackend(backend)
        self.model_name = model_name or self._DEFAULTS[self.backend]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_builder = PromptBuilder()
        logger.info("AnswerGenerator ready | backend=%s  model=%s", self.backend, self.model_name)

    def generate(self, query: str, context_chunks: List[Dict[str, Any]]) -> GenerationResult:
        """
        Synthesise an answer for *query* using *context_chunks*.

        Parameters
        ----------
        query:
            Original user question.
        context_chunks:
            Top-N reranked results from :class:`CrossEncoderReranker`.

        Returns
        -------
        GenerationResult
            Grounded answer with citation references.
        """
        system_prompt, user_message = self.prompt_builder.build(query, context_chunks)

        if self.backend == LLMBackend.CLAUDE:
            return self._call_claude(system_prompt, user_message)
        return self._call_openai(system_prompt, user_message)

    # ------------------------------------------------------------------

    def _call_claude(self, system: str, user: str) -> GenerationResult:
        try:
            import anthropic  # type: ignore

            client = anthropic.Anthropic()
            resp = client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            answer = resp.content[0].text
            return GenerationResult(
                answer=answer,
                model=self.model_name,
                usage={
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                },
            )
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

    def _call_openai(self, system: str, user: str) -> GenerationResult:
        try:
            import openai  # type: ignore

            client = openai.OpenAI()
            resp = client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            answer = resp.choices[0].message.content or ""
            return GenerationResult(
                answer=answer,
                model=self.model_name,
                usage={
                    "input_tokens": resp.usage.prompt_tokens,
                    "output_tokens": resp.usage.completion_tokens,
                },
            )
        except ImportError:
            raise ImportError("Install openai: pip install openai")

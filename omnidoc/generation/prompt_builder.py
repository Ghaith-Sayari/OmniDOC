"""
PromptBuilder — constructs the system + user messages for the LLM.

Enforces the zero-hallucination contract: the LLM is instructed to
answer ONLY from the supplied context and to include chunk-ID citations.
"""

from __future__ import annotations

from typing import Any, Dict, List

SYSTEM_PROMPT_TEMPLATE = """\
You are OmniDoc, an enterprise document assistant with zero-hallucination policy.

RULES (non-negotiable):
1. Answer ONLY using the CONTEXT provided below. Do NOT use external knowledge.
2. Every factual claim must be followed by a citation in the form [chunk:<chunk_id>].
3. If the answer cannot be found in the CONTEXT, respond: "I cannot answer this question
   based on the available documents."
4. Be concise, accurate, and structured. Use markdown when helpful.
5. Do not speculate, infer beyond the text, or paraphrase in a misleading way.
"""

USER_MESSAGE_TEMPLATE = """\
CONTEXT:
{context_block}

---
QUESTION: {query}

Provide a grounded answer with citations.
"""


class PromptBuilder:
    """Assembles system + user prompts from query and context chunks."""

    def build(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
    ) -> tuple[str, str]:
        """
        Parameters
        ----------
        query:
            User's question.
        context_chunks:
            List of reranked chunk dicts with at least ``id`` and ``text``.

        Returns
        -------
        (system_prompt, user_message)
        """
        context_block = self._format_context(context_chunks)
        user_message = USER_MESSAGE_TEMPLATE.format(
            context_block=context_block, query=query
        )
        return SYSTEM_PROMPT_TEMPLATE, user_message

    @staticmethod
    def _format_context(chunks: List[Dict[str, Any]]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            chunk_id = chunk.get("id", f"chunk_{i}")
            source = chunk.get("metadata", {}).get("source", "unknown")
            text = chunk.get("text", "").strip()
            parts.append(f"[chunk:{chunk_id}] (source: {source})\n{text}")
        return "\n\n---\n\n".join(parts)

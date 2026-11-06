"""
STEP 5 -- ANSWER   (question + retrieved chunks -> final answer from Claude)

Job: paste the retrieved chunks into a prompt as numbered sources and ask Claude
to answer ONLY from them, citing [1], [2]... This is the "G" (generation) in RAG.

Needs:  pip install anthropic   and   export ANTHROPIC_API_KEY=...
Without a key the app still works in retrieval-only mode (shows the chunks).
"""
import os

import config

SYSTEM = (
    "You answer questions using ONLY the numbered sources provided. "
    "Cite the sources you use like [1] or [2]. "
    "If the sources do not contain the answer, say you could not find it in the documents. "
    "Be concise."
)


def format_context(hits) -> str:
    blocks = []
    for n, (_, c) in enumerate(hits, 1):
        blocks.append(f"[{n}] ({c['doc']} | {c['location']} | {c['sub_type']})\n{c['text']}")
    return "\n\n".join(blocks)


def answer(question: str, hits) -> str | None:
    """Returns Claude's answer, or None if no API key / library is available."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        return None

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=config.MAX_ANSWER_TOKENS,
        system=SYSTEM,
        messages=[{"role": "user",
                   "content": f"Sources:\n{format_context(hits)}\n\nQuestion: {question}"}],
    )
    return msg.content[0].text

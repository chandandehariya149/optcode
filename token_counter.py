"""
token_counter.py
-----------------
Our first tool: measures how many tokens a conversation uses.
This is the baseline everything else will be compared against.
"""

import json
import tiktoken


def count_tokens(text: str, model_hint: str = "cl100k_base") -> int:
    """Count tokens in a piece of text using tiktoken."""
    enc = tiktoken.get_encoding(model_hint)
    return len(enc.encode(text))


def count_conversation_tokens(messages: list) -> dict:
    """
    Count tokens across a full conversation.
    messages = list of {"role": ..., "content": ...} dicts
    """
    breakdown = []
    total = 0

    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        tokens = count_tokens(content)
        breakdown.append({
            "index": i,
            "role": msg.get("role", "unknown"),
            "tokens": tokens,
            "chars": len(content),
        })
        total += tokens

    return {
        "total_tokens": total,
        "message_count": len(messages),
        "breakdown": breakdown,
    }


if __name__ == "__main__":
    sample_conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ]

    result = count_conversation_tokens(sample_conversation)
    print(json.dumps(result, indent=2))
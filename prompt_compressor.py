"""
prompt_compressor.py
---------------------
Day 4: Third compression technique.

Strips low-value bloat from text before it's sent to the model:
- Extra whitespace/blank lines
- Repeated/duplicate sentences
- Common filler phrases that add tokens but no meaning

This is the simplest technique but surprisingly effective on
real-world prompts (especially ones written/pasted by humans,
or system prompts that have grown messy over time).
"""

import re
from token_counter import count_tokens


# Filler phrases that add tokens but rarely change meaning.
# This list can grow over time as we find more patterns.
FILLER_PHRASES = [
    "please note that",
    "it is important to note that",
    "i just wanted to say that",
    "as i mentioned before",
    "in order to",
    "due to the fact that",
    "at this point in time",
    "for all intents and purposes",
]


def collapse_whitespace(text: str) -> str:
    """Collapse multiple blank lines and repeated spaces into single ones."""
    text = re.sub(r"[ \t]+", " ", text)          # multiple spaces/tabs -> one space
    text = re.sub(r"\n\s*\n+", "\n\n", text)      # multiple blank lines -> one blank line
    return text.strip()


def remove_duplicate_sentences(text: str) -> str:
    """
    Remove sentences that are exact duplicates of an earlier sentence
    in the same text (common when prompts get copy-pasted/edited over time).
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    seen = set()
    result = []
    for s in sentences:
        key = s.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(s)
    return " ".join(result)


def remove_filler_phrases(text: str) -> str:
    """Strip common low-value filler phrases (case-insensitive)."""
    for phrase in FILLER_PHRASES:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        text = pattern.sub("", text)
    return text


def compress_prompt(text: str) -> str:
    """Run all three cleanup steps in sequence."""
    text = remove_filler_phrases(text)
    text = remove_duplicate_sentences(text)
    text = collapse_whitespace(text)
    return text


def compare_compression(original: str) -> dict:
    """Compress the given text and report token savings."""
    compressed = compress_prompt(original)

    before_tokens = count_tokens(original)
    after_tokens = count_tokens(compressed)
    saved = before_tokens - after_tokens
    percent = (saved / before_tokens) * 100 if before_tokens > 0 else 0

    return {
        "before_tokens": before_tokens,
        "after_tokens": after_tokens,
        "saved_tokens": saved,
        "percent_saved": round(percent, 1),
        "compressed_text": compressed,
    }


if __name__ == "__main__":
    # A realistic messy prompt - the kind that grows over months of edits
    messy_prompt = """
    You are a helpful customer support assistant.


    Please note that you should always be polite to customers.
    You are a helpful customer support assistant.

    In order to resolve issues quickly, ask clarifying questions.
    It is important to note that refunds require a manager's approval.

    Due to the fact that customers get frustrated, stay calm and empathetic.
    At this point in time, we do not offer international shipping.


    Please note that you should always be polite to customers.
    """

    print("=== ORIGINAL PROMPT ===")
    print(repr(messy_prompt))
    print(f"\nOriginal tokens: {count_tokens(messy_prompt)}")

    result = compare_compression(messy_prompt)

    print("\n=== COMPRESSED PROMPT ===")
    print(result["compressed_text"])

    print(f"\nBefore: {result['before_tokens']} tokens")
    print(f"After: {result['after_tokens']} tokens")
    print(f"✅ Saved: {result['saved_tokens']} tokens ({result['percent_saved']}%)")
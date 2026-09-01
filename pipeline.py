"""
pipeline.py
------------
Day 5: The combined harness pipeline.

Takes a full realistic input - a system prompt, a conversation history,
and a tool/API result - and runs ALL THREE compression techniques
together:
  1. Prompt compression (whitespace/filler/duplicates) - on the system prompt
  2. History summarization (Gemini call) - on old conversation turns
  3. Tool-output trimming - on the API/tool result JSON

Then reports the total before/after token counts across everything.
This is the first version of our actual "harness" as a single unit.
"""

import json
from token_counter import count_conversation_tokens, count_tokens
from prompt_compressor import compress_prompt
from tool_trimmer import trim_json
from summarizer import compress_conversation


def run_pipeline(system_prompt: str, conversation: list, tool_result: dict, tool_keep_keys: list, keep_recent: int = 4):
    """
    Runs the full harness pipeline on a realistic bundle of inputs.
    Returns before/after stats for each technique plus a combined total.
    """
    # ---- BEFORE: measure everything raw ----
    before_system_tokens = count_tokens(system_prompt)
    before_conversation = count_conversation_tokens(conversation)
    before_tool_text = json.dumps(tool_result)
    before_tool_tokens = count_tokens(before_tool_text)

    before_total = before_system_tokens + before_conversation["total_tokens"] + before_tool_tokens

    # ---- APPLY: run each technique ----
    compressed_system_prompt = compress_prompt(system_prompt)
    compressed_conversation = compress_conversation(conversation, keep_recent=keep_recent)
    trimmed_tool_result = trim_json(tool_result, tool_keep_keys)

    # ---- AFTER: measure everything compressed ----
    after_system_tokens = count_tokens(compressed_system_prompt)
    after_conversation = count_conversation_tokens(compressed_conversation)
    after_tool_text = json.dumps(trimmed_tool_result)
    after_tool_tokens = count_tokens(after_tool_text)

    after_total = after_system_tokens + after_conversation["total_tokens"] + after_tool_tokens

    saved = before_total - after_total
    percent = (saved / before_total) * 100 if before_total > 0 else 0

    return {
        "before": {
            "system_prompt_tokens": before_system_tokens,
            "conversation_tokens": before_conversation["total_tokens"],
            "tool_result_tokens": before_tool_tokens,
            "total_tokens": before_total,
        },
        "after": {
            "system_prompt_tokens": after_system_tokens,
            "conversation_tokens": after_conversation["total_tokens"],
            "tool_result_tokens": after_tool_tokens,
            "total_tokens": after_total,
        },
        "saved_tokens": saved,
        "percent_saved": round(percent, 1),
        "compressed_system_prompt": compressed_system_prompt,
        "compressed_conversation": compressed_conversation,
        "trimmed_tool_result": trimmed_tool_result,
    }


if __name__ == "__main__":
    messy_system_prompt = """
    You are a helpful customer support assistant.

    Please note that you should always be polite to customers.
    In order to resolve issues quickly, ask clarifying questions.
    It is important to note that refunds require a manager's approval.
    """

    long_conversation = [
        {"role": "system", "content": "You are a helpful travel assistant."},
        {"role": "user", "content": "I'm planning a trip to Japan in April."},
        {"role": "assistant", "content": "Great choice! April is cherry blossom season. Do you want Tokyo, Kyoto, or both?"},
        {"role": "user", "content": "Both. About 10 days total."},
        {"role": "assistant", "content": "I'd suggest 5 days Tokyo, 5 days Kyoto, with a day trip to Nara."},
        {"role": "user", "content": "Sounds good. What about budget hotels?"},
        {"role": "assistant", "content": "Capsule hotels and business hotels like APA or Toyoko Inn are affordable and clean."},
        {"role": "user", "content": "Perfect. Now, what's the best way to get from Tokyo to Kyoto?"},
        {"role": "assistant", "content": "The Shinkansen bullet train, about 2.5 hours. Get a Japan Rail Pass if doing multiple trips."},
        {"role": "user", "content": "Do I need a JR pass for just this one trip?"},
        {"role": "assistant", "content": "For just Tokyo-Kyoto round trip, a single ticket is usually cheaper than a 7-day JR Pass."},
        {"role": "user", "content": "Good to know. What should I pack for April weather?"},
    ]

    fake_weather_response = {
        "location": {"name": "Tokyo", "region": "Kanto", "country": "Japan", "lat": 35.68, "lon": 139.69},
        "current": {
            "temp_c": 17.0,
            "condition": {"text": "Sunny", "icon": "//x.png", "code": 1000},
            "humidity": 55,
            "wind_kph": 10.0,
            "pressure_mb": 1012.0,
        }
    }
    important_weather_keys = ["name", "temp_c", "condition", "text", "humidity"]

    print("Running full pipeline... (this makes one Gemini API call for summarization)\n")
    result = run_pipeline(
        system_prompt=messy_system_prompt,
        conversation=long_conversation,
        tool_result=fake_weather_response,
        tool_keep_keys=important_weather_keys,
        keep_recent=4,
    )

    print("=== BEFORE (raw, unoptimized) ===")
    print(json.dumps(result["before"], indent=2))

    print("\n=== AFTER (through the harness) ===")
    print(json.dumps(result["after"], indent=2))

    print(f"\n✅ TOTAL SAVED: {result['saved_tokens']} tokens ({result['percent_saved']}%)")
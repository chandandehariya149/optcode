"""
verify_quality.py
------------------
Day 6: Quality verification loop.

Runs the SAME question through the model twice:
  1. With the original, uncompressed context
  2. With our compressed context (from pipeline.py)

Then shows both answers side by side so we can judge whether
compression cost us any real accuracy. This is what turns
"we save tokens" into "we save tokens safely."
"""

import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from pipeline import run_pipeline

load_dotenv()
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-3.6-flash")


def build_full_prompt(system_prompt: str, conversation: list, tool_result: dict, question: str) -> str:
    """Assemble a full prompt from system + history + tool result + new question."""
    convo_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in conversation)
    tool_text = json.dumps(tool_result)

    return (
        f"SYSTEM: {system_prompt}\n\n"
        f"CONVERSATION SO FAR:\n{convo_text}\n\n"
        f"RELEVANT DATA: {tool_text}\n\n"
        f"NEW QUESTION: {question}\n\n"
        f"Answer the new question using the conversation and data above."
    )


def ask_model(prompt: str) -> str:
    response = model.generate_content(prompt)
    return response.text.strip()


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

    new_question = "Given the current weather, what should I pack, and is an umbrella needed?"

    print("Running compression pipeline...\n")
    result = run_pipeline(
        system_prompt=messy_system_prompt,
        conversation=long_conversation,
        tool_result=fake_weather_response,
        tool_keep_keys=important_weather_keys,
        keep_recent=4,
    )

    # Build the ORIGINAL (uncompressed) full prompt
    original_prompt = build_full_prompt(
        messy_system_prompt, long_conversation, fake_weather_response, new_question
    )

    # Build the COMPRESSED full prompt using pipeline's output
    compressed_prompt = build_full_prompt(
        result["compressed_system_prompt"],
        result["compressed_conversation"],
        result["trimmed_tool_result"],
        new_question,
    )

    print("Asking model with ORIGINAL context...\n")
    original_answer = ask_model(original_prompt)

    print("Asking model with COMPRESSED context...\n")
    compressed_answer = ask_model(compressed_prompt)

    print("=" * 60)
    print("ANSWER WITH ORIGINAL (uncompressed) CONTEXT:")
    print("=" * 60)
    print(original_answer)

    print("\n" + "=" * 60)
    print("ANSWER WITH COMPRESSED CONTEXT:")
    print("=" * 60)
    print(compressed_answer)

    print(f"\n\n📊 Token savings from compression: {result['saved_tokens']} tokens ({result['percent_saved']}%)")
    print("\n👉 Compare the two answers above - are they substantively the same?")
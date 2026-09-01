"""
summarizer.py
-------------
Day 2: Our first real compression technique.

Instead of resending an entire long conversation history every time,
we keep recent messages as-is and replace older messages with a short
AI-generated summary. This is a core "harness" technique - we're not
touching the model, just being smarter about what we send it.
"""

import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from token_counter import count_conversation_tokens

# Load the API key from .env
load_dotenv()
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-3.6-flash")  # fast, cheap model - good for summarizing


def summarize_messages(messages: list) -> str:
    """
    Take a list of old messages and compress them into one short summary
    using a cheap/fast model call.
    """
    # Turn the messages into plain text for the summarizer to read
    conversation_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages
    )

    prompt = (
        "Summarize the following conversation history in 2-4 sentences. "
        "Keep only information that would matter for continuing the conversation "
        "later (facts stated, decisions made, context needed). Be concise.\n\n"
        f"{conversation_text}"
    )

    response = model.generate_content(prompt)
    return response.text.strip()


def compress_conversation(messages: list, keep_recent: int = 4) -> list:
    """
    Keep the last `keep_recent` messages untouched.
    Summarize everything before that into a single system-style message.
    """
    if len(messages) <= keep_recent:
        return messages  # nothing to compress yet

    old_messages = messages[:-keep_recent]
    recent_messages = messages[-keep_recent:]

    summary_text = summarize_messages(old_messages)

    summary_message = {
        "role": "system",
        "content": f"[Summary of earlier conversation]: {summary_text}"
    }

    return [summary_message] + recent_messages


if __name__ == "__main__":
    # A longer fake conversation to test compression on
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
        {"role": "assistant", "content": "April in Japan is mild, 10-19°C. Bring layers, a light jacket, and comfortable walking shoes."},
        {"role": "user", "content": "Will I need an umbrella?"},
        {"role": "assistant", "content": "It's not peak rainy season, but light showers happen. A compact umbrella is a good idea."},
        {"role": "user", "content": "What are must-see spots in Tokyo?"},
        {"role": "assistant", "content": "Senso-ji Temple, Shibuya Crossing, Shinjuku Gyoen, and teamLab Planets are top picks."},
        {"role": "user", "content": "What about food I should try?"},
        {"role": "assistant", "content": "Try sushi at Tsukiji Outer Market, ramen in Shinjuku, and conveyor belt sushi for budget meals."},
        {"role": "user", "content": "Any day trips from Tokyo besides Kyoto?"},
        {"role": "assistant", "content": "Nikko and Hakone are popular for nature and hot springs, both under 2 hours from Tokyo."},
        {"role": "user", "content": "Let's add a day in Hakone then."},
        {"role": "assistant", "content": "Great idea, adjust to 4 days Tokyo, 1 day Hakone, 5 days Kyoto."},
        {"role": "user", "content": "What can I do in Kyoto specifically?"},
        {"role": "assistant", "content": "Fushimi Inari Shrine, Arashiyama Bamboo Grove, Kinkaku-ji, and Gion district are must-sees."},
        {"role": "user", "content": "Is Kyoto walkable or should I rent a bike?"},
        {"role": "assistant", "content": "Central Kyoto is walkable, but renting a bike is popular for covering more ground easily."},
        {"role": "user", "content": "What about SIM cards or internet access?"},
        {"role": "assistant", "content": "Airport SIM cards or portable pocket wifi rentals are both affordable and easy to arrange."},
        {"role": "user", "content": "Do I need cash or can I use cards everywhere?"},
        {"role": "assistant", "content": "Cash is still important in Japan, especially in smaller shops, though cards are more common now."},
        {"role": "user", "content": "Great, this has been super helpful."},
        {"role": "assistant", "content": "Happy to help! Enjoy your Japan trip in April."},
        {"role": "user", "content": "One last thing - any etiquette tips I should know?"},
    ]

    print("=== BEFORE COMPRESSION ===")
    before = count_conversation_tokens(long_conversation)
    print(json.dumps(before, indent=2))

    print("\nCompressing... (this calls the Gemini API, takes a few seconds)\n")
    compressed = compress_conversation(long_conversation, keep_recent=4)

    print("=== AFTER COMPRESSION ===")
    after = count_conversation_tokens(compressed)
    print(json.dumps(after, indent=2))

    saved = before["total_tokens"] - after["total_tokens"]
    percent = (saved / before["total_tokens"]) * 100
    print(f"\n✅ Tokens saved: {saved} ({percent:.1f}%)")
    print(f"\nCompressed conversation:\n{json.dumps(compressed, indent=2)}")
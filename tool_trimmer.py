"""
tool_trimmer.py
----------------
Day 3: Second compression technique.

Tool/API results (weather data, database rows, search results, etc.)
often come back as large JSON blobs with far more fields than the
model actually needs to answer the user's question. This trims
big JSON down to only the relevant fields before it gets sent to
the LLM as context - pure harness-level work, no model involved.
"""

import json
from token_counter import count_tokens


def trim_json(data, keep_keys: list):
    """
    Recursively walk a JSON-like structure (dict or list of dicts)
    and keep only the keys listed in `keep_keys`, wherever they appear
    - even nested inside other keys we're not explicitly keeping.
    """
    if isinstance(data, dict):
        result = {}
        for k, v in data.items():
            if k in keep_keys:
                result[k] = v
            elif isinstance(v, (dict, list)):
                nested = trim_json(v, keep_keys)
                if nested:  # only keep this branch if something survived inside it
                    result[k] = nested
        return result
    elif isinstance(data, list):
        trimmed = [trim_json(item, keep_keys) for item in data]
        return [item for item in trimmed if item]
    else:
        return data


def compare_trim(original: dict, keep_keys: list) -> dict:
    """
    Trims the given data and reports token savings.
    """
    original_text = json.dumps(original)
    trimmed_data = trim_json(original, keep_keys)
    trimmed_text = json.dumps(trimmed_data)

    before_tokens = count_tokens(original_text)
    after_tokens = count_tokens(trimmed_text)
    saved = before_tokens - after_tokens
    percent = (saved / before_tokens) * 100 if before_tokens > 0 else 0

    return {
        "before_tokens": before_tokens,
        "after_tokens": after_tokens,
        "saved_tokens": saved,
        "percent_saved": round(percent, 1),
        "trimmed_data": trimmed_data,
    }


if __name__ == "__main__":
    # A realistic bloated API response - like a real weather API
    # returns WAY more than "is it going to rain today"
    fake_weather_api_response = {
        "location": {
            "name": "Bhopal",
            "region": "Madhya Pradesh",
            "country": "India",
            "lat": 23.26,
            "lon": 77.4,
            "tz_id": "Asia/Kolkata",
            "localtime_epoch": 1725172800,
            "localtime": "2026-09-01 10:00"
        },
        "current": {
            "last_updated_epoch": 1725172500,
            "last_updated": "2026-09-01 09:45",
            "temp_c": 27.0,
            "temp_f": 80.6,
            "is_day": 1,
            "condition": {
                "text": "Partly cloudy",
                "icon": "//cdn.weatherapi.com/weather/64x64/day/116.png",
                "code": 1003
            },
            "wind_mph": 8.1,
            "wind_kph": 13.0,
            "wind_degree": 210,
            "wind_dir": "SSW",
            "pressure_mb": 1006.0,
            "pressure_in": 29.71,
            "precip_mm": 0.2,
            "precip_in": 0.01,
            "humidity": 78,
            "cloud": 50,
            "feelslike_c": 29.5,
            "feelslike_f": 85.1,
            "vis_km": 6.0,
            "vis_miles": 3.0,
            "uv": 5.0,
            "gust_mph": 10.2,
            "gust_kph": 16.4
        }
    }

    print("=== ORIGINAL (bloated API response) ===")
    original_text = json.dumps(fake_weather_api_response, indent=2)
    print(original_text)
    print(f"\nOriginal tokens: {count_tokens(original_text)}")

    # We only actually need these fields to answer "what's the weather like"
    important_keys = ["name", "temp_c", "condition", "text", "humidity"]

    result = compare_trim(fake_weather_api_response, important_keys)

    print("\n=== TRIMMED (only what the model needs) ===")
    print(json.dumps(result["trimmed_data"], indent=2))

    print(f"\nBefore: {result['before_tokens']} tokens")
    print(f"After: {result['after_tokens']} tokens")
    print(f"✅ Saved: {result['saved_tokens']} tokens ({result['percent_saved']}%)")
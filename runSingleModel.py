import os
import datetime
import requests

# The local model server address
BASE_URL = "http://localhost:1234/v1"

# The single model used for every role
MODEL_ID = "qwen2.5-14b-instruct-1m"

# Set to True when using a deep thinking model (e.g. QwQ-32B)
DEEP_THINKING = False

# Set to True to add multi-perspective instructions to the prompt
PROMPTED = True

TEMPERATURE = 0.7

TOPIC = "Social media does more harm than good to society"

# Output folder (changes based on DEEP_THINKING and PROMPTED flag)
_suffix = ("_giliaimastymo" if DEEP_THINKING else "") + ("_prompted" if PROMPTED else "")
OUTPUT_DIR = f"vieno_modelio_isvestis{_suffix}"


def call_model(system_prompt, user_prompt, temperature=None):
    if temperature is None:
        temperature = TEMPERATURE

    url = f"{BASE_URL}/chat/completions"
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": temperature,
    }

    response = requests.post(url, json=payload)

    try:
        response.raise_for_status()
    except requests.HTTPError:
        print("HTTP error:", response.status_code)
        print("Response body:", response.text)
        raise

    return response.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    safe_chars = []
    for c in TOPIC:
        if c.isalnum() or c in " _-":
            safe_chars.append(c)
        else:
            safe_chars.append("_")
    safe_topic = "".join(safe_chars)[:60].strip()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(OUTPUT_DIR, f"{safe_topic}__{timestamp}.txt")

    user_prompt = f"What is the answer to: '{TOPIC}'."
    if PROMPTED:
        user_prompt += "Think of arguments from multiple perspectives before giving a conclusion."

    print(f"Topic:         {TOPIC}")
    print(f"Model:         {MODEL_ID}")
    print(f"Deep thinking: {DEEP_THINKING}")
    print(f"Prompted:      {PROMPTED}")
    print("=" * 70)

    result = call_model("", user_prompt, TEMPERATURE)

    print(result)

    output = "\n".join([
        f"Topic: {TOPIC}",
        f"Model: {MODEL_ID}",
        f"Deep thinking: {DEEP_THINKING}",
        f"Prompted: {PROMPTED}",
        f"Temperature: {TEMPERATURE}",
        "=" * 70,
        result,
    ])

    with open(filename, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"\n[Output saved to: {filename}]")
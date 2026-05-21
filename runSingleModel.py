import os
import datetime
import requests

# The local model server address
BASE_URL = "http://localhost:1234/v1"

# The single model used for every role
MODEL_ID = "qwen2.5-14b-instruct-1m"

TEMPERATURE = 0.7

TOPIC = "AI is beneficial for humanity"

# Set to True to add multi-perspective instructions to the prompt
PROMPTED = False

# Output folder (changes based on PROMPTED flag)
OUTPUT_DIR = "vieno_modelio_isvestis_prompted" if PROMPTED else "vieno_modelio_isvestis"


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

    safe_topic = "".join(c if c.isalnum() or c in " _-" else "_" for c in TOPIC)[:60].strip()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(OUTPUT_DIR, f"{safe_topic}__{timestamp}.txt")

    if PROMPTED:
        user_prompt = (
            f"Think carefully about the following topic: '{TOPIC}'. "
            "Consider strong arguments on multiple sides before reaching a conclusion."
        )
    else:
        user_prompt = f"What is the answer to: '{TOPIC}'."

    print(f"Topic:   {TOPIC}")
    print(f"Model:   {MODEL_ID}")
    print(f"Prompted: {PROMPTED}")
    print("=" * 70)

    result = call_model("", user_prompt, TEMPERATURE)

    print(result)

    output = "\n".join([
        f"Topic: {TOPIC}",
        f"Model: {MODEL_ID}",
        f"Prompted: {PROMPTED}",
        f"Temperature: {TEMPERATURE}",
        "=" * 70,
        result,
    ])

    with open(filename, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"\n[Output saved to: {filename}]")
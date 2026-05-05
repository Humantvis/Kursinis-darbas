import json
import requests

# The local model server address
BASE_URL = "http://localhost:1234/v1"

# The single model used for every role
MODEL_ID = "qwen2.5-14b-instruct-1m"

TEMPERATURE = 0.7

TOPIC = "Social media does more harm than good"

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
    user_prompt = f"What is the answer to: '{TOPIC}'."
    print(call_model("", user_prompt, TEMPERATURE))
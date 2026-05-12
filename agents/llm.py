import requests
import os

def grok_llm(prompt):
    api_key = os.getenv("GROQ_API_KEY")

    url = "https://api.x.ai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "grok-beta",
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(url, headers=headers, json=data)

    return response.json()["choices"][0]["message"]["content"]
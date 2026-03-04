import os, requests, json

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not API_KEY:
    raise RuntimeError("Please set OPENROUTER_API_KEY")

url = "https://openrouter.ai/api/v1/chat/completions"
payload = {
    "model": "meta-llama/llama-3.2-3b-instruct:free",
    "messages": [{"role": "user", "content": "Say only: ok"}],
    "temperature": 0,
    "max_tokens": 10,
}
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
print("status:", r.status_code)
print(r.text)

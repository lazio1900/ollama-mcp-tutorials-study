import json
import requests

POD_ID = "tpi63brhybnlxg"
BASE = f"https://{POD_ID}-11434.proxy.runpod.net"

with requests.post(
    f"{BASE}/api/generate",
    json={"model": "qwen3:8b", "prompt": "한국어로: LLM 서빙 구조를 단계별로 설명", "stream": True},
    stream=True,
    timeout=180,
) as r:
    r.raise_for_status()
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        data = json.loads(line)
        print(data.get("response", ""), end="", flush=True)

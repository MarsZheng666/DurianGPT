#!/usr/bin/env python3
"""Print sanitized LangGraph SSE event ordering for one exact query."""

from __future__ import annotations

import json
import sys

import httpx


headers = {"X-API-Key": "change-me", "X-User-Id": "admin2"}
base_url = "http://127.0.0.1:8001"
query = sys.argv[1] if len(sys.argv) > 1 else "猫山王的编码是什么"
with httpx.Client(timeout=120, trust_env=False) as client:
    created = client.post(f"{base_url}/conversations", headers=headers)
    created.raise_for_status()
    conversation_id = created.json()["id"]
    with client.stream(
        "POST",
        f"{base_url}/chat/stream3",
        headers=headers,
        json={
            "conversation_id": conversation_id,
            "messages": [{"role": "user", "content": query}],
            "stream": True,
            "use_rag": True,
            "temperature": 0,
            "max_tokens": 120,
            "response_language": "zh",
        },
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if raw == "[DONE]":
                print("DONE")
                break
            event = json.loads(raw)
            evidence = event.get("evidence")
            print(
                "EVENT"
                f"\ttype={event.get('type')}"
                f"\tphase={event.get('phase')}"
                f"\tquality={event.get('evidence_quality')}"
                f"\tevidence={len(evidence) if isinstance(evidence, list) else 'absent'}"
                f"\tcontent={str(event.get('content') or '')[:80]}"
            )

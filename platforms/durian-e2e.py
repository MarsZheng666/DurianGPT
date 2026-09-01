#!/usr/bin/env python3
"""End-to-end checks for LangGraph memory, RAGFlow evidence, and SSE streaming."""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx


BASE_URL = "http://127.0.0.1:8001"
USER_ID = "admin2"
ENV_FILES = [
    Path("/etc/durian-gpt/qwen-vl.env"),
    Path("/etc/durian-gpt-platform.env"),
]


def load_api_key() -> str:
    values: dict[str, str] = {}
    for path in ENV_FILES:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("'\"")
    return values.get("DURIAN_API_KEY") or values.get("API_KEY") or "change-me"


headers = {
    "X-API-Key": load_api_key(),
    "X-User-Id": USER_ID,
}


def stream_turn(
    client: httpx.Client,
    conversation_id: str,
    messages: list[dict],
    *,
    max_tokens: int = 160,
) -> dict:
    started = time.perf_counter()
    first_content_at: float | None = None
    chunks: list[str] = []
    evidence: list[dict] = []
    evidence_quality = "none"
    errors: list[str] = []
    done = False

    with client.stream(
        "POST",
        f"{BASE_URL}/chat/stream3",
        headers={**headers, "Content-Type": "application/json"},
        json={
            "conversation_id": conversation_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "max_new_tokens": max_tokens,
            "temperature": 0,
            "top_p": 0.8,
            "use_rag": True,
            "stream": True,
            "response_language": "zh",
        },
    ) as response:
        response.raise_for_status()
        orchestrator = response.headers.get("X-Durian-Orchestrator", "")
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            event_type = str(event.get("type") or "")
            if event_type == "content":
                if first_content_at is None:
                    first_content_at = time.perf_counter() - started
                chunks.append(str(event.get("content") or ""))
            elif event_type in {"metadata", "evidence"}:
                candidate = event.get("evidence")
                if isinstance(candidate, list) and candidate:
                    evidence = candidate
                candidate_quality = str(event.get("evidence_quality") or "")
                if candidate_quality:
                    evidence_quality = candidate_quality
            elif event_type == "error":
                errors.append(str(event.get("error") or "unknown error"))

    return {
        "text": "".join(chunks),
        "chunk_count": len(chunks),
        "first_content_seconds": first_content_at,
        "duration_seconds": time.perf_counter() - started,
        "evidence": evidence,
        "evidence_quality": evidence_quality,
        "errors": errors,
        "done": done,
        "orchestrator": orchestrator,
    }


with httpx.Client(timeout=180, follow_redirects=True, trust_env=False) as client:
    status = client.get(f"{BASE_URL}/rag/status", headers=headers)
    status.raise_for_status()
    status_payload = status.json()
    print(
        "RAG_STATUS"
        f"\tenabled={status_payload.get('enabled')}"
        f"\tragflow={status_payload.get('ragflow_enabled')}"
    )

    created = client.post(f"{BASE_URL}/conversations", headers=headers)
    created.raise_for_status()
    conversation_id = str(created.json().get("id") or "")
    if not conversation_id:
        raise RuntimeError("Conversation creation returned no ID")

    first = stream_turn(
        client,
        conversation_id,
        [{"role": "user", "content": "猫山王的编码是什么"}],
    )
    print(
        "EXACT_FIRST"
        f"\td200={'D200' in first['text']}"
        f"\td197={'D197' in first['text']}"
        f"\tevidence={len(first['evidence'])}"
        f"\torchestrator={first['orchestrator']}"
        f"\ttext={first['text'][:180]}"
    )
    if "D197" not in first["text"] or first["errors"]:
        raise RuntimeError(f"First exact answer failed: {first['errors']}")

    reverse = stream_turn(
        client,
        conversation_id,
        [{"role": "user", "content": "D197是什么榴莲编号"}],
    )
    print(
        "EXACT_REVERSE"
        f"\tmusang={'猫山王' in reverse['text']}"
        f"\td197={'D197' in reverse['text']}"
        f"\tevidence={len(reverse['evidence'])}"
        f"\tquality={reverse['evidence_quality']}"
        f"\ttext={reverse['text'][:180]}"
    )
    if (
        "猫山王" not in reverse["text"]
        or "D197" not in reverse["text"]
        or not reverse["evidence"]
        or reverse["errors"]
    ):
        raise RuntimeError(f"Reverse exact answer failed: {reverse['errors']}")

    short_code = stream_turn(
        client,
        conversation_id,
        [{"role": "user", "content": "D200呢"}],
    )
    print(
        "EXACT_SHORT_CODE"
        f"\tblack_thorn={'黑刺' in short_code['text']}"
        f"\td200={'D200' in short_code['text']}"
        f"\tevidence={len(short_code['evidence'])}"
        f"\tquality={short_code['evidence_quality']}"
        f"\ttext={short_code['text'][:180]}"
    )
    if (
        "黑刺" not in short_code["text"]
        or "D200" not in short_code["text"]
        or not short_code["evidence"]
        or short_code["errors"]
    ):
        raise RuntimeError(
            f"Short code exact answer failed: {short_code['errors']}"
        )

    second = stream_turn(
        client,
        conversation_id,
        [{"role": "user", "content": "黑刺的编码是什么"}],
    )
    evidence_docs = {
        str(item.get("doc") or item.get("source") or "")
        for item in second["evidence"]
        if isinstance(item, dict)
    }
    print(
        "EXACT_MEMORY_ISOLATION"
        f"\td200={'D200' in second['text']}"
        f"\td197={'D197' in second['text']}"
        f"\tevidence={len(second['evidence'])}"
        f"\tauthoritative={any('authoritative' in doc for doc in evidence_docs)}"
        f"\ttext={second['text'][:180]}"
    )
    if (
        "D200" not in second["text"]
        or "D197" in second["text"]
        or not second["evidence"]
        or second["errors"]
    ):
        raise RuntimeError(f"Memory isolation exact answer failed: {second['errors']}")

    unknown = stream_turn(
        client,
        conversation_id,
        [{"role": "user", "content": "火星榴莲的品种编码是什么"}],
    )
    explicit_not_found = "没有找到" in unknown["text"] or "未找到" in unknown["text"]
    print(
        "EXACT_NOT_FOUND"
        f"\tnot_found={explicit_not_found}"
        f"\tevidence={len(unknown['evidence'])}"
        f"\ttext={unknown['text'][:180]}"
    )
    if not explicit_not_found or unknown["evidence"] or unknown["errors"]:
        raise RuntimeError(f"Unknown exact answer gate failed: {unknown['errors']}")

    unsupported = stream_turn(
        client,
        conversation_id,
        [
            {
                "role": "user",
                "content": "火星榴莲的曲速引擎应该怎么维护",
            }
        ],
    )
    grounded_refusal = (
        "RAGFlow" in unsupported["text"]
        and (
            "不进行推测" in unsupported["text"]
            or "没有找到" in unsupported["text"]
        )
    )
    print(
        "GROUNDED_NO_SOURCE"
        f"\trefused={grounded_refusal}"
        f"\tevidence={len(unsupported['evidence'])}"
        f"\ttext={unsupported['text'][:180]}"
    )
    if (
        not grounded_refusal
        or unsupported["evidence"]
        or unsupported["errors"]
    ):
        raise RuntimeError(
            f"Grounded no-source gate failed: {unsupported['errors']}"
        )

    image_created = client.post(f"{BASE_URL}/conversations", headers=headers)
    image_created.raise_for_status()
    image_conversation_id = str(image_created.json().get("id") or "")
    image_history = [
        {
            "role": "user",
            "content": "请识别图片里的榴莲品种",
            "image_url": "/uploads/e2e-musang-king.jpg",
        },
        {
            "role": "assistant",
            "content": "图片中的榴莲品种初步判断为猫山王。",
            "image_url": "/uploads/e2e-musang-king.jpg",
            "active_context_card": {
                "context_id": "ctx-e2e-musang-image",
                "source": "image",
                "image_url": "/uploads/e2e-musang-king.jpg",
                "scenario": "variety",
                "category_type": "variety",
                "main_subject": "猫山王",
                "last_user_query": "请识别图片里的榴莲品种",
                "last_answer_summary": "图片中的榴莲品种初步判断为猫山王。",
                "key_facts": ["图片识别主体：猫山王"],
            },
        },
        {"role": "user", "content": "它是什么编号"},
    ]
    image_exact = stream_turn(
        client,
        image_conversation_id,
        image_history,
    )
    print(
        "IMAGE_EXACT_FOLLOWUP"
        f"\td197={'D197' in image_exact['text']}"
        f"\tevidence={len(image_exact['evidence'])}"
        f"\tquality={image_exact['evidence_quality']}"
        f"\ttext={image_exact['text'][:180]}"
    )
    if (
        "D197" not in image_exact["text"]
        or not image_exact["evidence"]
        or image_exact["errors"]
    ):
        raise RuntimeError(
            f"Image exact follow-up failed: {image_exact['errors']}"
        )

    image_exact_again = stream_turn(
        client,
        image_conversation_id,
        [{"role": "user", "content": "再确认一下，它是什么编号"}],
    )
    print(
        "IMAGE_CONTEXT_PERSISTENCE"
        f"\td197={'D197' in image_exact_again['text']}"
        f"\tevidence={len(image_exact_again['evidence'])}"
        f"\tquality={image_exact_again['evidence_quality']}"
        f"\ttext={image_exact_again['text'][:180]}"
    )
    if (
        "D197" not in image_exact_again["text"]
        or not image_exact_again["evidence"]
        or image_exact_again["errors"]
    ):
        raise RuntimeError(
            "Image context did not survive the first text follow-up: "
            f"{image_exact_again['errors']}"
        )

    streamed = stream_turn(
        client,
        conversation_id,
        [{"role": "user", "content": "请用两句话说明猫山王的特点"}],
        max_tokens=180,
    )
    print(
        "ASYNC_STREAM"
        f"\tchunks={streamed['chunk_count']}"
        f"\tfirst={streamed['first_content_seconds']}"
        f"\tduration={streamed['duration_seconds']:.3f}"
        f"\tdone={streamed['done']}"
        f"\tchars={len(streamed['text'])}"
    )
    if (
        streamed["chunk_count"] < 2
        or streamed["first_content_seconds"] is None
        or not streamed["done"]
        or not streamed["evidence"]
        or streamed["errors"]
    ):
        raise RuntimeError(f"Async streaming failed: {streamed['errors']}")

print("DURIAN_E2E_PASSED")

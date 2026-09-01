#!/usr/bin/env python3
"""Read-only validation for the private Durian GPT RAGFlow dataset."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import httpx


PLATFORM_ROOT = Path("/home/admin01/桌面/Desktop/durian-training/platforms")
CREDENTIAL_FILE = PLATFORM_ROOT / ".admin-credentials"
RAGFLOW_URL = "http://127.0.0.1:19380"


def load_pairs(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def require_success(response: httpx.Response, operation: str) -> dict:
    response.raise_for_status()
    payload = response.json()
    if int(payload.get("code", 0) or 0) != 0:
        raise RuntimeError(
            f"{operation} failed: {payload.get('message') or payload}"
        )
    return payload


credentials = load_pairs(CREDENTIAL_FILE)
api_key = credentials.get("RAGFLOW_API_KEY", "")
dataset_id = credentials.get("RAGFLOW_DATASET_ID", "")
if not api_key or not dataset_id:
    raise SystemExit("RAGFlow API key or dataset ID is missing")

headers = {"Authorization": f"Bearer {api_key}"}
with httpx.Client(timeout=90, trust_env=False) as client:
    listing = require_success(
        client.get(
            f"{RAGFLOW_URL}/api/v1/datasets/{dataset_id}/documents",
            headers=headers,
            params={"page": 1, "page_size": 200},
        ),
        "RAGFlow document list",
    )
    data = listing.get("data") or {}
    if isinstance(data, dict):
        documents = data.get("docs") or data.get("documents") or []
    else:
        documents = data
    if not isinstance(documents, list):
        documents = []

    print("DOCUMENT_STATUS")
    all_complete = bool(documents)
    for document in documents:
        name = str(document.get("name") or "")
        run = str(document.get("run") or "")
        progress = float(document.get("progress") or 0)
        chunk_count = int(document.get("chunk_count") or 0)
        message = " ".join(str(document.get("progress_msg") or "").split())
        if len(message) > 180:
            message = message[:177] + "..."
        print(
            f"{name}\trun={run}\tprogress={progress:.3f}"
            f"\tchunks={chunk_count}\t{message}"
        )
        if progress < 0.999 or run not in {"3", "DONE", "done"}:
            all_complete = False

    if not all_complete:
        print("PARSING_PENDING")
        raise SystemExit(2)

    cases = [
        ("黑刺的编码是什么", r"(?:黑刺|Black\s*Thorn|Duri\s*Hitam|Ochee).{0,80}\bD[\s-]*200\b"),
        ("猫山王的编码是什么", r"(?:猫山王|Musang\s*King|Raja\s*Kunyit).{0,80}\bD[\s-]*197\b"),
        (
            "D197是什么榴莲编号",
            r"(?:(?:猫山王|Musang\s*King|Raja\s*Kunyit).{0,120}\bD[\s-]*197\b|\bD[\s-]*197\b.{0,120}(?:猫山王|Musang\s*King|Raja\s*Kunyit))",
        ),
        ("苏丹王D24和猫山王是不是同一品种", r"\bD[\s-]*24\b"),
    ]
    failures: list[str] = []
    for question, expected_pattern in cases:
        retrieval = require_success(
            client.post(
                f"{RAGFLOW_URL}/api/v1/retrieval",
                headers=headers,
                json={
                    "question": question,
                    "dataset_ids": [dataset_id],
                    "page": 1,
                    "page_size": 8,
                    "similarity_threshold": 0.03,
                    "vector_similarity_weight": 0.18,
                    "top_k": 32,
                    "keyword": True,
                    "highlight": False,
                },
            ),
            f"RAGFlow retrieval for {question}",
        )
        retrieval_data = retrieval.get("data") or {}
        chunks = (
            retrieval_data.get("chunks") or []
            if isinstance(retrieval_data, dict)
            else []
        )
        combined = "\n".join(
            str(chunk.get("content") or chunk.get("content_with_weight") or "")
            for chunk in chunks
            if isinstance(chunk, dict)
        )
        match = re.search(expected_pattern, combined, flags=re.I | re.S)
        print(f"QUERY\t{question}\tchunks={len(chunks)}\tmatch={bool(match)}")
        for index, chunk in enumerate(chunks[:3], start=1):
            source = str(
                chunk.get("document_keyword")
                or chunk.get("document_name")
                or chunk.get("doc_name")
                or ""
            )
            score = float(chunk.get("similarity") or chunk.get("score") or 0)
            content = " ".join(
                str(
                    chunk.get("content")
                    or chunk.get("content_with_weight")
                    or ""
                ).split()
            )
            print(f"TOP{index}\t{score:.4f}\t{source}\t{content[:260]}")
        if not match:
            failures.append(question)

if failures:
    print("VALIDATION_FAILED\t" + " | ".join(failures))
    raise SystemExit(1)

print("VALIDATION_PASSED")

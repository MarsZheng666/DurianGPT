#!/usr/bin/env python3
"""Read-only validation for the private Durian GPT Dify retrieval workflow."""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx


CREDENTIAL_FILE = Path(
    "/home/admin01/桌面/Desktop/durian-training/platforms/.admin-credentials"
)
DIFY_URL = "http://127.0.0.1:15001"


def load_pairs(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


credentials = load_pairs(CREDENTIAL_FILE)
api_key = credentials.get("DIFY_RAGFLOW_APP_API_KEY", "")
if not api_key:
    raise SystemExit("Dify workflow API key is missing")

cases = [
    ("黑刺的编码是什么", r"(?:黑刺|Black\s*Thorn|Duri\s*Hitam|Ochee).{0,80}\bD[\s-]*200\b"),
    ("猫山王的编码是什么", r"(?:猫山王|Musang\s*King|Raja\s*Kunyit).{0,80}\bD[\s-]*197\b"),
]
failed: list[str] = []
with httpx.Client(timeout=120, trust_env=False) as client:
    for question, expected_pattern in cases:
        response = client.post(
            f"{DIFY_URL}/v1/workflows/run",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "inputs": {"query": question},
                "response_mode": "blocking",
                "user": "durian-ragflow-validation",
            },
        )
        response.raise_for_status()
        payload = response.json()
        outputs = ((payload.get("data") or {}).get("outputs") or {})
        status_code = int(outputs.get("status_code") or 0)
        raw_retrieval = str(outputs.get("retrieval_json") or "")
        try:
            retrieval = json.loads(raw_retrieval)
        except json.JSONDecodeError:
            retrieval = {}
        records = retrieval.get("records") or []
        combined = "\n".join(
            str(record.get("content") or "")
            for record in records
            if isinstance(record, dict)
        )
        matched = bool(re.search(expected_pattern, combined, flags=re.I | re.S))
        print(
            f"DIFY_QUERY\t{question}\tstatus={status_code}"
            f"\trecords={len(records)}\tmatch={matched}"
        )
        for index, record in enumerate(records[:3], start=1):
            title = str(record.get("title") or "")
            score = float(record.get("score") or 0)
            content = " ".join(str(record.get("content") or "").split())
            print(f"TOP{index}\t{score:.4f}\t{title}\t{content[:260]}")
        if status_code != 200 or not matched:
            failed.append(question)

if failed:
    print("DIFY_VALIDATION_FAILED\t" + " | ".join(failed))
    raise SystemExit(1)

print("DIFY_VALIDATION_PASSED")

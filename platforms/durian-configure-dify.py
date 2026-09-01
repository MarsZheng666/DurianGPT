#!/usr/bin/env python3
"""Create and validate the private Dify control-plane workflow for Durian GPT."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import httpx


PLATFORM_ROOT = Path("/home/admin01/桌面/Desktop/durian-training/platforms")
CREDENTIAL_FILE = PLATFORM_ROOT / ".admin-credentials"
PLATFORM_ENV_FILE = Path("/etc/durian-gpt-platform.env")
DIFY_URL = "http://127.0.0.1:15001"
WORKFLOW_NAME = "Durian GPT - RAGFlow Retrieval Control"


def load_pairs(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def write_pairs(path: Path, values: dict[str, str]) -> None:
    payload = "\n".join(f"{key}={value}" for key, value in sorted(values.items()))
    path.write_text(payload + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


def check_response(response: httpx.Response, operation: str) -> dict:
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if response.status_code >= 400:
        message = str(payload.get("message") or payload.get("error") or "")
        raise RuntimeError(
            f"{operation} failed: HTTP {response.status_code} {message[:300]}"
        )
    return payload


credentials = load_pairs(CREDENTIAL_FILE)
email = credentials.get("DIFY_ADMIN_EMAIL", "")
password = credentials.get("DIFY_ADMIN_PASSWORD", "")
ragflow_api_key = credentials.get("RAGFLOW_API_KEY", "")
dataset_id = credentials.get("RAGFLOW_DATASET_ID", "")
if not all((email, password, ragflow_api_key, dataset_id)):
    raise SystemExit("Required Dify or RAGFlow credentials are missing")

encoded_password = base64.b64encode(password.encode("utf-8")).decode("ascii")
workflow_dsl = f"""app:
  description: Private visual control plane for the Durian GPT RAGFlow retriever.
  icon: "🔎"
  icon_background: "#D1F7C4"
  mode: workflow
  name: "{WORKFLOW_NAME}"
  use_icon_as_answer_icon: false
dependencies: []
kind: app
version: 0.6.0
workflow:
  conversation_variables: []
  environment_variables: []
  features:
    file_upload:
      enabled: false
    opening_statement: ''
    retriever_resource:
      enabled: false
    sensitive_word_avoidance:
      enabled: false
    speech_to_text:
      enabled: false
    suggested_questions: []
    suggested_questions_after_answer:
      enabled: false
    text_to_speech:
      enabled: false
  graph:
    edges:
    - data:
        isInIteration: false
        isInLoop: false
        sourceType: start
        targetType: http-request
      id: start-to-ragflow
      source: start_node
      sourceHandle: source
      target: ragflow_node
      targetHandle: target
      type: custom
      zIndex: 0
    - data:
        isInIteration: false
        isInLoop: false
        sourceType: http-request
        targetType: end
      id: ragflow-to-end
      source: ragflow_node
      sourceHandle: source
      target: end_node
      targetHandle: target
      type: custom
      zIndex: 0
    nodes:
    - data:
        desc: Question sent to the private RAGFlow knowledge base.
        selected: false
        title: Query
        type: start
        variables:
        - label: query
          max_length: 4096
          options: []
          required: true
          type: paragraph
          variable: query
      height: 116
      id: start_node
      position:
        x: 30
        y: 227
      positionAbsolute:
        x: 30
        y: 227
      selected: false
      sourcePosition: right
      targetPosition: left
      type: custom
      width: 244
    - data:
        authorization:
          config:
            api_key: "{ragflow_api_key}"
            type: bearer
          type: api-key
        body:
          data: []
          type: none
        desc: Hybrid retrieval from the private Durian GPT RAGFlow dataset.
        headers: ''
        method: get
        params: |-
          knowledge_id:{dataset_id}
          query:{{{{#start_node.query#}}}}
          top_k:8
          score_threshold:0.03
        retry_config:
          max_retries: 2
          retry_enabled: true
          retry_interval: 300
        selected: false
        ssl_verify: true
        timeout:
          max_connect_timeout: 10
          max_read_timeout: 60
          max_write_timeout: 10
        title: RAGFlow Hybrid Retrieval
        type: http-request
        url: http://host.docker.internal:18002/api/v1/dify/retrieval
        variables: []
      height: 150
      id: ragflow_node
      position:
        x: 350
        y: 227
      positionAbsolute:
        x: 350
        y: 227
      selected: false
      sourcePosition: right
      targetPosition: left
      type: custom
      width: 244
    - data:
        desc: Raw Dify-compatible RAGFlow response with records and citations.
        outputs:
        - value_selector:
          - ragflow_node
          - status_code
          value_type: number
          variable: status_code
        - value_selector:
          - ragflow_node
          - body
          value_type: string
          variable: retrieval_json
        selected: false
        title: Retrieval Result
        type: end
      height: 116
      id: end_node
      position:
        x: 670
        y: 227
      positionAbsolute:
        x: 670
        y: 227
      selected: false
      sourcePosition: right
      targetPosition: left
      type: custom
      width: 244
    viewport:
      x: 0
      y: 0
      zoom: 0.8
"""

with httpx.Client(timeout=90, follow_redirects=True, trust_env=False) as client:
    login = client.post(
        f"{DIFY_URL}/console/api/login",
        json={
            "email": email,
            "password": encoded_password,
            "remember_me": True,
        },
    )
    check_response(login, "Dify login")
    csrf_token = client.cookies.get("csrf_token")
    if not csrf_token:
        raise RuntimeError("Dify login returned no CSRF token")
    console_headers = {"X-CSRF-Token": csrf_token}

    import_payload: dict[str, str] = {
        "mode": "yaml-content",
        "yaml_content": workflow_dsl,
        "name": WORKFLOW_NAME,
        "description": "Durian GPT private RAGFlow retrieval control plane.",
    }
    existing_app_id = credentials.get("DIFY_RAGFLOW_APP_ID", "")
    if existing_app_id:
        import_payload["app_id"] = existing_app_id
    imported_response = client.post(
        f"{DIFY_URL}/console/api/apps/imports",
        headers=console_headers,
        json=import_payload,
    )
    imported = check_response(imported_response, "Dify workflow import")
    if imported_response.status_code == 202:
        import_id = str(imported.get("id") or imported.get("import_id") or "")
        if not import_id:
            raise RuntimeError("Dify workflow import requires confirmation but returned no ID")
        imported_response = client.post(
            f"{DIFY_URL}/console/api/apps/imports/{import_id}/confirm",
            headers=console_headers,
        )
        imported = check_response(imported_response, "Dify workflow import confirmation")
    app_id = str(imported.get("app_id") or existing_app_id)
    if not app_id:
        raise RuntimeError("Dify workflow import returned no app ID")
    print("Dify RAGFlow retrieval workflow imported.")

    publish = client.post(
        f"{DIFY_URL}/console/api/apps/{app_id}/workflows/publish",
        headers=console_headers,
        json={
            "marked_name": "RAGFlow",
            "marked_comment": "Private hybrid retrieval control workflow",
        },
    )
    check_response(publish, "Dify workflow publish")
    print("Dify RAGFlow retrieval workflow published.")

    key_listing = client.get(
        f"{DIFY_URL}/console/api/apps/{app_id}/api-keys",
        headers=console_headers,
    )
    keys_payload = check_response(key_listing, "Dify app API key list")
    keys = keys_payload.get("data") or []
    app_api_key = next(
        (
            str(item.get("token") or "")
            for item in keys
            if isinstance(item, dict) and str(item.get("token") or "").startswith("app-")
        ),
        "",
    )
    if not app_api_key:
        created_key = client.post(
            f"{DIFY_URL}/console/api/apps/{app_id}/api-keys",
            headers=console_headers,
        )
        key_payload = check_response(created_key, "Dify app API key creation")
        app_api_key = str(key_payload.get("token") or "")
    if not app_api_key.startswith("app-"):
        raise RuntimeError("Dify returned an invalid app API key")

credentials["DIFY_RAGFLOW_APP_ID"] = app_id
credentials["DIFY_RAGFLOW_APP_API_KEY"] = app_api_key
write_pairs(CREDENTIAL_FILE, credentials)
platform_env = load_pairs(PLATFORM_ENV_FILE)
platform_env["DURIAN_DIFY_ENABLED"] = "false"
platform_env["DURIAN_DIFY_BASE_URL"] = DIFY_URL
platform_env["DURIAN_DIFY_APP_API_KEY"] = app_api_key
write_pairs(PLATFORM_ENV_FILE, platform_env)
print("Dify workflow credentials stored; live GPT routing remains disabled.")

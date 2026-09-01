#!/usr/bin/env python3
"""Configure the private RAGFlow model, dataset, and initial Durian documents."""

from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path
import os
import subprocess
import sys

import httpx


PROJECT_ROOT = Path("/home/admin01/桌面/Desktop/durian-training")
PLATFORM_ROOT = PROJECT_ROOT / "platforms"
CREDENTIAL_FILE = PLATFORM_ROOT / ".admin-credentials"
PLATFORM_ENV_FILE = Path("/etc/durian-gpt-platform.env")
RAGFLOW_URL = "http://127.0.0.1:19380"
DATASET_NAME = "Durian GPT Knowledge Base"


def load_pairs(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def write_pairs(path: Path, values: dict[str, str]) -> None:
    payload = "\n".join(
        f"{key}={value}"
        for key, value in sorted(values.items())
    )
    path.write_text(payload + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


def encrypt_password(password: str) -> str:
    result = subprocess.run(
        [
            "docker",
            "exec",
            "durian-ragflow-ragflow-cpu-1",
            "python",
            "-c",
            "from api.utils.crypt import crypt; import sys; print(crypt(sys.argv[1]))",
            password,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().splitlines()[-1]


def require_success(response: httpx.Response, operation: str) -> dict:
    response.raise_for_status()
    payload = response.json()
    if int(payload.get("code", 0) or 0) != 0:
        raise RuntimeError(
            f"{operation} failed: {payload.get('message') or payload}"
        )
    return payload


credentials = load_pairs(CREDENTIAL_FILE)
ragflow_api_key = credentials.get("RAGFLOW_API_KEY", "")
embedding_api_key = credentials.get("DURIAN_INTERNAL_EMBEDDING_API_KEY", "")
email = credentials.get("RAGFLOW_ADMIN_EMAIL", "")
password = credentials.get("RAGFLOW_ADMIN_PASSWORD", "")
if not all((ragflow_api_key, embedding_api_key, email, password)):
    raise SystemExit("Required RAGFlow credentials are missing")

with httpx.Client(timeout=60, follow_redirects=True, trust_env=False) as client:
    login = require_success(
        client.post(
            f"{RAGFLOW_URL}/api/v1/auth/login",
            json={
                "email": email,
                "password": encrypt_password(password),
            },
        ),
        "RAGFlow login",
    )
    authorization = client.headers.get("Authorization", "")
    if not authorization:
        authorization = str(login.get("authorization") or "")
    response_authorization = client.cookies.get("Authorization")
    if not authorization and response_authorization:
        authorization = response_authorization
    # httpx stores response headers on the response, not the client.
    login_response = client.post(
        f"{RAGFLOW_URL}/api/v1/auth/login",
        json={
            "email": email,
            "password": encrypt_password(password),
        },
    )
    login_payload = require_success(login_response, "RAGFlow login")
    authorization = login_response.headers.get("Authorization", "")
    if not authorization:
        raise RuntimeError(f"RAGFlow login returned no authorization token: {login_payload}")

    login_headers = {"Authorization": authorization}
    model_payload = require_success(
        client.post(
            f"{RAGFLOW_URL}/v1/llm/add_llm",
            headers=login_headers,
            json={
                "llm_factory": "OpenAI-API-Compatible",
                "llm_name": "durian-e5",
                "model_type": "embedding",
                "api_base": "http://host.docker.internal:18001/v1",
                "api_key": embedding_api_key,
                "max_tokens": 8192,
            },
            timeout=90,
        ),
        "RAGFlow embedding model registration",
    )
    if model_payload.get("data") is not True:
        raise RuntimeError("RAGFlow embedding model registration was not confirmed")
    print("RAGFlow embedding model verified.")

    chat_model_payload = require_success(
        client.post(
            f"{RAGFLOW_URL}/v1/llm/add_llm",
            headers=login_headers,
            json={
                "llm_factory": "OpenAI-API-Compatible",
                "llm_name": "durian-lora",
                "model_type": "chat",
                "api_base": "http://host.docker.internal:18003/v1",
                "api_key": "durian-docker-internal",
                "max_tokens": 8192,
            },
            timeout=90,
        ),
        "RAGFlow chat model registration",
    )
    if chat_model_payload.get("data") is not True:
        raise RuntimeError("RAGFlow chat model registration was not confirmed")

    tenant_payload = require_success(
        client.get(
            f"{RAGFLOW_URL}/api/v1/users/me/models",
            headers=login_headers,
        ),
        "RAGFlow tenant model settings",
    )
    tenant = tenant_payload.get("data") or {}
    tenant_id = str(tenant.get("tenant_id") or tenant.get("id") or "")
    if not tenant_id:
        raise RuntimeError(f"RAGFlow tenant ID is missing: {tenant}")
    model_settings = {
        "tenant_id": tenant_id,
        "llm_id": "durian-lora@OpenAI-API-Compatible",
        "embd_id": str(
            tenant.get("embd_id")
            or "durian-e5___OpenAI-API@OpenAI-API-Compatible"
        ),
        "asr_id": str(tenant.get("asr_id") or ""),
        "img2txt_id": str(tenant.get("img2txt_id") or ""),
    }
    require_success(
        client.patch(
            f"{RAGFLOW_URL}/api/v1/users/me/models",
            headers=login_headers,
            json=model_settings,
        ),
        "RAGFlow default chat model selection",
    )
    print("RAGFlow existing Durian chat model verified and selected.")

    api_headers = {"Authorization": f"Bearer {ragflow_api_key}"}
    datasets_payload = require_success(
        client.get(
            f"{RAGFLOW_URL}/api/v1/datasets",
            headers=api_headers,
            params={"page": 1, "page_size": 100},
        ),
        "RAGFlow dataset list",
    )
    datasets_data = datasets_payload.get("data") or []
    if isinstance(datasets_data, dict):
        datasets = datasets_data.get("datasets") or datasets_data.get("items") or []
    else:
        datasets = datasets_data
    dataset = next(
        (
            item
            for item in datasets
            if isinstance(item, dict) and item.get("name") == DATASET_NAME
        ),
        None,
    )
    if dataset is None:
        create_payload = require_success(
            client.post(
                f"{RAGFLOW_URL}/api/v1/datasets",
                headers=api_headers,
                json={
                    "name": DATASET_NAME,
                    "description": (
                        "Durian GPT source documents. Retrieval is hybrid "
                        "vector + keyword and exact questions require source matches."
                    ),
                    "embedding_model": (
                        "durian-e5___OpenAI-API@OpenAI-API-Compatible"
                    ),
                    "permission": "me",
                    "chunk_method": "naive",
                    "parser_config": {
                        "chunk_token_num": 512,
                        "delimiter": "\\n!?;。；！？",
                        "layout_recognize": "DeepDOC",
                    },
                },
            ),
            "RAGFlow dataset creation",
        )
        dataset = create_payload.get("data") or {}
        print("RAGFlow dataset created.")
    dataset_id = str(dataset.get("id") or "")
    if not dataset_id:
        raise RuntimeError(f"RAGFlow dataset ID is missing: {dataset}")

    documents_payload = require_success(
        client.get(
            f"{RAGFLOW_URL}/api/v1/datasets/{dataset_id}/documents",
            headers=api_headers,
            params={"page": 1, "page_size": 200},
        ),
        "RAGFlow document list",
    )
    documents_data = documents_payload.get("data") or {}
    existing_documents = (
        documents_data.get("documents")
        or documents_data.get("docs")
        or documents_data
        if isinstance(documents_data, dict)
        else documents_data
    )
    if not isinstance(existing_documents, list):
        existing_documents = []
    existing_names = {
        str(item.get("name") or "")
        for item in existing_documents
        if isinstance(item, dict)
    }

    unique_files: list[Path] = []
    seen_hashes: set[str] = set()
    for path in sorted((PROJECT_ROOT / "rag_pdfs").iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".pdf", ".md", ".txt"}:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        unique_files.append(path)

    new_document_ids: list[str] = []
    for path in unique_files:
        if path.name in existing_names:
            continue
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        upload_payload = require_success(
            client.post(
                f"{RAGFLOW_URL}/api/v1/datasets/{dataset_id}/documents",
                headers=api_headers,
                files={"file": (path.name, path.read_bytes(), mime_type)},
                timeout=180,
            ),
            f"RAGFlow upload {path.name}",
        )
        uploaded = upload_payload.get("data") or []
        if isinstance(uploaded, dict):
            uploaded = [uploaded]
        for item in uploaded:
            if isinstance(item, dict) and item.get("id"):
                new_document_ids.append(str(item["id"]))
        print(f"Uploaded {path.name}.")

    if new_document_ids:
        require_success(
            client.post(
                f"{RAGFLOW_URL}/api/v1/datasets/{dataset_id}/documents/parse",
                headers=api_headers,
                json={"document_ids": new_document_ids},
                timeout=60,
            ),
            "RAGFlow document parsing",
        )
        print(f"Started parsing {len(new_document_ids)} unique documents.")
    else:
        print("No new RAGFlow documents needed upload.")

credentials["RAGFLOW_DATASET_ID"] = dataset_id
write_pairs(CREDENTIAL_FILE, credentials)
platform_env = load_pairs(PLATFORM_ENV_FILE)
platform_env["DURIAN_RAGFLOW_DATASET_IDS"] = dataset_id
if "--enable" in sys.argv:
    platform_env["DURIAN_RAGFLOW_ENABLED"] = "true"
else:
    platform_env.setdefault("DURIAN_RAGFLOW_ENABLED", "false")
write_pairs(PLATFORM_ENV_FILE, platform_env)
if "--enable" in sys.argv:
    print("RAGFlow dataset ID stored and live retrieval enabled.")
else:
    print("RAGFlow dataset ID stored; current live retrieval setting preserved.")

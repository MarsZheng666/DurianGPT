#!/usr/bin/env python3
"""Initialize the private Durian GPT RAGFlow and Dify control planes."""

from __future__ import annotations

import http.cookiejar
import json
import os
from pathlib import Path
import secrets
import string
import subprocess
import urllib.error
import urllib.request


PLATFORM_ROOT = Path("/home/admin01/桌面/Desktop/durian-training/platforms")
CREDENTIAL_FILE = PLATFORM_ROOT / ".admin-credentials"
RAGFLOW_URL = "http://127.0.0.1:19380"
DIFY_URL = "http://127.0.0.1:15001"


def load_credentials() -> dict[str, str]:
    credentials: dict[str, str] = {}
    if CREDENTIAL_FILE.exists():
        for line in CREDENTIAL_FILE.read_text(encoding="utf-8").splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            key, value = line.split("=", 1)
            credentials[key.strip()] = value.strip()
    return credentials


def random_password(length: int = 28) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def save_credentials(credentials: dict[str, str]) -> None:
    CREDENTIAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(f"{key}={value}" for key, value in sorted(credentials.items()))
    CREDENTIAL_FILE.write_text(payload + "\n", encoding="utf-8")
    os.chmod(CREDENTIAL_FILE, 0o600)


def request_json(
    opener: urllib.request.OpenerDirector,
    url: str,
    *,
    method: str = "GET",
    payload: dict | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict, dict[str, str]]:
    body = None
    request_headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(
        url,
        data=body,
        headers=request_headers,
        method=method,
    )
    try:
        with opener.open(request, timeout=30) as response:
            raw = response.read().decode("utf-8", errors="replace")
            data = json.loads(raw) if raw else {}
            return response.status, data, dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            data = {"message": raw[:1000]}
        return exc.code, data, dict(exc.headers.items())


def initialize_dify(credentials: dict[str, str]) -> None:
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    status, setup, _ = request_json(opener, f"{DIFY_URL}/console/api/setup")
    if status != 200:
        raise RuntimeError(f"Dify setup status failed: HTTP {status} {setup}")
    if setup.get("step") == "finished":
        print("Dify is already initialized.")
        return

    init_password = credentials.get("DIFY_INIT_PASSWORD")
    if not init_password:
        raise RuntimeError("DIFY_INIT_PASSWORD is missing")

    credentials.setdefault("DIFY_ADMIN_EMAIL", "admin@durian-gpt.example.com")
    credentials.setdefault("DIFY_ADMIN_PASSWORD", random_password())
    save_credentials(credentials)

    status, result, _ = request_json(
        opener,
        f"{DIFY_URL}/console/api/init",
        method="POST",
        payload={"password": init_password},
    )
    if status != 201:
        raise RuntimeError(f"Dify init validation failed: HTTP {status} {result}")

    status, result, _ = request_json(
        opener,
        f"{DIFY_URL}/console/api/setup",
        method="POST",
        payload={
            "email": credentials["DIFY_ADMIN_EMAIL"],
            "name": "Durian GPT Admin",
            "password": credentials["DIFY_ADMIN_PASSWORD"],
            "language": "zh-Hans",
        },
    )
    if status != 201:
        raise RuntimeError(f"Dify setup failed: HTTP {status} {result}")
    print("Dify administrator initialized.")


def ragflow_encrypt(password: str) -> str:
    command = [
        "docker",
        "exec",
        "durian-ragflow-ragflow-cpu-1",
        "python",
        "-c",
        "from api.utils.crypt import crypt; import sys; print(crypt(sys.argv[1]))",
        password,
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    encrypted = result.stdout.strip().splitlines()[-1]
    if not encrypted:
        raise RuntimeError("RAGFlow password encryption returned no data")
    return encrypted


def initialize_ragflow(credentials: dict[str, str]) -> None:
    credentials.setdefault("RAGFLOW_ADMIN_EMAIL", "admin@durian-gpt.example.com")
    credentials.setdefault("RAGFLOW_ADMIN_PASSWORD", random_password())
    save_credentials(credentials)

    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    encrypted = ragflow_encrypt(credentials["RAGFLOW_ADMIN_PASSWORD"])

    status, result, response_headers = request_json(
        opener,
        f"{RAGFLOW_URL}/api/v1/users",
        method="POST",
        payload={
            "nickname": "Durian GPT Admin",
            "email": credentials["RAGFLOW_ADMIN_EMAIL"],
            "password": encrypted,
        },
    )
    if result.get("code") != 0:
        message = str(result.get("message") or "")
        if "already registered" not in message:
            raise RuntimeError(f"RAGFlow registration failed: HTTP {status} {result}")
        status, result, response_headers = request_json(
            opener,
            f"{RAGFLOW_URL}/api/v1/auth/login",
            method="POST",
            payload={
                "email": credentials["RAGFLOW_ADMIN_EMAIL"],
                "password": encrypted,
            },
        )
        if result.get("code") != 0:
            raise RuntimeError(f"RAGFlow login failed: HTTP {status} {result}")

    authorization = (
        response_headers.get("Authorization")
        or response_headers.get("authorization")
        or ""
    )
    auth_headers = {"Authorization": authorization} if authorization else {}
    status, token_result, _ = request_json(
        opener,
        f"{RAGFLOW_URL}/api/v1/system/tokens",
        headers=auth_headers,
    )
    if token_result.get("code") != 0:
        raise RuntimeError(f"RAGFlow token list failed: HTTP {status} {token_result}")
    tokens = token_result.get("data") or []
    api_key = next(
        (
            str(token.get("token"))
            for token in tokens
            if isinstance(token, dict) and str(token.get("token") or "").startswith("ragflow-")
        ),
        "",
    )
    if not api_key:
        status, token_result, _ = request_json(
            opener,
            f"{RAGFLOW_URL}/api/v1/system/tokens",
            method="POST",
            headers=auth_headers,
        )
        if token_result.get("code") != 0:
            raise RuntimeError(f"RAGFlow token creation failed: HTTP {status} {token_result}")
        api_key = str((token_result.get("data") or {}).get("token") or "")
    if not api_key.startswith("ragflow-"):
        raise RuntimeError("RAGFlow returned an invalid API key")
    credentials["RAGFLOW_API_KEY"] = api_key
    save_credentials(credentials)
    print("RAGFlow administrator and API token initialized.")


def main() -> None:
    credentials = load_credentials()
    initialize_dify(credentials)
    initialize_ragflow(credentials)
    print(f"Credentials saved to {CREDENTIAL_FILE} with mode 600.")


if __name__ == "__main__":
    main()

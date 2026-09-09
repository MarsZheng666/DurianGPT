"""LLM Provider 抽象。

约定（对齐 memory 的远端禁改约束）：
- OpenAICompatibleLLM 只用**显式配置**的端点（DURIAN_AGENT_LLM_* 环境变量），
  任一缺失直接抛错——绝不默认落到 127.0.0.1:8010/19380（那是到 durian-server
  的隧道端口，往那发请求 = 消耗远端线上算力）；
- 测试一律用 FakeLLM，离线可跑、零成本、可断言。
"""

from __future__ import annotations

from typing import Callable, Dict, List, Union


class LLMProvider:
    """最小接口：system + user → 文本补全。子类实现 complete()。"""

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:  # pragma: no cover
        raise NotImplementedError


class FakeLLM(LLMProvider):
    """测试替身：返回预设响应，并记录每次调用供断言。

    response 可以是字符串，或是 callable(system, user) -> str。
    """

    def __init__(self, response: Union[str, Callable[[str, str], str]]):
        self._response = response
        self.calls: List[Dict[str, str]] = []

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        self.calls.append({"system": system, "user": user, "temperature": temperature})
        if callable(self._response):
            return self._response(system, user)
        return self._response


class LLMConfigError(RuntimeError):
    """LLM 端点未显式配置。"""


class OpenAICompatibleLLM(LLMProvider):
    """OpenAI 兼容 /chat/completions 端点（vLLM/网关/外部 API 均可）。

    环境变量：
        DURIAN_AGENT_LLM_BASE_URL   例如 https://api.example.com/v1
        DURIAN_AGENT_LLM_API_KEY
        DURIAN_AGENT_LLM_MODEL
    也可以在构造时直接传参覆盖。
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
    ):
        import os

        self.base_url = (base_url or os.environ.get("DURIAN_AGENT_LLM_BASE_URL") or "").rstrip("/")
        self.api_key = api_key or os.environ.get("DURIAN_AGENT_LLM_API_KEY") or ""
        self.model = model or os.environ.get("DURIAN_AGENT_LLM_MODEL") or ""
        self.timeout = timeout
        if not (self.base_url and self.model):
            raise LLMConfigError(
                "LLM 端点未配置：请设置 DURIAN_AGENT_LLM_BASE_URL / "
                "DURIAN_AGENT_LLM_MODEL（不要指向 127.0.0.1:8010/19380 隧道端口）"
            )

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        import json

        import requests

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            json={
                "model": self.model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

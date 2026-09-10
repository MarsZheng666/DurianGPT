"""图片处理（架构文档 §21，任务 #18）：Image → VLM → 结构化描述。

产出三要素（§21）：图片描述 + 图中文字 + 农业语义信息，
供建库侧把图片知识转成可检索文本（chunk_text 进索引）。

通道约定：
- VLMFn 依赖注入：(image_bytes, prompt) -> str；
- 默认通道 qwen_vl()：读 .env / 环境变量 QWEN_VL_MAX_*（外部 API，
  项目既有 VLM 工作流同款，端点构造与 qwen_vl_max_client.py 一致：
  base_url 补 /v1/chat/completions，图片走 base64 data URL）；
- 未配置时抛 VLMConfigError——测试一律用 FakeVLM，不产生真实调用费用。
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

VLMFn = Callable[[bytes, str], str]

VLM_DESCRIBE_PROMPT = """You are analyzing an image from a durian plantation document.

Return JSON only:
{
  "description": "<what the image shows, one or two sentences>",
  "text_in_image": "<any visible text, labels, captions, numbers; empty string if none>",
  "agricultural_semantics": "<agricultural meaning: e.g. disease symptom on leaf, \
pest damage on fruit, irrigation setup, fertilization demo; empty string if not agricultural>"
}

Do not guess facts that are not visible. Answer in Chinese when the image contains Chinese text.
"""


class VLMConfigError(RuntimeError):
    """VLM 通道未配置。"""


def _load_dotenv_quietly() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
    except Exception:
        pass


def qwen_vl(
    *,
    api_key: str = "",
    base_url: str = "",
    model: str = "",
    timeout: float = 60.0,
) -> VLMFn:
    """外部 VLM 通道（OpenAI 兼容 + base64 data URL，对齐项目既有配置）。"""
    _load_dotenv_quietly()
    api_key = api_key or os.environ.get("QWEN_VL_MAX_API_KEY", "")
    base_url = (base_url or os.environ.get("QWEN_VL_MAX_BASE_URL", "")).rstrip("/")
    model = model or os.environ.get("QWEN_VL_MAX_MODEL", "")
    if not (api_key and base_url and model):
        raise VLMConfigError(
            "VLM 通道未配置：需要 QWEN_VL_MAX_API_KEY / QWEN_VL_MAX_BASE_URL / "
            "QWEN_VL_MAX_MODEL（外部付费 API，测试请用替身，不要隐式真实调用）"
        )
    url = (f"{base_url}/chat/completions" if base_url.endswith("/v1")
           else f"{base_url}/v1/chat/completions")

    def call(image_bytes: bytes, prompt: str) -> str:
        import requests

        data_url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    return call


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str):
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if not brace:
            return None
        text = brace.group(0)
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def describe_image(image_bytes: bytes, vlm_fn: VLMFn) -> Dict[str, str]:
    """图片 → {description, text_in_image, agricultural_semantics}。

    VLM 输出非法时保守降级：整体描述放 description，其余字段空串
    （不让一张解析失败的图片拖垮整条建库管线）。
    """
    raw = vlm_fn(image_bytes, VLM_DESCRIBE_PROMPT)
    parsed = _extract_json(raw)
    if parsed is None:
        return {"description": (raw or "").strip()[:500],
                "text_in_image": "", "agricultural_semantics": ""}

    def field(name: str) -> str:
        value = parsed.get(name)
        return value.strip() if isinstance(value, str) else ""

    return {
        "description": field("description"),
        "text_in_image": field("text_in_image"),
        "agricultural_semantics": field("agricultural_semantics"),
    }


def image_description_to_text(info: Dict[str, str]) -> str:
    """三要素 → 可检索文本（进索引的 chunk_text 形态）。"""
    parts = [info.get("description", "")]
    if info.get("text_in_image"):
        parts.append(f"图中文字：{info['text_in_image']}")
    if info.get("agricultural_semantics"):
        parts.append(f"农业语义：{info['agricultural_semantics']}")
    return "\n".join(p for p in parts if p).strip()


def enrich_image_blocks(
    blocks: Sequence[Dict[str, Any]],
    vlm_fn: VLMFn,
    *,
    image_loader: Optional[Callable[[str], bytes]] = None,
) -> Sequence[Dict[str, Any]]:
    """建库接入：对 image/scan_page 块做 VLM 富化（text ← 三要素文本）。

    image_loader：image_path → bytes（默认按本地路径读取）。
    单块失败不中断整批（该块保持原状，附 enrich_error 标记）。
    """
    def _default_loader(path: str) -> bytes:
        return Path(path).read_bytes()

    loader = image_loader or _default_loader
    enriched = []
    for block in blocks:
        if block.get("block_type") not in ("image", "scan_page"):
            enriched.append(block)
            continue
        new_block = dict(block)
        image_path = block.get("image_path")
        if not image_path:
            enriched.append(new_block)
            continue
        try:
            info = describe_image(loader(image_path), vlm_fn)
            text = image_description_to_text(info)
            if text:
                new_block["text"] = text
                new_block["vlm_info"] = info
        except Exception as exc:
            new_block["enrich_error"] = str(exc)[:200]
        enriched.append(new_block)
    return enriched

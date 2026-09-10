"""ReAct 循环（架构文档 §17/§49，任务 #39）。

    Reason → Select Tool → Action → Observation → Reason → …

- 文本协议（provider 无关）：LLM 每步输出恰好一个 JSON——
  {"action": {"tool": ..., "args": {...}}} 或 {"final_answer": "..."};
  不依赖 OpenAI 原生 function calling，FakeLLM 可完全模拟；
- §49 约束进 system prompt：专业结论先取证 / 不执行未授权工具 /
  敏感写操作需确认（会被 Registry 拦截为草稿）；
- 步数上限 max_steps：防工具环（每步一次 LLM 调用 + 至多一次工具执行）；
- LLM 失败/输出非法：降级为诚实说明（不猜、不自由发挥）。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage

from durian_agent.llm import LLMProvider
from durian_agent.tools.base import ToolRegistry

MAX_STEPS = 6

_SYSTEM_TEMPLATE = """For complex plantation tasks, dynamically use available tools.

If the task requires agricultural professional judgment and no validated \
agricultural evidence exists in the current state, call agriculture_rag \
before forming a professional conclusion.

Never execute a tool the user's role is not authorized to access.

Sensitive write operations require confirmation — propose them; the system \
will ask the user.

Available tools:
{tool_specs}

Respond with EXACTLY ONE JSON object per step, nothing else:
{{"action": {{"tool": "<name>", "args": {{...}}}}}}
or
{{"final_answer": "<your complete answer in the user's language>"}}

Rules:
- One tool call per step; the tool result will be shown to you as the next step.
- Cite evidence chunk ids like [CHUNK_1] when you used agriculture_rag.
"""


def build_system_prompt(registry: ToolRegistry, role: str = "manager") -> str:
    """§39：只列该角色允许的工具——Agent 看不到未授权工具。"""
    return _SYSTEM_TEMPLATE.format(tool_specs=registry.spec_text(role))


def parse_step(text: str) -> Optional[Dict[str, Any]]:
    """解析一步 LLM 输出：action 或 final_answer。非法返回 None。"""
    if not isinstance(text, str):
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    raw = fenced.group(1) if fenced else text.strip()
    brace = re.search(r"\{.*\}", raw, re.DOTALL)
    if not brace:
        return None
    try:
        parsed = json.loads(brace.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    action = parsed.get("action")
    if isinstance(action, dict) and isinstance(action.get("tool"), str):
        return {"action": {"tool": action["tool"],
                           "args": action.get("args") or {}}}
    if isinstance(parsed.get("final_answer"), str) and parsed["final_answer"].strip():
        return {"final_answer": parsed["final_answer"].strip()}
    return None


class ReActEngine:
    """一步一推理的 ReAct 引擎（作为 LangGraph reactAgent 节点内核）。"""

    def __init__(self, llm: LLMProvider, registry: ToolRegistry,
                 max_steps: int = MAX_STEPS):
        self.llm = llm
        self.registry = registry
        self.max_steps = max_steps
        self._prompt_cache: Dict[str, str] = {}   # role → prompt

    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """一步推理。返回图状态补丁：
        - {"pending_tool_calls": [action], "messages": [AIMessage]}
        - {"final_answer": ..., "react_done": True, "messages": [AIMessage]}
        - 降级：{"final_answer": 诚实说明, "react_done": True, ...}
        """
        if state.get("react_steps", 0) >= self.max_steps:
            return self._degrade("已达到工具调用步数上限，未能完成任务编排。")

        role = state.get("role", "worker")
        if role not in self._prompt_cache:
            self._prompt_cache[role] = build_system_prompt(self.registry, role)
        transcript = self._render_transcript(state)
        try:
            raw = self.llm.complete(self._prompt_cache[role], transcript)
        except Exception:
            return self._degrade("推理服务不可用，本次复杂任务未能完成。")

        parsed = parse_step(raw)
        if parsed is None:
            return self._degrade("推理输出无法解析为工具动作，本次未能完成。")

        if "action" in parsed:
            return {
                "pending_tool_calls": [parsed["action"]],
                "messages": [AIMessage(content=json.dumps(
                    parsed, ensure_ascii=False))],
            }
        return {
            "final_answer": parsed["final_answer"],
            "react_done": True,
            "messages": [AIMessage(content=parsed["final_answer"])],
        }

    # ──────────── 内部 ────────────

    def _degrade(self, message: str) -> Dict[str, Any]:
        return {"final_answer": message, "react_done": True,
                "messages": [AIMessage(content=message)]}

    def _render_transcript(self, state: Dict[str, Any]) -> str:
        lines = []
        for msg in state.get("messages") or []:
            from langchain_core.messages import HumanMessage, ToolMessage
            if isinstance(msg, HumanMessage):
                lines.append(f"[用户] {msg.content}")
            elif isinstance(msg, ToolMessage):
                lines.append(f"[工具结果] {msg.content}")
            elif isinstance(msg, AIMessage):
                content = str(msg.content)
                if content.strip().startswith("{"):
                    lines.append(f"[上一步动作] {content}")
                # 最终回答不进 transcript（react_done 后不再调用）
        return "\n".join(lines)

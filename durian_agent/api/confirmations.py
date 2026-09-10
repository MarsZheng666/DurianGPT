"""敏感操作确认存储（架构文档 §41/§60，任务 #10）。

§41 流程的 API 侧承接：
    Agent 生成待执行操作（toolNode 拦截为草稿）→ API 返回
    pending_confirmation → 用户经 /api/chat/confirm 回传 →
    approved=true 时以 ctx.confirmed=True 重新执行工具。

内存版：进程生命周期内有效；持久化在阶段五 #64 存储层。
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional


class ConfirmationStore:
    """confirmation_id → 待确认操作（含 thread 归属）。"""

    def __init__(self):
        self._items: Dict[str, Dict[str, Any]] = {}

    def register(self, thread_id: str, tool: str, args: Dict[str, Any],
                 confirmation_id: Optional[str] = None,
                 role: str = "manager", user_id: str = "") -> str:
        cid = confirmation_id or f"cfm-{uuid.uuid4().hex[:10]}"
        self._items[cid] = {
            "thread_id": thread_id, "tool": tool, "role": role,
            "user_id": user_id,
            "args": args or {}, "created": time.time(),
        }
        return cid

    def get(self, confirmation_id: str) -> Optional[Dict[str, Any]]:
        return self._items.get(confirmation_id)

    def pop(self, confirmation_id: str) -> Optional[Dict[str, Any]]:
        return self._items.pop(confirmation_id, None)

    def __len__(self) -> int:
        return len(self._items)

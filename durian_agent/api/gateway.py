"""Conversation Gateway（架构文档 §2，任务 #6）——阶段一范围。

网关职责（§2）：Auth / User / Role / Thread / Language。

阶段一实现（诚实标注边界）：
- 身份：请求头 X-User-Id / X-Role（开发模式）。**身份可伪造是前提问题**
  （已知约束），真正的认证与权限系统在阶段五（#49 Tool Permission /
  用户级权限隔离方案）落地；
- 角色：白名单 {worker, manager, admin}，非法值拒绝（不给静默降级）；
- Thread：内存注册表（thread_id → 元数据），缺省自动分配；
  持久化会话元数据是 #64 存储层的职责；
- Language：透传（默认 auto，实际语言由 SemanticParse 判定）。
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

VALID_ROLES = ("worker", "manager", "admin")
DEFAULT_ROLE = "worker"


class GatewayError(ValueError):
    """网关层输入错误（角色非法/消息为空等）。"""


class ThreadRegistry:
    """会话线程注册表（内存版；持久化在 #64 存储层）。"""

    def __init__(self):
        self._threads: Dict[str, Dict[str, Any]] = {}

    def resolve(self, thread_id: Optional[str]) -> str:
        """缺省分配新 thread_id；已存在则原样返回。"""
        if thread_id and thread_id.strip():
            resolved = thread_id.strip()
        else:
            resolved = f"thread-{uuid.uuid4().hex[:12]}"
        self._threads.setdefault(resolved, {"created": True})
        return resolved

    def get(self, thread_id: str) -> Optional[Dict[str, Any]]:
        return self._threads.get(thread_id)

    def __len__(self) -> int:
        return len(self._threads)


class ConversationGateway:
    """解析请求上下文：身份/角色/线程/语言 → 图调用参数。"""

    def __init__(self, registry: Optional[ThreadRegistry] = None):
        self.registry = registry or ThreadRegistry()

    def resolve_context(
        self,
        *,
        user_id: Optional[str] = None,
        role: Optional[str] = None,
        thread_id: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """校验并归一请求上下文。

        - role 缺省 worker（最小权限）；非法值抛 GatewayError（不静默降级）；
        - user_id 缺省 anonymous（开发模式，身份认证属阶段五）；
        - language 缺省 auto（SemanticParse 判定）。
        """
        normalized_role = (role or DEFAULT_ROLE).strip().lower()
        if normalized_role not in VALID_ROLES:
            raise GatewayError(f"非法角色: {role!r}，可选: {VALID_ROLES}")

        normalized_lang = (language or "auto").strip().lower()
        if normalized_lang not in ("auto", "zh", "en", "th", "ms"):
            raise GatewayError(f"非法语言: {language!r}，可选: auto/zh/en/th/ms")

        return {
            "user_id": (user_id or "anonymous").strip() or "anonymous",
            "role": normalized_role,
            "thread_id": self.registry.resolve(thread_id),
            "language": normalized_lang,
        }

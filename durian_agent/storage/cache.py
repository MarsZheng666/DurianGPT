"""Redis 语义缓存（架构文档 §58，任务 #62）。

§58 职责：session hot state / short cache / idempotency / rate limit。

本机未装 redis 客户端——提供**同接口的进程内实现**（RedisLike），
生产切真 Redis 时按同方法签名替换连接层即可（接口与 redis-py 的
get/set/expire/incr 语义对齐）。
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional


class RedisLike:
    """进程内 Redis 语义实现：TTL / 幂等 / 计数限流。"""

    def __init__(self):
        self._store: Dict[str, Any] = {}
        self._expires: Dict[str, float] = {}

    def _alive(self, key: str) -> bool:
        deadline = self._expires.get(key)
        if deadline is not None and time.time() >= deadline:
            self._store.pop(key, None)
            self._expires.pop(key, None)
            return False
        return key in self._store

    def set(self, key: str, value: Any, ex: Optional[int] = None) -> None:
        self._store[key] = value
        if ex is not None:
            self._expires[key] = time.time() + ex
        else:
            self._expires.pop(key, None)

    def get(self, key: str) -> Any:
        return self._store[key] if self._alive(key) else None

    def delete(self, key: str) -> None:
        self._store.pop(key, None)
        self._expires.pop(key, None)

    def get_or_set(self, key: str, factory, ex: Optional[int] = None) -> Any:
        """幂等语义（§42 网络层去重的缓存形态）。"""
        if self._alive(key):
            return self._store[key]
        value = factory()
        self.set(key, value, ex=ex)
        return value

    def incr(self, key: str, ex: Optional[int] = None) -> int:
        current = int(self._store.get(key, 0)) if self._alive(key) else 0
        current += 1
        self.set(key, current, ex=ex)
        return current

    def rate_allow(self, key: str, limit: int, window_sec: int) -> bool:
        """固定窗口限流：窗口内第 limit 次之后拒绝。"""
        count = self.incr(key, ex=window_sec)
        return count <= limit

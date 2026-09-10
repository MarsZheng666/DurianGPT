"""关系存储（架构文档 §58，任务 #62）：八表 SQLite。

    users / roles / permissions / orchards / plots / tasks /
    tool_audit / conversation_metadata

- SQLite 直连（与 #56 checkpointer 同栈，部署最简；换 PG 时仅换 DDL 方言）；
- §58 关系库职责：账号/权限基准数据、工单持久化（替代 InMemoryTasks
  的进程态）、tool_audit 审计留痕（§52 的持久化落地）、会话元数据。
- Milvus（VectorIndex #25）与 Checkpoint store（#56）已在各自模块交付。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

#: §58 八表 DDL（冻结清单）
SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    role TEXT NOT NULL DEFAULT 'worker',
    display_name TEXT NOT NULL DEFAULT '',
    orchard_scope TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS roles (
    role TEXT PRIMARY KEY,
    description TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS permissions (
    role TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    PRIMARY KEY (role, tool_name)
);
CREATE TABLE IF NOT EXISTS orchards (
    orchard_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    name TEXT NOT NULL DEFAULT '',
    area_ha REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS plots (
    plot_id TEXT PRIMARY KEY,
    orchard_id TEXT NOT NULL,
    cultivar TEXT NOT NULL DEFAULT '',
    trees INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    orchard TEXT NOT NULL DEFAULT '',
    plot TEXT NOT NULL DEFAULT '',
    priority TEXT NOT NULL DEFAULT 'normal',
    status TEXT NOT NULL DEFAULT 'pending',
    assignee TEXT NOT NULL DEFAULT '',
    idempotency_key TEXT UNIQUE,
    created_by TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS tool_audit (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id TEXT NOT NULL DEFAULT '',
    thread_id TEXT NOT NULL DEFAULT '',
    user_id TEXT NOT NULL DEFAULT '',
    tool_name TEXT NOT NULL,
    arguments TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS conversation_metadata (
    thread_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    user_id TEXT NOT NULL DEFAULT '',
    route TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

TABLES = ("users", "roles", "permissions", "orchards", "plots",
          "tasks", "tool_audit", "conversation_metadata")


class RelationalStore:
    """八表 SQLite 存储（线程安全连接，进程级持有）。"""

    def __init__(self, path: Union[str, Path] = ":memory:"):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA_DDL)
        self.conn.commit()

    # ── 通用 ─────────────────────────────────────────────
    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        self.conn.execute(sql, params)
        self.conn.commit()

    def query(self, sql: str,
              params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        rows = self.conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def tables(self) -> List[str]:
        rows = self.query("SELECT name FROM sqlite_master "
                          "WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        return sorted(r["name"] for r in rows)

    # ── users / permissions ───────────────────────────────
    def upsert_user(self, user_id: str, *, tenant_id: str = "default",
                    role: str = "worker", display_name: str = "",
                    orchard_scope: str = "") -> None:
        self.execute(
            "INSERT OR REPLACE INTO users "
            "(user_id, tenant_id, role, display_name, orchard_scope) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, tenant_id, role, display_name, orchard_scope))

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        rows = self.query("SELECT * FROM users WHERE user_id = ?", (user_id,))
        return rows[0] if rows else None

    def set_permission(self, role: str, tool_name: str) -> None:
        self.execute("INSERT OR IGNORE INTO permissions VALUES (?, ?)",
                     (role, tool_name))

    def permissions_of(self, role: str) -> List[str]:
        return [r["tool_name"] for r in self.query(
            "SELECT tool_name FROM permissions WHERE role = ?", (role,))]

    # ── tasks（含 §42 幂等）──────────────────────────────
    def create_task(self, draft: Dict[str, Any],
                    idempotency_key: str) -> Optional[Dict[str, Any]]:
        existing = self.query(
            "SELECT * FROM tasks WHERE idempotency_key = ?",
            (idempotency_key,))
        if existing:
            return existing[0]                       # 幂等命中
        task_id = draft.get("task_id") or f"T-{id(abs(hash(idempotency_key))) % 100000}"
        try:
            self.execute(
                "INSERT INTO tasks (task_id, title, description, orchard, "
                "plot, priority, status, idempotency_key, created_by) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (task_id, draft.get("title", ""), draft.get("description", ""),
                 draft.get("orchard", ""), draft.get("plot", ""),
                 draft.get("priority", "normal"), "pending",
                 idempotency_key, draft.get("created_by", "")))
        except sqlite3.IntegrityError:
            return self.query("SELECT * FROM tasks WHERE idempotency_key = ?",
                              (idempotency_key,))[0]
        return self.get_task(task_id)

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        rows = self.query("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
        return rows[0] if rows else None

    def update_task(self, task_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        allowed = {k: v for k, v in patch.items()
                   if k in ("status", "assignee", "title", "description")}
        for column, value in allowed.items():
            self.execute(f"UPDATE tasks SET {column} = ? WHERE task_id = ?",
                         (value, task_id))
        return self.get_task(task_id)

    # ── tool_audit ────────────────────────────────────────
    def audit_tool(self, *, tool_name: str, arguments: Dict[str, Any],
                   status: str = "ok", trace_id: str = "", thread_id: str = "",
                   user_id: str = "") -> None:
        self.execute(
            "INSERT INTO tool_audit (trace_id, thread_id, user_id, tool_name, "
            "arguments, status) VALUES (?, ?, ?, ?, ?, ?)",
            (trace_id, thread_id, user_id, tool_name,
             json.dumps(arguments, ensure_ascii=False), status))

    def audit_trail(self, thread_id: str) -> List[Dict[str, Any]]:
        return self.query(
            "SELECT * FROM tool_audit WHERE thread_id = ? "
            "ORDER BY audit_id", (thread_id,))

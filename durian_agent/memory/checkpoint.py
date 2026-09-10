"""Checkpoint 持久化（架构文档 §32，任务 #56）。

- thread_id 是**业务会话 ID**（非系统线程 ID）——LangGraph Checkpointer
  按 configurable.thread_id 隔离，图的 invoke 入口已统一注入；
- sqlite_checkpointer(path)：SqliteSaver（langgraph-checkpoint-sqlite），
  进程重启后会话可续——这是「持久化」的验收点（两个图实例共享同一
  sqlite 文件，会话上下文跨实例延续）；
- memory_checkpointer()：MemorySaver（开发/测试，进程内）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / \
    "durian_agent_checkpoints.sqlite"


def sqlite_checkpointer(path: Union[str, Path] = DEFAULT_DB_PATH):
    """SQLite 持久化 checkpointer（长生命周期连接，服务进程持有）。

    from_conn_string 返回上下文管理器（with 语义），不适合进程级持有；
    直接用 sqlite3 连接构造（check_same_thread=False 兼容 FastAPI 线程池）。
    """
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn)


def memory_checkpointer():
    """进程内 checkpointer（开发/测试）。"""
    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver()

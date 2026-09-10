"""任务 #56 验证：Checkpoint 持久化（§32：thread_id 隔离，业务会话 ID）。

验证点：
1. 跨实例续接：图实例 A 写入会话 → 全新图实例 B（同 sqlite 文件）
   读到历史——持久化的核心价值（进程重启可续）；
2. thread_id 隔离：不同 thread 互不可见；
3. 同 thread 多轮累积。
"""

import tempfile
import unittest
from pathlib import Path

from durian_agent.graph import DurianAgentGraph
from durian_agent.llm import FakeLLM
from durian_agent.memory.checkpoint import (
    memory_checkpointer,
    sqlite_checkpointer,
)


def simple_graph(checkpointer):
    """SIMPLE 路径图（历史消息可观察）。"""
    return DurianAgentGraph(
        llm=FakeLLM("好的。"), checkpointer=checkpointer)


class TestSqliteCheckpoint(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = Path(self.tmp) / "cp.sqlite"

    def test_cross_instance_continuity(self):
        """持久化验收：实例 A 的会话在全新实例 B 中延续。"""
        graph_a = simple_graph(sqlite_checkpointer(self.db))
        r1 = graph_a.invoke("第一句", thread_id="persist-t")
        self.assertEqual(len(r1["messages"]), 2)   # human + ai

        # 全新图实例（模拟进程重启），同一 sqlite 文件
        graph_b = simple_graph(sqlite_checkpointer(self.db))
        r2 = graph_b.invoke("第二句", thread_id="persist-t")
        # 历史延续：第一轮的 Human/AI 仍在
        contents = [m.content for m in r2["messages"]]
        self.assertIn("第一句", contents)
        self.assertIn("第二句", contents)
        self.assertEqual(len(r2["messages"]), 4)

    def test_thread_isolation(self):
        graph = simple_graph(sqlite_checkpointer(self.db))
        graph.invoke("线程甲的消息", thread_id="thread-a")
        r = graph.invoke("线程乙的消息", thread_id="thread-b")
        contents = [m.content for m in r["messages"]]
        self.assertNotIn("线程甲的消息", contents)

    def test_same_thread_accumulates(self):
        graph = simple_graph(sqlite_checkpointer(self.db))
        graph.invoke("一", thread_id="acc")
        graph.invoke("二", thread_id="acc")
        r3 = graph.invoke("三", thread_id="acc")
        self.assertEqual(len(r3["messages"]), 6)

    def test_memory_checkpointer_in_process(self):
        graph = simple_graph(memory_checkpointer())
        graph.invoke("一", thread_id="m")
        r = graph.invoke("二", thread_id="m")
        self.assertEqual(len(r["messages"]), 4)


if __name__ == "__main__":
    unittest.main()

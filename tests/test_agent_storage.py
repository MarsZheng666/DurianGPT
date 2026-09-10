"""任务 #62 验证：存储层（§58：关系库八表 + Redis 语义 + 已有 Milvus/Checkpoint）。"""

import unittest

from durian_agent.storage.cache import RedisLike
from durian_agent.storage.relational import SCHEMA_DDL, TABLES, RelationalStore


class TestRelationalStore(unittest.TestCase):

    def setUp(self):
        self.store = RelationalStore()   # 内存库

    def test_eight_tables_created(self):
        self.assertEqual(len(TABLES), 8)
        self.assertEqual(self.store.tables(), sorted(TABLES))

    def test_users_and_permissions(self):
        self.store.upsert_user("u-1", tenant_id="IOI", role="manager",
                               display_name="阿明", orchard_scope="ORCHARD_3")
        user = self.store.get_user("u-1")
        self.assertEqual(user["role"], "manager")
        self.assertEqual(user["tenant_id"], "IOI")

        self.store.set_permission("manager", "weather")
        self.store.set_permission("manager", "weather")   # 幂等
        self.assertIn("weather", self.store.permissions_of("manager"))
        self.assertEqual(len(self.store.permissions_of("manager")), 1)

    def test_task_idempotency(self):
        """§42 工单持久化 + 幂等键去重（网络/Agent 重放）。"""
        draft = {"title": "排水巡检", "orchard": "ORCHARD_3",
                 "created_by": "u-1"}
        first = self.store.create_task(draft, "req-1")
        replay = self.store.create_task(draft, "req-1")   # 重放
        self.assertEqual(first["task_id"], replay["task_id"])
        rows = self.store.query("SELECT * FROM tasks")
        self.assertEqual(len(rows), 1)

    def test_task_update(self):
        task = self.store.create_task({"title": "施肥"}, "k")
        updated = self.store.update_task(
            task["task_id"], {"status": "done", "assignee": "阿明"})
        self.assertEqual(updated["status"], "done")
        self.assertEqual(updated["assignee"], "阿明")

    def test_tool_audit_trail(self):
        self.store.audit_tool(tool_name="weather",
                              arguments={"days": 3}, status="ok",
                              thread_id="t-a", user_id="u-1")
        self.store.audit_tool(tool_name="task_create",
                              arguments={"title": "巡检"}, status="ok",
                              thread_id="t-a", user_id="u-1")
        trail = self.store.audit_trail("t-a")
        self.assertEqual([t["tool_name"] for t in trail],
                         ["weather", "task_create"])
        self.assertEqual(trail[0]["user_id"], "u-1")

    def test_conversation_metadata_upsert(self):
        self.store.execute(
            "INSERT OR REPLACE INTO conversation_metadata "
            "(thread_id, tenant_id, user_id, route) VALUES (?, ?, ?, ?)",
            ("t-1", "IOI", "u-1", "COMPLEX_TASK"))
        rows = self.store.query(
            "SELECT * FROM conversation_metadata WHERE thread_id = 't-1'")
        self.assertEqual(rows[0]["route"], "COMPLEX_TASK")


class TestRedisLike(unittest.TestCase):

    def test_ttl_expiry(self):
        cache = RedisLike()
        cache.set("k", "v", ex=0)          # 立即过期
        self.assertIsNone(cache.get("k"))

    def test_get_or_set_idempotency(self):
        cache = RedisLike()
        calls = []

        def factory():
            calls.append(1)
            return "computed"

        self.assertEqual(cache.get_or_set("idem", factory), "computed")
        self.assertEqual(cache.get_or_set("idem", factory), "computed")
        self.assertEqual(len(calls), 1)    # 第二次走缓存

    def test_rate_limit(self):
        cache = RedisLike()
        allowed = [cache.rate_allow("u-1:chat", limit=3, window_sec=60)
                   for _ in range(5)]
        self.assertEqual(allowed, [True, True, True, False, False])

    def test_persistence_to_disk(self):
        import tempfile
        from pathlib import Path

        tmp = Path(tempfile.mkdtemp()) / "rel.db"
        store = RelationalStore(tmp)
        store.upsert_user("u-9", role="admin")
        # 新实例（模拟重启）读同一文件
        reopened = RelationalStore(tmp)
        self.assertEqual(reopened.get_user("u-9")["role"], "admin")


if __name__ == "__main__":
    unittest.main()

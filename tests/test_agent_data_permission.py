"""任务 #53 验证：Data Permission（§40：tenant/role_scope/orchard_scope 联合过滤）。"""

import tempfile
import unittest
from pathlib import Path

from durian_agent.rag.index import VectorIndex
from durian_agent.rag.permissions import (
    ALL_ORCHARDS,
    DataScope,
    build_scope_expr,
    normalize_orchard_scope,
)


def fake_embed(texts):
    """按租户关键词分桶的确定性嵌入（3 维正交）。"""
    vectors = []
    for t in texts:
        if "IOI" in t or "ioi" in t:
            vectors.append([1.0, 0.0, 0.0])
        elif "manager" in t or "管理" in t:
            vectors.append([0.0, 1.0, 0.0])
        else:
            vectors.append([0.0, 0.0, 1.0])
    return vectors


RECORDS = [
    {"chunk_id": "pub-default", "text": "IOI 公开手册：榴莲种植通识",
     "document_id": "d1", "tenant_id": "IOI",
     "role_scope": ["worker", "manager"], "orchard_scope": []},
    {"chunk_id": "mgr-only", "text": "IOI 管理手册：manager 专读",
     "document_id": "d2", "tenant_id": "IOI",
     "role_scope": ["manager"], "orchard_scope": []},
    {"chunk_id": "orchard3", "text": "IOI 三号园专区：管理专属",
     "document_id": "d3", "tenant_id": "IOI",
     "role_scope": ["worker", "manager"], "orchard_scope": ["ORCHARD_3"]},
    {"chunk_id": "other-tenant", "text": "IOI 别家租户资料",
     "document_id": "d4", "tenant_id": "OTHER",
     "role_scope": ["worker", "manager"], "orchard_scope": []},
]


class TestScopeExpr(unittest.TestCase):

    def test_expr_three_conditions(self):
        expr = build_scope_expr(DataScope(
            tenant_id="IOI", role="worker", orchard_scope=["ORCHARD_3"]))
        self.assertIn('tenant_id == "IOI"', expr)
        self.assertIn('array_contains(role_scope, "worker")', expr)
        self.assertIn('array_contains(orchard_scope, "ORCHARD_3")', expr)
        self.assertIn(f'array_contains(orchard_scope, "{ALL_ORCHARDS}")', expr)
        self.assertIn(" and ", expr)

    def test_admin_skips_role_filter(self):
        expr = build_scope_expr(DataScope(tenant_id="IOI", role="admin"))
        self.assertNotIn("role_scope", expr)

    def test_normalize_sentinel(self):
        self.assertEqual(normalize_orchard_scope([]), [ALL_ORCHARDS])
        self.assertEqual(normalize_orchard_scope(None), [ALL_ORCHARDS])
        self.assertEqual(normalize_orchard_scope(["ORCHARD_3"]), ["ORCHARD_3"])


class TestVectorIndexScope(unittest.TestCase):
    """milvus-lite 实测：三条件联合过滤的真实隔离。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.index = VectorIndex(Path(cls.tmp) / "scope.db",
                                embed_fn=fake_embed, dim=3)
        cls.index.build(RECORDS)

    def _search(self, scope):
        return [h["chunk_id"] for h in self.index.search(
            "IOI", top_k=10, expr=build_scope_expr(scope))]

    def test_worker_of_tenant(self):
        # 用户园区范围为空 = 全部园区权限（可看受限文档）
        hits = self._search(DataScope(tenant_id="IOI", role="worker"))
        self.assertIn("pub-default", hits)
        self.assertIn("orchard3", hits)            # 全部园区权限含三号园
        self.assertNotIn("mgr-only", hits)         # manager 专读不可见
        self.assertNotIn("other-tenant", hits)     # 跨租户不可见

    def test_tenant_isolation(self):
        hits = self._search(DataScope(tenant_id="OTHER", role="worker"))
        self.assertEqual(hits, ["other-tenant"])

    def test_orchard_restriction(self):
        """用户园区范围=ORCHARD_5：三号园专区不可见，全园区文档可见。"""
        hits = self._search(DataScope(tenant_id="IOI", role="worker",
                                      orchard_scope=["ORCHARD_5"]))
        self.assertIn("pub-default", hits)
        self.assertNotIn("orchard3", hits)

    def test_manager_sees_more(self):
        hits = self._search(DataScope(tenant_id="IOI", role="manager"))
        self.assertIn("mgr-only", hits)

    def test_admin_no_role_filter(self):
        hits = self._search(DataScope(tenant_id="IOI", role="admin"))
        self.assertIn("mgr-only", hits)
        self.assertNotIn("other-tenant", hits)     # 租户墙仍在


if __name__ == "__main__":
    unittest.main()

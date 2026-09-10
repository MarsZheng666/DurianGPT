"""任务 #66 验证：三服务拆分（§59）。"""

import unittest

from fastapi.testclient import TestClient

from durian_agent.api.app import create_app
from durian_agent.services import create_rag_service, create_tool_service
from durian_agent.tools import build_default_registry


HITS = {
    "hits": {
        "bm25_original": [
            {"chunk_id": "c1", "score": 4.0, "text": "炭疽病用波尔多液防治",
             "record": {"document_id": "植保手册"}},
            {"chunk_id": "c2", "score": 2.0, "text": "炭疽病雨季高发注意排水",
             "record": {}},
        ],
        "dense_canonical": [], "dense_original": [], "bm25_expanded": [],
    },
    "rankings": {"bm25_original": ["c1", "c2"]},
    "queries": {},
}


class FakeRetriever:
    def recall(self, query, *, top_k_each=10, expr=None):
        return HITS


class TestServiceSplit(unittest.TestCase):

    def test_agent_service(self):
        """Agent Service（§59：LangGraph/NLU/Router/ReAct/Memory）。"""
        client = TestClient(create_app())
        self.assertEqual(client.get("/health").json()["status"], "ok")
        resp = client.post("/api/chat", json={"message": "你好"})
        self.assertEqual(resp.status_code, 200)
        self.assertIn("route", resp.json())

    def test_rag_service_standalone(self):
        """RAG Service 独立：检索即服务，契约与 Agent 内嵌一致。"""
        client = TestClient(create_rag_service(FakeRetriever()))
        self.assertEqual(client.get("/health").json()["service"], "rag")
        resp = client.post("/api/rag/search",
                           json={"query": "炭疽病怎么防治", "top_k": 1})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["documents"][0]["chunk_id"], "c1")
        self.assertTrue(body["evidence_sufficient"])

    def test_rag_service_validation(self):
        client = TestClient(create_rag_service(FakeRetriever()))
        self.assertEqual(client.post("/api/rag/search", json={"query": ""})
                         .status_code, 422)

    def test_tool_service_standalone(self):
        """Business Tool Service 独立：工具清单 + 执行。"""
        registry = build_default_registry()
        client = TestClient(create_tool_service(registry))
        health = client.get("/health").json()
        self.assertEqual(health["service"], "tool")
        self.assertIn("weather", health["tools"])

        tools = client.get("/tools").json()
        names = {t["name"] for t in tools}
        self.assertIn("agriculture_rag", names)
        self.assertIn("task_create", names)

        resp = client.post("/tools/execute", json={
            "tool": "weather",
            "args": {"location": "ORCHARD_3", "days": 2}})
        self.assertEqual(resp.status_code, 200)
        self.assertIn("降雨", resp.json()["observation"])

    def test_tool_service_unknown_tool_404(self):
        client = TestClient(create_tool_service(build_default_registry()))
        self.assertEqual(client.post("/tools/execute",
                                     json={"tool": "nope"}).status_code, 404)


if __name__ == "__main__":
    unittest.main()

"""任务 #52 验证：多用户隔离（§37 五维 + checkpoint key）。"""

import unittest

from langgraph.checkpoint.memory import MemorySaver

from durian_agent.api.gateway import ConversationGateway
from durian_agent.graph import DurianAgentGraph
from durian_agent.llm import FakeLLM


def make_graph():
    return DurianAgentGraph(llm=FakeLLM("好的。"),
                            checkpointer=MemorySaver())


class TestCheckpointKeyComposition(unittest.TestCase):

    def test_gateway_composes_key(self):
        context = ConversationGateway().resolve_context(
            tenant_id="IOI", user_id="u-1", role="manager",
            thread_id="t-9")
        self.assertEqual(context["checkpoint_thread_id"], "IOI:u-1:t-9")
        self.assertEqual(context["tenant_id"], "IOI")

    def test_defaults_fill_placeholders(self):
        context = ConversationGateway().resolve_context(thread_id="t")
        self.assertEqual(context["checkpoint_thread_id"],
                         "default:anonymous:t")


class TestIsolation(unittest.TestCase):

    def test_same_thread_different_users_isolated(self):
        """同名 thread：不同用户的会话互不可见（§37 核心）。"""
        graph = make_graph()
        graph.invoke("用户甲的悄悄话", thread_id="shared",
                     user_id="alice", tenant_id="IOI")
        bob = graph.invoke("用户乙的问题", thread_id="shared",
                           user_id="bob", tenant_id="IOI")
        contents = [m.content for m in bob["messages"]]
        self.assertNotIn("用户甲的悄悄话", contents)
        # 甲续接自己的会话不受影响
        alice = graph.invoke("继续", thread_id="shared",
                             user_id="alice", tenant_id="IOI")
        alice_contents = [m.content for m in alice["messages"]]
        self.assertIn("用户甲的悄悄话", alice_contents)

    def test_same_thread_different_tenants_isolated(self):
        graph = make_graph()
        graph.invoke("租户A的内部讨论", thread_id="t",
                     user_id="u-1", tenant_id="TENANT_A")
        other = graph.invoke("租户B的问题", thread_id="t",
                             user_id="u-1", tenant_id="TENANT_B")
        contents = [m.content for m in other["messages"]]
        self.assertNotIn("租户A的内部讨论", contents)

    def test_same_user_same_thread_continues(self):
        graph = make_graph()
        graph.invoke("第一句", thread_id="t", user_id="u-1",
                     tenant_id="IOI")
        second = graph.invoke("第二句", thread_id="t", user_id="u-1",
                              tenant_id="IOI")
        self.assertEqual(len(second["messages"]), 4)


class TestAPITenantHeader(unittest.TestCase):

    def test_tenant_header_isolation_end_to_end(self):
        from fastapi.testclient import TestClient

        from durian_agent.api.app import create_app
        client = TestClient(create_app(graph=make_graph()))
        client.post("/api/chat",
                    json={"thread_id": "shared", "message": "甲的消息"},
                    headers={"X-Tenant-Id": "IOI", "X-User-Id": "alice"})
        resp = client.post(
            "/api/chat",
            json={"thread_id": "shared", "message": "乙的问题"},
            headers={"X-Tenant-Id": "IOI", "X-User-Id": "bob"})
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn("甲的消息", resp.json()["answer"])


if __name__ == "__main__":
    unittest.main()

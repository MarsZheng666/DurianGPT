"""任务 #65 验证：端到端复杂任务场景回归（§43 示例，16 步全链路）。

§43 场景：
    "未来三天会下雨吗？结合现在的土壤湿度和猫山王开花期要求，
     判断今天需不需要灌水。如果有积水风险，帮我安排排水巡检。"

步骤链：Normalize → SemanticParse → COMPLEX_TASK → ReAct
    → WeatherTool → SoilSensorTool → PolicyGate/强制取证
    → AgricultureRAG（真实四路召回→RRF→重排→Evidence）
    → ReAct 综合判断 → AssetTool → Task 草稿 → 确认 → TaskCreate
    → Final Answer

真实组件：milvus-lite 向量索引 + BM25 四路召回 + Weighted RRF +
证据判断；LLM/重排器为确定性替身（不触外部服务）。
"""

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from durian_agent.api.app import create_app
from durian_agent.api.confirmations import ConfirmationStore
from langgraph.checkpoint.memory import MemorySaver
from durian_agent.graph import DurianAgentGraph
from durian_agent.llm import FakeLLM
from durian_agent.rag.index import BM25Index, VectorIndex
from durian_agent.rag.recall import FourWayRetriever
from durian_agent.tools import ToolRegistry
from durian_agent.tools import (
    alarm as alarm_tool,
    asset as asset_tool,
    rag_tool,
    sensor as sensor_tool,
    task as task_tool,
    user_context as user_context_tool,
    weather as weather_tool,
)
from durian_agent.tools.providers import (
    InMemoryAlarms,
    InMemoryAssets,
    InMemorySensors,
    InMemoryTasks,
    InMemoryWeather,
)

CORPUS = [
    {"chunk_id": "CHUNK_1", "document_id": "灌溉手册", "language": "zh",
     "domain": "irrigation", "source_type": "sop",
     "role_scope": ["worker", "manager"], "orchard_scope": [],
     "text": "猫山王开花期土壤含水量宜保持田间持水量的60%-70%，"
             "低于50%应及时灌水；花期忌积水，雨前停灌并清理排水沟。"},
    {"chunk_id": "CHUNK_2", "document_id": "灌溉手册", "language": "zh",
     "domain": "irrigation", "source_type": "sop",
     "role_scope": ["worker", "manager"], "orchard_scope": [],
     "text": "猫山王花期连续降雨时有积水烂根风险，应提前开沟排水，"
             "雨后巡检排水沟畅通情况。"},
]

QUERY = ("未来三天会下雨吗？结合现在的土壤湿度和猫山王开花期要求，"
         "判断今天需不需要灌水。如果有积水风险，帮我安排排水巡检。")


def _embed(texts):
    """确定性嵌入：含「花期/灌水/积水」语义桶 → 三维正交。"""
    vectors = []
    for t in texts:
        if any(k in t for k in ("花期", "灌水", "积水", "排水")):
            vectors.append([1.0, 0.0, 0.0])
        elif "天气" in t or "下雨" in t or "降雨" in t:
            vectors.append([0.0, 1.0, 0.0])
        else:
            vectors.append([0.0, 0.0, 1.0])
    return vectors


def _rerank(query, texts):
    """确定性重排：覆盖查询关键词多者分高。"""
    scores = []
    for text in texts:
        score = 0.5
        for kw in ("猫山王", "花期", "灌水", "积水", "排水"):
            if kw in query and kw in text:
                score += 0.1
        scores.append(round(min(score, 1.0), 2))
    return scores


class ScriptedReActLLM(FakeLLM):
    """§43 步序脚本：weather → sensor → rag → asset → task_create → 收尾。"""

    SCRIPT = [
        '{"action": {"tool": "weather", "args": {"location": "ORCHARD_3", "days": 3}}}',
        '{"action": {"tool": "sensor", "args": {"orchard": "ORCHARD_3", "plot": "PLOT_5"}}}',
        '{"action": {"tool": "agriculture_rag", "args": '
        '{"query": "猫山王开花期灌水要求 积水风险"}}}',
        '{"action": {"tool": "asset_query", "args": '
        '{"asset_type": "workers", "orchard": "ORCHARD_3"}}}',
        '{"action": {"tool": "task_create", "args": '
        '{"title": "排水巡检", "orchard": "ORCHARD_3", "plot": "PLOT_5", '
        '"priority": "high", "description": "花期积水风险，巡检排水沟"}}}',
    ]
    FINAL = ('{"final_answer": "明天有雨（28mm）、土壤湿度42%处于适宜区间，'
             '今天不需灌水；花期有积水风险，已生成排水巡检工单待确认。'
             '依据 [CHUNK_1] [CHUNK_2]。"}')

    def __init__(self):
        super().__init__("")
        self.react_calls = 0

    SIMPLE_ANSWER = ("综合判断：明天有雨（28mm）、当前土壤湿度42%处于花期适宜区间，"
                     "今天不需灌水；排水巡检工单已确认创建。依据 [CHUNK_1] [CHUNK_2]。")

    def __init__(self):
        super().__init__("")
        self.react_calls = 0
        self.simple_calls = []

    def complete(self, system, user, *, temperature=0.0):
        if "Respond with EXACTLY ONE JSON" in system:
            index = min(self.react_calls, len(self.SCRIPT))
            self.react_calls += 1
            return self.SCRIPT[index] if index < len(self.SCRIPT) else self.FINAL
        if "durian plantation" in system:      # SimpleAgent 轮
            self.simple_calls.append(system)
            return self.SIMPLE_ANSWER
        return "不是JSON"   # SemanticParse → 规则层


class TestEndToEndSection43(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 真实四路召回：milvus-lite 向量索引 + BM25
        tmp = tempfile.mkdtemp()
        vector = VectorIndex(Path(tmp) / "e2e.db", embed_fn=_embed, dim=3)
        vector.build(CORPUS)
        bm25 = BM25Index(CORPUS)
        retriever = FourWayRetriever(vector_index=vector, bm25_index=bm25)

        # 工具装配（共享工单存储供断言）
        cls.tasks = InMemoryTasks()
        registry = ToolRegistry()
        weather_tool.register(registry, InMemoryWeather())
        sensor_tool.register(registry, InMemorySensors())
        asset_tool.register(registry, InMemoryAssets())
        alarm_tool.register(registry, InMemoryAlarms())
        task_tool.register_all(registry, cls.tasks)
        rag_tool.register(registry, retriever, reranker=_rerank)
        user_context_tool.register(registry)

        cls.graph = DurianAgentGraph(llm=ScriptedReActLLM(), tools=registry,
                                     checkpointer=MemorySaver())
        cls.client = TestClient(create_app(
            graph=cls.graph, confirmations=ConfirmationStore()))

    def test_full_16_step_scenario(self):
        # 1-4. 输入 → 归一 → 语义解析 → COMPLEX_TASK 路由
        resp = self.client.post(
            "/api/chat", json={"thread_id": "e2e-43", "message": QUERY},
            headers={"X-Role": "manager", "X-User-Id": "u-mgr",
                     "X-Tenant-Id": "IOI"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["route"], "COMPLEX_TASK")

        # 5-12. ReAct 工具链 + 真实 RAG 取证（§43 步序，查 checkpoint）
        snap = self.graph.graph.get_state({"configurable": {
            "thread_id": "IOI:u-mgr:e2e-43"}})
        values = snap.values
        self.assertEqual(
            [r["tool_name"] for r in values["tool_calls"]],
            ["weather", "sensor", "agriculture_rag", "asset_query",
             "task_create"])          # §43 步序：天气→土壤→取证→资产→草稿

        # 真实四路召回取证成功（PolicyGate 未触发强制——Agent 主动取证）
        self.assertTrue(values["evidence_sufficient"])
        self.assertEqual(
            {d["chunk_id"] for d in values["reranked_docs"]},
            {"CHUNK_1", "CHUNK_2"})

        # 13. 工单草稿 → 展示给用户（pending_confirmation）
        pending = body["pending_confirmation"]
        self.assertIsNotNone(pending)
        self.assertEqual(pending["tool"], "task_create")
        self.assertEqual(pending["args"]["title"], "排水巡检")
        # 确认前未创建
        self.assertEqual(self.tasks.query(), [])

        # 14. 用户确认 → TaskCreateTool 真执行
        confirm = self.client.post("/api/chat/confirm", json={
            "thread_id": "e2e-43",
            "confirmation_id": pending["confirmation_id"],
            "approved": True})
        self.assertEqual(confirm.status_code, 200)
        self.assertEqual(confirm.json()["status"], "executed")
        task = self.tasks.query()[0]
        self.assertEqual(task["title"], "排水巡检")
        self.assertEqual(task["priority"], "high")

        # 15. 确认回合的回答带证据引用（§43：判断依据随草稿展示）
        self.assertIn("[CHUNK_1]", body["answer"])
        self.assertTrue(body["sources"])
        cited = {s["chunk_id"] for s in body["sources"]}
        self.assertTrue(cited & {"CHUNK_1", "CHUNK_2"})

        # 16. §43 Final Answer：确认后的总结轮，SimpleAgent 带记忆上下文
        summary = self.client.post(
            "/api/chat",
            json={"thread_id": "e2e-43", "message": "总结今天的判断和工单安排"},
            headers={"X-Role": "manager", "X-User-Id": "u-mgr",
                     "X-Tenant-Id": "IOI"})
        self.assertEqual(summary.status_code, 200)
        summary_body = summary.json()
        self.assertIn("28mm", summary_body["answer"])
        self.assertIn("42%", summary_body["answer"])
        self.assertIn("[CHUNK_1]", summary_body["answer"])
        # 记忆上下文生效：工具观察（天气/土壤数据）进入了总结轮 prompt
        self.assertTrue(self.graph._simple.llm.simple_calls)
        prompt = self.graph._simple.llm.simple_calls[-1]
        self.assertIn("28.0mm", prompt)        # weather 观察在窗口里
        self.assertIn("42.0%", prompt)         # sensor 观察在窗口里


if __name__ == "__main__":
    unittest.main()

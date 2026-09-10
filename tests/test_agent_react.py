"""任务 #39 验证：ReAct 循环（§17/§49）+ 图接线。

验证点：
1. 工具链式调用：weather → final answer（Observation 进 transcript）；
2. §16/§38/§64 联动：想下专业结论且无证据 → PolicyGate 强制注入
   agriculture_rag → 取证后收尾，回答带 [CHUNK] 引用与 sources；
3. 敏感操作：task_create 被 Registry 拦截为草稿 → 图收尾并携带
   pending_confirmation（工单未创建）；
4. 步数上限：LLM 永远要求调工具 → 降级不无限循环；
5. LLM 输出非法 → 降级；协议解析（action/final/围栏/杂文）。
"""

import unittest

from durian_agent.graph import DurianAgentGraph
from durian_agent.llm import FakeLLM
from durian_agent.tools.react import parse_step

RAG_HITS = {
    "hits": {
        "bm25_original": [
            {"chunk_id": "CHUNK_1", "score": 4.0,
             "text": "猫山王开花期土壤含水量保持田间持水量的60%-70%，低于50%需灌水",
             "record": {"document_id": "灌溉手册"}},
            {"chunk_id": "CHUNK_2", "score": 2.0,
             "text": "猫山王花期忌积水，雨季注意排水",
             "record": {"document_id": "栽培指南"}},
        ],
        "dense_canonical": [], "dense_original": [], "bm25_expanded": [],
    },
    "rankings": {"bm25_original": ["CHUNK_1", "CHUNK_2"]},
    "queries": {},
}


class FakeRetriever:
    def __init__(self, result):
        self.result = result

    def recall(self, query, *, top_k_each=10, expr=None):
        return self.result


class SmartLLM(FakeLLM):
    """按 system prompt 分流的 LLM：
    - ReAct 调用（"Respond with EXACTLY ONE JSON"）→ 按脚本逐个返回；
    - 其余（SemanticParse 等）→ 返回非法串，让规则层接管。"""

    def __init__(self, react_responses):
        super().__init__("")
        self.react_responses = list(react_responses)
        self.react_calls = 0

    def complete(self, system, user, *, temperature=0.0):
        self.calls.append({"system": system, "user": user})
        if "Respond with EXACTLY ONE JSON" in system:
            index = min(self.react_calls, len(self.react_responses) - 1)
            self.react_calls += 1
            return self.react_responses[index]
        return "不是JSON"


def make_graph(responses, retriever_result=RAG_HITS):
    return DurianAgentGraph(
        llm=SmartLLM(responses),
        retriever=FakeRetriever(retriever_result),
    )


class TestParseStep(unittest.TestCase):

    def test_action(self):
        parsed = parse_step('{"action": {"tool": "weather", "args": {"days": 3}}}')
        self.assertEqual(parsed, {"action": {"tool": "weather", "args": {"days": 3}}})

    def test_final_answer(self):
        parsed = parse_step('{"final_answer": "今天需要灌水"}')
        self.assertEqual(parsed, {"final_answer": "今天需要灌水"})

    def test_fenced_json(self):
        parsed = parse_step('```json\n{"final_answer": "x"}\n```')
        self.assertEqual(parsed, {"final_answer": "x"})

    def test_invalid(self):
        for bad in ["随便说说", "[]", '{"foo": 1}', "", None,
                    '{"final_answer": ""}']:
            self.assertIsNone(parse_step(bad), msg=bad)


class TestReActFlow(unittest.TestCase):

    def test_tool_chain_then_answer(self):
        """weather 工具 → 观察进对话 → 最终回答（§17 循环）。"""
        graph = make_graph([
            '{"action": {"tool": "weather", '
            '"args": {"location": "ORCHARD_3", "days": 3}}}',
            '{"final_answer": "未来三天第二天有雨（28mm），可推迟灌水。"}',
        ])
        result = graph.invoke("看下传感器数据和未来三天天气再决定浇水",
                              thread_id="t-react-1", role="manager")
        self.assertEqual(result["route"], "COMPLEX_TASK")
        self.assertIn("28mm", result["final_answer"])
        observations = [m for m in result["messages"]
                        if m.__class__.__name__ == "ToolMessage"]
        self.assertTrue(any("28.0mm" in m.content for m in observations))

    def test_policy_gate_forces_rag_before_professional_conclusion(self):
        """§43 式场景：无证据直接想下专业结论 → 强制 RAG → 带引用收尾。"""
        graph = make_graph([
            # ReAct 第一步：想直接回答（无证据）——被 gate 拦下强制取证
            '{"final_answer": "今天需要灌水"}',
            # 看到证据后带引用回答
            '{"final_answer": "当前湿度42%高于阈值，且明天有雨，'
            '今天不需灌水 [CHUNK_1]"}',
        ])
        result = graph.invoke(
            "根据未来三天天气和当前土壤湿度，判断3号园猫山王今天要不要灌水",
            thread_id="t-react-2")
        self.assertEqual(result["route"], "COMPLEX_TASK")
        self.assertIn("[CHUNK_1]", result["final_answer"])
        self.assertEqual(result["rag_sources"][0]["chunk_id"], "CHUNK_1")
        # 强制取证确实发生且证据写入状态
        self.assertTrue(result["rag_forced"])
        self.assertTrue(result["evidence_sufficient"])

    def test_sensitive_task_create_intercepted(self):
        """task_create → Registry 拦截为草稿 → pending_confirmation，不执行。"""
        graph = make_graph([
            '{"action": {"tool": "task_create", "args": '
            '{"title": "排水巡检", "orchard": "ORCHARD_3", "priority": "high"}}}',
        ])
        result = graph.invoke("帮我创建一个排水巡检工单",
                          thread_id="t-react-3", role="manager")
        pending = result.get("pending_confirmation")
        self.assertIsNotNone(pending, msg=result["final_answer"])
        self.assertEqual(pending["tool"], "task_create")
        self.assertEqual(pending["args"]["title"], "排水巡检")
        self.assertTrue(pending["confirmation_id"].startswith("cfm-"))
        self.assertIn("需要您确认", result["final_answer"])

    def test_max_steps_degrades(self):
        """LLM 永远要调工具 → 步数上限后降级，不死循环。"""
        graph = make_graph([
            '{"action": {"tool": "weather", "args": {"days": 1}}}',
        ])
        result = graph.invoke("看下传感器数据和未来三天天气再决定浇水",
                              thread_id="t-react-4")
        self.assertIn("步数上限", result["final_answer"])
        # 降级后 gate 还会做一次性强制取证（+1），有界不无限
        self.assertLessEqual(result.get("react_steps", 0), 7)

    def test_invalid_llm_output_degrades(self):
        graph = make_graph(["我觉得吧这个事情不好说"])
        result = graph.invoke("看下传感器数据和未来三天天气再决定浇水",
                              thread_id="t-react-5")
        self.assertIn("未能完成", result["final_answer"])


if __name__ == "__main__":
    unittest.main()

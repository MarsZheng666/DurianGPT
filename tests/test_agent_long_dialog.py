"""任务 #64 验证：长对话成功率（§57 失败口径）。"""

import unittest

from langgraph.checkpoint.memory import MemorySaver

from durian_agent.evaluation.long_dialog import (
    LongConversationHarness,
    classify_failure,
)
from durian_agent.graph import DurianAgentGraph
from durian_agent.llm import FakeLLM


class TestClassifyFailure(unittest.TestCase):

    def test_success_cases(self):
        self.assertIsNone(classify_failure("波尔多液可防治炭疽病。"))
        # 诚实拒答是正常生成（§57 是请求级口径）
        self.assertIsNone(classify_failure("知识库中暂无足够证据回答该问题。"))

    def test_failure_cases(self):
        self.assertEqual(classify_failure(""), "NO_RESPONSE")
        self.assertEqual(classify_failure("推理输出无法解析，本次未能完成。"),
                         "NO_RESPONSE")
        self.assertEqual(classify_failure("请求超出上下文长度限制"),
                         "LLM_TOKEN_OVERFLOW")


class TestHarness(unittest.TestCase):

    def test_long_conversation_success_rate(self):
        """20 轮长会话：checkpoint 续接，全部成功 → 成功率 1.0。"""
        graph = DurianAgentGraph(llm=FakeLLM("好的。"),
                                 checkpointer=MemorySaver())
        turns = [f"第{i}轮问题：猫山王花期管理要点" for i in range(20)]
        report = LongConversationHarness(
            graph, turns, thread_id="t-long-20").run()
        self.assertEqual(report["total_requests"], 20)
        self.assertEqual(report["successful"], 20)
        self.assertEqual(report["success_rate"], 1.0)
        # 历史真实累积（长会话不因轮数增长而失败）
        self.assertGreater(len(graph.graph.get_state(
            {"configurable": {
                "thread_id": "default:anonymous:t-long-20"}}).values
            ["messages"]), 20)

    def test_failure_classification_in_report(self):
        """LLM 输出非法（降级文案）按 §57 计为未正常生成。"""

        class FlakyLLM(FakeLLM):
            def complete(self, system, user, *, temperature=0.0):
                return "随便聊聊"   # 非法输出 → 引擎/解析降级

        # COMPLEX 查询 + 无效 LLM → reactAgent 降级「未能完成」
        graph = DurianAgentGraph(llm=FlakyLLM(""))
        turns = ["根据传感器数据决定浇水"] * 3
        report = LongConversationHarness(
            graph, turns, thread_id="t-fail").run()
        self.assertEqual(report["total_requests"], 3)
        self.assertLess(report["success_rate"], 1.0)
        self.assertIn("NO_RESPONSE", report["failures_by_type"])

    def test_mixed_success_rate(self):
        by_type = {"NO_RESPONSE": 2, "LLM_TOKEN_OVERFLOW": 1}
        report = LongConversationHarness.report([
            {"turn": 1, "answer": "ok", "failure": None},
            {"turn": 2, "answer": "ok", "failure": None},
            {"turn": 3, "answer": "", "failure": "NO_RESPONSE"},
            {"turn": 4, "answer": "x 未能完成", "failure": "NO_RESPONSE"},
            {"turn": 5, "answer": "overflow", "failure": "LLM_TOKEN_OVERFLOW"},
        ])
        self.assertEqual(report["success_rate"], 0.4)
        self.assertEqual(report["failures_by_type"], by_type)


if __name__ == "__main__":
    unittest.main()

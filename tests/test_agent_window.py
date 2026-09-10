"""任务 #55 验证：滑动窗口与摘要（§35）。"""

import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from durian_agent.graph import DurianAgentGraph
from durian_agent.llm import FakeLLM
from durian_agent.memory.window import (
    MAX_RECENT_MESSAGES,
    SUMMARY_SYSTEM_PROMPT,
    compress,
    llm_summarize,
    rule_summarize,
    split_window,
)


def make_turn(i, q=None, a=None):
    return [HumanMessage(content=q or f"第{i}个问题关于猫山王施肥",
                         id=f"h{i}"),
            AIMessage(content=a or f"第{i}个回答要点", id=f"a{i}")]


class TestSplitWindow(unittest.TestCase):

    def test_within_window_no_overflow(self):
        msgs = [m for turn in (make_turn(i) for i in range(5)) for m in turn]
        result = split_window(msgs)
        self.assertEqual(result["overflow"], [])
        self.assertEqual(len(result["recent"]), 10)

    def test_overflow_beyond_window(self):
        msgs = [m for turn in (make_turn(i) for i in range(12)) for m in turn]
        result = split_window(msgs)
        self.assertEqual(len(result["recent"]), MAX_RECENT_MESSAGES)
        self.assertEqual(len(result["overflow"]), 8)
        # 溢出的是最早的消息
        self.assertEqual(result["overflow"][0].content, "第0个问题关于猫山王施肥")


class TestRuleSummarize(unittest.TestCase):

    def test_keeps_questions_and_answers(self):
        msgs = [HumanMessage(content="3号园的猫山王怎么施肥"),
                AIMessage(content="开花期以磷钾肥为主，每株复合肥两公斤")]
        summary = rule_summarize(msgs)
        self.assertIn("猫山王怎么施肥", summary)
        self.assertIn("磷钾肥", summary)

    def test_drops_smalltalk_and_duplicates(self):
        msgs = [
            HumanMessage(content="你好"),                    # 寒暄
            HumanMessage(content="谢谢"),                    # 寒暄
            HumanMessage(content="浇水频率"),
            AIMessage(content="旱季每七天一次，灌透根系层"),
            AIMessage(content="旱季每七天一次，灌透根系层"),  # 重复回答
        ]
        summary = rule_summarize(msgs)
        self.assertNotIn("你好", summary)
        self.assertNotIn("谢谢", summary)
        self.assertEqual(summary.count("旱季每七天一次"), 1)

    def test_tool_output_bounded(self):
        msgs = [ToolMessage(content="很长的工具输出" * 200, tool_call_id="t1")]
        summary = rule_summarize(msgs)
        self.assertLessEqual(len(summary), 200)   # 限幅

    def test_merges_previous_summary(self):
        summary = rule_summarize(
            [HumanMessage(content="新问题")], previous_summary="旧摘要内容")
        self.assertIn("旧摘要内容", summary)
        self.assertIn("新问题", summary)


class TestLLMSummarize(unittest.TestCase):

    def test_prompt_contains_keep_drop_lists(self):
        self.assertIn("KEEP", SUMMARY_SYSTEM_PROMPT)
        self.assertIn("confirmed parameters", SUMMARY_SYSTEM_PROMPT)
        self.assertIn("unfinished tasks", SUMMARY_SYSTEM_PROMPT)
        self.assertIn("DROP", SUMMARY_SYSTEM_PROMPT)
        self.assertIn("small talk", SUMMARY_SYSTEM_PROMPT)
        self.assertIn("verbose raw tool output", SUMMARY_SYSTEM_PROMPT)

    def test_llm_summary_used(self):
        fake = FakeLLM("用户在3号园种猫山王，正在讨论花期施肥。")
        result = llm_summarize(
            make_turn(0), "", fake)
        self.assertEqual(result, "用户在3号园种猫山王，正在讨论花期施肥。")

    def test_llm_failure_returns_none(self):
        class Broken(FakeLLM):
            def complete(self, system, user, *, temperature=0.0):
                raise RuntimeError("LLM 不可用")
        self.assertIsNone(llm_summarize(make_turn(0), "", Broken("")))


class TestCompressEntry(unittest.TestCase):

    def test_compress_trims_and_summarizes(self):
        msgs = [m for turn in (make_turn(i) for i in range(12)) for m in turn]
        result = compress(msgs, previous_summary="", llm=None,
                          char_threshold=10)   # 低阈值触发压缩
        self.assertEqual(len(result["recent"]), MAX_RECENT_MESSAGES)
        self.assertTrue(result["history_summary"])
        self.assertEqual(len(result["trimmed_ids"]), 8)
        self.assertIn("第0个问题", result["history_summary"])

    def test_below_threshold_no_compression_no_loss(self):
        """总量未超阈值：完全不压缩（原文全保留，零丢失零成本）。"""
        msgs = [m for turn in (make_turn(i, q="问", a="答") for i in range(12))
                for m in turn]
        result = compress(msgs, previous_summary="旧摘要")
        self.assertEqual(result["history_summary"], "旧摘要")   # 未动
        self.assertEqual(result["trimmed_ids"], [])              # 未裁
        self.assertEqual(len(result["recent"]), len(msgs))       # 全保留


class TestGraphIntegration(unittest.TestCase):

    def test_long_conversation_bounded(self):
        """图内 memoryCompress：长会话 messages 有界 + 摘要沉淀。"""
        base = ("第{i}个问题：猫山王开花期在高温高湿环境下出现叶片边缘卷曲，"
                "请问应当如何调整水肥管理方案，需要注意哪些病虫害风险，"
                "园区排水与遮荫措施如何配合，请给出具体的操作建议。")
        long_q = "".join(f"补充{i}：" + base for i in range(7))   # ~600字/轮
        class SummarizingLLM(FakeLLM):
            """聊天回答「好的。」；摘要调用回显既有摘要要点。"""
            def complete(self, system, user, *, temperature=0.0):
                if system.startswith("Summarize this durian plantation"):
                    return f"摘要：{user[:100]}"
                return "好的。"

        graph = DurianAgentGraph(llm=SummarizingLLM("好的。"),
                                 checkpointer=MemorySaver())
        for i in range(14):
            graph.invoke(long_q.format(i=i), thread_id="t-long")
        state = graph.graph.get_state({"configurable": {"thread_id": "t-long"}})
        messages = state.values["messages"]
        # 超阈值后窗口裁剪生效：有界不无限膨胀
        self.assertLessEqual(len(messages), MAX_RECENT_MESSAGES + 4)
        # 早期内容沉入摘要（先摘后剪不丢失）
        self.assertIn("第0个问题", state.values["history_summary"])


if __name__ == "__main__":
    unittest.main()

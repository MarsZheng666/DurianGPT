"""任务 #48 验证：Token 膨胀五级降级（§36）。"""

import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from durian_agent.memory.degrade import (
    apply_degrade,
    critical_messages,
    degrade_until_fits,
    drop_tool_outputs,
    estimate_context_tokens,
    next_level_on_error,
    shrink_window,
)


def msgs(n_tool=6, n_turns=8):
    out = []
    for i in range(n_turns):
        out.append(HumanMessage(content=f"第{i}个问题" + "背景" * 30))
        out.append(AIMessage(content=f"第{i}个回答"))
    for t in range(n_tool):
        out.append(ToolMessage(content="工具输出" * 100, tool_call_id=f"t{t}"))
    return out


class TestLevelTransforms(unittest.TestCase):

    def test_level0_full(self):
        result = apply_degrade(0, recent_messages=msgs(), summary="摘要",
                               business_state="状态", query="问题")
        self.assertEqual(len(result["recent_messages"]), 22)   # 全保留

    def test_level1_shrinks_window(self):
        result = apply_degrade(1, recent_messages=msgs(6, 8), summary="",
                               business_state="", query="q")
        self.assertLess(len(result["recent_messages"]), 22)
        # 保最近的（后半段内容保留，早期被裁）
        kept = " ".join(str(m.content)[:10] for m in result["recent_messages"])
        self.assertIn("第7", kept)
        self.assertNotIn("第0个问题", kept)

    def test_level2_drops_tool_outputs(self):
        result = apply_degrade(2, recent_messages=msgs(), summary="",
                               business_state="", query="q")
        tools = [m for m in result["recent_messages"]
                 if isinstance(m, ToolMessage)]
        self.assertTrue(all(m.content == "[工具输出已省略]" for m in tools))
        # token 量显著下降
        full = estimate_context_tokens(
            {"recent_messages": msgs(), "summary": "", "query": "", "business_state": ""})
        degraded = estimate_context_tokens(result)
        self.assertLess(degraded, full * 0.5)

    def test_level3_summary_plus_critical(self):
        result = apply_degrade(3, recent_messages=msgs(), summary="历史摘要",
                               business_state="", query="q")
        self.assertEqual(len(result["recent_messages"]), 2)
        self.assertEqual(result["summary"], "历史摘要")

    def test_level4_minimal(self):
        result = apply_degrade(4, recent_messages=msgs(), summary="摘要",
                               business_state="状态", query="问题")
        self.assertEqual(result["recent_messages"], [])
        self.assertEqual(result["summary"], "")
        self.assertEqual(result["business_state"], "状态")   # 必要业务状态保留
        self.assertEqual(result["query"], "问题")

    def test_monotonic_shrinking(self):
        """各级别上下文估计量单调不增（L0≥L1≥…≥L4）。"""
        base = dict(recent_messages=msgs(), summary="摘" * 200,
                    business_state="状态", query="问题")
        sizes = [estimate_context_tokens(apply_degrade(lv, **base))
                 for lv in range(5)]
        self.assertEqual(sizes, sorted(sizes, reverse=True))


class TestErrorDrivenLevels(unittest.TestCase):

    def test_overflow_steps_up(self):
        self.assertEqual(next_level_on_error("LLM_TOKEN_OVERFLOW", 0), 1)
        self.assertEqual(next_level_on_error("LLM_TOKEN_OVERFLOW", 3), 4)
        self.assertEqual(next_level_on_error("LLM_TOKEN_OVERFLOW", 4), 4)   # 封顶

    def test_summary_failure_to_level3(self):
        self.assertEqual(next_level_on_error("SUMMARY_FAILED", 0), 3)
        self.assertEqual(next_level_on_error("SUMMARY_FAILED", 3), 3)

    def test_timeout_no_degrade(self):
        self.assertEqual(next_level_on_error("LLM_TIMEOUT", 0), 0)
        self.assertEqual(next_level_on_error("TOOL_TIMEOUT", 2), 2)


class TestDegradeUntilFits(unittest.TestCase):

    def test_iterates_to_fit(self):
        parts = {"recent_messages": msgs(), "summary": "摘" * 200,
                 "business_state": "状态", "query": "问题"}
        result = degrade_until_fits(parts, token_limit=300)
        self.assertLessEqual(estimate_context_tokens(result), 300)
        self.assertGreaterEqual(int(result["degrade_level"]), 1)

    def test_already_fits_stays_level0(self):
        parts = {"recent_messages": [HumanMessage(content="短")],
                 "summary": "", "business_state": "", "query": "问"}
        result = degrade_until_fits(parts, token_limit=10000)
        self.assertEqual(int(result["degrade_level"]), 0)


class TestHelpers(unittest.TestCase):

    def test_shrink_keeps_minimum(self):
        kept = shrink_window([HumanMessage(content=str(i)) for i in range(3)])
        self.assertGreaterEqual(len(kept), 2)

    def test_critical_prefers_recent_and_skips_tools(self):
        messages = [HumanMessage(content="早"), AIMessage(content="中"),
                    ToolMessage(content="巨大的工具输出" * 100, tool_call_id="t"),
                    HumanMessage(content="近")]
        self.assertEqual([m.content for m in critical_messages(messages)],
                         ["中", "近"])   # 关键消息=对话轮次，不含工具原始输出


if __name__ == "__main__":
    unittest.main()

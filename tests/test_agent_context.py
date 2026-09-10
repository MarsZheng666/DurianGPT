"""任务 #51 验证：记忆结构五段式组装（§33）+ SimpleAgent 接入。"""

import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from durian_agent.agent.simple import SimpleAgent
from durian_agent.llm import FakeLLM
from durian_agent.memory.budget import TokenBudget
from durian_agent.memory.context import (
    build_context,
    business_state_from_graph_state,
    render_messages,
)


class TestBuildContext(unittest.TestCase):

    def test_five_segments_in_order(self):
        context = build_context(
            system_prompt="你是榴莲种植助手。",
            query="怎么施肥",
            summary="用户在3号园种猫山王。",
            recent_messages=[HumanMessage(content="上次问过浇水"),
                             AIMessage(content="旱季七天一次")],
            business_state="涉及实体：cultivar=猫山王",
        )
        # 五段齐备且顺序正确
        positions = [
            context.index("你是榴莲种植助手。"),
            context.index("【会话摘要】"),
            context.index("【最近对话】"),
            context.index("【当前业务状态】"),
            context.index("【当前问题】"),
        ]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("[用户] 上次问过浇水", context)
        self.assertIn("3号园", context)

    def test_empty_segments_omitted(self):
        context = build_context(system_prompt="系统", query="问题")
        self.assertNotIn("【会话摘要】", context)
        self.assertNotIn("【最近对话】", context)
        self.assertIn("【当前问题】问题", context)

    def test_budget_applied_per_section(self):
        budget = TokenBudget(total_tokens=1000)   # summary cap=150
        context = build_context(
            system_prompt="系统", query="问题",
            summary="摘" * 500, budget=budget)
        summary_part = context.split("【会话摘要】")[1].split("【当前问题】")[0]
        self.assertLessEqual(len(summary_part.strip()), 150)   # 段间隔符除外

    def test_render_messages_roles(self):
        text = render_messages([
            HumanMessage(content="用户问题"),
            AIMessage(content="助手回答"),
            ToolMessage(content="工具输出", tool_call_id="t"),
        ])
        self.assertIn("[用户] 用户问题", text)
        self.assertIn("[助手] 助手回答", text)
        self.assertIn("[工具] 工具输出", text)


class TestBusinessState(unittest.TestCase):

    def test_extracts_from_graph_state(self):
        state = {
            "semantic": {"entities": {"cultivar": "猫山王", "disease": None},
                         "growth_stage": "flowering"},
        }
        text = business_state_from_graph_state(state)
        self.assertIn("cultivar=猫山王", text)
        self.assertIn("生育阶段：flowering", text)
        self.assertNotIn("disease", text)          # 空槽不出现

    def test_pending_confirmation_included(self):
        state = {"pending_confirmation": {
            "tool": "task_create", "args": {"title": "排水巡检"}}}
        self.assertIn("待确认操作：task_create 排水巡检",
                      business_state_from_graph_state(state))

    def test_empty_state(self):
        self.assertEqual(business_state_from_graph_state({}), "")


class TestSimpleAgentWithContext(unittest.TestCase):

    def test_history_goes_into_prompt(self):
        fake = FakeLLM("好的。")
        SimpleAgent(fake).answer(
            "今天浇多少水",
            history_summary="用户在3号园种猫山王，关注花期管理。",
            recent_messages=[HumanMessage(content="上次说旱季七天一次")],
            business_state="生育阶段：flowering",
        )
        system = fake.calls[0]["system"]
        self.assertIn("3号园", system)
        self.assertIn("旱季七天一次", system)
        self.assertIn("flowering", system)
        self.assertIn("【当前问题】今天浇多少水", system)

    def test_no_context_backward_compatible(self):
        fake = FakeLLM("好的。")
        SimpleAgent(fake).answer("普通问题")
        self.assertIn("durian plantation", fake.calls[0]["system"])
        self.assertNotIn("【会话摘要】", fake.calls[0]["system"])


if __name__ == "__main__":
    unittest.main()

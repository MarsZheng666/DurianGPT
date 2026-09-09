"""任务 #7 验证：AgentState 完整状态结构（架构文档 §4）。

验证点：
1. 23 个字段与文档 §4 一一对应；
2. RouteType 三值、ErrorType 九种错误码（§50）、DegradeLevel 五级（§36）；
3. init_state 工厂默认值安全（最小权限 / 零计数 / Level 0）；
4. 状态可直接作为 StateGraph 的 state_schema 使用（LangGraph 1.2.10 兼容）。
"""

import unittest

from langgraph.graph import END, START, StateGraph

from durian_agent.state import (
    STATE_FIELDS,
    AgentState,
    DegradeLevel,
    RetrievedDocument,
    ToolCallRecord,
    init_state,
)


class TestAgentState(unittest.TestCase):

    def test_fields_cover_architecture_doc(self):
        """AgentState 的 23 个字段与 §4 冻结清单完全一致（无多无漏）。"""
        declared = set(AgentState.__annotations__.keys())
        self.assertEqual(declared, set(STATE_FIELDS))
        self.assertEqual(len(STATE_FIELDS), 23)

    def test_substructures_defined(self):
        """§4 引用的 RetrievedDocument / ToolCallRecord 子结构存在且为 TypedDict。"""
        self.assertIn("chunk_id", RetrievedDocument.__annotations__)
        self.assertIn("source_route", RetrievedDocument.__annotations__)
        self.assertIn("tool_name", ToolCallRecord.__annotations__)

    def test_init_state_defaults(self):
        """工厂默认值：最小权限 worker、零计数、Level 0、空集合。"""
        state = init_state(thread_id="t-1", user_id="u-1", original_query="你好")
        self.assertEqual(state["thread_id"], "t-1")
        self.assertEqual(state["role"], "worker")
        self.assertEqual(state["retrieval_count"], 0)
        self.assertEqual(state["retry_count"], 0)
        self.assertEqual(state["degrade_level"], DegradeLevel.FULL_CONTEXT)
        self.assertFalse(state["evidence_sufficient"])
        self.assertEqual(state["allowed_tools"], set())
        self.assertEqual(state["allowed_knowledge_partitions"], set())
        self.assertEqual(state["normalized_query"], "你好")

    def test_degrade_level_ordering(self):
        """五级降级按 §36 语义取值 0~4，且可比较大小。"""
        levels = [DegradeLevel.FULL_CONTEXT, DegradeLevel.SHRINK_WINDOW,
                  DegradeLevel.DROP_TOOL_OUTPUT, DegradeLevel.SUMMARY_ONLY,
                  DegradeLevel.MINIMAL]
        self.assertEqual([int(lv) for lv in levels], [0, 1, 2, 3, 4])
        self.assertLess(DegradeLevel.SHRINK_WINDOW, DegradeLevel.MINIMAL)

    def test_error_type_literal_members(self):
        """ErrorType 覆盖 §50 的 9 种错误码。"""
        from typing import get_args
        from durian_agent.state import ErrorType
        expected = {
            "RAG_NO_RESULT", "RAG_LOW_CONFIDENCE", "TOOL_TIMEOUT",
            "TOOL_PERMISSION_DENIED", "TOOL_INVALID_ARGUMENT",
            "LLM_TIMEOUT", "LLM_TOKEN_OVERFLOW", "SUMMARY_FAILED",
            "TASK_CREATE_FAILED",
        }
        self.assertEqual(set(get_args(ErrorType)), expected)

    def test_route_type_literal_members(self):
        """RouteType 恰为三路（§12）。"""
        from typing import get_args
        from durian_agent.state import RouteType
        self.assertEqual(set(get_args(RouteType)), {"MUST_RAG", "SIMPLE", "COMPLEX_TASK"})

    def test_state_usable_in_state_graph(self):
        """AgentState 可直接作为 StateGraph 的 schema 编译并运行。"""
        builder = StateGraph(AgentState)
        builder.add_node("echo", lambda s: {"final_answer": s.get("normalized_query", "")})
        builder.add_edge(START, "echo")
        builder.add_edge("echo", END)
        graph = builder.compile()
        result = graph.invoke(init_state(original_query="测试问句"))
        self.assertEqual(result["final_answer"], "测试问句")

    def test_messages_reducer_appends(self):
        """messages 走 add_messages reducer：两次写入应累积而非覆盖。"""
        from langchain_core.messages import HumanMessage
        builder = StateGraph(AgentState)
        builder.add_node("a", lambda s: {"messages": [HumanMessage(content="一", id="m1")]})
        builder.add_node("b", lambda s: {"messages": [HumanMessage(content="二", id="m2")]})
        builder.add_edge(START, "a")
        builder.add_edge("a", "b")
        builder.add_edge("b", END)
        result = builder.compile().invoke(init_state())
        self.assertEqual(len(result["messages"]), 2)


if __name__ == "__main__":
    unittest.main()

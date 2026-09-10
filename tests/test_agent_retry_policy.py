"""任务 #50 验证：Retry 原则（§51：上限 + 瞬态-only + 幂等配合）。"""

import unittest

from durian_agent.llm import FakeLLM
from durian_agent.retry import DEFAULT_POLICY, RetryPolicy, with_retry
from durian_agent.tools.react import ReActEngine
from durian_agent.tools import ToolRegistry


class TestRetryPolicy(unittest.TestCase):

    def test_section51_limits(self):
        self.assertEqual(DEFAULT_POLICY, {"llm": 2, "tool": 2, "rag_rewrite": 2})
        policy = RetryPolicy()
        self.assertTrue(policy.allows("llm", 1))
        self.assertFalse(policy.allows("llm", 2))     # 上限 2 次尝试

    def test_rag_rewrite_limit_matches_graph(self):
        from durian_agent.graph import MAX_RETRIEVALS
        self.assertEqual(RetryPolicy().max_attempts("rag_rewrite"),
                         MAX_RETRIEVALS)


class TestWithRetry(unittest.TestCase):

    def test_transient_retries_then_succeeds(self):
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 2:
                raise TimeoutError("request timed out")
            return "ok"

        self.assertEqual(with_retry(flaky, kind="llm"), "ok")
        self.assertEqual(len(calls), 2)

    def test_gives_up_after_budget(self):
        calls = []

        def always_timeout():
            calls.append(1)
            raise TimeoutError("timed out")

        with self.assertRaises(TimeoutError):
            with_retry(always_timeout, kind="llm")
        self.assertEqual(len(calls), 2)               # 上限即停

    def test_permission_denied_no_retry(self):
        calls = []

        def denied():
            calls.append(1)
            raise RuntimeError("权限不足: worker")

        with self.assertRaises(RuntimeError):
            with_retry(denied, kind="tool")
        self.assertEqual(len(calls), 1)               # 立即失败

    def test_invalid_argument_no_retry(self):
        calls = []

        def bad():
            calls.append(1)
            raise ValueError("参数错误")

        with self.assertRaises(ValueError):
            with_retry(bad, kind="tool")
        self.assertEqual(len(calls), 1)


class TestEngineLLMRetry(unittest.TestCase):

    def test_engine_retries_transient_llm_failure(self):
        """ReAct 引擎：首次超时重试成功（§51 LLM retry 1~2）。"""
        from langchain_core.messages import HumanMessage

        class FlakyLLM(FakeLLM):
            def __init__(self):
                super().__init__("")
                self.calls = 0

            def complete(self, system, user, *, temperature=0.0):
                if "Respond with EXACTLY ONE JSON" not in system:
                    return "不是JSON"
                self.calls += 1
                if self.calls == 1:
                    raise TimeoutError("llm timeout")
                return '{"final_answer": "重试后的回答"}'

        registry = ToolRegistry()
        engine = ReActEngine(FlakyLLM(), registry)
        result = engine.step({
            "messages": [HumanMessage(content="问题")],
            "react_steps": 0, "role": "manager",
        })
        self.assertEqual(result["final_answer"], "重试后的回答")
        self.assertTrue(result["react_done"])


if __name__ == "__main__":
    unittest.main()

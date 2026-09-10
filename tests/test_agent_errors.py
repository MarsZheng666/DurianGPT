"""任务 #46 验证：9 种错误分类与处理策略（§50）。"""

import unittest

from durian_agent.errors import (
    ACTION_DIRECT_REPLY,
    ACTION_MEMORY_DEGRADE,
    ACTION_REWRITE,
    ACTION_RETRY_ONCE,
    ALL_ERROR_TYPES,
    ERROR_STRATEGIES,
    LLM_TIMEOUT,
    LLM_TOKEN_OVERFLOW,
    RAG_NO_RESULT,
    SUMMARY_FAILED,
    TASK_CREATE_FAILED,
    TOOL_INVALID_ARGUMENT,
    TOOL_PERMISSION_DENIED,
    TOOL_TIMEOUT,
    classify_exception,
    resolve_strategy,
    should_degrade_memory,
)


class TestErrorCatalog(unittest.TestCase):

    def test_nine_error_types_complete(self):
        self.assertEqual(len(ALL_ERROR_TYPES), 9)
        self.assertEqual(set(ERROR_STRATEGIES), set(ALL_ERROR_TYPES))

    def test_section50_strategies(self):
        self.assertEqual(resolve_strategy(RAG_NO_RESULT)["action"],
                         ACTION_REWRITE)
        self.assertEqual(resolve_strategy(TOOL_TIMEOUT)["action"],
                         ACTION_RETRY_ONCE)
        self.assertEqual(resolve_strategy(TOOL_PERMISSION_DENIED)["action"],
                         ACTION_DIRECT_REPLY)          # 不重试直接回应
        self.assertEqual(resolve_strategy(TOOL_INVALID_ARGUMENT)["action"],
                         ACTION_DIRECT_REPLY)
        for error in (LLM_TOKEN_OVERFLOW, SUMMARY_FAILED):
            self.assertEqual(resolve_strategy(error)["action"],
                             ACTION_MEMORY_DEGRADE)    # → #48 降级
        self.assertEqual(resolve_strategy(TASK_CREATE_FAILED)["action"],
                         "idempotent_retry")           # 幂等键重试


class TestClassifyException(unittest.TestCase):

    def test_timeout(self):
        self.assertEqual(
            classify_exception(TimeoutError("request timed out"), "llm"),
            LLM_TIMEOUT)
        self.assertEqual(
            classify_exception(TimeoutError("read timeout"), "tool"),
            TOOL_TIMEOUT)

    def test_token_overflow(self):
        exc = RuntimeError("maximum context length exceeded")
        self.assertEqual(classify_exception(exc, "llm"),
                         LLM_TOKEN_OVERFLOW)

    def test_permission(self):
        self.assertEqual(
            classify_exception(RuntimeError("权限不足: worker"), "tool"),
            TOOL_PERMISSION_DENIED)

    def test_invalid_argument(self):
        self.assertEqual(
            classify_exception(ValueError("bad args"), "tool"),
            TOOL_INVALID_ARGUMENT)


class TestDegradeLinkage(unittest.TestCase):

    def test_memory_errors_flag_degrade(self):
        self.assertTrue(should_degrade_memory(LLM_TOKEN_OVERFLOW))
        self.assertTrue(should_degrade_memory(SUMMARY_FAILED))
        self.assertFalse(should_degrade_memory(LLM_TIMEOUT))
        self.assertFalse(should_degrade_memory(RAG_NO_RESULT))


if __name__ == "__main__":
    unittest.main()

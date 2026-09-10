"""任务 #28 验证：Query Rewrite 两层机制（§24/§47）。

验证点：
1. 规则层：标准化+实体标准名（离线）；
2. LLM 层：§47 禁令进 prompt（不诊断/不编剂量/不编农药）；
   已知实体与已检文档摘要进上下文；
3. 护栏：LLM 改写引入原查询没有的数字 → 拒绝，退规则层；
4. LLM 失败/空输出/原样返回 → 规则层兜底；
5. 图 rewriteQuery 节点接新内核（retry 计数正确）。
"""

import unittest

from durian_agent.llm import FakeLLM
from durian_agent.rag.rewrite import (
    REWRITE_SYSTEM_PROMPT,
    llm_rewrite,
    rewrite_query,
    rule_rewrite,
)

SEMANTIC = {"entities": {"cultivar": "猫山王", "disease": None}}
DOCS = [{"text": "金枕施肥以复合肥为主，开花期增施磷钾肥。"}]


class TestRuleRewrite(unittest.TestCase):

    def test_rule_layer_offline(self):
        rewritten = rule_rewrite("Musang King 怎么施肥", SEMANTIC)
        self.assertIn("猫山王", rewritten)
        self.assertIn("施肥", rewritten)


class TestLLMRewrite(unittest.TestCase):

    def test_prompt_contains_section47_constraints(self):
        for fragment in ["Rewrite the query only for retrieval.",
                         "You must NOT:", "- diagnose",
                         "- invent dosage", "- invent pesticides",
                         "- add unsupported facts"]:
            self.assertIn(fragment, REWRITE_SYSTEM_PROMPT)

    def test_llm_rewrite_used_with_context(self):
        fake = FakeLLM("猫山王 Musang King 施肥管理 开花期")
        result = llm_rewrite("猫山王怎么施肥", SEMANTIC, DOCS, fake)
        self.assertEqual(result, "猫山王 Musang King 施肥管理 开花期")
        # 上下文：已知实体 + 已检文档摘要都进了 user 消息
        user = fake.calls[0]["user"]
        self.assertIn("cultivar=猫山王", user)
        self.assertIn("金枕施肥", user)
        self.assertIn(REWRITE_SYSTEM_PROMPT, fake.calls[0]["system"])

    def test_guardrail_rejects_invented_numbers(self):
        """改写引入原查询没有的数字（疑似编剂量）→ 拒绝。"""
        fake = FakeLLM("猫山王施肥 每株25kg复合肥")
        result = llm_rewrite("猫山王怎么施肥", SEMANTIC, DOCS, fake)
        self.assertIsNone(result)

    def test_existing_numbers_allowed(self):
        fake = FakeLLM("每株施2kg复合肥 猫山王 开花期施肥")
        result = llm_rewrite("猫山王每株施2kg什么肥", SEMANTIC, DOCS, fake)
        self.assertIsNotNone(result)   # 2kg 原查询已有，不触发护栏

    def test_llm_failure_falls_back_to_rule(self):
        class BrokenLLM(FakeLLM):
            def complete(self, system, user, *, temperature=0.0):
                raise RuntimeError("LLM 不可用")

        result = rewrite_query("猫山王怎么施肥", SEMANTIC, DOCS, llm=BrokenLLM(""))
        self.assertEqual(result, rule_rewrite("猫山王怎么施肥", SEMANTIC))

    def test_llm_empty_or_same_output_falls_back(self):
        for response in ["", '"猫山王怎么施肥"']:   # 空输出 / 原样返回
            result = llm_rewrite("猫山王怎么施肥", SEMANTIC, DOCS, FakeLLM(response))
            self.assertIsNone(result)

    def test_no_llm_uses_rule_layer(self):
        result = rewrite_query("猫山王怎么施肥", SEMANTIC)
        self.assertEqual(result, rule_rewrite("猫山王怎么施肥", SEMANTIC))


class TestGraphIntegration(unittest.TestCase):

    def test_rewrite_node_uses_new_kernel(self):
        from durian_agent.graph import DurianAgentGraph

        class TrackingLLM(FakeLLM):
            def __init__(self):
                super().__init__("猫山王 Musang King 施肥管理")
                self.rewrite_calls = 0

        graph = DurianAgentGraph(llm=FakeLLM("好的。"), retriever=None)
        # 无检索命中 → MUST_RAG 走 rewrite 分支
        result = graph.invoke("炭疽病用什么农药防治", thread_id="t-rw")
        self.assertEqual(result["retry_count"], 1)
        # 第二轮查询已是规则层改写结果（含标准名）
        self.assertTrue(result["search_queries"][0])


if __name__ == "__main__":
    unittest.main()

"""任务 #35 验证：SimpleAgent（架构文档 §14 SIMPLE 路径）。

验证点：
1. 无检索时纯 LLM 直答（§14 示例类问题）；
2. rag_fn 证据充分 → 证据注入 prompt、回答带 sources、used_rag=True；
3. rag_fn 证据不足/抛错 → 不注入、不阻塞直答（SIMPLE 不做强制闭环）；
4. prompt 含「不得编造剂量/法规/阈值」约束；
5. 无 LLM 构造直接报错（SIMPLE 的定义就是 LLM 直答）。
"""

import unittest

from durian_agent.agent.simple import SIMPLE_SYSTEM_PROMPT, SimpleAgent
from durian_agent.llm import FakeLLM

GOOD_RAG = {
    "documents": [
        {"document_id": "DOC001", "chunk_id": "C1", "text": "滴灌是局部灌溉方式",
         "metadata": {"title": "灌溉方式", "section": "1.1"}},
        {"document_id": "DOC002", "chunk_id": "C2", "text": "喷灌覆盖面积大",
         "metadata": {"title": "灌溉方式", "section": "1.2"}},
    ],
    "evidence_sufficient": True,
}


class TestSimpleAgent(unittest.TestCase):

    def test_direct_answer_without_rag(self):
        fake = FakeLLM("坐果是雌花受精后形成幼果的过程。")
        result = SimpleAgent(fake).answer("什么是榴莲坐果？")
        self.assertEqual(result["answer"], "坐果是雌花受精后形成幼果的过程。")
        self.assertFalse(result["used_rag"])
        self.assertEqual(result["sources"], [])
        # 系统提示含安全约束
        self.assertIn("Do NOT invent specific pesticide dosages",
                      fake.calls[0]["system"])

    def test_rag_evidence_injected_when_sufficient(self):
        fake = FakeLLM("滴灌局部灌溉省水，喷灌覆盖大。")
        result = SimpleAgent(fake).answer(
            "滴灌和喷灌有什么区别？", rag_fn=lambda q: GOOD_RAG)
        self.assertTrue(result["used_rag"])
        # 证据文本进了 system prompt
        self.assertIn("滴灌是局部灌溉方式", fake.calls[0]["system"])
        self.assertIn("Reference evidence", fake.calls[0]["system"])
        # sources 与证据一一对应（§31 引用格式）
        self.assertEqual(len(result["sources"]), 2)
        self.assertEqual(result["sources"][0]["document_id"], "DOC001")
        self.assertEqual(result["sources"][0]["chunk_id"], "C1")

    def test_rag_insufficient_not_injected(self):
        fake = FakeLLM("修剪为了通风透光。")
        result = SimpleAgent(fake).answer(
            "为什么需要修剪？", rag_fn=lambda q: {"documents": [], "evidence_sufficient": False})
        self.assertFalse(result["used_rag"])
        self.assertNotIn("Reference evidence", fake.calls[0]["system"])

    def test_rag_failure_does_not_block(self):
        def broken_rag(q):
            raise RuntimeError("检索服务不可用")
        fake = FakeLLM("好的。")
        result = SimpleAgent(fake).answer("总结一下刚才的内容", rag_fn=broken_rag)
        self.assertFalse(result["used_rag"])
        self.assertEqual(result["answer"], "好的。")

    def test_requires_llm(self):
        with self.assertRaises(ValueError):
            SimpleAgent(None)

    def test_prompt_safety_constraint_present(self):
        self.assertIn("pesticide dosages", SIMPLE_SYSTEM_PROMPT)
        self.assertIn("thresholds", SIMPLE_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()

"""任务 #31/#33 验证：Final Generation 完整验收 + 引用一一对应（§30/§31/§48）。

#31 验证点：
1. §48 prompt 全禁令；证据块带 [chunk_id] 标注；
2. 无证据 → 固定声明（不调 LLM 不硬答）；
3. 诱导编造剂量的防线在 prompt（真 LLM 验证属阶段五 Faithfulness 评估）。

#33 验证点：
4. 引用解析：[CHUNK_x] ↔ 提供证据一一对应，未提供的标记不收；
5. 未被引用的证据不进 sources（图 answer 节点过滤）。
"""

import unittest

from durian_agent.graph import DurianAgentGraph
from durian_agent.llm import FakeLLM
from durian_agent.rag.generation import (
    ABSTAIN_ANSWER,
    GROUNDED_SYSTEM_PROMPT,
    build_evidence_block,
    generate_grounded_answer,
    generate_with_citations,
    parse_citations,
)

DOCS = [
    {"chunk_id": "CHUNK_1", "text": "炭疽病发病初期用波尔多液防治",
     "record": {"document_id": "植保手册"}},
    {"chunk_id": "CHUNK_2", "text": "炭疽病雨季高发，注意排水通风",
     "record": {"document_id": "栽培指南"}},
]


class TestFinalGeneration(unittest.TestCase):
    """#31：证据约束生成（§30/§48）。"""

    def test_prompt_contains_section48_constraints(self):
        for fragment in [
            "Answer agricultural professional claims only from the supplied evidence.",
            "state that the current knowledge base does not provide enough evidence.",
            "Do not use model prior knowledge to invent:",
            "- dosage", "- pesticide recommendations", "- thresholds",
            "- legal requirements", "- SOP requirements",
        ]:
            self.assertIn(fragment, GROUNDED_SYSTEM_PROMPT)

    def test_evidence_block_annotated_with_chunk_ids(self):
        block = build_evidence_block(DOCS)
        self.assertIn("[CHUNK_1]", block)
        self.assertIn("(植保手册)", block)
        self.assertIn("波尔多液", block)

    def test_no_evidence_abstains_without_llm_call(self):
        fake = FakeLLM("不该被调用")
        answer = generate_grounded_answer(fake, "炭疽病怎么治", [])
        self.assertEqual(answer, ABSTAIN_ANSWER)
        self.assertEqual(len(fake.calls), 0)   # 无证据不调用 LLM

    def test_evidence_goes_into_prompt(self):
        fake = FakeLLM("波尔多液防治 [CHUNK_1]")
        generate_grounded_answer(fake, "炭疽病怎么治", DOCS)
        system = fake.calls[0]["system"]
        self.assertIn("Evidence:", system)
        self.assertIn("[CHUNK_1]", system)

    def test_induce_dosage_is_forbidden_by_prompt(self):
        """诱导场景：查询问剂量，prompt 仍禁止编造（防线在提示词层）。"""
        fake = FakeLLM("依据证据回答 [CHUNK_1]")
        generate_grounded_answer(fake, "炭疽病用什么药多少浓度", DOCS)
        self.assertIn("Do not use model prior knowledge to invent",
                      fake.calls[0]["system"])


class TestCitationCorrespondence(unittest.TestCase):
    """#33：引用与证据一一对应（§31）。"""

    def test_citations_parsed(self):
        answer = "波尔多液可防治 [CHUNK_1]，雨季注意排水 [CHUNK_2]。"
        self.assertEqual(parse_citations(answer, DOCS), ["CHUNK_1", "CHUNK_2"])

    def test_unknown_marker_rejected(self):
        answer = "某种说法 [CHUNK_99] 不可信"
        self.assertEqual(parse_citations(answer, DOCS), [])

    def test_dedup_preserves_order(self):
        answer = "先 [CHUNK_2] 后 [CHUNK_1] 再 [CHUNK_2]"
        self.assertEqual(parse_citations(answer, DOCS), ["CHUNK_2", "CHUNK_1"])

    def test_no_citation_empty(self):
        self.assertEqual(parse_citations("没有任何引用", DOCS), [])
        self.assertEqual(parse_citations("", DOCS), [])

    def test_generate_with_citations(self):
        result = generate_with_citations(
            FakeLLM("用波尔多液 [CHUNK_1]。"), "炭疽病怎么治", DOCS)
        self.assertEqual(result["cited_chunk_ids"], ["CHUNK_1"])

    def test_graph_sources_only_cited(self):
        """图 answer 节点：未被引用的 CHUNK_2 不进 sources（§31 一一对应）。"""
        class Retriever:
            def recall(self, q, *, top_k_each=10, expr=None):
                return {
                    "hits": {
                        "bm25_original": [
                            {"chunk_id": "CHUNK_1", "score": 3.0,
                             "text": "炭疽病发病初期用波尔多液防治",
                             "record": {"document_id": "植保手册"}},
                            {"chunk_id": "CHUNK_2", "score": 2.0,
                             "text": "炭疽病雨季高发注意排水通风",
                             "record": {"document_id": "栽培指南"}},
                        ],
                        "dense_canonical": [], "dense_original": [], "bm25_expanded": [],
                    },
                    "rankings": {"bm25_original": ["CHUNK_1", "CHUNK_2"]},
                    "queries": {},
                }

        graph = DurianAgentGraph(
            llm=FakeLLM("用波尔多液防治 [CHUNK_1]。"), retriever=Retriever())
        result = graph.invoke("炭疽病怎么防治", thread_id="t-cite")
        self.assertEqual(result["rag_sources"][0]["chunk_id"], "CHUNK_1")
        self.assertEqual(len(result["rag_sources"]), 1)   # CHUNK_2 未被引用不进
        self.assertIn("[CHUNK_1]", result["final_answer"])


if __name__ == "__main__":
    unittest.main()

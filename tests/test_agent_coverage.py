"""任务 #63 验证：有效语料解析覆盖率（§56 四类型分层）。"""

import unittest
from pathlib import Path

from durian_agent.evaluation.coverage import (
    COVERAGE_TYPES,
    indexed_counts,
    load_annotations,
    parse_coverage,
)

CORPUS = Path(__file__).resolve().parent.parent / "rag_build" / "chunks.jsonl"


class TestCoverage(unittest.TestCase):

    def test_four_types_closed(self):
        self.assertEqual(COVERAGE_TYPES,
                         ("text", "table", "image", "scan_page"))

    def test_indexed_counts_fixture(self):
        chunks = [
            {"block_type": "text"}, {"block_type": "text"},
            {"block_type": "qa_pair"},                     # 计入 text
            {"block_type": "table"},
            {"block_type": "table_broken"},                # 未成功解析，不计
            {"block_type": "image"},
        ]
        counts = indexed_counts(chunks)
        self.assertEqual(counts, {"text": 3, "table": 1, "image": 1,
                                  "scan_page": 0})

    def test_parse_coverage_layered(self):
        chunks = ([{"block_type": "text"}] * 80
                  + [{"block_type": "table"}] * 4
                  + [{"block_type": "image"}] * 2)
        report = parse_coverage(chunks, {"text": 100, "table": 5})
        self.assertEqual(report["layers"]["text"]["coverage"], 0.8)
        self.assertEqual(report["layers"]["table"]["coverage"], 0.8)
        self.assertIsNone(report["layers"]["image"]["coverage"])  # 无基准不评
        self.assertIsNone(report["layers"]["scan_page"]["coverage"])
        # overall 只计有基准的层
        self.assertEqual(report["overall"], 0.8)

    def test_over_coverage_capped_reported(self):
        """索引数超标注（标注不全）如实报告 >1，提示基准需补。"""
        report = parse_coverage([{"block_type": "text"}] * 10,
                                {"text": 5})
        self.assertEqual(report["layers"]["text"]["coverage"], 2.0)

    def test_load_annotations(self):
        self.assertEqual(load_annotations('{"text": 10, "table": 2}'),
                         {"text": 10, "table": 2, "image": 0,
                          "scan_page": 0})
        self.assertEqual(load_annotations({"text": 3}),
                         {"text": 3, "table": 0, "image": 0,
                          "scan_page": 0})

    def test_real_corpus_counts(self):
        """真实语料的索引侧计数（基准待人工盘点后接入）。"""
        import json

        chunks = []
        with open(CORPUS, encoding="utf-8") as fh:
            for line in fh:
                chunks.append(json.loads(line))
        counts = indexed_counts(chunks)
        # 实测分布（2026-09-10）：text 7298 + qa_pair 1000；table 41
        self.assertEqual(counts["text"], 8298)
        self.assertEqual(counts["table"], 41)
        # 图片/扫描页当前语料为 0（§21 链路已备，语料待接入）
        self.assertEqual(counts["image"], 0)
        self.assertEqual(counts["scan_page"], 0)


if __name__ == "__main__":
    unittest.main()

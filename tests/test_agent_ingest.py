"""任务 #21 验证：Chunk 实体识别与 Metadata（§22）。

验证点：
1. §22 十键齐备（元数据完整性）；
2. 实体识别嵌入建库流程（含实体的 chunk 正确标注）；
3. domain 派生：实体优先、关键词兜底、无信号 general；
4. source_type 映射（provenance / qa_pair）；
5. 真实语料接入（chunks.jsonl 前 800 条）：键完整、语言正确、
   chunk_id 唯一、实体覆盖率合理、role_scope 默认开放。
"""

import unittest
from pathlib import Path

from durian_agent.rag.ingest import (
    CHUNK_METADATA_KEYS,
    build_chunk_record,
    infer_domain,
    ingest_corpus,
)

CORPUS = Path(__file__).resolve().parent.parent / "rag_build" / "chunks.jsonl"


class TestBuildChunkRecord(unittest.TestCase):

    def test_metadata_keys_complete(self):
        record = build_chunk_record("普通文本", chunk_id="c1", document_id="d1")
        for key in CHUNK_METADATA_KEYS:
            self.assertIn(key, record, msg=f"缺键: {key}")
        self.assertEqual(len(CHUNK_METADATA_KEYS), 10)

    def test_entity_recognition_embedded(self):
        record = build_chunk_record(
            "猫山王开花期要注意炭疽病防治",
            chunk_id="c2", document_id="d2")
        self.assertEqual(record["entities"]["cultivar"], "猫山王")
        self.assertEqual(record["entities"]["disease"], "炭疽病")
        self.assertEqual(record["domain"], "plant_protection")  # 实体优先

    def test_domain_keyword_fallback(self):
        self.assertEqual(infer_domain("灌溉制度与排水管理"), "irrigation")
        self.assertEqual(infer_domain("施肥方案说明"), "fertilization")
        self.assertEqual(infer_domain("你好世界"), "general")

    def test_source_type_mapping(self):
        self.assertEqual(
            build_chunk_record("x", chunk_id="c", document_id="d",
                               provenance="legacy_faq")["source_type"], "faq")
        self.assertEqual(
            build_chunk_record("x", chunk_id="c", document_id="d",
                               provenance="legacy_lit_salvage")["source_type"],
            "literature")
        self.assertEqual(
            build_chunk_record("x", chunk_id="c", document_id="d",
                               block_type="qa_pair")["source_type"], "faq")
        self.assertEqual(
            build_chunk_record("x", chunk_id="c", document_id="d",
                               source_type="sop")["source_type"], "sop")

    def test_default_scopes_open(self):
        """权限系统（#47）就绪前默认开放，防止误伤召回。"""
        record = build_chunk_record("x", chunk_id="c", document_id="d")
        self.assertEqual(record["role_scope"], ["worker", "manager"])
        self.assertEqual(record["orchard_scope"], [])   # 空 = 全园区


class TestIngestCorpus(unittest.TestCase):
    """真实语料接入验证（前 800 条，控制测试耗时）。"""

    @classmethod
    def setUpClass(cls):
        cls.records = ingest_corpus(CORPUS, limit=800)

    def test_corpus_loaded(self):
        self.assertEqual(len(self.records), 800)

    def test_all_keys_and_unique_ids(self):
        ids = set()
        for record in self.records:
            for key in CHUNK_METADATA_KEYS:
                self.assertIn(key, record)
            self.assertTrue(record["chunk_id"])
            self.assertTrue(record["document_id"])
            self.assertTrue(record["text"])
            ids.add(record["chunk_id"])
        self.assertEqual(len(ids), 800, msg="chunk_id 必须唯一")

    def test_language_distribution(self):
        langs = {r["language"] for r in self.records}
        # 前 800 条覆盖 legacy_thai（th）与文献（zh/en）
        self.assertIn("th", langs)
        for lang in langs:
            self.assertIn(lang, {"zh", "en", "th", "ms"})

    def test_entity_coverage_reasonable(self):
        """部分 chunk 应识别出实体（术语表覆盖品种/病害/虫害等）。"""
        with_entity = sum(
            1 for r in self.records
            if any(v for v in r["entities"].values())
        )
        self.assertGreater(with_entity, 50,
                           msg=f"实体覆盖率异常低: {with_entity}/800")

    def test_domain_not_all_general(self):
        domains = {r["domain"] for r in self.records}
        self.assertIn("plant_protection", domains)
        self.assertTrue(domains & {"fertilization", "variety", "irrigation"})


if __name__ == "__main__":
    unittest.main()

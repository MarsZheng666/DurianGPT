"""任务 #17 验证：文档解析与结构识别（§18：正文/标题/表格/图片/扫描页）。

验证点：
1. 显式 type 字段正确映射（MinerU 常见值）；
2. 缺 type 时的启发式：markdown 管道符=表格、图片特征=图片、标题模式；
3. 整页图像 + page_scan → 扫描页分流；
4. iter_document_blocks 保留溯源字段并附 block_type；
5. 混合类型文档整体分类正确。
"""

import unittest

from durian_agent.rag.chunking import (
    BLOCK_TYPES,
    CHUNK_MAX_CHARS,
    CHUNK_MIN_CHARS,
    chunk_document,
    chunk_text,
    classify_block,
    iter_document_blocks,
)


class TestClassifyBlock(unittest.TestCase):

    def test_explicit_types(self):
        cases = {
            "text": "text", "paragraph": "text", "title": "title",
            "section_header": "title", "table": "table", "image": "image",
            "figure": "image",
        }
        for raw_type, expected in cases.items():
            self.assertEqual(classify_block({"type": raw_type, "text": "x"}), expected)

    def test_heuristic_markdown_table(self):
        block = {"text": "|品种|株距|\n|---|---|\n|金枕|8米|"}
        self.assertEqual(classify_block(block), "table")

    def test_heuristic_image(self):
        self.assertEqual(
            classify_block({"image_path": "images/page3.jpg", "text": ""}), "image")
        self.assertEqual(
            classify_block({"text": "![图1](output/fig2.png)"}), "image")

    def test_heuristic_title(self):
        for text in ["第三章 施肥管理", "3.2 灌溉制度", "IV. 结论"]:
            self.assertEqual(classify_block({"text": text}), "title", msg=text)
        # 正文（句末标点/长段）不能误判为标题
        for text in ["施肥应在雨季进行。", "这段正文很长" * 20 + "。"]:
            self.assertEqual(classify_block({"text": text}), "text", msg=text[:20])

    def test_scan_page_detection(self):
        block = {"type": "image", "page_idx": 7, "page_scan": True, "text": ""}
        self.assertEqual(classify_block(block), "scan_page")
        # 普通插图（非整页扫描）仍是 image
        block = {"type": "image", "text": ""}
        self.assertEqual(classify_block(block), "image")

    def test_block_types_closed_set(self):
        self.assertEqual(set(BLOCK_TYPES),
                         {"text", "title", "table", "image", "scan_page"})


class TestIterDocumentBlocks(unittest.TestCase):

    def test_mixed_document(self):
        content = [
            {"type": "title", "text": "第二章 病虫害防治"},
            {"type": "text", "text": "炭疽病是常见病害。"},
            {"type": "table", "text": "|病害|药剂|"},
            {"text": "![症状图](figs/fig1.jpg)"},
            {"type": "image", "page_idx": 9, "page_scan": True, "text": ""},
        ]
        blocks = iter_document_blocks(content)
        self.assertEqual([b["block_type"] for b in blocks],
                         ["title", "text", "table", "image", "scan_page"])
        # 溯源字段透传（§22/§56 需要）
        self.assertEqual(blocks[4]["page_idx"], 9)

    def test_empty_and_invalid_input(self):
        self.assertEqual(iter_document_blocks([]), [])
        self.assertEqual(iter_document_blocks(["不是dict", None]), [])


class TestChunkText(unittest.TestCase):
    """任务 #20：正文递归分块（§19：300～800，overlap 10%～20%）。"""

    @classmethod
    def setUpClass(cls):
        # 生成带句号边界的长中文正文（约 20 句 × 60 字 ≈ 1200+ 字）
        sentences = [f"榴莲施肥管理第{i}条要点，需要根据树龄和生育阶段调整用肥量。" for i in range(30)]
        cls.long_text = "".join(sentences)

    def test_short_text_single_piece(self):
        short = ("施肥应在雨季进行，注意排水防涝，避免积水烂根；旱季则需适当补水，"
                 "并配合覆盖保墒；幼树薄肥勤施，结果树按生育阶段调整氮磷钾比例。")
        self.assertGreaterEqual(len(short), 60)   # 前置：确在单块区间
        self.assertEqual(chunk_text(short), [short])
        self.assertEqual(chunk_text("太短"), [])   # 低于 60 字噪声下限

    def test_pieces_within_bounds(self):
        pieces = chunk_text(self.long_text)
        self.assertGreater(len(pieces), 1)
        for piece in pieces:
            self.assertLessEqual(len(piece), CHUNK_MAX_CHARS + 5,
                                 msg=f"超出上限: {len(piece)}")
        # 可分块的长文：除最后一块外均应达到下限附近
        for piece in pieces[:-1]:
            self.assertGreaterEqual(len(piece), CHUNK_MIN_CHARS * 0.5,
                                    msg=f"碎块: {len(piece)}")

    def test_sentence_boundary_respected(self):
        pieces = chunk_text(self.long_text)
        for piece in pieces[:-1]:
            self.assertTrue(
                piece.rstrip().endswith(("。", "；", ";", ".", "\n")),
                msg=f"句中切断: ...{piece[-20:]!r}",
            )

    def test_overlap_in_range(self):
        """相邻块共享 10%～20% size 的重叠文本。"""
        pieces = chunk_text(self.long_text)
        for prev, cur in zip(pieces, pieces[1:]):
            # 在前块尾部找当前块开头的重叠
            head = cur[:20]
            self.assertIn(head, prev[-int(CHUNK_MAX_CHARS * 0.25):],
                          msg="相邻块无重叠")


class TestChunkDocument(unittest.TestCase):

    def test_section_aware_chunking(self):
        blocks = iter_document_blocks([
            {"type": "title", "text": "第三章 病虫害防治"},
            {"type": "text", "text": "炭疽病防治要点一。" * 100},
            {"type": "title", "text": "3.1 常见病害"},
            {"type": "text", "text": "根腐病多发生于排水不良的果园地块，雨季前应做好排水沟清理与高垄栽培管理，发病初期可见叶片黄化脱落，严重时整株萎蔫枯死，需及时挖除病株并对土壤消毒。"},
        ])
        chunks = chunk_document(blocks)
        self.assertGreaterEqual(len(chunks), 2)
        sections = {c["section"] for c in chunks}
        self.assertIn("第三章 病虫害防治", sections)
        self.assertIn("3.1 常见病害", sections)

    def test_table_block_kept_whole(self):
        long_table = "|病害|药剂|\n|---|---|\n" + "|炭疽病|波尔多液|\n" * 200
        blocks = iter_document_blocks([
            {"type": "title", "text": "附录"},
            {"type": "table", "text": long_table},
            {"type": "text", "text": "表格说明见上文。"},
        ])
        chunks = chunk_document(blocks)
        tables = [c for c in chunks if c["block_type"] == "table"]
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0]["text"], long_table.strip())   # 整块保留不切分
        self.assertEqual(tables[0]["section"], "附录")

    def test_image_block_own_chunk(self):
        blocks = iter_document_blocks([
            {"type": "text", "text": "症状如图所示。"},
            {"text": "![fig](f1.jpg)"},
        ])
        chunks = chunk_document(blocks)
        images = [c for c in chunks if c["block_type"] == "image"]
        self.assertEqual(len(images), 1)

    def test_text_blocks_merged_within_section(self):
        blocks = iter_document_blocks([
            {"type": "title", "text": "灌溉"},
            {"type": "text", "text": "滴灌省水，适合缺水园区，配合水肥一体化效果更好，可精准控制每株用水量与施肥量。"},
            {"type": "text", "text": "喷灌覆盖广，适合苗期降温，但会增加叶面湿度需注意病害防控，不宜在花期使用。"},
        ])
        chunks = chunk_document(blocks)
        self.assertEqual(len(chunks), 1)
        self.assertIn("滴灌省水", chunks[0]["text"])
        self.assertIn("喷灌覆盖广", chunks[0]["text"])


if __name__ == "__main__":
    unittest.main()

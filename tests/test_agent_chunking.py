"""任务 #17 验证：文档解析与结构识别（§18：正文/标题/表格/图片/扫描页）。

验证点：
1. 显式 type 字段正确映射（MinerU 常见值）；
2. 缺 type 时的启发式：markdown 管道符=表格、图片特征=图片、标题模式；
3. 整页图像 + page_scan → 扫描页分流；
4. iter_document_blocks 保留溯源字段并附 block_type；
5. 混合类型文档整体分类正确。
"""

import unittest

from durian_agent.rag.chunking import BLOCK_TYPES, classify_block, iter_document_blocks


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


if __name__ == "__main__":
    unittest.main()

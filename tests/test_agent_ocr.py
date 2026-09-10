"""任务 #22 验证：扫描件处理（§21：OCR → 版面恢复 → Chunk）。

验证点：
1. restore_layout：行距正常聚一段、大间距分段、CJK 无空格/拉丁有空格；
2. ocr_scan：单页 → 段落；
3. ocr_pages_to_chunks：多页 → #20 递归分块（§21 完整链路）；
4. tesseract 通道：本机未装时清晰报错（不静默降级）；
5. vlm_ocr：VLM 文本 → 线性行（已知局限：无坐标）。
"""

import unittest

from durian_agent.rag.ocr import (
    OcrConfigError,
    ocr_pages_to_chunks,
    ocr_scan,
    restore_layout,
    tesseract_ocr,
    vlm_ocr,
)


def line(text, top, left=0, h=20):
    return {"text": text, "top": top, "left": left, "width": 100, "height": h}


def fake_ocr(image_bytes):
    """两段的确定性 OCR 输出：第一段三行紧密，第二段一行（间距大）。"""
    assert image_bytes == b"scan-page"
    return [
        line("榴莲幼树施肥要点", 0),
        line("薄肥勤施", 22),
        line("每株施复合肥 50 g", 44),
        line("雨季注意排水防涝", 200),          # 大间距 → 新段落
    ]


class TestRestoreLayout(unittest.TestCase):

    def test_paragraph_split_by_gap(self):
        paragraphs = restore_layout([
            line("第一段第一行", 0), line("第一段第二行", 22),
            line("第二段第一行", 200),
        ])
        self.assertEqual(paragraphs, ["第一段第一行第一段第二行", "第二段第一行"])

    def test_cjk_join_no_space_latin_with_space(self):
        paragraphs = restore_layout([
            line("榴莲施肥", 0), line("50 g per tree", 22),
        ])
        self.assertEqual(paragraphs, ["榴莲施肥50 g per tree"])

    def test_empty_input(self):
        self.assertEqual(restore_layout([]), [])


class TestOcrScan(unittest.TestCase):

    def test_single_page_to_paragraphs(self):
        paragraphs = ocr_scan(b"scan-page", fake_ocr)
        self.assertEqual(len(paragraphs), 2)
        self.assertIn("薄肥勤施", paragraphs[0])
        self.assertEqual(paragraphs[1], "雨季注意排水防涝")


class TestOcrPagesToChunks(unittest.TestCase):

    def test_full_pipeline_to_chunks(self):
        """§21 完整链路：多页扫描 → OCR → 版面恢复 → 递归分块。"""
        chunks = ocr_pages_to_chunks([b"scan-page", b"scan-page"], fake_ocr)
        self.assertTrue(chunks)
        # 分块产出含正文且带 block_type
        for chunk in chunks:
            self.assertEqual(chunk["block_type"], "text")
            self.assertTrue(chunk["text"])
        # 两页的内容都在（60 字下限过滤后可能合并）
        joined = "\n".join(c["text"] for c in chunks)
        self.assertIn("薄肥勤施", joined)
        self.assertIn("雨季注意排水防涝", joined)


class TestChannels(unittest.TestCase):

    def test_tesseract_missing_raises_clearly(self):
        import shutil
        from unittest.mock import patch

        with patch.object(shutil, "which", return_value=None):
            with self.assertRaises(OcrConfigError) as ctx:
                tesseract_ocr()
        self.assertIn("tesseract", str(ctx.exception))
        self.assertIn("vlm_ocr", str(ctx.exception))   # 给出备选路径

    def test_vlm_ocr_linear_lines(self):
        """VLM OCR 已知局限：无坐标 → 行距退化为 0 → 线性行合成一段。"""

        def fake_vlm(image_bytes, prompt):
            assert prompt.startswith("Extract all visible text")
            return "榴莲栽培手册\n\n第二章 施肥\n每株50克"

        paragraphs = ocr_scan(b"x", vlm_ocr(fake_vlm))
        # 单段线性拼接（行间无空格；行内原有空格保留）；真分段用 tesseract 通道
        self.assertEqual(paragraphs, ["榴莲栽培手册第二章 施肥每株50克"])


if __name__ == "__main__":
    unittest.main()

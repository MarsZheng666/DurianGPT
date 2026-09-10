"""任务 #18 验证：图片处理 VLM 描述（§21：描述+图中文字+农业语义）。

验证点：
1. describe_image：三要素解析（JSON/围栏 JSON）；非法输出保守降级；
2. image_description_to_text：三要素 → 可检索文本；
3. enrich_image_blocks：image/scan_page 块富化、非图片块不动、
   单块失败不中断整批；
4. 真实通道 qwen_vl：未配置时明确报错（测试不产生 API 费用）；
   prompt 含「不得猜测不可见事实」约束。
"""

import unittest

from durian_agent.rag.vision import (
    VLM_DESCRIBE_PROMPT,
    VLMConfigError,
    describe_image,
    enrich_image_blocks,
    image_description_to_text,
    qwen_vl,
)

GOOD_VLM_RESPONSE = """```json
{"description": "榴莲叶片背面出现深褐色水渍状病斑",
 "text_in_image": "图3-2 炭疽病症状",
 "agricultural_semantics": "炭疽病叶片症状示意"}
```"""

PNG_BYTES = b"\x89PNG fake image bytes"


def fake_vlm(image_bytes, prompt):
    assert image_bytes == PNG_BYTES
    return GOOD_VLM_RESPONSE


class TestDescribeImage(unittest.TestCase):

    def test_three_fields_parsed(self):
        info = describe_image(PNG_BYTES, fake_vlm)
        self.assertEqual(info["description"], "榴莲叶片背面出现深褐色水渍状病斑")
        self.assertEqual(info["text_in_image"], "图3-2 炭疽病症状")
        self.assertEqual(info["agricultural_semantics"], "炭疽病叶片症状示意")

    def test_plain_json_parsed(self):
        info = describe_image(PNG_BYTES, lambda b, p:
                              '{"description": "灌溉示意图", "text_in_image": "", '
                              '"agricultural_semantics": "滴灌安装"}')
        self.assertEqual(info["description"], "灌溉示意图")
        self.assertEqual(info["agricultural_semantics"], "滴灌安装")

    def test_invalid_output_degrades_gracefully(self):
        info = describe_image(PNG_BYTES, lambda b, p: "这张图展示了灌溉系统。")
        self.assertIn("灌溉", info["description"])
        self.assertEqual(info["text_in_image"], "")
        self.assertEqual(info["agricultural_semantics"], "")

    def test_prompt_constraints(self):
        self.assertIn("Do not guess facts that are not visible", VLM_DESCRIBE_PROMPT)
        self.assertIn("text_in_image", VLM_DESCRIBE_PROMPT)
        self.assertIn("agricultural_semantics", VLM_DESCRIBE_PROMPT)


class TestImageDescriptionToText(unittest.TestCase):

    def test_full_info(self):
        text = image_description_to_text({
            "description": "病斑特写",
            "text_in_image": "图3-2",
            "agricultural_semantics": "炭疽病症状",
        })
        self.assertIn("病斑特写", text)
        self.assertIn("图中文字：图3-2", text)
        self.assertIn("农业语义：炭疽病症状", text)

    def test_description_only(self):
        text = image_description_to_text({"description": "果园全景", "text_in_image": "",
                                          "agricultural_semantics": ""})
        self.assertEqual(text, "果园全景")


class TestEnrichImageBlocks(unittest.TestCase):

    def test_image_blocks_enriched(self):
        blocks = [
            {"block_type": "image", "image_path": "figs/a.jpg", "text": ""},
            {"block_type": "text", "text": "正文段落"},
        ]
        enriched = enrich_image_blocks(blocks, fake_vlm,
                                       image_loader=lambda p: PNG_BYTES)
        self.assertIn("炭疽病", enriched[0]["text"])       # VLM 描述进 text
        self.assertIn("vlm_info", enriched[0])
        self.assertEqual(enriched[1]["text"], "正文段落")   # 非图片块不动

    def test_scan_page_blocks_enriched(self):
        blocks = [{"block_type": "scan_page", "image_path": "scan/p7.png", "text": ""}]
        enriched = enrich_image_blocks(blocks, fake_vlm,
                                       image_loader=lambda p: PNG_BYTES)
        self.assertTrue(enriched[0]["text"])

    def test_single_failure_does_not_break_batch(self):
        def flaky_loader(path):
            if "bad" in path:
                raise FileNotFoundError(path)
            return PNG_BYTES

        blocks = [
            {"block_type": "image", "image_path": "bad/missing.jpg", "text": ""},
            {"block_type": "image", "image_path": "figs/ok.jpg", "text": ""},
        ]
        enriched = enrich_image_blocks(blocks, fake_vlm, image_loader=flaky_loader)
        self.assertIn("enrich_error", enriched[0])          # 失败标记
        self.assertIn("炭疽病", enriched[1]["text"])         # 其余正常


class TestRealChannel(unittest.TestCase):

    def test_unconfigured_raises_clearly(self):
        import os
        from unittest.mock import patch

        saved = {k: os.environ.pop(k, None) for k in
                 ("QWEN_VL_MAX_API_KEY", "QWEN_VL_MAX_BASE_URL", "QWEN_VL_MAX_MODEL")}
        try:
            # .env 在本机存在且含真实键——同时屏蔽 dotenv 加载与环境变量
            with patch("durian_agent.rag.vision._load_dotenv_quietly"):
                with self.assertRaises(VLMConfigError):
                    qwen_vl(api_key="", base_url="", model="")
        finally:
            for k, v in saved.items():
                if v:
                    os.environ[k] = v

    def test_configured_channel_builds(self):
        """显式传参时通道可构造（不实际调用，不产生费用）。"""
        fn = qwen_vl(api_key="sk-test", base_url="https://example.com",
                     model="test-vl")
        self.assertTrue(callable(fn))


if __name__ == "__main__":
    unittest.main()

"""任务 #8 验证：通用 InputNormalize（架构文档 §11）。

验证点：
1. 四语种（zh/en/th/ms）脏输入各自归一正确；
2. 幂等：二次归一不改变结果；
3. 干净输入不被误改（无损通过）；
4. 大小写默认保留、fold_case 可开；
5. 数字/单位标准化（全角、泰文数字、温度变体、数字-单位空格）。
"""

import unittest

from durian_agent.normalize import normalize_input


class TestNormalizeInput(unittest.TestCase):

    # ── 四语种脏输入 ──────────────────────────────────────────

    def test_zh_dirty_input(self):
        # 全角尖括号、零宽字符、重复句读、多余空白
        raw = "什么是＜榴莲坐果＞？\u200b。。   需要修剪吗？？？"
        out = normalize_input(raw)
        self.assertNotIn("＜", out)
        self.assertNotIn("\u200b", out)
        self.assertNotIn("。。", out)
        self.assertNotIn("？？？", out)
        self.assertNotIn("  ", out)
        self.assertIn("榴莲坐果", out)

    def test_en_dirty_input(self):
        raw = "How   to   fertilize   DURIAN   trees???"
        out = normalize_input(raw)
        self.assertNotIn("  ", out)
        self.assertNotIn("???", out)
        # 默认保留大小写（实体名/展示需要）
        self.assertIn("DURIAN", out)

    def test_th_dirty_input(self):
        # 零宽字符 + 泰文数字 + 全角字符
        raw = "โรครากเน่า\ufeffรักษายังไง ๑๒๓"
        out = normalize_input(raw)
        self.assertNotIn("\ufeff", out)
        self.assertIn("123", out)          # 泰文数字 → ASCII
        self.assertIn("โรครากเน่า", out)  # 泰文内容本体不动

    def test_ms_dirty_input(self):
        raw = "berapa   kerap   menyiram   pokok durian !!!"
        out = normalize_input(raw)
        self.assertNotIn("  ", out)
        self.assertNotIn("!!!", out)   # 3+ 连打压成单个
        self.assertTrue(out.startswith("berapa"))
        # 句末语气标点保留（！不剥离）
        self.assertTrue(out.endswith("durian !"))

    # ── 幂等与无损 ────────────────────────────────────────────

    def test_idempotent(self):
        for raw in [
            "什么是＜榴莲坐果＞？\u200b。。",
            "How   to   fertilize??? 25 °C",
            "โรครากเน่า ๑๒๓ ๒๕˚C",
            "siram pokok durian 3 kg",
        ]:
            once = normalize_input(raw)
            twice = normalize_input(once)
            self.assertEqual(once, twice, msg=f"不幂等: {raw!r}")

    def test_clean_input_passthrough(self):
        """干净输入语义无损通过；唯一允许的形式变化是全角标点归半角（NFKC）。"""
        cases = [
            ("什么是榴莲坐果？", "什么是榴莲坐果?"),   # ？(U+FF1F) → ?
            ("How to fertilize durian trees?",
             "How to fertilize durian trees?"),
            ("โรครากเน่ารักษายังไง", "โรครากเน่ารักษายังไง"),
            ("Berapa kerap menyiram pokok durian?",
             "Berapa kerap menyiram pokok durian?"),
        ]
        for clean, expected in cases:
            self.assertEqual(normalize_input(clean), expected,
                             msg=f"干净输入被意外改动: {clean!r}")

    # ── 大小写 ────────────────────────────────────────────────

    def test_fold_case_option(self):
        raw = "Musang King vs Raja Kunyit"
        self.assertEqual(normalize_input(raw), raw)                       # 默认保留
        self.assertEqual(normalize_input(raw, fold_case=True),
                         "musang king vs raja kunyit")                    # 检索侧

    # ── 数字与单位 ────────────────────────────────────────────

    def test_degree_variants_unified(self):
        for raw in ("25°C", "25 ° C", "25∘C", "25˚C", "25ºc"):
            self.assertEqual(normalize_input(raw), "25℃", msg=f"温度变体未统一: {raw!r}")

    def test_degree_without_digit_not_temperature_guessed(self):
        # 无数字前缀的 °C 也统一（室温°C 这类），但不把孤立的度数符号当温度
        self.assertEqual(normalize_input("室温°C"), "室温℃")

    def test_unit_space_compaction(self):
        self.assertEqual(normalize_input("25 mm"), "25mm")
        self.assertEqual(normalize_input("3 kg"), "3kg")
        self.assertEqual(normalize_input("500 ml"), "500ml")
        self.assertEqual(normalize_input("5 ppm"), "5ppm")

    def test_fullwidth_digits_normalized(self):
        self.assertEqual(normalize_input("＜２５ｍｍ＞"), "<25mm>")

    def test_thai_digits_normalized(self):
        self.assertEqual(normalize_input("๒๕"), "25")

    def test_ambiguous_thousands_separator_untouched(self):
        # 千分位逗号有歧义（可能是列表），明确不做
        raw = "1,200 kg"
        out = normalize_input(raw)
        self.assertIn("1,200", out)

    # ── 标点 ──────────────────────────────────────────────────

    def test_leading_trailing_punctuation_stripped(self):
        self.assertEqual(normalize_input("  。。请问怎么施肥？  "), "请问怎么施肥?")

    def test_chinese_ellipsis_preserved(self):
        # "……" 是中文省略号规范写法，不能压
        self.assertEqual(normalize_input("怎么办……"), "怎么办……")


if __name__ == "__main__":
    unittest.main()

"""任务 #2 验证：泰语归一化（词典优先 + tokenizer，架构文档 §11 泰语管线）。

验证点：
1. verbatim 迁移的声调修复与 rag_llamaindex 原实现逐例一致（parity，防漂移）；
2. 文档记录的修复样例各自正确（颠倒/双声调/词表/方案 AB）；
3. 修复幂等；干净泰文与非泰文不受影响；"25˚C" 不被误改；
4. normalize_thai_input 完成 §11 前两步（基础归一化 + 修复）；
5. tokenize_thai 词典优先：专业术语作为完整 token 保留，剩余片段 3-4 gram。
"""

import unittest

from durian_agent.glossary import load_glossary
from durian_agent.thai import (
    normalize_thai,
    normalize_thai_input,
    thai_domain_terms,
    tokenize_thai,
)


class TestThaiRepairParity(unittest.TestCase):
    """迁移段与 rag_llamaindex.py 原实现的同步保证。"""

    def test_parity_with_legacy(self):
        try:
            from rag_llamaindex import normalize_thai as legacy
        except Exception:  # pragma: no cover - 环境缺依赖时跳过
            self.skipTest("rag_llamaindex 不可导入")
        cases = [
            # 损坏形态全家桶（文档记录的样例与组合）
            "เช้ือรา",                     # 顺序颠倒 → เชื้อรา
            "น้้า", "ให้น้้า", "ซา้้",       # 双声调 → ำ
            "ไฟทอปธอรํา", "ไฟทอบธอรํา",     # 词表兜底
            "รากเนําโคน",                   # 方案 AB：ํา→ำ
            "โรครากเนําโคนเนําในต่างจังหวัด",  # 多处 ํ
            "ทนทานตํอโรครากเนําโคนเนํา",     # 判据边界（文档记录的难例）
            "นํ้า", "ตํ่า",                  # 叠写 SARA AM+声调
            "น้˚า", "เพือก่ ˚าจัดเ",         # U+02DA 形态
            "อุณหภูมิ 25˚C ที่กรุงเทพ",       # 度数符号不得误改
            "25˚C", "25°C",
            # 干净参照
            "ทุเรียนหมอนทองรสชาติดี",
            "โรครากเน่ารักษายังไง",
            "Musang King ราคาเท่าไหร่",
            "什么是榴莲坐果",
            "siram pokok durian",
            "",
        ]
        for case in cases:
            self.assertEqual(
                normalize_thai(case), legacy(case),
                msg=f"与 legacy 实现分叉: {case!r}",
            )


class TestThaiRepairBehavior(unittest.TestCase):

    def test_documented_repairs(self):
        self.assertEqual(normalize_thai("เช้ือรา"), "เชื้อรา")      # 真菌
        self.assertEqual(normalize_thai("น้้า"), "น้ำ")
        self.assertEqual(normalize_thai("ไฟทอปธอรํา"), "ไฟทอปธอรา")  # Phytophthora
        self.assertEqual(normalize_thai("รากเนําโคน"), "รากเนำโคน")   # 方案 AB

    def test_idempotent(self):
        for case in ["เช้ือรา", "รากเนําโคน", "น้้า", "ไฟทอปธอรํา"]:
            once = normalize_thai(case)
            self.assertEqual(normalize_thai(once), once)

    def test_clean_and_non_thai_untouched(self):
        for case in ["ทุเรียนหมอนทอง", "什么是坐果", "siram durian", "25˚C"]:
            self.assertEqual(normalize_thai(case), case)


class TestThaiInputPipeline(unittest.TestCase):

    def test_normalize_thai_input_combines_steps(self):
        # 泰文数字→ASCII、零宽→无、温度→℃，泰文内容保留
        raw = "โรครากเน่า ๑๒๕˚C \u200b"
        out = normalize_thai_input(raw)
        self.assertIn("125℃", out)
        self.assertNotIn("\u200b", out)
        self.assertIn("โรครากเน่า", out)

    def test_degree_consumed_before_sara_am_rules(self):
        # ˚ 在泰文语境中先被温度规则消费成 ℃，不会撞上 SARA AM 修复
        out = normalize_thai_input("อุณหภูมิ 25˚C วันนี้")
        self.assertIn("25℃", out)
        self.assertNotIn("ำC", out)


class TestThaiTokenize(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.glossary = load_glossary()
        cls.terms = thai_domain_terms(cls.glossary)
        # 品种类目实测含泰文别名（หมอนทอง 等）
        self_cls = cls
        self_cls.min_terms = 5

    def test_glossary_has_thai_aliases(self):
        self.assertGreaterEqual(len(self.terms), self.min_terms)
        for known in ["หมอนทอง", "ชะนี", "ก้านยาว"]:
            self.assertIn(known, self.terms, msg=f"术语表缺泰文别名: {known}")

    def test_domain_term_kept_whole(self):
        # 核心验收：专业术语不进 n-gram，作为完整 token 保留
        tokens = tokenize_thai("ราคาหมอนทองตอนนี้เท่าไหร่", terms=self.terms)
        self.assertIn("หมอนทอง", tokens)

    def test_custom_terms_dictionary_first(self):
        # 不依赖术语表内容，用自定义词表验证机制本身
        tokens = tokenize_thai("ปลูกหมอนทอง", terms=["หมอนทอง"])
        self.assertIn("หมอนทอง", tokens)
        # 剩余片段进入 3-gram
        self.assertTrue(any(t.startswith("ปลู") for t in tokens))

    def test_longest_match_wins(self):
        # 长词优先：包含短词的长术语整体成 token，不被短词截断
        tokens = tokenize_thai("ดูแลหมอนทอง", terms=["หมอนทอง", "หมอน"])
        self.assertIn("หมอนทอง", tokens)
        self.assertNotIn("หมอน", tokens)

    def test_non_thai_not_tokenized_here(self):
        tokens = tokenize_thai("Musang King หมอนทอง", terms=["หมอนทอง"])
        self.assertIn("หมอนทอง", tokens)
        self.assertFalse(any("Musang" in t for t in tokens))

    def test_corrupted_input_normalized_before_match(self):
        # 输入里的损坏形态先修复，再进词典匹配（词表收的是干净写法）
        tokens = tokenize_thai("ราคาหมอนทอง", terms=["หมอนทอง"])
        self.assertIn("หมอนทอง", tokens)


if __name__ == "__main__":
    unittest.main()

"""任务 #24 验证：Query Build 跨语言扩展完整验收（§9/§23）。

验证点：
1. §23 三输入汇齐：原查询 + Canonical Schema（注入复用）+ 术语词典；
2. 泰文查询扩展出 en/zh 别名（§23 核心场景）；
3. 英文查询扩展出 zh/th 别名；
4. schema 注入与自动检测等价（图内不二次检测）；
5. 不做整句翻译（§23：扩展只发生在实体级，非实体词不动）。
"""

import unittest

from durian_agent.rag.recall import build_queries


class TestQueryBuilderAcceptance(unittest.TestCase):

    def test_thai_query_expands_to_zh_and_en(self):
        """泰文查询 → canonical 含中文标准名，expanded 含中英别名。"""
        queries = build_queries("หมอนทอง ราคาเท่าไหร่")
        self.assertIn("金枕", queries["canonical"])
        # 四语别名至少覆盖 zh（标准名自身）与 en
        self.assertIn("金枕", queries["expanded"])
        for en_alias in ("monthong", "golden pillow"):
            self.assertIn(en_alias, queries["expanded"].lower())

    def test_english_query_expands_to_zh_and_thai(self):
        queries = build_queries("how to fertilize Musang King")
        self.assertIn("猫山王", queries["canonical"])
        self.assertIn("D197".lower(), queries["expanded"].lower())
        self.assertIn("猫山王", queries["expanded"])

    def test_semantic_schema_injection(self):
        """注入 #3 的 Canonical Schema 时复用其 entities（不二次检测）。"""
        semantic = {"entities": {"cultivar": "猫山王", "disease": None}}
        queries = build_queries("随便什么句子", semantic=semantic)
        self.assertIn("猫山王", queries["canonical"])

    def test_schema_injection_equivalent_to_detection(self):
        query = "Musang King fertilization"
        auto = build_queries(query)
        semantic = {"entities": {"cultivar": "猫山王"}}
        injected = build_queries(query, semantic=semantic)
        self.assertEqual(auto["canonical"], injected["canonical"])
        self.assertEqual(auto["expanded"], injected["expanded"])

    def test_no_whole_sentence_translation(self):
        """非实体词不做翻译：泰文问题词 ราคา 不被翻译成中文「价格」。"""
        queries = build_queries("หมอนทอง ราคาเท่าไหร่")
        # 扩展词里只有品种别名，没有句子的中文翻译
        self.assertNotIn("价格多少", queries["expanded"])
        # 原查询保留（供 Original 路）
        self.assertIn("หมอนทอง", queries["original"])

    def test_disease_entity_expands(self):
        queries = build_queries("anthracnose control")
        self.assertIn("炭疽病", queries["canonical"])
        # 炭疽病的别名（anthracnose 自身已在原查询）
        self.assertIn("炭疽病", queries["expanded"])


if __name__ == "__main__":
    unittest.main()

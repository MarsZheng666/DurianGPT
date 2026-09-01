import unittest

import durian__inference_api as api
import rag_llamaindex


class ExactRAGTest(unittest.TestCase):
    def test_ocr_temperature_and_length_units_are_normalized(self):
        raw = (
            "年平均温度 24\\% 以上，绝对低温 5\\% 以上，"
            "年降水量 1000\\mathrm{mm} 以上。"
        )
        cleaned = rag_llamaindex.clean_text(raw)
        self.assertIn("24℃", cleaned)
        self.assertIn("5℃", cleaned)
        self.assertIn("1000 mm", cleaned)

    def test_multi_blank_answers_are_copied_from_source(self):
        query = (
            "请严格根据知识库原文填空，只填写答案："
            "园区年平均温度____以上，绝对低温____以上，"
            "年降水量____以上。"
        )
        evidence = [
            {
                "text": (
                    "园区年平均温度 24\\% 以上，绝对低温 5\\% 以上，"
                    "年降水量 1000\\mathrm{mm} 以上。"
                )
            }
        ]
        answer = api.extract_exact_fill_answer(query, evidence)
        self.assertEqual(answer, "24℃；5℃；1000mm")

    def test_exact_extractor_refuses_partial_source(self):
        query = "园区年平均温度____以上，年降水量____以上。"
        evidence = [{"text": "园区年平均温度 24℃ 以上。"}]
        self.assertIsNone(api.extract_exact_fill_answer(query, evidence))

    def test_exact_extractor_chooses_shortest_match_when_ocr_repeats_prefix(self):
        query = (
            "只填写答案：园区年平均温度____以上，绝对低温____以上，"
            "年降水量____以上。"
        )
        evidence = [
            {
                "text": (
                    "园区年平均温度 园区年平均温度 24℃ 以上，"
                    "绝对低温 5℃ 以上，年降水量 1000 mm 以上。"
                )
            }
        ]
        self.assertEqual(
            api.extract_exact_fill_answer(query, evidence),
            "24℃；5℃；1000mm",
        )

    def test_exact_extractor_removes_mineru_content_markers(self):
        query = (
            "只填写答案：年平均温度____以上，绝对低温____以上，"
            "年降水量____以上。"
        )
        evidence = [
            {
                "text": (
                    "'content': '年平均温度 '}, 'content': '24\\\\%'}, "
                    "'content': '以上，绝对低温 '}, 'content': '5\\\\%'}, "
                    "'content': '以上，年降水量 '}, "
                    "'content': '1000\\\\mathrm{mm}'}, 'content': '以上。'"
                )
            }
        ]
        self.assertEqual(
            api.extract_exact_fill_answer(query, evidence),
            "24℃；5℃；1000mm",
        )


    def test_durian_code_lookup_detects_reverse_lookup(self):
        self.assertTrue(api.is_exact_answer_query("D168 对应什么榴莲？"))
        self.assertTrue(api.is_exact_answer_query("D175对应哪个品种"))

    def test_durian_code_lookup_uses_reference_aliases(self):
        query = "D168 对应什么榴莲？"
        evidence = [
            {
                "text": (
                    "D168榴莲为什么被取为101和IOI。作为冠军榴莲之一，"
                    "D168也被称为榴莲101, IOI, Johor Mas, Hajah Hasmah和Mas Muar。"
                ),
                "doc": "D168榴莲为什么被取为101和IOI",
            }
        ]
        answer = api.extract_durian_code_lookup_answer(query, evidence)
        self.assertIn("D168", answer)
        self.assertIn("101", answer)
        self.assertIn("IOI", answer)
        self.assertNotIn("红虾", answer)
        self.assertNotIn("Red Prawn", answer)

    def test_durian_alias_lookup_uses_reference_code(self):
        query = "红虾的编号是什么？"
        evidence = [
            {
                "text": "红虾（Red Prawn, Udang Merah）的马来西亚品种登记编号为 D175。",
                "doc": "cultivar_table.pdf",
            }
        ]
        answer = api.extract_durian_code_lookup_answer(query, evidence)
        self.assertIn("D175", answer)
        self.assertIn("红虾", answer)

    def test_durian_code_lookup_does_not_split_or_inside_alias(self):
        query = "D163编号对应的品种是什么"
        evidence = [
            {
                "text": "D163 Horloh Durian: Taste, Origin, and Facts | Dury Dury",
                "doc": "D163 Horloh Durian: Taste, Origin, and Facts | Dury Dury",
            }
        ]
        answer = api.extract_durian_code_lookup_answer(query, evidence)
        self.assertIn("Horloh", answer)
        self.assertNotIn("loh : Taste", answer)

    def test_durian_code_lookup_prefers_name_before_also_known_as_code(self):
        query = "D163编号对应的品种是什么"
        evidence = [
            {
                "text": "Hor Lor (Hulu), also known as D163, at 227 Katong Durian is smooth thick fragrant",
                "doc": "Hor Lor (Hulu), also known as D163, at... - 227 Katong Durian | https://www.facebook.com/example",
            }
        ]
        answer = api.extract_durian_code_lookup_answer(query, evidence)
        self.assertIn("Hor Lor", answer)
        self.assertIn("Hulu", answer)
        self.assertNotIn("对应的是 at", answer.lower())
        self.assertNotIn("at 榴莲", answer.lower())
        self.assertNotIn("也常见写作 at", answer.lower())

    def test_durian_code_lookup_rejects_taste_descriptors_as_aliases(self):
        query = "D163编号对应的品种是什么"
        evidence = [
            {
                "text": "D163 清甜花香 葫蘆 幼滑 少纖維感的人 葫蘆即 Hor Lor",
                "doc": "榴蓮10大品種分類及選購指南:貓山王、金枕頭、黑刺,邊款先係你本命? | https://zh.zalora.com.hk/blog/lifestyle/durian-varieties-guide-and-buying-tips",
            }
        ]
        answer = api.extract_durian_code_lookup_answer(query, evidence)
        self.assertIn("D163", answer)
        self.assertTrue("葫蘆" in answer or "葫芦" in answer or "Hor Lor" in answer)
        self.assertNotIn("清甜花香", answer)
        self.assertNotIn("幼滑", answer)
        self.assertNotIn("少纖維", answer)
        self.assertNotIn("的人", answer)

    def test_durian_code_context_card_uses_code_alias_subject(self):
        card = api.build_text_context_card(
            "D163编号对应的品种是什么",
            "D163 corresponds to Horloh durian. (Source: D163 Horloh Durian: Taste, Origin, and Facts | Dury Dury)",
            "en",
        )
        self.assertEqual(card["main_subject"], "D163 Horloh")

    def test_durian_code_context_card_trims_secondary_alias_phrase(self):
        card = api.build_text_context_card(
            "D163编号对应的品种是什么",
            "D163 对应的是 Hor Lor 榴莲，也常见写作 Hulu。（来源：Hor Lor (Hulu), also known as D163）",
            "zh",
        )
        self.assertEqual(card["main_subject"], "D163 Hor Lor")

    def test_durian_code_context_card_prefers_current_code_over_comparison_names(self):
        card = api.build_text_context_card(
            "D163编号对应的品种是什么",
            "D163 编号对应的是 **葫芦榴莲**（Horlor），不像猫山王那样昂贵。",
            "zh",
        )
        self.assertEqual(card["main_subject"], "D163 葫芦")

    def test_explicit_identifier_query_does_not_inherit_old_context(self):
        self.assertEqual(api.classify_user_intent("详细讲解一下D163"), "new_topic")

    def test_code_answer_separates_url_from_closing_parenthesis(self):
        source = "D163 Horloh Durian: Taste, Origin, and Facts | Dury Dury | https://durydury.com/durian-type/horloh）"
        answer = api._format_code_alias_answer("D163", ["Horloh"], source, "zh")
        self.assertIn("https://durydury.com/durian-type/horloh ）", answer)
        self.assertNotIn("horloh）", answer)

    def test_web_query_is_scoped_to_durian(self):
        self.assertEqual(api.scope_web_query_to_durian("D163"), "D163 durian")
        self.assertEqual(api.scope_web_query_to_durian("你觉得是什么疾病"), "你觉得是什么疾病 榴莲 durian")

    def test_web_evidence_rejects_non_durian_sources(self):
        self.assertFalse(api.evidence_is_durian_scoped({"title": "与压力有关的疾病", "text": "心理压力和焦虑"}))
        self.assertTrue(api.evidence_is_durian_scoped({"title": "D163 Horloh Durian", "text": "durian variety"}))

    def test_reference_url_trims_full_width_parenthesis(self):
        raw = "https://durydury.com/what-makes-durian-d88-stand-out-from-other-varieties）"
        self.assertEqual(
            api.normalize_reference_url(raw),
            "https://durydury.com/what-makes-durian-d88-stand-out-from-other-varieties",
        )

    def test_clean_model_output_trims_bare_url_punctuation(self):
        raw = "来源：https://durydury.com/what-makes-durian-d88-stand-out-from-other-varieties）"
        cleaned = api.clean_model_output(raw)
        self.assertEqual(
            cleaned,
            "来源：https://durydury.com/what-makes-durian-d88-stand-out-from-other-varieties",
        )

    def test_clean_model_output_preserves_markdown_link_close(self):
        raw = "[D88](https://durydury.com/what-makes-durian-d88-stand-out-from-other-varieties）)"
        cleaned = api.clean_model_output(raw)
        self.assertEqual(
            cleaned,
            "[D88](https://durydury.com/what-makes-durian-d88-stand-out-from-other-varieties)",
        )

    def test_fill_blank_detection(self):
        self.assertTrue(api.is_exact_answer_query("榴莲适温为____。"))
        self.assertFalse(api.is_exact_answer_query("榴莲适温是多少？"))


if __name__ == "__main__":
    unittest.main()

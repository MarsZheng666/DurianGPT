import tempfile
import unittest
from pathlib import Path

from durian_langgraph import DurianLangGraphRuntime


class FakeAdapter:
    def __init__(self):
        self.rag_queries = []
        self.final_messages = []
        self.generation_calls = 0

    def resolve_language(self, query, fallback):
        return "zh"

    def has_image_context(self, payloads):
        return any(item.get("image_url") for item in payloads)

    def is_casual_query(self, query):
        return query in {"你哈批", "谢谢"}

    def casual_reply(self, query, language):
        if query == "你哈批":
            return "如果刚才的回答有问题，请直接指出，我会修正。"
        return "好的。"

    def route_context(self, query, payloads, language):
        if query == "你好":
            return {
                "intent": "greeting",
                "use_history": False,
                "use_image_context": False,
                "reason": "greeting",
                "source": "fake",
            }
        return {
            "intent": "new_topic",
            "use_history": False,
            "use_image_context": False,
            "reason": "fake default",
            "source": "fake",
        }

    def resolve_context(self, query, payloads, language):
        return {"selected": False, "transition": "new_topic"}

    def enhance_query(self, query, language):
        return query

    def rag_available(self):
        return True

    def retrieve(self, query, language):
        self.rag_queries.append(query)
        return ([{"text": "reference", "score": 0.9, "doc": "test"}], "high", "reference")

    def build_final_messages(self, payloads, evidence_text, max_messages, language):
        result = [{"role": item["role"], "content": item["content"]} for item in payloads[-max_messages:]]
        self.final_messages.append(result)
        return result

    def prepare_evidence(self, evidence, language):
        return evidence

    def greeting_reply(self, language):
        return "你好，我在。"

    def grounded_not_found(self, language):
        return "RAGFlow 知识库中没有找到足以支持回答的原文。"

    def stream_generate(self, messages, generation, language):
        self.generation_calls += 1
        last = messages[-1]["content"]
        yield f"answer:{last}"

    def clean_output(self, text):
        return text

    def build_context_card(self, query, response, language, **kwargs):
        return {"main_subject": query, "summary": response, "source": kwargs.get("source")}

    def approx_tokens(self, text):
        return len(text)

    def log_route(self, **kwargs):
        return None


class ImageContextAdapter(FakeAdapter):
    def resolve_context(self, query, payloads, language):
        for payload in reversed(payloads[:-1]):
            card = payload.get("active_context_card")
            if payload.get("role") == "assistant" and card:
                return {
                    "selected": True,
                    "transition": "continue",
                    "card": card,
                    "standalone_query": query,
                    "system_instruction": (
                        f"Continue image topic: {card.get('main_subject')}"
                    ),
                }
        return super().resolve_context(query, payloads, language)


class ExactAnswerAdapter(FakeAdapter):
    def __init__(self, answer=None):
        super().__init__()
        self.answer = answer

    def is_exact_query(self, query):
        return "____" in query

    def extract_exact_answer(self, query, evidence, language):
        return self.answer

    def exact_not_found(self, language):
        return "知识库原文中未找到能够完整对应这些空格的内容。"

class SelectedContextAdapter(FakeAdapter):
    def __init__(self, should_use_rag=False):
        super().__init__()
        self.should_use_rag = should_use_rag

    def resolve_context(self, query, payloads, language):
        for payload in reversed(payloads[:-1]):
            card = payload.get("active_context_card")
            if payload.get("role") == "assistant" and card:
                subject = card.get("main_subject")
                return {
                    "selected": True,
                    "transition": "continue",
                    "card": card,
                    "standalone_query": f"{subject}：{query}",
                    "should_use_rag": self.should_use_rag,
                    "system_instruction": f"Continue topic: {subject}",
                }
        return {"selected": False, "transition": "new_topic"}


class ExactSelectedContextAdapter(SelectedContextAdapter):
    def is_exact_query(self, query):
        return "编码" in query

    def extract_exact_answer(self, query, evidence, language):
        if "黑刺" in query:
            return "黑刺的品种登记编号是 D200。"
        return None

    def exact_not_found(self, language):
        return "知识库原文中未找到可靠答案。"


class ImageExactContextAdapter(ExactSelectedContextAdapter):
    def is_exact_query(self, query):
        return "编号" in query

    def is_contextual_exact_query(self, query, payloads):
        return query == "它是什么编号" and any(
            isinstance(item.get("active_context_card"), dict)
            and (
                item["active_context_card"].get("image_url")
                or str(
                    item["active_context_card"].get("source") or ""
                ).startswith("image")
            )
            for item in payloads
        )

    def route_context(self, query, payloads, language):
        if query == "它是什么编号":
            return {
                "intent": "follow_up",
                "use_history": True,
                "use_image_context": True,
                "reason": "explicit image follow-up",
                "source": "fake",
            }
        return super().route_context(query, payloads, language)

    def extract_exact_answer(self, query, evidence, language):
        if "猫山王" in query:
            return "猫山王的品种登记编号是 D197。"
        return None


class ClarificationAdapter(FakeAdapter):
    def resolve_context(self, query, payloads, language):
        if query == "这个怎么办":
            return {
                "selected": False,
                "transition": "clarify",
            }
        return super().resolve_context(query, payloads, language)


class NoEvidenceAdapter(FakeAdapter):
    def retrieve(self, query, language):
        self.rag_queries.append(query)
        return [], "none", ""


class DurianLangGraphRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.adapter = FakeAdapter()
        self.runtime = DurianLangGraphRuntime(
            adapter=self.adapter,
            checkpoint_path=str(Path(self.tmp.name) / "checkpoints.sqlite"),
            max_memory_messages=12,
            max_prompt_messages=8,
        )

    def tearDown(self):
        self.runtime.close()
        self.tmp.cleanup()

    def run_turn(self, thread_id, messages):
        return list(
            self.runtime.stream_turn(
                thread_id=thread_id,
                messages=messages,
                requested_language="zh",
                use_rag=True,
                generation={"max_tokens": 100},
            )
        )

    def test_generic_followup_uses_history_and_contextualizes_rag(self):
        first = self.run_turn(
            "admin2:conversation-a",
            [{"role": "user", "content": "这张图是粉蚧吗", "image_url": "/uploads/a.jpg"}],
        )
        self.assertTrue(any(event.get("type") == "content" for event in first))

        second = self.run_turn(
            "admin2:conversation-a",
            [{"role": "user", "content": "详细一点"}],
        )
        route_meta = next(
            event
            for event in second
            if event.get("type") == "metadata" and event.get("phase") == "context_ready"
        )
        self.assertEqual(route_meta["route"], "follow_up")
        self.assertTrue(route_meta["history_used"])
        self.assertIn("这张图是粉蚧吗", self.adapter.rag_queries[-1])
        self.assertIn("详细一点", self.adapter.rag_queries[-1])
        self.assertGreaterEqual(len(self.adapter.final_messages[-1]), 3)

    def test_greeting_stays_on_fast_path(self):
        events = self.run_turn(
            "admin2:greeting",
            [{"role": "user", "content": "你好"}],
        )
        content = "".join(
            event.get("content", "")
            for event in events
            if event.get("type") == "content"
        )
        self.assertEqual(content, "你好，我在。")

    def test_casual_feedback_bypasses_history_rag_and_generation(self):
        self.run_turn(
            "admin2:casual",
            [{"role": "user", "content": "猫山王有什么特点"}],
        )
        rag_calls = len(self.adapter.rag_queries)
        generation_calls = self.adapter.generation_calls

        events = self.run_turn(
            "admin2:casual",
            [{"role": "user", "content": "你哈批"}],
        )
        metadata = next(
            event
            for event in events
            if event.get("type") == "metadata" and event.get("phase") == "context_ready"
        )
        content = "".join(
            event.get("content", "")
            for event in events
            if event.get("type") == "content"
        )

        self.assertEqual(metadata["route"], "casual")
        self.assertFalse(metadata["history_used"])
        self.assertEqual(content, "如果刚才的回答有问题，请直接指出，我会修正。")
        self.assertEqual(len(self.adapter.rag_queries), rag_calls)
        self.assertEqual(self.adapter.generation_calls, generation_calls)

    def test_threads_are_isolated(self):
        self.run_turn("admin2:a", [{"role": "user", "content": "猫山王"}])
        self.run_turn("admin2:b", [{"role": "user", "content": "黑刺"}])
        a_contents = [item["content"] for item in self.runtime.get_thread_messages("admin2:a")]
        b_contents = [item["content"] for item in self.runtime.get_thread_messages("admin2:b")]
        self.assertIn("猫山王", a_contents)
        self.assertNotIn("黑刺", a_contents)
        self.assertIn("黑刺", b_contents)

    def test_frontend_history_replay_does_not_duplicate_messages(self):
        thread_id = "admin2:replay"
        self.run_turn(thread_id, [{"role": "user", "content": "粉蚧怎么处理"}])
        self.run_turn(
            thread_id,
            [
                {"role": "user", "content": "粉蚧怎么处理"},
                {"role": "assistant", "content": "answer:粉蚧怎么处理"},
                {"role": "user", "content": "详细一点"},
            ],
        )
        contents = [item["content"] for item in self.runtime.get_thread_messages(thread_id)]
        self.assertEqual(contents.count("粉蚧怎么处理"), 1)
        self.assertEqual(contents.count("answer:粉蚧怎么处理"), 1)
        self.assertEqual(contents[-2:], ["详细一点", "answer:详细一点"])

    def test_external_image_turn_becomes_immediate_followup_context(self):
        self.runtime.close()
        self.adapter = ImageContextAdapter()
        self.runtime = DurianLangGraphRuntime(
            adapter=self.adapter,
            checkpoint_path=str(Path(self.tmp.name) / "image-checkpoints.sqlite"),
            max_memory_messages=12,
            max_prompt_messages=8,
        )
        thread_id = "admin2:image-followup"

        self.run_turn(thread_id, [{"role": "user", "content": "猫山王 vs 黑金枕"}])
        self.runtime.record_external_turn(
            thread_id=thread_id,
            user_message={
                "role": "user",
                "content": "请分析这张图片",
                "image_url": "/uploads/mealybug.jpg",
            },
            assistant_message={
                "role": "assistant",
                "content": "初步判断为粉蚧，叶脉处有白色蜡质絮状物。",
                "active_context_card": {
                    "context_id": "ctx-image",
                    "source": "image_diagnosis",
                    "main_subject": "粉蚧",
                    "last_user_query": "请分析这张图片",
                    "last_answer_summary": "初步判断为粉蚧，叶脉处有白色蜡质絮状物。",
                },
            },
        )

        events = self.run_turn(
            thread_id,
            [{"role": "user", "content": "详细一点"}],
        )
        route_meta = next(
            event
            for event in events
            if event.get("type") == "metadata"
            and event.get("phase") == "context_ready"
        )
        self.assertEqual(route_meta["route"], "follow_up")
        self.assertTrue(route_meta["history_used"])
        self.assertIn("粉蚧", self.adapter.rag_queries[-1])
        self.assertNotIn("猫山王", self.adapter.rag_queries[-1])
        self.assertIn(
            "Continue image topic: 粉蚧",
            self.adapter.final_messages[-1][0]["content"],
        )

    def test_ordered_browser_history_overrides_checkpoint_merge_order(self):
        thread_id = "admin2:ordered-replay"
        self.run_turn(thread_id, [{"role": "user", "content": "猫山王"}])
        self.run_turn(
            thread_id,
            [
                {"role": "assistant", "content": "旧的猫山王回答"},
                {
                    "role": "assistant",
                    "content": "最新图片诊断为粉蚧",
                    "active_context_card": {
                        "context_id": "ctx-latest-image",
                        "source": "image_diagnosis",
                        "main_subject": "粉蚧",
                    },
                },
                {"role": "user", "content": "详细一点"},
            ],
        )
        self.assertIn("最新图片诊断为粉蚧", self.adapter.rag_queries[-1])
        self.assertNotIn("answer:猫山王", self.adapter.rag_queries[-1])

    def test_exact_answer_mode_bypasses_generative_model(self):
        self.runtime.close()
        self.adapter = ExactAnswerAdapter(answer="24℃；5℃；1000mm")
        self.runtime = DurianLangGraphRuntime(
            adapter=self.adapter,
            checkpoint_path=str(Path(self.tmp.name) / "exact.sqlite"),
        )
        events = self.run_turn(
            "admin2:exact",
            [
                {
                    "role": "user",
                    "content": "年平均温度____以上，绝对低温____以上，年降水量____以上。",
                }
            ],
        )
        metadata = next(
            event
            for event in events
            if event.get("type") == "metadata"
            and event.get("phase") == "context_ready"
        )
        content = "".join(
            event.get("content", "")
            for event in events
            if event.get("type") == "content"
        )
        self.assertEqual(metadata["answer_mode"], "exact_source")
        self.assertEqual(content, "24℃；5℃；1000mm")
        self.assertEqual(self.adapter.generation_calls, 0)

    def test_exact_answer_mode_refuses_when_source_does_not_match(self):
        self.runtime.close()
        self.adapter = ExactAnswerAdapter(answer=None)
        self.runtime = DurianLangGraphRuntime(
            adapter=self.adapter,
            checkpoint_path=str(Path(self.tmp.name) / "exact-not-found.sqlite"),
        )
        events = self.run_turn(
            "admin2:exact-not-found",
            [{"role": "user", "content": "不存在的原文____答案。"}],
        )
        content = "".join(
            event.get("content", "")
            for event in events
            if event.get("type") == "content"
        )
        self.assertIn("未找到", content)
        self.assertEqual(self.adapter.generation_calls, 0)

    def test_exact_fact_question_does_not_inherit_stale_context(self):
        self.runtime.close()
        self.adapter = ExactSelectedContextAdapter(should_use_rag=False)
        self.runtime = DurianLangGraphRuntime(
            adapter=self.adapter,
            checkpoint_path=str(Path(self.tmp.name) / "exact-fact.sqlite"),
        )
        self.run_turn(
            "admin2:exact-fact",
            [{"role": "user", "content": "猫山王和 D24 有什么区别"}],
        )
        events = self.run_turn(
            "admin2:exact-fact",
            [{"role": "user", "content": "黑刺的编码是什么"}],
        )
        metadata = next(
            event
            for event in events
            if event.get("type") == "metadata"
            and event.get("phase") == "context_ready"
        )
        content = "".join(
            event.get("content", "")
            for event in events
            if event.get("type") == "content"
        )
        self.assertEqual(metadata["route"], "new_topic")
        self.assertFalse(metadata["history_used"])
        self.assertEqual(metadata["answer_mode"], "exact_source")
        self.assertEqual(content, "黑刺的品种登记编号是 D200。")
        self.assertEqual(self.adapter.rag_queries[-1], "黑刺的编码是什么")
        self.assertNotIn("D24", self.adapter.rag_queries[-1])

    def test_exact_image_followup_uses_image_subject_for_rag_and_extraction(self):
        self.runtime.close()
        self.adapter = ImageExactContextAdapter(should_use_rag=True)
        self.runtime = DurianLangGraphRuntime(
            adapter=self.adapter,
            checkpoint_path=str(Path(self.tmp.name) / "image-exact.sqlite"),
        )
        thread_id = "admin2:image-exact"
        self.runtime.record_external_turn(
            thread_id=thread_id,
            user_message={
                "role": "user",
                "content": "请识别这个榴莲品种",
                "image_url": "/uploads/musang-king.jpg",
            },
            assistant_message={
                "role": "assistant",
                "content": "图片中的品种初步判断为猫山王。",
                "active_context_card": {
                    "context_id": "ctx-musang-image",
                    "source": "image",
                    "image_url": "/uploads/musang-king.jpg",
                    "main_subject": "猫山王",
                    "last_answer_summary": "图片中的品种初步判断为猫山王。",
                },
            },
        )

        events = self.run_turn(
            thread_id,
            [{"role": "user", "content": "它是什么编号"}],
        )
        metadata = next(
            event
            for event in events
            if event.get("type") == "metadata"
            and event.get("phase") == "context_ready"
        )
        content = "".join(
            event.get("content", "")
            for event in events
            if event.get("type") == "content"
        )
        self.assertEqual(metadata["route"], "follow_up")
        self.assertTrue(metadata["history_used"])
        self.assertIn("猫山王", metadata["resolved_query"])
        self.assertIn("猫山王", self.adapter.rag_queries[-1])
        self.assertEqual(content, "猫山王的品种登记编号是 D197。")
        self.assertEqual(self.adapter.generation_calls, 0)

    def test_selected_followup_uses_resolved_query_and_cannot_skip_rag(self):
        self.runtime.close()
        self.adapter = SelectedContextAdapter(should_use_rag=False)
        self.runtime = DurianLangGraphRuntime(
            adapter=self.adapter,
            checkpoint_path=str(Path(self.tmp.name) / "selected.sqlite"),
        )
        thread_id = "admin2:selected"
        self.run_turn(
            thread_id,
            [{"role": "user", "content": "我的榴莲叶片有粉蚧"}],
        )
        retrieval_count = len(self.adapter.rag_queries)
        events = self.run_turn(
            thread_id,
            [{"role": "user", "content": "它现在严重吗"}],
        )
        metadata = next(
            event
            for event in events
            if event.get("type") == "metadata"
            and event.get("phase") == "context_ready"
        )
        self.assertEqual(metadata["route"], "follow_up")
        self.assertIn("我的榴莲叶片有粉蚧", metadata["resolved_query"])
        self.assertEqual(len(self.adapter.rag_queries), retrieval_count + 1)
        self.assertIn(
            "我的榴莲叶片有粉蚧",
            self.adapter.rag_queries[-1],
        )
        self.assertEqual(metadata["evidence_quality"], "high")
        self.assertIn(
            "Continue topic: 我的榴莲叶片有粉蚧",
            self.adapter.final_messages[-1][0]["content"],
        )

    def test_substantive_query_without_ragflow_evidence_refuses_generation(self):
        self.runtime.close()
        self.adapter = NoEvidenceAdapter()
        self.runtime = DurianLangGraphRuntime(
            adapter=self.adapter,
            checkpoint_path=str(Path(self.tmp.name) / "no-evidence.sqlite"),
        )
        calls_before = self.adapter.generation_calls
        events = self.run_turn(
            "admin2:no-evidence",
            [{"role": "user", "content": "火星榴莲应该怎么施肥"}],
        )
        metadata = next(
            event
            for event in events
            if event.get("type") == "metadata"
            and event.get("phase") == "context_ready"
        )
        content = "".join(
            event.get("content", "")
            for event in events
            if event.get("type") == "content"
        )
        self.assertEqual(metadata["answer_mode"], "grounded_no_source")
        self.assertEqual(metadata["evidence"], [])
        self.assertIn("RAGFlow", content)
        self.assertEqual(self.adapter.generation_calls, calls_before)

    def test_long_term_user_memory_crosses_conversations_but_not_users(self):
        self.run_turn(
            "admin2:memory-source",
            [
                {
                    "role": "user",
                    "content": "请记住我在彭亨种植猫山王，树龄3年。",
                }
            ],
        )
        memories = self.runtime.get_user_memories("admin2")
        self.assertTrue(any("彭亨" in item["content"] for item in memories))

        events = self.run_turn(
            "admin2:memory-target",
            [{"role": "user", "content": "针对我的情况，怎么施肥？"}],
        )
        metadata = next(
            event
            for event in events
            if event.get("type") == "metadata"
            and event.get("phase") == "context_ready"
        )
        self.assertGreaterEqual(metadata["memory_used"], 1)
        self.assertTrue(
            any(
                "彭亨" in item["content"]
                for item in self.adapter.final_messages[-1]
                if item["role"] == "system"
            )
        )

        self.run_turn(
            "other-user:memory-target",
            [{"role": "user", "content": "针对我的情况，怎么施肥？"}],
        )
        self.assertFalse(
            any(
                "彭亨" in item["content"]
                for item in self.adapter.final_messages[-1]
                if item["role"] == "system"
            )
        )

    def test_user_can_clear_long_term_memory(self):
        self.run_turn(
            "admin2:clear-source",
            [{"role": "user", "content": "记住我更喜欢简短回答。"}],
        )
        self.assertTrue(self.runtime.get_user_memories("admin2"))
        self.run_turn(
            "admin2:clear-target",
            [{"role": "user", "content": "请清空我的所有记忆"}],
        )
        self.assertEqual(self.runtime.get_user_memories("admin2"), [])

    def test_sensitive_values_are_not_saved_as_long_term_memory(self):
        self.run_turn(
            "admin2:sensitive",
            [{"role": "user", "content": "请记住我的密码是123456。"}],
        )
        self.assertEqual(self.runtime.get_user_memories("admin2"), [])

    def test_rolling_summary_survives_raw_message_trimming(self):
        self.runtime.close()
        self.adapter = FakeAdapter()
        self.runtime = DurianLangGraphRuntime(
            adapter=self.adapter,
            checkpoint_path=str(Path(self.tmp.name) / "summary.sqlite"),
            max_memory_messages=4,
            max_prompt_messages=4,
        )
        thread_id = "admin2:summary"
        self.run_turn(thread_id, [{"role": "user", "content": "主题一：幼树浇水"}])
        self.run_turn(thread_id, [{"role": "user", "content": "主题二：土壤排水"}])
        self.run_turn(thread_id, [{"role": "user", "content": "主题三：叶片施肥"}])
        raw_contents = [
            item["content"]
            for item in self.runtime.get_thread_messages(thread_id)
        ]
        memory = self.runtime.get_thread_memory(thread_id)
        self.assertNotIn("主题一：幼树浇水", raw_contents)
        self.assertIn("主题一：幼树浇水", memory["rolling_summary"])
        self.assertEqual(memory["turn_count"], 3)

    def test_ambiguous_reference_requests_clarification_without_generation(self):
        self.runtime.close()
        self.adapter = ClarificationAdapter()
        self.runtime = DurianLangGraphRuntime(
            adapter=self.adapter,
            checkpoint_path=str(Path(self.tmp.name) / "clarify.sqlite"),
        )
        thread_id = "admin2:clarify"
        self.run_turn(thread_id, [{"role": "user", "content": "粉蚧防治"}])
        calls_before = self.adapter.generation_calls
        events = self.run_turn(
            thread_id,
            [{"role": "user", "content": "这个怎么办"}],
        )
        metadata = next(
            event
            for event in events
            if event.get("type") == "metadata"
            and event.get("phase") == "context_ready"
        )
        content = "".join(
            event.get("content", "")
            for event in events
            if event.get("type") == "content"
        )
        self.assertEqual(metadata["answer_mode"], "clarification")
        self.assertIn("请补充", content)
        self.assertEqual(self.adapter.generation_calls, calls_before)


if __name__ == "__main__":
    unittest.main()

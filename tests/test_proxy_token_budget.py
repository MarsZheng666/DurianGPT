import unittest

import durian__inference_api as api


class ProxyTokenBudgetTest(unittest.TestCase):
    def make_model(self):
        model = api.DurianGPTModel({})
        model.max_model_len = 6144
        return model

    def test_exact_boundary_reduces_output_instead_of_sending_400(self):
        model = self.make_model()
        model._proxy_tokenize_messages = lambda messages: (5121, 6144)
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
        ]
        fitted, output_tokens = model._fit_proxy_budget(messages, 1024)
        self.assertEqual(fitted, messages)
        self.assertEqual(output_tokens, 959)

    def test_old_history_is_removed_before_latest_exchange(self):
        model = self.make_model()

        def count(messages):
            return (
                sum(len(str(item.get("content") or "")) for item in messages),
                6144,
            )

        model._proxy_tokenize_messages = count
        messages = [{"role": "system", "content": "S" * 2600}]
        for index in range(6):
            messages.append(
                {
                    "role": "assistant" if index % 2 else "user",
                    "content": str(index) * 700,
                }
            )
        messages.append({"role": "user", "content": "LATEST"})
        fitted, output_tokens = model._fit_proxy_budget(messages, 512)
        self.assertEqual(fitted[-1]["content"], "LATEST")
        self.assertLess(len(fitted), len(messages))
        self.assertGreaterEqual(output_tokens, 256)

    def test_400_context_error_parser_returns_safe_retry_budget(self):
        message = (
            "This model's maximum context length is 6144 tokens. "
            "However, you requested 1024 output tokens and your prompt "
            "contains at least 5121 input tokens."
        )
        self.assertEqual(
            api.DurianGPTModel._retry_output_tokens_from_400(message, 1024),
            991,
        )


if __name__ == "__main__":
    unittest.main()

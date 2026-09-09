"""任务 #6 验证：Conversation Gateway（§2：Auth/User/Role/Thread/Language）。

验证点：
1. 角色白名单：合法角色透传，缺省最小权限 worker，非法值拒绝不降级；
2. 语言校验：auto/zh/en/th/ms，非法拒绝；
3. Thread 注册表：缺省自动分配、已有线程复用、元数据可查；
4. 身份透传到图调用参数（anonymous 为开发模式默认，认证属阶段五）。
"""

import unittest

from durian_agent.api.gateway import (
    ConversationGateway,
    GatewayError,
    ThreadRegistry,
    VALID_ROLES,
)


class TestGateway(unittest.TestCase):

    def setUp(self):
        self.gateway = ConversationGateway()

    def test_role_whitelist(self):
        self.assertEqual(VALID_ROLES, ("worker", "manager", "admin"))
        for role in ("worker", "manager", "admin"):
            self.assertEqual(
                self.gateway.resolve_context(role=role)["role"], role)
        # 大小写归一
        self.assertEqual(
            self.gateway.resolve_context(role="Manager")["role"], "manager")

    def test_role_defaults_to_least_privilege(self):
        self.assertEqual(self.gateway.resolve_context()["role"], "worker")

    def test_invalid_role_rejected(self):
        with self.assertRaises(GatewayError):
            self.gateway.resolve_context(role="superadmin")
        with self.assertRaises(GatewayError):
            self.gateway.resolve_context(role="")

    def test_language_validation(self):
        for lang in ("auto", "zh", "en", "th", "ms"):
            self.assertEqual(
                self.gateway.resolve_context(language=lang)["language"], lang)
        self.assertEqual(self.gateway.resolve_context()["language"], "auto")
        with self.assertRaises(GatewayError):
            self.gateway.resolve_context(language="jp")

    def test_user_id_passthrough_and_anonymous(self):
        self.assertEqual(
            self.gateway.resolve_context(user_id="u-123")["user_id"], "u-123")
        self.assertEqual(self.gateway.resolve_context()["user_id"], "anonymous")


class TestThreadRegistry(unittest.TestCase):

    def test_auto_assign_thread_id(self):
        registry = ThreadRegistry()
        tid = registry.resolve(None)
        self.assertTrue(tid.startswith("thread-"))
        self.assertEqual(len(registry), 1)

    def test_existing_thread_reused(self):
        registry = ThreadRegistry()
        self.assertEqual(registry.resolve("t-1"), "t-1")
        self.assertEqual(registry.resolve("t-1"), "t-1")
        self.assertEqual(len(registry), 1)     # 不重复注册

    def test_thread_metadata_lookup(self):
        registry = ThreadRegistry()
        registry.resolve("t-1")
        self.assertIn("created", registry.get("t-1"))
        self.assertIsNone(registry.get("不存在的线程"))


if __name__ == "__main__":
    unittest.main()

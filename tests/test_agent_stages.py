"""任务 #14 验证：生育阶段识别（架构文档 §8，10 个阶段）。

验证点：
1. 阶段集与 §8 完全一致；
2. 每个阶段 ≥2 条多语言样例正确识别；
3. 最长优先（花芽分化≠开花、post-harvest≠harvest）；
4. 无阶段信息回落 unknown，不猜；
5. normalize_growth_stage 非法值回落。
"""

import unittest

from durian_agent.semantic.stages import (
    GROWTH_STAGES,
    UNKNOWN_STAGE,
    detect_growth_stage,
    normalize_growth_stage,
)


class TestGrowthStages(unittest.TestCase):

    def test_stage_set_matches_doc_section8(self):
        expected = [
            "seedling", "vegetative", "pre_flowering", "flowering", "fruit_set",
            "fruit_development", "pre_harvest", "harvest", "post_harvest", "unknown",
        ]
        self.assertEqual(list(GROWTH_STAGES), expected)
        self.assertEqual(len(GROWTH_STAGES), 10)

    def test_each_stage_detected(self):
        cases = {
            "seedling": ["苗期怎么管理", "seedling care tips", "ดูแลต้นกล้ายังไง"],
            "vegetative": ["抽梢期要施肥吗", "vegetative stage pruning"],
            "pre_flowering": ["花芽分化期注意什么", "pre-flowering fertilizer plan"],
            "flowering": ["开花期可以浇水吗", "anthesis stage irrigation", "ช่วงออกดอกให้น้ำได้ไหม"],
            "fruit_set": ["坐果后怎么保果", "fruit set stage management"],
            "fruit_development": ["膨果期施什么肥", "fruit development nutrients"],
            "pre_harvest": ["采前要停水吗", "pre-harvest irrigation stop"],
            "harvest": ["采收期怎么判断成熟", "harvest timing by sound", "เก็บเกี่ยวเมื่อไหร่"],
            "post_harvest": ["采后处理流程", "post-harvest ripening temperature"],
        }
        for stage, queries in cases.items():
            for query in queries:
                self.assertEqual(
                    detect_growth_stage(query), stage,
                    msg=f"阶段识别错误: {query!r} 期望 {stage}",
                )

    def test_longest_term_wins(self):
        self.assertEqual(detect_growth_stage("花芽分化期用药"), "pre_flowering")
        self.assertEqual(detect_growth_stage("post-harvest handling"), "post_harvest")
        self.assertEqual(detect_growth_stage("开花期管理"), "flowering")

    def test_no_stage_returns_unknown(self):
        for query in ["什么是榴莲", "帮我创建工单", "how much is a durian", ""]:
            self.assertEqual(detect_growth_stage(query), UNKNOWN_STAGE)

    def test_normalize_growth_stage(self):
        self.assertEqual(normalize_growth_stage("flowering"), "flowering")
        self.assertEqual(normalize_growth_stage("  flowering  "), "flowering")
        self.assertEqual(normalize_growth_stage("blooming"), UNKNOWN_STAGE)
        self.assertEqual(normalize_growth_stage(""), UNKNOWN_STAGE)
        self.assertEqual(normalize_growth_stage(None), UNKNOWN_STAGE)


if __name__ == "__main__":
    unittest.main()

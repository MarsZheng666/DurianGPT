"""语义理解模块（架构文档 §5-§10）。

- labels    Intent / 生育阶段 / 植株部位标签集（§7/§8）
- entities  实体归一化 alias→canonical（§9，任务 #12）
- stages    生育阶段识别（§8，任务 #14）
- risks     risk_features 风险标志（§6，任务 #19）
- parse     SemanticParse：LLM + 规则合成 Canonical Schema（§6，任务 #3）
"""

from durian_agent.semantic.entities import detect_entities

__all__ = ["detect_entities"]

"""生成质量指标（架构文档 §54，任务 #58）。

五项指标（高风险场景重点：Faithfulness 与 Abstention Accuracy）：

    Answer Correctness    答案与金标语义一致（LLM 评审）
    Faithfulness          答案是否有证据支撑、无编造（LLM 评审，重点）
    Citation Accuracy     引用与提供的证据一一对应（确定性）
    Evidence Coverage     金标要点被证据覆盖比例（确定性）
    Abstention Accuracy   无证据场景下正确拒答的比例（确定性，重点）

确定性部分离线可测；LLM 评审部分 judge 注入（FakeLLM 可测契约）。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from durian_agent.llm import LLMProvider

JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for a durian plantation QA system.

Given a question, reference answer, evidence, and the system's answer,
score ONE aspect from 0 to 1 (one decimal).

Respond with the score only, no explanation."""


def _parse_score(text: str) -> Optional[float]:
    match = re.search(r"\b(0(?:\.\d)?|1(?:\.0)?)\b", text or "")
    return float(match.group(1)) if match else None


def llm_judge(llm: LLMProvider, *, aspect: str, question: str,
              reference: str, evidence: str, answer: str) -> Optional[float]:
    """LLM 评审单维打分（Answer Correctness / Faithfulness）。"""
    user = (f"Aspect: {aspect}\nQuestion: {question}\n"
            f"Reference: {reference}\nEvidence: {evidence}\n"
            f"Answer: {answer}")
    try:
        return _parse_score(llm.complete(JUDGE_SYSTEM_PROMPT, user))
    except Exception:
        return None


def citation_accuracy(answer: str, cited_ids: Sequence[str],
                      provided_ids: Sequence[str]) -> float:
    """引用精度：被引用的 chunk 中确实提供的比例（非法引用=编造）。"""
    if not cited_ids:
        return 0.0
    provided = set(provided_ids)
    valid = [c for c in cited_ids if c in provided]
    return len(valid) / len(cited_ids)


def evidence_coverage(gold_points: Sequence[str],
                      evidence_texts: Sequence[str]) -> float:
    """金标要点覆盖：金标关键词组在证据文本中的命中比例。"""
    if not gold_points:
        return 0.0
    joined = "\n".join(evidence_texts)
    hits = sum(1 for point in gold_points
               if all(kw in joined for kw in str(point).split()))
    return hits / len(gold_points)


ABSTAIN_MARKERS = ("暂无足够", "没有提供足够", "无法回答", "证据不足",
                   "not provide enough evidence", "insufficient evidence")


def is_abstention(answer: str) -> bool:
    return any(marker in (answer or "") for marker in ABSTAIN_MARKERS)


def abstention_accuracy(cases: Sequence[Dict[str, Any]]) -> float:
    """拒答准确率：无证据场景应拒答、有证据场景不应拒答，双向正确率。

    cases: [{"has_evidence": bool, "answer": str}]
    """
    if not cases:
        return 0.0
    correct = 0
    for case in cases:
        should_abstain = not case.get("has_evidence")
        if should_abstain == is_abstention(case.get("answer", "")):
            correct += 1
    return correct / len(cases)


def evaluate_generation(
    llm: Optional[LLMProvider],
    cases: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    """cases: [{"question", "reference", "evidence_texts", "provided_ids",
                 "cited_ids", "gold_points", "answer"}] → 五指标汇总。"""
    correctness_scores: List[float] = []
    faithfulness_scores: List[float] = []
    citation_scores: List[float] = []
    coverage_scores: List[float] = []

    for case in cases:
        answer = case.get("answer", "")
        evidence = "\n".join(case.get("evidence_texts") or [])
        if llm is not None:
            correctness = llm_judge(
                llm, aspect="answer_correctness",
                question=case.get("question", ""),
                reference=case.get("reference", ""),
                evidence=evidence, answer=answer)
            faithfulness = llm_judge(
                llm, aspect="faithfulness",
                question=case.get("question", ""),
                reference=case.get("reference", ""),
                evidence=evidence, answer=answer)
            if correctness is not None:
                correctness_scores.append(correctness)
            if faithfulness is not None:
                faithfulness_scores.append(faithfulness)
        citation_scores.append(citation_accuracy(
            answer, case.get("cited_ids") or [],
            case.get("provided_ids") or []))
        coverage_scores.append(evidence_coverage(
            case.get("gold_points") or [],
            case.get("evidence_texts") or []))

    def _avg(values: List[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    return {
        "answer_correctness": _avg(correctness_scores),
        "faithfulness": _avg(faithfulness_scores),
        "citation_accuracy": _avg(citation_scores),
        "evidence_coverage": _avg(coverage_scores),
    }

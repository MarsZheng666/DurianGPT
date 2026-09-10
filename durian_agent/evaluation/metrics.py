"""RAG 离线指标（架构文档 §53，任务 #61）。

    Recall@K / Hit@1 / Hit@3 / MRR / NDCG —— zh/en/th/ms 分别统计。

- ranked_ids：检索系统的排序 chunk_id 列表（按相关性降序）；
- gold_ids：金标相关 chunk 集合；
- evaluate_dataset：{query, language, gold_ids} 数据集 × search_fn →
  分语种报告 + 总表（§53「分别统计」的交付形态）。
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Sequence

SearchFn = Callable[[str], Sequence[str]]


def hit_at_k(ranked_ids: Sequence[str], gold_ids: Sequence[str],
             k: int) -> bool:
    return bool(set(ranked_ids[:k]) & set(gold_ids))


def recall_at_k(ranked_ids: Sequence[str], gold_ids: Sequence[str],
                k: int) -> float:
    gold = set(gold_ids)
    if not gold:
        return 0.0
    return len(set(ranked_ids[:k]) & gold) / len(gold)


def mrr(ranked_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    gold = set(gold_ids)
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in gold:
            return 1.0 / rank
    return 0.0


def ndcg(ranked_ids: Sequence[str], gold_ids: Sequence[str],
         k: int = 10) -> float:
    """二元相关性 NDCG：金标命中=1。"""
    gold = set(gold_ids)
    if not gold:
        return 0.0
    dcg = sum((1.0 / math.log2(rank + 1))
              for rank, chunk_id in enumerate(ranked_ids[:k], start=1)
              if chunk_id in gold)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(rank + 1)
               for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_query(ranked_ids: Sequence[str], gold_ids: Sequence[str],
                   k: int = 10) -> Dict[str, float]:
    return {
        "hit@1": 1.0 if hit_at_k(ranked_ids, gold_ids, 1) else 0.0,
        "hit@3": 1.0 if hit_at_k(ranked_ids, gold_ids, 3) else 0.0,
        f"recall@{k}": recall_at_k(ranked_ids, gold_ids, k),
        "mrr": mrr(ranked_ids, gold_ids),
        "ndcg": ndcg(ranked_ids, gold_ids, k),
    }


def evaluate_dataset(
    dataset: List[Dict[str, Any]],
    search_fn: SearchFn,
    k: int = 10,
) -> Dict[str, Dict[str, float]]:
    """数据集评估：四语种分别统计 + overall（§53）。

    dataset 条目：{"query", "language", "gold_ids"}。
    """
    by_language: Dict[str, List[Dict[str, float]]] = {}
    for entry in dataset:
        ranked = list(search_fn(entry["query"]))
        metrics = evaluate_query(ranked, entry.get("gold_ids") or [], k)
        by_language.setdefault(entry.get("language", "unknown"),
                               []).append(metrics)

    def _average(rows: List[Dict[str, float]]) -> Dict[str, float]:
        if not rows:
            return {}
        keys = rows[0].keys()
        return {key: round(sum(r[key] for r in rows) / len(rows), 4)
                for key in keys}

    report = {lang: _average(rows) for lang, rows in sorted(by_language.items())}
    report["overall"] = _average(
        [m for rows in by_language.values() for m in rows])
    report["__count__"] = {lang: len(rows)                     # type: ignore[dict-item]
                           for lang, rows in by_language.items()}
    return report

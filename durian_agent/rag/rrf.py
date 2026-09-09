"""Weighted RRF（架构文档 §26，任务 #27）。

公式：score(d) = Σ wi / (k + rank_i(d))

- rank 1-based，某路未召回的文档该路贡献 0；
- 初始权重（§26）：Original Dense 0.20 / Canonical Dense 0.30 /
  Original BM25 0.15 / Expanded BM25 0.35，可配置，
  最终以离线评估集调优（任务 #59/#61）；
- 融合键是 chunk_id：四路召回（#23）共用同一套 chunk_id，
  从结构上避免「改写文本导致同一 node 在 dense/bm25 两侧裂成两条、
  各自掉出 Top-K」的历史坑（见 rag_llamaindex 的 RRF 去重教训）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

#: §26 初始权重（路名 → 权重）
DEFAULT_RRF_WEIGHTS: Dict[str, float] = {
    "dense_original": 0.20,
    "dense_canonical": 0.30,
    "bm25_original": 0.15,
    "bm25_expanded": 0.35,
}

DEFAULT_K = 60


def weighted_rrf(
    rankings: Dict[str, Sequence[Any]],
    *,
    weights: Optional[Dict[str, float]] = None,
    k: int = DEFAULT_K,
    top_n: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """多路排名融合。

    rankings: {路名: [chunk_id 按相关性降序]}（#23 四路召回的输出）。
    返回按融合分降序的 [{"chunk_id", "score", "rrf_rank", "sources"}]，
    sources 记录该文档被哪些路召回及各路名次（§52 Trace 需要）。
    """
    weights = weights or DEFAULT_RRF_WEIGHTS
    scores: Dict[Any, float] = {}
    sources: Dict[Any, Dict[str, int]] = {}

    for route, ranking in rankings.items():
        weight = float(weights.get(route, 0.0))
        if weight <= 0 or not ranking:
            continue
        for position, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + position)
            sources.setdefault(chunk_id, {})[route] = position

    fused = sorted(scores.items(), key=lambda kv: -kv[1])
    if top_n is not None:
        fused = fused[:top_n]

    return [
        {
            "chunk_id": chunk_id,
            "score": score,
            "rrf_rank": rank,
            "sources": sources[chunk_id],
        }
        for rank, (chunk_id, score) in enumerate(fused, start=1)
    ]

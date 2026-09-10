"""四路召回（架构文档 §25，任务 #23）。

    Query → 语义归一（#8/#2）→ 术语扩展（§9/§23）
        ├─ Original Dense     向量检索 × 原查询
        ├─ Canonical Dense    向量检索 × 规范化查询（实体 → 标准名）
        ├─ Original BM25      词法检索 × 原查询
        └─ Expanded BM25      词法检索 × 多语扩展查询（标准名 → 四语别名）
        （→ Weighted RRF，#27）

设计要点：
- 不把所有问题统一翻译成中文（§23）：跨语言桥接靠**实体级**别名扩展，
  只扩展已确定性识别的实体，不引入整句翻译的噪声与语义漂移；
- 泰文查询的 dense 塌缩（跨语言向量鸿沟，实测四问余弦相似度恒为 1）
  由 Canonical/Expanded 两路兜底——这正是四路设计的动机；
- canonical 查询 = 原查询 + 实体标准名（追加而非改写，保留原词信息）；
- expanded 查询 = 原查询 + 实体的全部四语别名。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

from durian_agent.glossary import load_glossary
from durian_agent.normalize import normalize_input
from durian_agent.semantic.entities import detect_entities

#: 四路名（与 #27 DEFAULT_RRF_WEIGHTS 的键一致）
ROUTES = ("dense_original", "dense_canonical", "bm25_original", "bm25_expanded")


@lru_cache(maxsize=1)
def _canonical_aliases() -> Dict[str, List[str]]:
    """标准名 → 全部别名（四语，含标准名自身）。"""
    mapping: Dict[str, List[str]] = {}
    for alias, canonical in load_glossary().items():
        aliases = mapping.setdefault(canonical, [])
        if alias != canonical:
            aliases.append(alias)
    return mapping


def build_queries(query: str, semantic: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """原查询 → {original, canonical, expanded}（§23 跨语言扩展）。

    §23 的三输入在此汇齐：
    - Original Query：归一化后的原查询；
    - Canonical Schema：可选注入（#3 的输出）——已有 entities 时直接复用，
      避免图内二次检测；
    - Terminology Dictionary：标准名 → 四语别名（_canonical_aliases）。
    """
    original = normalize_input(query)
    if semantic and isinstance(semantic.get("entities"), dict):
        entities = {k: v for k, v in semantic["entities"].items() if v}
    else:
        entities = detect_entities(original)
    canonical_names = [name for name in entities.values() if name]

    canonical = original
    expanded_terms: List[str] = []
    for name in canonical_names:
        if name not in canonical:
            canonical = f"{canonical} {name}".strip()
        expanded_terms.append(name)
        expanded_terms.extend(_canonical_aliases().get(name, []))

    expanded = " ".join([original] + _dedupe(expanded_terms))
    return {"original": original, "canonical": canonical, "expanded": expanded}


def _dedupe(items: List[str]) -> List[str]:
    seen, result = set(), []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


#: dense 路相似度下限：Milvus 会返回相似度≈0 的垃圾命中（无阈值语义），
#: 直接送进 RRF 会按名次给分污染融合——低于此值丢弃。
DENSE_MIN_SCORE = 0.3


class FourWayRetriever:
    """四路召回执行器：输出各路 chunk_id 排名 + 完整命中（供 RRF 与 Trace）。"""

    def __init__(self, vector_index, bm25_index,
                 dense_min_score: float = DENSE_MIN_SCORE):
        self.vector_index = vector_index
        self.bm25_index = bm25_index
        self.dense_min_score = dense_min_score

    def recall(
        self,
        query: str,
        *,
        top_k_each: int = 10,
        expr: Optional[str] = None,
    ) -> Dict[str, Any]:
        """返回 {queries, rankings, hits}。

        rankings: {路名: [chunk_id...]}，直接喂给 #27 weighted_rrf；
        hits: {路名: [{chunk_id, score, text, record}...]}，保留完整
        命中信息（Trace §52 与 Evidence Check §29 需要）。
        expr: Milvus 过滤表达式，只作用于两路 dense（§40 数据权限）。
        """
        queries = build_queries(query)
        rankings: Dict[str, List[str]] = {}
        hits: Dict[str, List[Dict[str, Any]]] = {}

        for route, q in (("dense_original", queries["original"]),
                         ("dense_canonical", queries["canonical"])):
            found = self.vector_index.search(q, top_k=top_k_each, expr=expr) if q else []
            # 垃圾命中过滤：相似度低于阈值的丢弃，不进 RRF
            found = [h for h in found if h.get("score", 0.0) >= self.dense_min_score]
            hits[route] = found
            rankings[route] = [h["chunk_id"] for h in found]

        for route, q in (("bm25_original", queries["original"]),
                         ("bm25_expanded", queries["expanded"])):
            found = self.bm25_index.search(q, top_k=top_k_each) if q else []
            hits[route] = found
            rankings[route] = [h["chunk_id"] for h in found]

        return {"queries": queries, "rankings": rankings, "hits": hits}

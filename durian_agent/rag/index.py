"""Milvus + BM25 双索引（架构文档 §18/§58，任务 #25）。

- VectorIndex：milvus-lite 本地嵌入式 Milvus（无服务器依赖），
  COSINE + FLAT，元数据字段齐全（language/domain/source_type/
  role_scope/orchard_scope），支持 §40 的 array_contains 数据权限过滤；
- BM25Index：纯 Python BM25（k1=1.5, b=0.75），多语分词复用
  #2 泰语词典优先分词与既有实测的 zh 2-3 gram / 拉丁词策略；
- 嵌入函数依赖注入：默认本地 bge-small-zh-v1.5（models/ 目录，离线），
  测试注入确定性假嵌入器——绝不调用远端嵌入 API。
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from durian_agent.normalize import normalize_input
from durian_agent.thai import tokenize_thai

EmbedFn = Callable[[Sequence[str]], List[List[float]]]

_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "bge-small-zh-v1.5"
_INDEX_DIR = Path(__file__).resolve().parent.parent.parent / "rag_build" / "durian_agent_index"

_LATIN_RE = re.compile(r"[a-z][a-z0-9_.-]+")
_ZH_RE = re.compile(r"[\u4e00-\u9fff]+")
_THAI_RE = re.compile(r"[\u0e00-\u0e7f]+")

_STOPWORDS = {"the", "a", "an", "of", "for", "and", "or", "is", "are", "to",
              "in", "on", "how", "what", "do", "i", "my", "me"}


def multilingual_tokens(text: str) -> List[str]:
    """多语分词（对齐 rag_llamaindex._lexical_terms 的实测策略）。

    拉丁词（小写化）、中文 2-3 gram、泰文词典优先 token + 3-4 gram。
    """
    value = normalize_input(text, fold_case=True)
    tokens: List[str] = []
    for word in _LATIN_RE.findall(value):
        if word not in _STOPWORDS:
            tokens.append(word)
    for span in _ZH_RE.findall(value):
        if 2 <= len(span) <= 12:
            tokens.append(span)
        for size in (2, 3):
            for i in range(len(span) - size + 1):
                tokens.append(span[i:i + size])
    if _THAI_RE.search(value):
        tokens.extend(tokenize_thai(value))
    seen, result = set(), []
    for token in tokens:
        if token and token not in seen:
            seen.add(token)
            result.append(token)
    return result[:120]


# ══════════════════ 嵌入（依赖注入）══════════════════


class EmbeddingConfigError(RuntimeError):
    pass


def local_bge_embedder() -> EmbedFn:
    """本地 bge-small-zh-v1.5 嵌入（离线模型目录，懒加载）。"""
    if not _MODEL_PATH.exists():
        raise EmbeddingConfigError(f"本地嵌入模型不存在: {_MODEL_PATH}")
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    model = HuggingFaceEmbedding(model_name=str(_MODEL_PATH))

    def embed(texts: Sequence[str]) -> List[List[float]]:
        return [list(map(float, v)) for v in
                model.get_text_embedding_batch(list(texts))]

    return embed


# ══════════════════ BM25（纯 Python）══════════════════


class BM25Index:
    """内存 BM25。语料规模（万级 chunk）下毫秒级，无需外置服务。"""

    def __init__(self, records: Sequence[Dict[str, Any]]):
        self.chunk_ids: List[str] = []
        self.texts: List[str] = []
        doc_tokens: List[List[str]] = []
        for record in records:
            self.chunk_ids.append(str(record["chunk_id"]))
            self.texts.append(str(record["text"]))
            doc_tokens.append(multilingual_tokens(str(record["text"])))
        self.doc_count = max(len(doc_tokens), 1)
        self.doc_len = [len(t) for t in doc_tokens]
        self.avg_len = (sum(self.doc_len) / self.doc_count) if doc_tokens else 1.0
        self.term_docs: Dict[str, int] = {}
        self.term_freqs: List[Dict[str, int]] = []
        for tokens in doc_tokens:
            freq: Dict[str, int] = {}
            for token in tokens:
                freq[token] = freq.get(token, 0) + 1
            self.term_freqs.append(freq)
            for term in freq:
                self.term_docs[term] = self.term_docs.get(term, 0) + 1
        self.k1, self.b = 1.5, 0.75
        self._records = list(records)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        query_terms = multilingual_tokens(query)
        scores: List[float] = []
        for index, freq in enumerate(self.term_freqs):
            score = 0.0
            for term in query_terms:
                tf = freq.get(term, 0)
                if not tf:
                    continue
                df = self.term_docs.get(term, 0)
                idf = math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))
                norm = 1.0 - self.b + self.b * (self.doc_len[index] / self.avg_len)
                score += idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm)
            scores.append(score)
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [
            {"chunk_id": self.chunk_ids[i], "score": scores[i], "text": self.texts[i],
             "record": self._records[i]}
            for i in ranked[:top_k] if scores[i] > 0.0
        ]


# ══════════════════ Milvus 向量索引 ══════════════════


class VectorIndex:
    """milvus-lite 本地向量索引（§18/§58：vector index + metadata + partition）。

    partition 语义由 filter 表达式承担（role_scope/orchard_scope 的
    array_contains 联合过滤，§40），暂不建物理 partition——
    单租户语料下 filter 即可，多租户时再切 partition key。
    """

    COLLECTION = "durian_chunks"
    TEXT_MAX = 4096

    def __init__(self, db_path: Union[str, Path], embed_fn: EmbedFn, dim: int = 512):
        from pymilvus import MilvusClient

        self.db_path = str(db_path)
        self.embed_fn = embed_fn
        self.dim = dim
        self.client = MilvusClient(self.db_path)

    def build(self, records: Sequence[Dict[str, Any]]) -> int:
        """重建集合并写入全量记录（含向量）。返回写入条数。"""
        from pymilvus import DataType

        if self.client.has_collection(self.COLLECTION):
            self.client.drop_collection(self.COLLECTION)
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field("tenant_id", DataType.VARCHAR, max_length=64)
        schema.add_field("text", DataType.VARCHAR, max_length=self.TEXT_MAX)
        schema.add_field("document_id", DataType.VARCHAR, max_length=512)
        schema.add_field("language", DataType.VARCHAR, max_length=8)
        schema.add_field("domain", DataType.VARCHAR, max_length=64)
        schema.add_field("source_type", DataType.VARCHAR, max_length=32)
        schema.add_field("role_scope", DataType.ARRAY,
                         element_type=DataType.VARCHAR, max_capacity=8, max_length=32)
        schema.add_field("orchard_scope", DataType.ARRAY,
                         element_type=DataType.VARCHAR, max_capacity=16, max_length=32)
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")
        self.client.create_collection(self.COLLECTION, schema=schema, index_params=index_params)

        rows: List[Dict[str, Any]] = []
        texts = [str(r["text"])[: self.TEXT_MAX - 1] for r in records]
        vectors = self.embed_fn(texts)
        for record, vector, text in zip(records, vectors, texts):
            from durian_agent.rag.permissions import normalize_orchard_scope

            rows.append({
                "chunk_id": str(record["chunk_id"])[:63],
                "vector": vector,
                "tenant_id": str(record.get("tenant_id", "default"))[:63],
                "text": text,
                "document_id": str(record.get("document_id", ""))[:511],
                "language": str(record.get("language", "zh"))[:7],
                "domain": str(record.get("domain", "general"))[:63],
                "source_type": str(record.get("source_type", "document"))[:31],
                "role_scope": list(record.get("role_scope") or [])[:8],
                "orchard_scope": normalize_orchard_scope(
                    record.get("orchard_scope"))[:16],
            })
        for start in range(0, len(rows), 256):
            self.client.insert(self.COLLECTION, rows[start:start + 256])
        return len(rows)

    def search(
        self,
        query: str,
        top_k: int = 10,
        expr: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """向量检索；expr 为 Milvus 过滤表达式（如 array_contains(role_scope, "worker")）。"""
        vector = self.embed_fn([query])[0]
        results = self.client.search(
            self.COLLECTION,
            data=[vector],
            limit=top_k,
            filter=expr,
            output_fields=["text", "document_id", "language", "domain",
                           "source_type", "role_scope", "orchard_scope",
                           "tenant_id"],
        )
        hits: List[Dict[str, Any]] = []
        for hit in results[0]:
            entity = hit.get("entity", {})
            # milvus-lite 的 Hit 以主键字段名暴露 id（非通用 "id" 键）
            chunk_id = hit.get("chunk_id") or entity.get("chunk_id")
            hits.append({
                "chunk_id": chunk_id,
                "score": float(hit.get("distance", 0.0)),
                "text": entity.get("text", ""),
                "record": {
                    "chunk_id": chunk_id,
                    "tenant_id": entity.get("tenant_id", "default"),
                    "text": entity.get("text", ""),
                    "document_id": entity.get("document_id", ""),
                    "language": entity.get("language", ""),
                    "domain": entity.get("domain", ""),
                    "source_type": entity.get("source_type", ""),
                    "role_scope": entity.get("role_scope", []),
                    "orchard_scope": entity.get("orchard_scope", []),
                },
            })
        return hits


def default_index_dir() -> Path:
    return _INDEX_DIR

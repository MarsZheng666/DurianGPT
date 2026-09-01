from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "rag_pdfs"
STORAGE_DIR = BASE_DIR / "rag_llamaindex_storage"

EXTRA_CHUNK_FILES = [
    BASE_DIR / "rag_auto" / "chunks_cache.json",
    BASE_DIR / "rag_auto" / "auto_chunks.jsonl",
    BASE_DIR / "clean_chunks.jsonl",
    BASE_DIR / "chunks.jsonl",
]

# 先用 CPU，避免和 vLLM 抢 5090 显存。
# 支持环境变量覆盖（本地部署时指向 models/bge-small-zh-v1.5）
EMBED_MODEL_NAME = os.getenv(
    "DURIAN_LI_EMBED_MODEL_PATH",
    str(BASE_DIR / "models" / "bge-small-zh-v1.5"),
)

# LlamaIndex 检索分数一般是相似度分数，不同模型范围会有差异。
# 先保守一点，太低就不返回 evidence。
DEFAULT_MIN_SCORE = 0.42

_INDEX_CACHE = None
_LEXICAL_NODE_CACHE: List[Dict[str, Any]] | None = None
_NEIGHBOR_CACHE: Dict[tuple, List[Dict[str, Any]]] | None = None



GENERIC_STOPWORDS = {
    "榴莲", "榴莲树", "请问", "哪些", "什么", "怎么", "如何", "是否", "有没有",
    "介绍", "说明", "一下", "相关", "资料", "内容", "的是", "有哪", "可以",
    "the", "a", "an", "of", "and", "or", "to", "in", "for", "about", "durian",
}


INTENT_RULES = {
    "code": {
        "triggers": ["编号", "代号", "代码", "品种号", "品种编号"],
        "required": [
            "编号", "代号", "代码", "品种号", "品种编号",
            "D24", "D197", "D200", "D101",
            "猫山王", "苏丹王", "黑刺", "金枕", "青尼", "干尧", "托曼尼",
        ],
    },
    "disease": {
        "triggers": ["病害", "病虫害", "炭疽病", "疫病", "根腐病", "叶斑", "防治", "症状", "病原"],
        "required": [
            "病害", "病虫害", "炭疽病", "疫病", "根腐病", "叶斑",
            "防治", "症状", "病原", "病斑", "菌", "药剂",
            "Colletotrichum", "Phytophthora",
        ],
    },
    "variety": {
        "triggers": ["品种", "品系", "种类", "猫山王", "金枕", "黑刺", "苏丹王"],
        "required": [
            "品种", "品系", "种类", "猫山王", "金枕", "黑刺", "苏丹王",
            "D24", "D197", "D200",
        ],
    },
    "planting": {
        "triggers": ["种植", "栽培", "定植", "施肥", "修剪", "灌溉", "管理"],
        "required": ["种植", "栽培", "定植", "施肥", "修剪", "灌溉", "管理", "土壤", "温度", "水分"],
    },
}


def setup_llamaindex() -> None:
    """Configure local embedding model."""
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        device="cpu",
    )

    # 这里只用 LlamaIndex 做检索，不让它调用默认 LLM。
    Settings.llm = None


def normalize_ocr_units(text: str) -> str:
    """Normalize common MinerU/LaTeX unit artifacts before retrieval."""
    value = str(text or "")
    value = re.sub(
        r"(\d+(?:\.\d+)?)\s*\\+mathrm\{mm\}",
        r"\1 mm",
        value,
        flags=re.I,
    )
    value = re.sub(
        r"(\d+(?:\.\d+)?)\s*\\+mathrm\{cm\}",
        r"\1 cm",
        value,
        flags=re.I,
    )
    value = re.sub(
        r"(\d+(?:\.\d+)?)\s*\\+sim\s*(\d+(?:\.\d+)?)"
        r"\s*(?:\^\{\\+circ\})?\s*\\+mathrm\{C\}",
        r"\1–\2℃",
        value,
        flags=re.I,
    )
    value = re.sub(
        r"(\d+(?:\.\d+)?)\s*(?:\^\{\\+circ\})?\s*\\+mathrm\{C\}",
        r"\1℃",
        value,
        flags=re.I,
    )
    value = value.replace("\\geqslant", "≥").replace("\\leqslant", "≤")

    # MinerU occasionally emits temperature degree symbols as ``\%``.
    # Convert only when the surrounding text clearly describes temperature;
    # keep real humidity/percentage values unchanged.
    bad_degree = re.compile(r"(\d+(?:\.\d+)?)\s*\\+%")

    def replace_bad_degree(match: re.Match) -> str:
        start, end = match.span()
        context = value[max(0, start - 28) : min(len(value), end + 16)]
        temperature_markers = (
            "温度",
            "低温",
            "高温",
            "积温",
            "摄氏",
            "temperature",
        )
        if any(marker in context.lower() for marker in temperature_markers):
            return f"{match.group(1)}℃"
        return match.group(0)

    return bad_degree.sub(replace_bad_degree, value)


def clean_text(text: str) -> str:
    text = str(text or "")

    # 清理常见 OCR / parser 泄漏
    text = re.sub(r"\{?\s*['\"]bbox['\"]\s*:\s*\[[^\]]*\]\s*,?", " ", text)
    text = re.sub(r"['\"](?:page_idx|page_size|image_path|angle|index|type)['\"]\s*:\s*[^,}\]]+", " ", text)
    text = re.sub(r"['\"](?:content|text|type)['\"]\s*:\s*", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = normalize_ocr_units(text)

    # 统一空白
    text = re.sub(r"\s+", " ", text).strip()
    return text


FILL_BLANK_RE = re.compile(
    r"(?:_{2,}|＿{2,}|（\s*）|\(\s*\)|\[\s*\]|【\s*】|<\s*空\s*>)"
)


def is_exact_answer_query(query: str) -> bool:
    value = str(query or "")
    return bool(
        FILL_BLANK_RE.search(value)
        or ("填空" in value and any(mark in value for mark in ("___", "（", "(")))
    )


def is_noise_text(text: str) -> bool:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    lower = compact.lower()

    if len(compact) < 20:
        return True

    parser_markers = [
        "'bbox'", '"bbox"', "bbox:",
        "'spans'", '"spans"',
        "'lines'", '"lines"',
        "'para_blocks'", '"para_blocks"',
        "'discarded_blocks'", '"discarded_blocks"',
        "'page_idx'", '"page_idx"',
        "'image_path'", '"image_path"',
        "'content'", '"content"',
    ]
    if sum(1 for m in parser_markers if m in lower) >= 3:
        return True

    front_matter = [
        "本书由", "主编", "副研究员", "助理研究员", "负责图书框架",
        "主要负责", "资料整理", "责任编辑", "出版",
    ]
    if any(m in compact for m in front_matter):
        return True

    return False


def load_pdf_documents() -> List[Document]:
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    docs: List[Document] = []
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    for pdf_path in pdf_files:
        try:
            pdf = fitz.open(str(pdf_path))
        except Exception as exc:
            print(f"[WARN] cannot open PDF: {pdf_path.name}: {exc}")
            continue

        for page_idx, page in enumerate(pdf, start=1):
            try:
                text = page.get_text("text")
            except Exception as exc:
                print(f"[WARN] cannot read page: {pdf_path.name} p{page_idx}: {exc}")
                continue

            text = clean_text(text)

            if is_noise_text(text):
                continue

            docs.append(
                Document(
                    text=text,
                    metadata={
                        "source_file": pdf_path.name,
                        "doc": pdf_path.name,
                        "page": page_idx,
                    },
                )
            )

        pdf.close()

    return docs



def load_extra_chunk_documents() -> List[Document]:
    """Load existing RAG JSON / JSONL chunks into LlamaIndex."""
    docs: List[Document] = []

    def iter_objects(path: Path):
        suffix = path.suffix.lower()

        if suffix == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"[WARN] cannot read JSON chunks: {path}: {exc}")
                return

            if isinstance(data, list):
                for obj in data:
                    if isinstance(obj, dict):
                        yield obj
                return

            if isinstance(data, dict):
                for key in ["chunks", "data", "items", "documents"]:
                    val = data.get(key)
                    if isinstance(val, list):
                        for obj in val:
                            if isinstance(obj, dict):
                                yield obj
                        return
            return

        # JSONL
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        yield obj
        except Exception as exc:
            print(f"[WARN] cannot read JSONL chunks: {path}: {exc}")
            return

    for path in EXTRA_CHUNK_FILES:
        if not path.exists():
            continue

        print(f"[INFO] loading extra chunks: {path}")
        before = len(docs)

        for obj in iter_objects(path):
            text = (
                obj.get("display_text")
                or obj.get("text")
                or obj.get("index_text")
                or obj.get("original_text")
                or obj.get("content")
                or obj.get("evidence")
                or ""
            )

            text = clean_text(text)

            if is_noise_text(text):
                continue

            docs.append(
                Document(
                    text=text,
                    metadata={
                        "source_file": obj.get("source_file") or path.name,
                        "doc": obj.get("doc") or obj.get("source_file") or path.name,
                        "page": obj.get("page") or obj.get("page_idx"),
                        "item_index": obj.get("item_index"),
                        "chunk_index": obj.get("chunk_index"),
                        "text_sha1": obj.get("text_sha1"),
                        "source_path": str(path),
                    },
                )
            )

        print(f"[INFO] loaded from {path.name}: {len(docs) - before}")

    return docs



def rebuild_index() -> None:
    setup_llamaindex()

    pdf_docs = load_pdf_documents()
    extra_docs = load_extra_chunk_documents()
    docs = pdf_docs + extra_docs

    if not docs:
        raise RuntimeError(f"No valid documents found in {PDF_DIR} or EXTRA_CHUNK_FILES")

    print(f"[INFO] loaded PDF pages: {len(pdf_docs)}")
    print(f"[INFO] loaded extra chunks: {len(extra_docs)}")
    print(f"[INFO] loaded total documents: {len(docs)}")

    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=80,
    )

    index = VectorStoreIndex.from_documents(
        docs,
        transformations=[splitter],
        show_progress=True,
    )

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))

    print(f"[INFO] LlamaIndex RAG index persisted to: {STORAGE_DIR}")


def load_index() -> VectorStoreIndex:
    """Load LlamaIndex index once and reuse it in backend."""
    global _INDEX_CACHE

    if _INDEX_CACHE is not None:
        return _INDEX_CACHE

    setup_llamaindex()

    if not STORAGE_DIR.exists():
        raise RuntimeError(f"Index storage not found: {STORAGE_DIR}. Run rebuild first.")

    storage_context = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
    _INDEX_CACHE = load_index_from_storage(storage_context)
    return _INDEX_CACHE


def extract_query_terms(query: str) -> List[str]:
    q = str(query or "").strip()
    if not q:
        return []

    terms = re.findall(r"[\u4e00-\u9fff]{2,8}|[A-Za-z][A-Za-z0-9_-]{1,}|D\s*-?\s*\d{1,4}", q)

    result = []
    seen = set()

    for term in terms:
        t = re.sub(r"\s+", "", term).strip()
        if not t:
            continue
        if t in GENERIC_STOPWORDS or t.lower() in GENERIC_STOPWORDS:
            continue
        key = t.lower()
        if key not in seen:
            seen.add(key)
            result.append(t)

    return result


def detect_intents(query: str) -> List[str]:
    q = str(query or "")
    intents = []

    for name, rule in INTENT_RULES.items():
        if any(t in q for t in rule["triggers"]):
            intents.append(name)

    return intents


def evidence_matches_query(query: str, text: str, score: float, min_score: float = DEFAULT_MIN_SCORE) -> bool:
    text = str(text or "")
    query = str(query or "")

    if not text or is_noise_text(text):
        return False

    # 对精确原文/长句查询：只要相似度够高，直接保留。
    # 例如：“叶片病斑多从叶尖或叶缘开始，个别从叶内发生”
    if len(query) >= 18 and score >= 0.55:
        return True

    intents = detect_intents(query)
    terms = extract_query_terms(query)

    # 编号/代号类问题必须严格，避免“3～50朵”这种数字误命中。
    if "code" in intents:
        required = INTENT_RULES["code"]["required"]
        return score >= 0.55 and any(word in text for word in required)

    # 病害类问题：只要证据里有病害/症状/防治相关词，并且分数不是太低，就保留。
    if "disease" in intents:
        disease_required = INTENT_RULES["disease"]["required"]
        if score >= 0.45 and any(word in text for word in disease_required):
            return True
        return False

    # 品种类问题：要求有品种相关词。
    if "variety" in intents:
        variety_required = INTENT_RULES["variety"]["required"]
        if score >= 0.45 and any(word in text for word in variety_required):
            return True
        return False

    # 种植/栽培类问题：要求有管理相关词。
    if "planting" in intents:
        planting_required = INTENT_RULES["planting"]["required"]
        if score >= 0.45 and any(word in text for word in planting_required):
            return True
        return False

    # 没有明确意图的泛问题，不强行返回 RAG evidence。
    if not terms:
        return False

    matched = sum(1 for t in terms if t in text)
    coverage = matched / max(len(terms), 1)

    # 多关键词问题，至少覆盖一部分，并且分数过线。
    if len(terms) >= 2:
        return score >= 0.45 and coverage >= 0.34

    # 单关键词太宽，默认不返回，避免“榴莲”到处乱命中。
    return False


def _normalized_dedupe_key(text: str) -> str:
    return re.sub(r"[\W_]+", "", str(text or "").lower(), flags=re.UNICODE)[:900]


def _exact_anchor_coverage(query: str, text: str) -> float:
    if not is_exact_answer_query(query):
        return 0.0
    body = str(query or "")
    if "：" in body:
        prefix, suffix = body.split("：", 1)
        if any(mark in prefix for mark in ("填空", "原文", "知识库", "只填")):
            body = suffix
    elif ":" in body:
        prefix, suffix = body.split(":", 1)
        if any(mark in prefix for mark in ("填空", "原文", "知识库", "只填")):
            body = suffix
    anchors = [
        _normalized_dedupe_key(segment)
        for segment in FILL_BLANK_RE.split(body)
    ]
    anchors = [anchor for anchor in anchors if len(anchor) >= 2]
    if not anchors:
        return 0.0
    haystack = _normalized_dedupe_key(text)
    matched = sum(1 for anchor in anchors if anchor in haystack)
    return matched / len(anchors)


def _lexical_terms(text: str) -> List[str]:
    """Build compact Chinese/Latin terms suitable for an in-memory BM25 scan."""
    value = clean_text(text)
    value = FILL_BLANK_RE.sub(" ", value)
    value = re.sub(
        r"^(?:请|请严格)?(?:根据|依据)?(?:知识库|参考资料|原文)?"
        r"(?:进行)?(?:填空|回答)?(?:，|,|：|:|\s)*(?:只填写答案[：:]?)?",
        " ",
        value,
    )

    terms: List[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_.-]{1,}|\d+(?:\.\d+)?", value):
        lowered = token.lower()
        if lowered not in GENERIC_STOPWORDS:
            terms.append(lowered)

    for span in re.findall(r"[\u4e00-\u9fff]+", value):
        if 2 <= len(span) <= 12 and span not in GENERIC_STOPWORDS:
            terms.append(span)
        for size in (2, 3):
            if len(span) < size:
                continue
            for index in range(0, len(span) - size + 1):
                gram = span[index : index + size]
                if gram not in GENERIC_STOPWORDS:
                    terms.append(gram)

    seen = set()
    result = []
    for term in terms:
        if term and term not in seen:
            seen.add(term)
            result.append(term)
    return result[:80]


def _build_lexical_node_cache(index: VectorStoreIndex) -> List[Dict[str, Any]]:
    global _LEXICAL_NODE_CACHE, _NEIGHBOR_CACHE
    if _LEXICAL_NODE_CACHE is not None:
        return _LEXICAL_NODE_CACHE

    nodes: List[Dict[str, Any]] = []
    neighbors: Dict[tuple, List[Dict[str, Any]]] = {}
    for node_id, node in index.docstore.docs.items():
        try:
            text = clean_text(node.get_content())
        except Exception:
            continue
        if not text or is_noise_text(text):
            continue
        metadata = dict(getattr(node, "metadata", {}) or {})
        item = {
            "node_id": str(node_id),
            "text": text,
            "metadata": metadata,
            "length": max(1, len(re.sub(r"\s+", "", text))),
        }
        nodes.append(item)

        doc = str(
            metadata.get("doc")
            or metadata.get("source_file")
            or "unknown"
        )
        if metadata.get("item_index") is not None:
            try:
                sequence = ("item", doc, int(metadata["item_index"]))
                neighbors.setdefault(sequence, []).append(item)
            except (TypeError, ValueError):
                pass
        elif metadata.get("page") is not None:
            try:
                sequence = ("page", doc, int(metadata["page"]))
                neighbors.setdefault(sequence, []).append(item)
            except (TypeError, ValueError):
                pass

    _LEXICAL_NODE_CACHE = nodes
    _NEIGHBOR_CACHE = neighbors
    return nodes


def _bm25_retrieve(
    index: VectorStoreIndex,
    query: str,
    limit: int,
) -> List[Dict[str, Any]]:
    nodes = _build_lexical_node_cache(index)
    terms = _lexical_terms(query)
    if not nodes or not terms:
        return []

    document_frequency: Counter = Counter()
    matched_rows = []
    for node in nodes:
        text_lower = node["text"].lower()
        frequencies = {}
        for term in terms:
            count = text_lower.count(term)
            if count:
                frequencies[term] = count
                document_frequency[term] += 1
        if frequencies:
            matched_rows.append((node, frequencies))

    if not matched_rows:
        return []

    total_docs = len(nodes)
    average_length = sum(node["length"] for node in nodes) / max(total_docs, 1)
    k1 = 1.5
    b = 0.75
    scored = []
    for node, frequencies in matched_rows:
        score = 0.0
        for term, tf in frequencies.items():
            df = document_frequency[term]
            inverse_frequency = math.log(
                1.0 + (total_docs - df + 0.5) / (df + 0.5)
            )
            denominator = tf + k1 * (
                1.0 - b + b * node["length"] / max(average_length, 1.0)
            )
            score += inverse_frequency * (tf * (k1 + 1.0)) / denominator
        if score > 0:
            scored.append(
                {
                    **node,
                    "bm25_score": float(score),
                }
            )

    scored.sort(key=lambda item: item["bm25_score"], reverse=True)
    return scored[: max(1, int(limit))]


def _dense_retrieve(
    index: VectorStoreIndex,
    query: str,
    limit: int,
) -> List[Dict[str, Any]]:
    retriever = index.as_retriever(similarity_top_k=max(12, int(limit)))
    output = []
    for node in retriever.retrieve(query):
        text = clean_text(node.node.get_content())
        if not text or is_noise_text(text):
            continue
        output.append(
            {
                "node_id": str(getattr(node.node, "node_id", "") or ""),
                "text": text,
                "metadata": dict(node.node.metadata or {}),
                "dense_score": float(node.score or 0.0),
            }
        )
    return output


def _expand_adjacent_text(item: Dict[str, Any], max_chars: int = 2600) -> str:
    if not _NEIGHBOR_CACHE:
        return str(item.get("text") or "")
    metadata = dict(item.get("metadata") or {})
    doc = str(
        metadata.get("doc")
        or metadata.get("source_file")
        or "unknown"
    )
    sequence_type = ""
    sequence_value = None
    if metadata.get("item_index") is not None:
        sequence_type = "item"
        sequence_value = metadata.get("item_index")
    elif metadata.get("page") is not None:
        sequence_type = "page"
        sequence_value = metadata.get("page")
    try:
        sequence_value = int(sequence_value)
    except (TypeError, ValueError):
        return str(item.get("text") or "")

    pieces = []
    seen = set()
    for offset in (-1, 0, 1):
        key = (sequence_type, doc, sequence_value + offset)
        for neighbor in _NEIGHBOR_CACHE.get(key, []):
            text = str(neighbor.get("text") or "").strip()
            dedupe = _normalized_dedupe_key(text)
            if text and dedupe and dedupe not in seen:
                seen.add(dedupe)
                pieces.append(text)
    combined = " ".join(pieces).strip()
    return combined[:max_chars] if combined else str(item.get("text") or "")


def _hybrid_retrieve(
    query: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    index = load_index()
    exact_mode = is_exact_answer_query(query)
    candidate_limit = max(36, int(top_k) * 8)
    dense = _dense_retrieve(index, query, candidate_limit)
    lexical = _bm25_retrieve(index, query, candidate_limit)

    merged: Dict[str, Dict[str, Any]] = {}
    rrf_constant = 60.0

    def merge_candidates(items, source, weight):
        for rank, item in enumerate(items, start=1):
            key = _normalized_dedupe_key(item.get("text") or "")
            if not key:
                continue
            target = merged.setdefault(
                key,
                {
                    "text": item.get("text") or "",
                    "metadata": dict(item.get("metadata") or {}),
                    "dense_score": 0.0,
                    "bm25_score": 0.0,
                    "hybrid_score": 0.0,
                },
            )
            target["hybrid_score"] += weight / (rrf_constant + rank)
            if source == "dense":
                target["dense_score"] = max(
                    target["dense_score"],
                    float(item.get("dense_score") or 0.0),
                )
            else:
                target["bm25_score"] = max(
                    target["bm25_score"],
                    float(item.get("bm25_score") or 0.0),
                )

    merge_candidates(dense, "dense", 1.0)
    merge_candidates(lexical, "bm25", 1.35 if exact_mode else 1.0)

    ranked = sorted(
        merged.values(),
        key=lambda item: (
            float(item.get("hybrid_score") or 0.0),
            float(item.get("bm25_score") or 0.0),
            float(item.get("dense_score") or 0.0),
        ),
        reverse=True,
    )

    results = []
    seen_locations = set()
    seen_expanded = set()
    for item in ranked:
        dense_score = float(item.get("dense_score") or 0.0)
        bm25_score = float(item.get("bm25_score") or 0.0)
        text = str(item.get("text") or "")
        if not exact_mode and not (
            evidence_matches_query(query, text, dense_score)
            or bm25_score > 1.2
        ):
            continue
        metadata = dict(item.get("metadata") or {})
        if exact_mode:
            text = _expand_adjacent_text(item)
            if _exact_anchor_coverage(query, text) < 0.6:
                continue

            doc = str(
                metadata.get("doc")
                or metadata.get("source_file")
                or "unknown"
            )
            sequence_value = metadata.get("item_index")
            sequence_type = "item"
            if sequence_value is None:
                sequence_value = metadata.get("page")
                sequence_type = "page"
            try:
                numeric_value = int(sequence_value)
            except (TypeError, ValueError):
                numeric_value = None
            if numeric_value is not None:
                nearby_locations = {
                    (sequence_type, doc, numeric_value + offset)
                    for offset in (-1, 0, 1)
                }
                if seen_locations.intersection(nearby_locations):
                    continue
                seen_locations.add((sequence_type, doc, numeric_value))

            expanded_key = _normalized_dedupe_key(text)
            if expanded_key in seen_expanded:
                continue
            seen_expanded.add(expanded_key)
        results.append(
            {
                "source_file": (
                    metadata.get("source_file")
                    or metadata.get("doc")
                    or "unknown"
                ),
                "doc": (
                    metadata.get("doc")
                    or metadata.get("source_file")
                    or "unknown"
                ),
                "page": metadata.get("page"),
                "score": round(
                    dense_score
                    if dense_score
                    else float(item.get("hybrid_score") or 0.0),
                    4,
                ),
                "dense_score": round(dense_score, 4),
                "bm25_score": round(bm25_score, 4),
                "hybrid_score": round(
                    float(item.get("hybrid_score") or 0.0),
                    6,
                ),
                "text": text,
                "display_text": text,
                "metadata": metadata,
                "neighbor_expanded": bool(exact_mode),
            }
        )
        if len(results) >= top_k:
            break
    return results



def retrieve(query: str, top_k: int = 5, min_score: float = DEFAULT_MIN_SCORE) -> List[Dict[str, Any]]:
    return _hybrid_retrieve(query, top_k=max(1, int(top_k)))



def raw_retrieve_debug(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Return raw LlamaIndex retrieval results without conservative filtering."""
    index = load_index()
    retriever = index.as_retriever(similarity_top_k=top_k)
    raw_nodes = retriever.retrieve(query)

    results = []
    for i, node in enumerate(raw_nodes, start=1):
        text = clean_text(node.node.get_content())
        meta = dict(node.node.metadata or {})
        results.append({
            "rank": i,
            "score": round(float(node.score or 0.0), 4),
            "doc": meta.get("doc") or meta.get("source_file") or "unknown",
            "page": meta.get("page"),
            "text_preview": text[:500],
        })
    return results

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python rag_llamaindex.py rebuild")
        print("  python rag_llamaindex.py query '问题'")
        raise SystemExit(1)

    cmd = sys.argv[1]

    if cmd == "rebuild":
        rebuild_index()
        return

    if cmd == "query":
        if len(sys.argv) < 3:
            raise SystemExit("missing query")
        query = sys.argv[2]
        results = retrieve(query)
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    if cmd == "raw":
        if len(sys.argv) < 3:
            raise SystemExit("missing query")
        query = sys.argv[2]
        results = raw_retrieve_debug(query)
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    raise SystemExit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()


def retrieve_for_backend(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Return evidence objects compatible with DurianGPT frontend/backend."""
    items = retrieve(query, top_k=top_k)

    evidence = []
    for i, item in enumerate(items):
        text = item.get("text") or item.get("display_text") or ""
        doc = item.get("doc") or item.get("source_file") or "unknown"

        evidence.append({
            "id": f"llamaindex_{i}_{abs(hash((doc, text[:80])))}",
            "source_file": item.get("source_file") or doc,
            "doc": doc,
            "page": item.get("page"),
            "item_index": item.get("metadata", {}).get("item_index"),
            "chunk_index": item.get("metadata", {}).get("chunk_index"),
            "score": item.get("score", 0.0),
            "dense_score": item.get("dense_score", item.get("score", 0.0)),
            "lexical_score": item.get("bm25_score", 0.0),
            "bm25_score": item.get("bm25_score", 0.0),
            "hybrid_score": item.get("hybrid_score", 0.0),
            "neighbor_expanded": bool(item.get("neighbor_expanded")),
            "text": text,
            "index_text": text,
            "original_text": text,
            "display_text": text,
            "canonical_language": "zh",
            "text_language": "zh",
            "display_language": "zh",
        })

    return evidence

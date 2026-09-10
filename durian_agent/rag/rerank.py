"""Cross Encoder Reranker（架构文档 §27，任务 #32）。

RRF 解决「找得到」，Reranker 解决「排得准」：
    四路召回 → RRF Top30-50 → Cross Encoder Rerank → Top3-5

- 默认本地 models/bge-reranker-v2-m3（离线，实测相关/无关判别 0.80/0.00）；
- 懒加载：首次 rerank 才加载模型（加载约 8s，服务进程内只发生一次）；
- 长文本截断 512 字符（对齐既有 DURIAN_RERANK_MAX_PASSAGE_CHARS 经验值：
  语料 chunk ≤2400 字符，超出部分对相关性判别无增益且拖慢推理）；
- rerank_fn 依赖注入：测试用确定性替身，不加载模型。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "bge-reranker-v2-m3"

#: RerankFn 契约：(query, texts) -> 与 texts 等长的相关性分数列表（越高越相关）
RerankFn = Callable[[str, Sequence[str]], List[float]]

#: 长文本截断（实测经验：超出部分无判别增益）
MAX_PASSAGE_CHARS = 512

#: §28 阈值：相关性低于该值的文档不进入最终上下文。
#: 默认 0.3 基于本地模型实测判别分布（相关 ≈0.8，无关 ≈0.0）；
#: 最终阈值须以离线评估集调优（任务 #60/#61，阶段五），此处可配置。
RERANK_THRESHOLD = 0.3


def local_bge_reranker(
    model_path: str | Path = _MODEL_PATH,
    device: str = "cpu",
) -> RerankFn:
    """本地 bge-reranker-v2-m3（sigmoid 分数，0~1）。"""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"本地重排模型不存在: {model_path}")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    def rerank(query: str, texts: Sequence[str]) -> List[float]:
        if not texts:
            return []
        pairs = [(query, str(t)[:MAX_PASSAGE_CHARS]) for t in texts]
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True,
                               return_tensors="pt", max_length=512)
            logits = model(**inputs).logits.view(-1)
            return [float(s) for s in torch.sigmoid(logits)]

    return rerank


def rerank_documents(
    query: str,
    docs: Sequence[Dict[str, Any]],
    rerank_fn: RerankFn,
    *,
    top_k: int = 5,
    min_score: float = RERANK_THRESHOLD,
) -> List[Dict[str, Any]]:
    """按 Cross Encoder 分数重排、阈值过滤（§28）、截取 Top-K。

    §27/§28：RRF Top30-50 → rerank → 低于 min_score 的不进最终上下文 →
    Top3-5。返回新列表（不修改入参）：按 rerank 分数降序，附 rerank_score。
    """
    if not docs:
        return []
    texts = [str(d.get("text", "")) for d in docs]
    scores = rerank_fn(query, texts)
    ranked = sorted(
        (dict(d, rerank_score=float(s)) for d, s in zip(docs, scores)),
        key=lambda d: -d["rerank_score"],
    )
    kept = [d for d in ranked if d["rerank_score"] >= min_score]
    return kept[:top_k]

"""临时探针（th-normalize）：泰文 known-item search 评测集。

为什么需要它：139 条泰文金标的 origin 全部是 faq、ref_source_file 全为 None，
证据 100% 落在 146 个泰文 FAQ node 上。所以「把 651 条 legacy 泰文正文的泰文原文
拼进 index_text」这个改动的收益，在现有金标上（主口径和知识库口径†）结构性地
测不出来。这个探针用已知项检索（known-item search）纯机械地构造一个能度量的集合。

构造方式（无 LLM、无翻译、可复现）：
  对每条 legacy 泰文 chunk 的 metadata.original_text（500 字符泰文原文）
    查询 = 前半部分（[0:250]）里泰文实词最密集的 60 字符连续片段
    锚点 = 后半部分（[250:]）经 rag_eval.extract_anchors() 提取的 24 字符窗口
  查询取自前半、锚点取自后半，物理不重叠，不会退化成平凡自匹配。

口径对齐：锚点匹配复用 rag_eval.norm_for_match() + rag_eval.hit_rank()。
因为这是新建的独立评测集、没有历史包袱，查询侧与锚点侧都显式调用
normalize_thai()，保证两侧同规范（rag_eval.py 本身不改）。

用法：
    ./.venv/bin/python _probe_th_knownitem_thnormalize.py build          # 只构集，不检索
    ./.venv/bin/python _probe_th_knownitem_thnormalize.py run --tag before
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rag_eval as E
import rag_llamaindex as R

BASE_DIR = Path(__file__).resolve().parent
CHUNKS = BASE_DIR / "rag_build" / "chunks.jsonl"
OUT_DIR = BASE_DIR / "rag_eval"


def set_path(side: str) -> Path:
    return OUT_DIR / f"_probe_thnormalize_ki_{side}_set.jsonl"


QUERY_LEN = 60
THAI_CHAR = re.compile(r"[\u0e00-\u0e7f]")
# 泰文辅音+元音（排除声调/符号类组合字符），用来度量「实词密度」
THAI_WORD = re.compile(r"[\u0e01-\u0e2e\u0e30-\u0e39\u0e40-\u0e44]")
CJK_WORD = re.compile(r"[\u4e00-\u9fff]")


def load_legacy_thai() -> List[Dict[str, Any]]:
    """取 block_type=='text' 且 src_lang=='th' 且有 original_text 的 chunk。"""
    rows: List[Dict[str, Any]] = []
    for line in CHUNKS.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        c = json.loads(line)
        meta = c.get("metadata") or {}
        if c.get("block_type") != "text":
            continue
        if meta.get("src_lang") != "th":
            continue
        original = meta.get("original_text")
        if not original or not THAI_CHAR.search(str(original)):
            continue
        rows.append(c)
    return rows


def pick_query(first_half: str, side: str = "th") -> str:
    """在前半部分里选实词最密集的 QUERY_LEN 字符窗口。

    避开纯数字/编号段：打分 = 实词字符数 - 2 * 数字字符数。
    """
    text = first_half
    if len(text) <= QUERY_LEN:
        return text.strip()
    word_re = THAI_WORD if side == "th" else CJK_WORD
    best_score = None
    best = text[:QUERY_LEN]
    for start in range(0, len(text) - QUERY_LEN + 1):
        window = text[start : start + QUERY_LEN]
        score = len(word_re.findall(window)) - 2 * sum(
            1 for ch in window if ch.isdigit()
        )
        if best_score is None or score > best_score:
            best_score = score
            best = window
    return best.strip()


def build(side: str = "th") -> List[Dict[str, Any]]:
    """side='th' 用泰文原文构集；side='zh' 用中文译文构集。

    side='zh' 是**中文回归检查**：同样这 651 个 node，拼接泰文原文会不会
    损害它们的中文可检索性。用同一套 known-item 方法度量，before/after 可比。
    """
    rows = load_legacy_thai()
    items: List[Dict[str, Any]] = []
    skipped = {"no_anchor": 0, "query_too_short": 0, "overlap": 0}

    for c in rows:
        meta = c.get("metadata") or {}
        if side == "th":
            source = R.normalize_thai(str(meta["original_text"]))
            min_word = 20
        else:
            source = str(c.get("index_text") or "")
            min_word = 0
        if len(source) < 80:
            skipped["query_too_short"] += 1
            continue

        mid = len(source) // 2
        first_half, second_half = source[:mid], source[mid:]

        query = pick_query(first_half, side)
        if side == "th" and len(THAI_WORD.findall(query)) < min_word:
            skipped["query_too_short"] += 1
            continue

        anchors = E.extract_anchors(second_half, window=24, max_anchors=8)
        if not anchors:
            skipped["no_anchor"] += 1
            continue

        # 硬保证不重叠：剔除任何与查询归一化形式有交集的锚点
        q_norm = E.norm_for_match(query)
        anchors = [a for a in anchors if a not in q_norm and q_norm not in a]
        if not anchors:
            skipped["overlap"] += 1
            continue

        items.append(
            {
                "kid": c.get("id"),
                "doc": c.get("doc"),
                "page": c.get("page"),
                "text_sha1": meta.get("text_sha1"),
                "query": query,
                "anchors": anchors,
                "n_anchors": len(anchors),
                "source_len": len(source),
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = set_path(side)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"[build side={side}] legacy 泰文 chunk: {len(rows)}")
    print(f"[build side={side}] 成功构造 known-item: {len(items)}  跳过: {skipped}")
    if items:
        qlen = [len(i["query"]) for i in items]
        alen = [i["n_anchors"] for i in items]
        print(
            f"[build side={side}] 查询长度 median {statistics.median(qlen)} "
            f"min {min(qlen)} max {max(qlen)}"
        )
        print(
            f"[build side={side}] 每题锚点数 median {statistics.median(alen)} "
            f"min {min(alen)} max {max(alen)}"
        )
        print(f"[build side={side}] 已写入 {path}")
    return items


def load_set(side: str = "th") -> List[Dict[str, Any]]:
    path = set_path(side)
    if not path.exists():
        return build(side)
    return [
        json.loads(line) for line in path.open(encoding="utf-8") if line.strip()
    ]


def run(
    tag: str,
    side: str = "th",
    top_k: int = 5,
    limit: int | None = None,
    storage: str | None = None,
) -> None:
    if storage:
        # 指向索引备份目录，用来在不动现网索引的前提下拿 before 基线。
        # 必须清 _INDEX_CACHE（不是 _INDEX），否则会静默复用已加载的索引。
        R.STORAGE_DIR = Path(storage)
        R._INDEX_CACHE = None
        R._LEXICAL_NODE_CACHE = None
        R._NEIGHBOR_CACHE = None
        print(f"[run] 使用索引目录: {storage}")

    items = load_set(side)
    if limit:
        items = items[:limit]

    rows: List[Dict[str, Any]] = []
    for i, it in enumerate(items, 1):
        try:
            evidences = R.retrieve(it["query"], top_k=top_k)
        except Exception as exc:
            print(f"  [WARN] kid={it['kid']} 检索异常: {exc}")
            evidences = []

        # 两侧同规范：证据侧也过 normalize_thai
        texts = [R.normalize_thai(str(e.get("text") or "")) for e in evidences]
        rank = E.hit_rank(it["anchors"], texts)

        # 是否召回到「正确的那一条」（用 text_sha1 / doc+page 精确核对）
        exact = None
        for pos, e in enumerate(evidences, 1):
            meta = e.get("metadata") or {}
            if it.get("text_sha1") and meta.get("text_sha1") == it["text_sha1"]:
                exact = pos
                break

        rows.append(
            {
                "kid": it["kid"],
                "doc": it["doc"],
                "hit_rank": rank,
                "exact_rank": exact,
                "n_returned": len(evidences),
                "top_docs": [str(e.get("doc") or "") for e in evidences],
            }
        )
        if i % 100 == 0:
            print(f"  ...{i}/{len(items)}")

    n = len(rows)

    def at(k: int, key: str = "hit_rank") -> float:
        return sum(1 for r in rows if r[key] and r[key] <= k) / max(n, 1)

    def mrr(key: str = "hit_rank") -> float:
        return sum(1.0 / r[key] for r in rows if r[key]) / max(n, 1)

    report = {
        "tag": tag,
        "side": side,
        "n": n,
        "top_k": top_k,
        "anchor_hit@1": at(1),
        "anchor_hit@3": at(3),
        "anchor_hit@5": at(5),
        "anchor_mrr": mrr(),
        "exact_hit@1": at(1, "exact_rank"),
        "exact_hit@3": at(3, "exact_rank"),
        "exact_hit@5": at(5, "exact_rank"),
        "exact_mrr": mrr("exact_rank"),
        "empty_rate": sum(1 for r in rows if r["n_returned"] == 0) / max(n, 1),
        "rows": rows,
    }
    out = OUT_DIR / f"_probe_thnormalize_ki_{side}_{tag}.json"
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    print(f"\n{'='*66}\nknown-item 检索  side={side}  tag={tag}  n={n}\n{'='*66}")
    print("锚点口径（召回到含该内容的任意 chunk）")
    for k in (1, 3, 5):
        print(f"  Hit@{k}: {report[f'anchor_hit@{k}']:7.1%}")
    print(f"  MRR   : {report['anchor_mrr']:7.4f}")
    print("精确口径（召回到 text_sha1 完全一致的那一条）")
    for k in (1, 3, 5):
        print(f"  Hit@{k}: {report[f'exact_hit@{k}']:7.1%}")
    print(f"  MRR   : {report['exact_mrr']:7.4f}")
    print(f"空结果率: {report['empty_rate']:.1%}")
    print(f"\n已写入 {out}")


def compare(before: str, after: str, side: str = "th") -> None:
    b = json.loads(
        (OUT_DIR / f"_probe_thnormalize_ki_{side}_{before}.json").read_text("utf-8")
    )
    a = json.loads(
        (OUT_DIR / f"_probe_thnormalize_ki_{side}_{after}.json").read_text("utf-8")
    )
    print(f"\n{'='*74}\nknown-item 对比 side={side}: {before} → {after}\n{'='*74}")
    print(f"{'指标':16s} {'before':>10s} {'after':>10s} {'变化':>12s}")
    for key in (
        "anchor_hit@1", "anchor_hit@3", "anchor_hit@5", "anchor_mrr",
        "exact_hit@1", "exact_hit@3", "exact_hit@5", "exact_mrr",
        "empty_rate",
    ):
        bv, av = b[key], a[key]
        if "mrr" in key:
            print(f"{key:16s} {bv:10.4f} {av:10.4f} {av-bv:+12.4f}")
        else:
            print(f"{key:16s} {bv:9.1%} {av:9.1%} {(av-bv)*100:+11.1f}pp")

    B = {r["kid"]: r for r in b["rows"]}
    A = {r["kid"]: r for r in a["rows"]}
    gained = [k for k in A if not B.get(k, {}).get("hit_rank") and A[k]["hit_rank"]]
    lost = [k for k in A if B.get(k, {}).get("hit_rank") and not A[k]["hit_rank"]]
    print(f"\n新召回成功: {len(gained)} 条    退化: {len(lost)} 条")
    if lost:
        print(f"  退化 kid（前 10）: {lost[:10]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run", "compare"])
    ap.add_argument("--side", choices=["th", "zh"], default="th")
    ap.add_argument("--tag", default="before")
    ap.add_argument("--before")
    ap.add_argument("--after")
    ap.add_argument("--storage", help="索引目录（用备份跑 before 基线时指定）")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    if args.cmd == "build":
        build(args.side)
    elif args.cmd == "run":
        run(args.tag, side=args.side, top_k=args.top_k,
            limit=args.limit, storage=args.storage)
    else:
        compare(args.before, args.after, side=args.side)


if __name__ == "__main__":
    main()

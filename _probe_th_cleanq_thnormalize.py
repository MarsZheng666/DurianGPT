"""临时探针（th-normalize）：规范化规则对「干净查询 vs 损坏语料」的价值测定。

为什么需要它：known-item 集两侧同源（查询与语料携带同样损坏），双侧规范化
不改变匹配关系，所以那个聚合结构性测不出规则价值（2×2 已证明边际=0.0pp）。
规则的真实价值场景是「用户打干净泰文 vs 语料存损坏形态」——本探针测这一格。

设计：
  从 651 条 legacy 泰文原文构造查询集：clean = normalize_thai(original)，
  只保留「查询窗口（前半 60 字符）内含至少一个被规则修复的位点」的 chunk，
  保证测试对规则敏感。锚点取后半 24 字符窗口（clean 形态）。
  A 臂：当前代码（运行时规范化修复语料侧）
  B 臂：monkeypatch normalize_thai = identity（语料侧保持损坏形态）
  预期 A 命中、B 显著劣化 —— 差值即规则对真实用户的贡献。

用法：
    ./.venv/bin/python _probe_th_cleanq_thnormalize.py build
    ./.venv/bin/python _probe_th_cleanq_thnormalize.py run --arm a --tag <tag>
    ./.venv/bin/python _probe_th_cleanq_thnormalize.py run --arm b --tag <tag>
    ./.venv/bin/python _probe_th_cleanq_thnormalize.py compare --before <a> --after <b>
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
SET_PATH = OUT_DIR / "_probe_thnormalize_cleanq_set.jsonl"

QUERY_LEN = 60
THAI_WORD = re.compile(r"[\u0e01-\u0e2e\u0e30-\u0e39\u0e40-\u0e44]")


def load_rows() -> List[Dict[str, Any]]:
    rows = []
    for line in CHUNKS.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        c = json.loads(line)
        m = c.get("metadata") or {}
        if c.get("block_type") == "text" and m.get("src_lang") == "th" and m.get("original_text"):
            rows.append(c)
    return rows


def pick_query(text: str) -> str:
    if len(text) <= QUERY_LEN:
        return text.strip()
    best_score = None
    best = text[:QUERY_LEN]
    for start in range(0, len(text) - QUERY_LEN + 1):
        w = text[start : start + QUERY_LEN]
        score = len(THAI_WORD.findall(w)) - 2 * sum(1 for ch in w if ch.isdigit())
        if best_score is None or score > best_score:
            best_score = score
            best = w
    return best.strip()


def build() -> None:
    items = []
    for c in load_rows():
        m = c.get("metadata") or {}
        original = str(m["original_text"])
        clean = R.normalize_thai(original)
        if clean == original:
            continue  # 无可修复位点，对本测试无信息量
        mid = len(clean) // 2
        query = pick_query(clean[:mid])
        # 窗口必须真的含被修复的位点，否则测不出规则价值
        raw_window = original[
            max(0, len(original) - len(clean[:mid])) : len(original)
        ]  # 近似对位；直接判 query 是否含 clean 形态且原 query 区间含 raw 形态
        changed_in_window = any(
            R.normalize_thai(original[i : i + 8]) != original[i : i + 8]
            for i in range(0, max(1, mid - QUERY_LEN), 4)
        )
        # 更直接：query 里是否存在这样的字符位置——它在 clean 里、且其原文对应处被改过。
        # 实现上用简化判据：原 chunk 前半（与 query 同源区间）含可修复位点
        first_half_raw = original[: len(original) // 2]
        first_half_clean = clean[: len(clean) // 2]
        if first_half_clean == first_half_raw:
            continue  # 前半没有修复位点，query 窗口选不到
        anchors = E.extract_anchors(clean[mid:], window=24, max_anchors=8)
        if not anchors:
            continue
        q_norm = E.norm_for_match(query)
        anchors = [a for a in anchors if a not in q_norm and q_norm not in a]
        if not anchors:
            continue
        items.append(
            {
                "kid": c.get("id"),
                "doc": c.get("doc"),
                "text_sha1": m.get("text_sha1"),
                "query": query,
                "anchors": anchors,
                "n_anchors": len(anchors),
            }
        )
    with SET_PATH.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"[build] 含可修复位点且 query 覆盖的 chunk: {len(items)}/651")
    print(f"[build] 已写入 {SET_PATH}")


def run(arm: str, tag: str, top_k: int = 5) -> None:
    if arm == "b":
        # B 臂：规则全关。必须在任何缓存构建之前打补丁。
        R.normalize_thai = lambda t: t  # type: ignore
        print("[run] B 臂：normalize_thai = identity（规则关闭）")
    items = [
        json.loads(line)
        for line in SET_PATH.open(encoding="utf-8")
        if line.strip()
    ]
    rows = []
    for i, it in enumerate(items, 1):
        try:
            ev = R.retrieve(it["query"], top_k=top_k)
        except Exception as exc:
            print(f"  [WARN] {it['kid']}: {exc}")
            ev = []
        texts = [str(e.get("text") or "") for e in ev]
        # 匹配口径：锚点是 clean 形态；A 臂语料侧被运行时修复（直接可比），
        # B 臂语料侧保持损坏。两侧都用 norm_for_match，不再额外规范化，
        # 保证两臂用同一把尺。
        rank = E.hit_rank(it["anchors"], texts)
        rows.append({"kid": it["kid"], "hit_rank": rank, "n": len(ev)})
        if i % 50 == 0:
            print(f"  ...{i}/{len(items)}")
    n = len(rows)
    rep = {
        "arm": arm,
        "tag": tag,
        "n": n,
        "hit@1": sum(1 for r in rows if r["hit_rank"] and r["hit_rank"] <= 1) / n,
        "hit@3": sum(1 for r in rows if r["hit_rank"] and r["hit_rank"] <= 3) / n,
        "hit@5": sum(1 for r in rows if r["hit_rank"] and r["hit_rank"] <= 5) / n,
        "mrr": sum(1.0 / r["hit_rank"] for r in rows if r["hit_rank"]) / n,
        "rows": rows,
    }
    out = OUT_DIR / f"_probe_thnormalize_cleanq_{arm}_{tag}.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{'='*60}\nclean-query 价值测定  arm={arm}  n={n}\n{'='*60}")
    for k in (1, 3, 5):
        print(f"  Hit@{k}: {rep[f'hit@{k}']:7.1%}")
    print(f"  MRR   : {rep['mrr']:7.4f}")
    print(f"已写入 {out}")


def compare(before_tag: str, after_tag: str) -> None:
    a = json.loads((OUT_DIR / f"_probe_thnormalize_cleanq_a_{before_tag}.json").read_text("utf-8"))
    b = json.loads((OUT_DIR / f"_probe_thnormalize_cleanq_b_{after_tag}.json").read_text("utf-8"))
    print(f"\n{'='*66}\nA 臂（规则开）vs B 臂（规则关）  n={a['n']}\n{'='*66}")
    for k in ("hit@1", "hit@3", "hit@5", "mrr"):
        print(f"  {k:8s} A={a[k]:.4f}  B={b[k]:.4f}  差={a[k]-b[k]:+.4f}")
    A = {r["kid"]: r for r in a["rows"]}
    B = {r["kid"]: r for r in b["rows"]}
    only_a = [k for k in A if A[k]["hit_rank"] and not B.get(k, {}).get("hit_rank")]
    only_b = [k for k in A if B.get(k, {}).get("hit_rank") and not A[k]["hit_rank"]]
    print(f"\n  仅 A 命中（规则救回）: {len(only_a)}")
    print(f"  仅 B 命中（规则反伤）: {len(only_b)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run", "compare"])
    ap.add_argument("--arm", choices=["a", "b"])
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--before")
    ap.add_argument("--after")
    args = ap.parse_args()
    if args.cmd == "build":
        build()
    elif args.cmd == "run":
        assert args.arm
        run(args.arm, args.tag)
    else:
        compare(args.before, args.after)


if __name__ == "__main__":
    main()

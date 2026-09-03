"""临时探针（th-normalize）：单受损词短查询探针 —— 规则检索价值的最后一个未测场景。

背景：#28 两臂测试（356 条 60 字符查询）显示规则检索聚合价值 ≈0，机制是长查询的
gram 冗余（损坏位点只毁少数 gram）。本探针测极端场景：**查询几乎全部 gram 都落在
损坏位**——单受损词或受损词主导的短查询。如果这里也 ≈0，规则的检索价值彻底归零、
纯卫生定性成立；如果 B 臂明显失配，规则在短查询场景有真实价值。

设计：
  索引：10:41 损坏索引（唯一正确实验条件，2c 重建后不可复现）。
  语料侧损坏取自 **docstore 已存文本**（不是 original_text——scheme 类损坏建索引时
  已修好，存储态只剩 reorder / 双声调 / 同声调叠写garble 三类可运行时修复的形态，
  以及 น่้า 这种不可修复形态）。
  查询 = 受损词的干净形态（用户会打的样子），锚点 = 该 chunk 干净化后的 24 字符窗口。
  A 臂规则开（运行时修复语料），B 臂 normalize_thai=identity（语料保持损坏）。
  额外记录：目标 chunk（按 text_sha1）是否进 top-5，独立于锚点口径。

用法：
    ./.venv/bin/python _probe_th_shortq_thnormalize.py build
    ./.venv/bin/python _probe_th_shortq_thnormalize.py run --arm a --tag v1
    ./.venv/bin/python _probe_th_shortq_thnormalize.py run --arm b --tag v1
    ./.venv/bin/python _probe_th_shortq_thnormalize.py compare --before v1 --after v1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rag_eval as E
import rag_llamaindex as R

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "rag_eval"
SET_PATH = OUT_DIR / "_probe_thnormalize_shortq_set.jsonl"
DOCSTORE = BASE_DIR / "rag_llamaindex_storage" / "docstore.json"

WORD = re.compile(r"[\u0e00-\u0e7f]+")


def load_stored_bilingual() -> Dict[str, str]:
    """text_sha1 -> 已存文本（651 个双语 node）。"""
    d = json.loads(DOCSTORE.read_text(encoding="utf-8"))["docstore/data"]
    out = {}
    for v in d.values():
        da = v.get("__data__", {})
        me = da.get("metadata") or {}
        if me.get("block_type") == "text" and me.get("src_lang") == "th" and me.get("display_text"):
            out[str(me.get("text_sha1"))] = da.get("text") or ""
    return out


def build() -> None:
    stored = load_stored_bilingual()
    items: List[Dict[str, Any]] = []
    stats = Counter()
    for sha, text in stored.items():
        clean_text = R.normalize_thai(text)
        if clean_text == text:
            continue
        anchors = E.extract_anchors(clean_text, window=24, max_anchors=6)
        if not anchors:
            continue
        seen_words = set()
        for w in WORD.findall(text):
            if w in seen_words or len(w) < 4:
                continue
            clean = R.normalize_thai(w)
            if clean == w:
                continue  # 该词无运行时可修复损坏
            seen_words.add(w)
            stats[f"class_{('reorder' if re.search(chr(0x0e48)+'-'+chr(0x0e4b)+chr(0x0e34)+'-'+chr(0x0e37), w) else ('dbl_garble' if re.search(r'([' + chr(0x0e48) + '-' + chr(0x0e4b) + r'])\1', w) else 'other'))}"] += 1
            items.append({
                "kid": sha,
                "word_raw": w,
                "query": clean,
                "anchors": anchors,
                "n_anchors": len(anchors),
            })
    # 每个 (chunk, 查询词) 一条；上限 60，优先不同查询词
    by_query: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        by_query.setdefault(it["query"], []).append(it)
    final = []
    for q in sorted(by_query, key=lambda q: -len(by_query[q])):
        for it in by_query[q]:
            final.append(it)
            if len(final) >= 60:
                break
        if len(final) >= 60:
            break
    with SET_PATH.open("w", encoding="utf-8") as f:
        for it in final:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"[build] 可运行时修复的受损词位: {sum(stats.values())} 处 {dict(stats)}")
    print(f"[build] 入选短查询: {len(final)} 条（覆盖 {len({i['query'] for i in final})} 个不同词形）")
    print(f"[build] 已写入 {SET_PATH}")


def run(arm: str, tag: str, top_k: int = 5) -> None:
    if arm == "b":
        R.normalize_thai = lambda t: t  # type: ignore
        print("[run] B 臂：normalize_thai = identity（规则关闭）")
    items = [json.loads(l) for l in SET_PATH.open(encoding="utf-8") if l.strip()]
    rows = []
    for it in items:
        try:
            ev = R.retrieve(it["query"], top_k=top_k)
        except Exception as exc:
            print(f"  [WARN] {it['kid'][:8]}: {exc}")
            ev = []
        texts = [str(e.get("text") or "") for e in ev]
        rank = E.hit_rank(it["anchors"], texts)
        target_rank = None
        for pos, e in enumerate(ev, 1):
            if str((e.get("metadata") or {}).get("text_sha1") or "") == it["kid"]:
                target_rank = pos
                break
        rows.append({
            "kid": it["kid"], "query": it["query"], "word_raw": it["word_raw"],
            "hit_rank": rank, "target_rank": target_rank,
        })
    n = len(rows)
    rep = {
        "arm": arm, "tag": tag, "n": n,
        "anchor_hit@5": sum(1 for r in rows if r["hit_rank"]) / n,
        "anchor_hit@1": sum(1 for r in rows if r["hit_rank"] == 1) / n,
        "target_in_top5": sum(1 for r in rows if r["target_rank"]) / n,
        "target_top1": sum(1 for r in rows if r["target_rank"] == 1) / n,
        "rows": rows,
    }
    out = OUT_DIR / f"_probe_thnormalize_shortq_{arm}_{tag}.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{'='*60}\n短查询探针  arm={arm}  n={n}\n{'='*60}")
    print(f"  anchor Hit@1: {rep['anchor_hit@1']:7.1%}   Hit@5: {rep['anchor_hit@5']:7.1%}")
    print(f"  目标chunk进top5: {rep['target_in_top5']:7.1%}   top1: {rep['target_top1']:7.1%}")
    print(f"已写入 {out}")


def compare(before_tag: str, after_tag: str) -> None:
    a = json.loads((OUT_DIR / f"_probe_thnormalize_shortq_a_{before_tag}.json").read_text("utf-8"))
    b = json.loads((OUT_DIR / f"_probe_thnormalize_shortq_b_{after_tag}.json").read_text("utf-8"))
    print(f"\n{'='*66}\n短查询 A 臂（规则开） vs B 臂（规则关）  n={a['n']}\n{'='*66}")
    for k in ("anchor_hit@1", "anchor_hit@5", "target_in_top5", "target_top1"):
        print(f"  {k:16s} A={a[k]:.4f}  B={b[k]:.4f}  差={a[k]-b[k]:+.4f}")
    A = {r["kid"] + r["query"]: r for r in a["rows"]}
    B = {r["kid"] + r["query"]: r for r in b["rows"]}
    only_a = [k for k in A if A[k]["target_rank"] and not B.get(k, {}).get("target_rank")]
    only_b = [k for k in A if B.get(k, {}).get("target_rank") and not A[k]["target_rank"]]
    print(f"\n  仅 A 把目标chunk送进top5（规则救回）: {len(only_a)}")
    for k in only_a[:10]:
        print(f"     {A[k]['word_raw']!r} -> 查询 {A[k]['query']!r}")
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

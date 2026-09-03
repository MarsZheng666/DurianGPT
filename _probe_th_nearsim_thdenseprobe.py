#!/usr/bin/env python3
# ⚠️ 归属已确认：th-dense-probe（会话日志 Write 调用取证；其否认系 grep 错对象的自查失误）
# ==========================================================================
#  临 时 探 针 —— 用完即删
#  两个目的：
#  (A) 用**泰文感知**的字符 n-gram Jaccard 测四语种金标的近重复密度。
#      不能用空格分词：泰文无词边界，按空格切会得到"整句一个 token"，
#      Jaccard 恒为 0（这正是先前泰文测出 0.000 的原因，那个数无效）。
#      n=3/4 与 _lexical_terms() 对泰文用 3-4 gram 的口径一致。
#  (B) 造一把**无污染**的命中判据，替代被近重复污染的 _ns：
#      _is_self_hit() 按"问题前 40 字符出现在证据里"判自命中，近重复问题
#      前缀高度雷同 → 非答案的重复项也被误扣。
#      改用 gold['ref_doc_id'] 与证据 metadata['doc'] **精确相等**来排除
#      自身（实测 th/ms 各 139/139 条都能精确对上）。这叫 xs 口径
#      (exact-self-excluded)，只排除真正的自身 chunk，零启发式。
#
#  不修改任何已有文件；不访问远端。
# ==========================================================================
from __future__ import annotations

import itertools
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))


def log(*a):
    print(*a, flush=True)


import rag_eval as E          # noqa: E402
import rag_llamaindex as R    # noqa: E402
sys.path.insert(0, str(BASE))
from _probe_th_dense_weight import hybrid_retrieve_w  # noqa: E402


def mtime() -> str:
    p = BASE / "rag_llamaindex_storage" / "docstore.json"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))


# ───────────────────── (A) 字符 n-gram 近重复密度 ─────────────────────
def char_ngrams(s: str, n: int) -> set:
    t = E.norm_for_match(s)          # 去标点/空白，四语种统一
    return {t[i:i + n] for i in range(max(0, len(t) - n + 1))}


def jac(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def nearest_neighbor_jaccard(items: List[Dict], n: int) -> List[float]:
    grams = [char_ngrams(g["question"], n) for g in items]
    out = []
    for i in range(len(grams)):
        best = 0.0
        for j in range(len(grams)):
            if i == j:
                continue
            v = jac(grams[i], grams[j])
            if v > best:
                best = v
        out.append(best)
    return out


def part_a():
    gold = E.load_gold()
    log("=" * 78)
    log("(A) 四语种金标近重复密度 —— 字符 n-gram Jaccard（泰文感知，非空格分词）")
    log("=" * 78)
    log("队列按 detect_lang(question) 取（金标 q_lang 有 237 条标注错误，不可用）")
    cohorts = {L: [g for g in gold if E.detect_lang(g["question"]) == L]
               for L in ("th", "ms", "en", "zh")}
    res = {}
    for n in (3, 4):
        log(f"\n--- 字符 {n}-gram ---")
        log(f"{'语种':<6}{'n':>6}{'最近邻中位':>11}{'均值':>9}"
            f"{'≥0.6占比':>10}{'≥0.8占比':>10}{'p90':>8}")
        for L, items in cohorts.items():
            nn = nearest_neighbor_jaccard(items, n)
            r = {
                "n": len(items),
                "median": statistics.median(nn),
                "mean": statistics.mean(nn),
                "ge6": sum(1 for v in nn if v >= 0.6) / len(nn) * 100,
                "ge8": sum(1 for v in nn if v >= 0.8) / len(nn) * 100,
                "p90": sorted(nn)[int(len(nn) * 0.9)],
            }
            res[f"{L}_{n}"] = r
            log(f"{L:<6}{r['n']:>6}{r['median']:>11.3f}{r['mean']:>9.3f}"
                f"{r['ge6']:>9.1f}%{r['ge8']:>9.1f}%{r['p90']:>8.3f}")
    return cohorts, res


# ───────── (B) 无污染判据 xs：按 ref_doc_id 精确排除自身 chunk ─────────
def hit_rank_xs(g: Dict, evidences: List[Dict]) -> Optional[int]:
    ref = str(g.get("ref_doc_id") or "")
    texts = []
    for ev in evidences:
        meta = ev.get("metadata") or {}
        doc = str(meta.get("doc") or ev.get("doc") or "")
        if ref and doc == ref:
            continue                      # 精确排除自身，零启发式
        texts.append(str(ev.get("text") or ""))
    return E.hit_rank(g["anchors"], texts)


def eval_variant(items: List[Dict], **kw) -> Dict[str, Any]:
    ranks_xs, rows = [], []
    n_excl = []
    for g in items:
        ev = hybrid_retrieve_w(g["question"], top_k=5, **kw)
        ranks_xs.append(hit_rank_xs(g, ev))
        rows.append(E.evaluate_one(g, ev))
        ref = str(g.get("ref_doc_id") or "")
        n_excl.append(sum(1 for e in ev
                          if str((e.get("metadata") or {}).get("doc") or "") == ref))
    n = len(items)
    pct = lambda k: sum(1 for r in rows if r.get(k)) / n * 100
    xs = lambda k: sum(1 for r in ranks_xs if r and r <= k) / n * 100
    return {
        "raw1": pct("hit@1"), "raw3": pct("hit@3"),
        "ns1": pct("hit@1_ns"), "ns3": pct("hit@3_ns"),
        "xs1": xs(1), "xs3": xs(3), "xs5": xs(5),
        "mrr_xs": statistics.mean(1.0 / r if r else 0.0 for r in ranks_xs),
        "mrr_ns": statistics.mean(r["mrr_ns"] for r in rows),
        "empty": pct("empty_result"),
        "ind": statistics.mean(r["ind_defect_ratio"] for r in rows) * 100,
        "n_self_excl_xs": statistics.mean(n_excl),
        "n_self_excl_ns": statistics.mean(r["n_self_hit"] for r in rows),
        "ranks_xs": ranks_xs, "rows": rows,
    }


def part_b(cohorts):
    log("\n" + "=" * 78)
    log("(B) 无污染判据 xs（按 ref_doc_id 精确排除自身）下重跑 A/B")
    log("=" * 78)
    log(f"索引 docstore mtime(开跑): {mtime()}")
    th, ms = cohorts["th"], cohorts["ms"]

    out = {}
    log(f"\n【泰文 n={len(th)}】")
    log(f"{'变体':<12}{'raw@3':>8}{'ns@3':>8}{'xs@1':>8}{'xs@3':>8}{'xs@5':>8}"
        f"{'MRRxs':>9}{'空':>7}{'扣ns':>7}{'扣xs':>7}")
    for lbl, kw in (("w=1.0", {"dense_weight": 1.0}), ("跳过dense", {"skip_dense": True})):
        r = eval_variant(th, **kw)
        out["th_" + lbl] = r
        log(f"{lbl:<12}{r['raw3']:7.1f}%{r['ns3']:7.1f}%{r['xs1']:7.1f}%"
            f"{r['xs3']:7.1f}%{r['xs5']:7.1f}%{r['mrr_xs']:9.3f}"
            f"{r['empty']:6.1f}%{r['n_self_excl_ns']:7.2f}{r['n_self_excl_xs']:7.2f}")
    a, b = out["th_w=1.0"], out["th_跳过dense"]
    for tag, key in (("xs@3", "ranks_xs"),):
        lost = [g["qid"] for g, x, y in zip(th, a["ranks_xs"], b["ranks_xs"])
                if (x and x <= 3) and not (y and y <= 3)]
        gain = [g["qid"] for g, x, y in zip(th, a["ranks_xs"], b["ranks_xs"])
                if not (x and x <= 3) and (y and y <= 3)]
    log(f"  逐题(xs@3 无污染): 靠dense才命中 {len(lost)} 条 {lost[:12]}")
    log(f"  逐题(xs@3 无污染): 因dense失手 {len(gain)} 条 {gain[:12]}")
    out["th_lost_xs"], out["th_gain_xs"] = lost, gain

    log(f"\n【马来文 n={len(ms)}】")
    log(f"{'变体':<12}{'raw@3':>8}{'ns@3':>8}{'xs@1':>8}{'xs@3':>8}{'xs@5':>8}"
        f"{'MRRxs':>9}{'空':>7}{'扣ns':>7}{'扣xs':>7}")
    for lbl, kw in (("w=1.0", {"dense_weight": 1.0}), ("w=0.5", {"dense_weight": 0.5}),
                    ("w=0.4", {"dense_weight": 0.4}), ("跳过dense", {"skip_dense": True})):
        r = eval_variant(ms, **kw)
        out["ms_" + lbl] = r
        log(f"{lbl:<12}{r['raw3']:7.1f}%{r['ns3']:7.1f}%{r['xs1']:7.1f}%"
            f"{r['xs3']:7.1f}%{r['xs5']:7.1f}%{r['mrr_xs']:9.3f}"
            f"{r['empty']:6.1f}%{r['n_self_excl_ns']:7.2f}{r['n_self_excl_xs']:7.2f}")
    return out


# ─────── (C) 命中组 vs 失手组的近重复密度（因果判据，照 glossary-th 的做法）───────
def part_c(cohorts, out):
    log("\n" + "=" * 78)
    log("(C) 命中组 vs 失手组 近重复密度对比（4-gram 最近邻 Jaccard）")
    log("=" * 78)
    for L in ("th", "ms"):
        items = cohorts[L]
        nn = nearest_neighbor_jaccard(items, 4)
        base = out[f"{L}_w=1.0"]
        for metric, getter in (("xs@1", lambda r: r and r <= 1),
                               ("xs@3", lambda r: r and r <= 3)):
            hit = [v for v, r in zip(nn, base["ranks_xs"]) if getter(r)]
            mis = [v for v, r in zip(nn, base["ranks_xs"]) if not getter(r)]
            if not hit or not mis:
                continue
            log(f"  [{L}] {metric}: 命中组 n={len(hit)} 中位 {statistics.median(hit):.3f}"
                f" ｜ 失手组 n={len(mis)} 中位 {statistics.median(mis):.3f}"
                f" ｜ 差 {statistics.median(mis)-statistics.median(hit):+.3f}")


def main():
    t0 = time.time()
    cohorts, ja = part_a()
    out = part_b(cohorts)
    part_c(cohorts, out)
    log(f"\n索引 docstore mtime(跑完): {mtime()}")
    log(f"总耗时 {time.time()-t0:.0f}s")
    (BASE / "_probe_nearsim.json").write_text(json.dumps(
        {"index_mtime": mtime(), "jaccard": ja,
         "ab": {k: {kk: vv for kk, vv in v.items()
                    if kk not in ("rows", "ranks_xs")}
                for k, v in out.items() if isinstance(v, dict)},
         "th_lost_xs": out.get("th_lost_xs"), "th_gain_xs": out.get("th_gain_xs")},
        ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

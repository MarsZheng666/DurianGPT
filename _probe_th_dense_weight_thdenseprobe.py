#!/usr/bin/env python3
# ==========================================================================
#  临 时 探 针 文 件 —— 用完即删，不属于项目正式代码
#  验证：dense 通路是否在 RRF 融合里主动拖累泰文检索
#
#  方法要点：走**真实完整链路**做 A/B。为了能调 dense 的 RRF 权重
#  （原实现里是 `merge_candidates(dense, "dense", 1.0)` 的字面量），
#  这里把 `_hybrid_retrieve` 忠实重实现一份并加 dense_weight 参数，
#  然后**先验证 w=1.0 时逐题输出与真实函数完全一致**——验证通过后，
#  其余权重档位的结果才可信。这样就不存在"近似"，除了下面这一条：
#
#  注意一个非平凡的副作用：后置过滤 `evidence_matches_query(query, text,
#  dense_score) or bm25_score > 1.2` 里用到了 dense_score。dense 跳过后，
#  那些同时被 BM25 召回的候选其 dense_score 会变成 0，可能因此被过滤掉。
#  这不是探针的人为偏差——生产环境真跳过 dense 也会这样，故如实保留。
#
#  不修改任何已有文件；不访问任何远端服务。
# ==========================================================================
from __future__ import annotations

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


# ───────────── 忠实重实现 _hybrid_retrieve，仅多一个 dense_weight ─────────────
def hybrid_retrieve_w(
    query: str,
    top_k: int,
    dense_weight: float = 1.0,
    skip_dense: bool = False,
    gate_mode: str = "asis",
    bm25_floor: float = 1.2,
) -> List[Dict[str, Any]]:
    """gate_mode 控制后置过滤如何处理 dense_score：
      asis     —— 原样（跳过 dense 时 dense_score=0）
      neutral  —— 给 evidence_matches_query 传 0.50 的中性分。
                  因为该函数**每一条 return True 路径都要求 score>=0.45**，
                  dense_score=0 时它恒为 False、门槛退化成只剩 bm25>1.2。
                  传中性分可让 intent/词覆盖那套判据重新生效。
      bm25only —— 显式只看 bm25>1.2（用于证明 asis 已等价于此）
    """
    index = R.load_index()
    exact_mode = R.is_exact_answer_query(query)
    candidate_limit = max(36, int(top_k) * 8)
    dense = [] if skip_dense else R._dense_retrieve(index, query, candidate_limit)
    lexical = R._bm25_retrieve(index, query, candidate_limit)

    merged: Dict[str, Dict[str, Any]] = {}
    rrf_constant = 60.0

    def merge_candidates(items, source, weight):
        for rank, item in enumerate(items, start=1):
            key = R._normalized_dedupe_key(item.get("text") or "")
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
                    target["dense_score"], float(item.get("dense_score") or 0.0))
            else:
                target["bm25_score"] = max(
                    target["bm25_score"], float(item.get("bm25_score") or 0.0))

    merge_candidates(dense, "dense", dense_weight)
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

    results: List[Dict[str, Any]] = []
    seen_locations = set()
    seen_expanded = set()
    for item in ranked:
        dense_score = float(item.get("dense_score") or 0.0)
        bm25_score = float(item.get("bm25_score") or 0.0)
        text = str(item.get("text") or "")
        if not exact_mode:
            if gate_mode == "bm25only":
                passed = bm25_score > 1.2
            else:
                # 注意阈值标定：泰文查询下 detect_intents=[] 且
                # extract_query_terms=[]，唯一可达的 return True 分支是
                # `len(query)>=18 and score>=0.55`。所以中性分必须 >0.55 才
                # 真正激活门槛；先前取 0.50 是无效对照（实测结果与 asis 全等）。
                if gate_mode.startswith("neutral"):
                    gate_score = float(gate_mode.split(":")[1]) if ":" in gate_mode else 0.60
                    if dense_score != 0.0:
                        gate_score = dense_score
                else:
                    gate_score = dense_score
                passed = (R.evidence_matches_query(query, text, gate_score)
                          or bm25_score > 1.2)
            if not passed:
                continue
        metadata = dict(item.get("metadata") or {})
        if exact_mode:
            text = R._expand_adjacent_text(item)
            if R._exact_anchor_coverage(query, text) < 0.6:
                continue
            doc = str(metadata.get("doc") or metadata.get("source_file") or "unknown")
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
                nearby = {(sequence_type, doc, numeric_value + off)
                          for off in (-1, 0, 1)}
                if seen_locations.intersection(nearby):
                    continue
                seen_locations.add((sequence_type, doc, numeric_value))
            expanded_key = R._normalized_dedupe_key(text)
            if expanded_key in seen_expanded:
                continue
            seen_expanded.add(expanded_key)
        results.append({
            "source_file": (metadata.get("source_file") or metadata.get("doc")
                            or "unknown"),
            "doc": metadata.get("doc") or metadata.get("source_file") or "unknown",
            "page": metadata.get("page"),
            "score": round(dense_score if dense_score
                           else float(item.get("hybrid_score") or 0.0), 4),
            "dense_score": round(dense_score, 4),
            "bm25_score": round(bm25_score, 4),
            "hybrid_score": round(float(item.get("hybrid_score") or 0.0), 6),
            "text": text,
            "display_text": text,
            "metadata": metadata,
            "neighbor_expanded": bool(exact_mode),
        })
        if len(results) >= top_k:
            break
    return results


# ───────────────────────── 重实现的等价性验证 ─────────────────────────
def validate_reimpl(items: List[Dict[str, Any]]) -> bool:
    log("\n[验证] 重实现在 dense_weight=1.0 时是否与真实 R.retrieve 完全一致")
    bad = 0
    for g in items:
        a = R.retrieve(g["question"], top_k=5)
        b = hybrid_retrieve_w(g["question"], top_k=5, dense_weight=1.0)
        ta = [x["text"] for x in a]
        tb = [x["text"] for x in b]
        if ta != tb:
            bad += 1
            if bad <= 2:
                log(f"   不一致 qid={g['qid']}: len {len(ta)} vs {len(tb)}")
    if bad == 0:
        log(f"   ✓ {len(items)} 条逐题证据序列完全一致，重实现可信")
    else:
        log(f"   ✗ {bad}/{len(items)} 条不一致 —— 权重档位结果不可信！")
    return bad == 0


# ───────────────────────────── 指标汇总 ─────────────────────────────
def run_variant(items, label, quiet: bool = False, **kw) -> Dict[str, Any]:
    rows = []
    t0 = time.time()
    for g in items:
        ev = hybrid_retrieve_w(g["question"], top_k=5, **kw)
        rows.append(E.evaluate_one(g, ev))
    n = len(rows)

    def pct(k):
        return sum(1 for r in rows if r.get(k)) / n * 100

    out = {
        "label": label,
        "h1": pct("hit@1"), "h3": pct("hit@3"), "h5": pct("hit@5"),
        "h1ns": pct("hit@1_ns"), "h3ns": pct("hit@3_ns"), "h5ns": pct("hit@5_ns"),
        "h3kb": pct("hit@3_kb"),
        "mrr": statistics.mean(r["mrr"] for r in rows),
        "mrrns": statistics.mean(r["mrr_ns"] for r in rows),
        "empty": pct("empty_result"),
        "ind": statistics.mean(r["ind_defect_ratio"] for r in rows) * 100,
        "nret": statistics.mean(r["n_returned"] for r in rows),
        "sec": time.time() - t0,
        "rows": rows,
    }
    if not quiet:
        log(f"   {label:<22} raw@3={out['h3']:5.1f}%  ns@3={out['h3ns']:5.1f}%  "
            f"kb@3={out['h3kb']:4.1f}%  MRRns={out['mrrns']:.3f}  "
            f"空={out['empty']:4.1f}%  均返回={out['nret']:.2f}  垃圾率={out['ind']:.2f}%")
    return out


def per_query_diff(base: Dict[str, Any], alt: Dict[str, Any], items, metric="hit@3"):
    """列出 base→alt 的得失题目。用 _ns 口径同时给一份（那是诚实口径）。"""
    lost, gained = [], []
    for g, rb, ra in zip(items, base["rows"], alt["rows"]):
        if rb.get(metric) and not ra.get(metric):
            lost.append(g["qid"])
        elif not rb.get(metric) and ra.get(metric):
            gained.append(g["qid"])
    return lost, gained


def main():
    lang = sys.argv[1] if len(sys.argv) > 1 else "th"
    gold = E.load_gold()
    items = [g for g in gold if g["q_lang"] == lang]
    log(f"=== 语种 {lang}，金标 {len(items)} 条 ===")
    mt0 = (BASE / "rag_llamaindex_storage" / "docstore.json").stat().st_mtime
    log(f"索引 docstore mtime(开跑): "
        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mt0))}")

    ok = validate_reimpl(items[:12])

    log("\n[A/B] 完整链路，仅改 dense 的 RRF 权重:")
    res = {}
    res["base"] = run_variant(items, "现状 dense_w=1.0")
    res["skip"] = run_variant(items, "跳过 dense", skip_dense=True)
    res["w05"] = run_variant(items, "dense_w=0.5", dense_weight=0.5)
    res["w03"] = run_variant(items, "dense_w=0.3", dense_weight=0.3)

    log("\n[逐题] 现状 → 跳过 dense 的得失（raw hit@3 口径）:")
    lost, gained = per_query_diff(res["base"], res["skip"], items, "hit@3")
    log(f"   靠 dense 才命中(跳过后丢失): {len(lost)} 条 {lost[:20]}")
    log(f"   因 dense 才失手(跳过后修回): {len(gained)} 条 {gained[:20]}")

    log("\n[逐题] 同上，_ns 诚实口径（已扣除自命中泄漏）:")
    lost2, gained2 = per_query_diff(res["base"], res["skip"], items, "hit@3_ns")
    log(f"   靠 dense 才命中: {len(lost2)} 条 {lost2[:20]}")
    log(f"   因 dense 才失手: {len(gained2)} 条 {gained2[:20]}")

    mt2 = (BASE / "rag_llamaindex_storage" / "docstore.json").stat().st_mtime
    log(f"\n索引 docstore mtime(跑完): "
        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mt2))}"
        f"  {'✓ 未变' if mt2 == mt0 else '✗ 索引被重建，本批作废！'}")
    out = BASE / f"_probe_weight_result_{lang}.json"
    out.write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in res.items()}
        | {"lost_raw": lost, "gained_raw": gained,
           "lost_ns": lost2, "gained_ns": gained2,
           "reimpl_validated": ok},
        ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n明细 → {out}")


if __name__ == "__main__" and (len(sys.argv) == 1 or sys.argv[1] in ("th", "ms", "zh", "en")):
    main()


# ==========================================================================
#  任务 2-1：马来文权重细扫（曲线形状：平台 or 尖峰？）
#  python _probe_th_dense_weight.py sweep [lang]
# ==========================================================================
def sweep(lang: str = "ms"):
    gold = E.load_gold()
    items = [g for g in gold if g["q_lang"] == lang]
    mt = (BASE / "rag_llamaindex_storage" / "docstore.json").stat().st_mtime
    log(f"=== {lang} 权重细扫，{len(items)} 条全跑 ===")
    log(f"索引 docstore mtime(开跑): {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mt))}")
    assert validate_reimpl(items[:8]), "重实现未通过等价性验证，结果不可信"

    grid = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    res = {}
    log(f"\n{'w':>5}{'raw@1':>8}{'raw@3':>8}{'ns@1':>8}{'ns@3':>8}{'ns@5':>8}"
        f"{'MRRns':>9}{'空':>7}{'垃圾率':>9}")
    for w in grid:
        kw = {"skip_dense": True} if w == 0.0 else {"dense_weight": w}
        r = run_variant_quiet(items, kw)
        res[str(w)] = r
        tag = " (=跳过)" if w == 0.0 else ""
        log(f"{w:>5.1f}{r['h1']:7.1f}%{r['h3']:7.1f}%{r['h1ns']:7.1f}%"
            f"{r['h3ns']:7.1f}%{r['h5ns']:7.1f}%{r['mrrns']:9.3f}"
            f"{r['empty']:6.1f}%{r['ind']:8.2f}%{tag}")

    mt2 = (BASE / "rag_llamaindex_storage" / "docstore.json").stat().st_mtime
    log(f"\n索引 docstore mtime(跑完): "
        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mt2))}"
        f"  {'✓ 未变' if mt2 == mt else '✗ 索引被重建，本批数字作废！'}")

    # 曲线形状判定：以 ns@3 为准
    xs = [w for w in grid if w > 0]
    ys = [res[str(w)]["h3ns"] for w in xs]
    best = max(range(len(xs)), key=lambda i: ys[i])
    base = res["1.0"]["h3ns"]
    log(f"\n[曲线形状] ns@3 基线(w=1.0)={base:.1f}%  最优 w={xs[best]}"
        f" 时 {ys[best]:.1f}%（+{ys[best]-base:.1f}pp）")
    # 平台判据（修正版）：锚定**基线**而非最大值。
    # 锚定最大值时，一根单点尖刺会把整段真实平台误判成"尖峰"——
    # 我第一版就犯了这个错。真正要问的是：
    #   「相对 w=1.0 的改善，是否在一段连续的权重区间上稳定成立？」
    gain_max = ys[best] - base
    thresh = base + 0.7 * gain_max          # 拿到最大改善的 70% 即算"在平台上"
    on = [xs[i] for i in range(len(xs)) if ys[i] >= thresh]
    # 求最长连续段（按 grid 相邻）
    runs, cur = [], []
    for w in xs:
        if w in on:
            cur.append(w)
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    longest = max(runs, key=len) if runs else []
    log(f"     最大改善 +{gain_max:.1f}pp（w={xs[best]}）；"
        f"达到其 70%（≥{thresh:.1f}%）的档位: {on}")
    log(f"     最长连续平台段: {longest}"
        f"（{len(longest)} 档，跨度 {min(longest) if longest else 0}~{max(longest) if longest else 0}）")
    log(f"     MRRns 各档: " + " ".join(
        f"{w}:{res[str(w)]['mrrns']:.3f}" for w in xs))
    log(f"     全档位 ns@3 极差: {max(ys)-min(ys):.1f}pp")
    if len(longest) >= 3:
        log(f"     → 判定：**平台**（改善在 w={min(longest)}~{max(longest)} 上稳定成立，"
            f"参数可放心定；最优点 {xs[best]} 的额外峰值应视为噪声）")
    else:
        log("     → 判定：**尖峰**（改善只在孤立档位成立，疑似过拟合，"
            "不值得引入分语种分支）")

    out = BASE / f"_probe_sweep_{lang}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"明细 → {out}")


def run_variant_quiet(items, kw) -> Dict[str, Any]:
    rows = [E.evaluate_one(g, hybrid_retrieve_w(g["question"], top_k=5, **kw))
            for g in items]
    n = len(rows)

    def pct(k):
        return sum(1 for r in rows if r.get(k)) / n * 100
    return {
        "h1": pct("hit@1"), "h3": pct("hit@3"), "h5": pct("hit@5"),
        "h1ns": pct("hit@1_ns"), "h3ns": pct("hit@3_ns"), "h5ns": pct("hit@5_ns"),
        "mrrns": statistics.mean(r["mrr_ns"] for r in rows),
        "mrr": statistics.mean(r["mrr"] for r in rows),
        "empty": pct("empty_result"),
        "ind": statistics.mean(r["ind_defect_ratio"] for r in rows) * 100,
    }


# ==========================================================================
#  任务 2-2：语种检测混淆矩阵 + 误判的净收益重算
#  python _probe_th_dense_weight.py langdetect
# ==========================================================================
def langdetect():
    gold = E.load_gold()
    log(f"=== rag_eval.detect_lang() 在 {len(gold)} 条金标提问上的混淆矩阵 ===")
    from collections import Counter, defaultdict
    conf = defaultdict(Counter)
    for g in gold:
        conf[g["q_lang"]][E.detect_lang(g["question"])] += 1

    langs = ["zh", "en", "ms", "th"]
    cols = langs + ["unknown"]
    log(f"\n{'真实\\判定':<10}" + "".join(f"{c:>9}" for c in cols)
        + f"{'合计':>8}{'正确率':>9}")
    for t in langs:
        row = conf[t]
        tot = sum(row.values())
        log(f"{t:<10}" + "".join(f"{row.get(c,0):>9}" for c in cols)
            + f"{tot:>8}{row.get(t,0)/tot*100:>8.1f}%")

    ms_as_en = conf["ms"].get("en", 0)
    ms_tot = sum(conf["ms"].values())
    en_as_ms = conf["en"].get("ms", 0)
    en_tot = sum(conf["en"].values())
    log(f"\n[关键比例]")
    log(f"  马来文被判成英文: {ms_as_en}/{ms_tot} = {ms_as_en/ms_tot*100:.1f}%"
        f"   （这些查询拿不到降权，收益打折）")
    log(f"  英文被判成马来文: {en_as_ms}/{en_tot} = {en_as_ms/en_tot*100:.1f}%"
        f"   （这些英文查询会被误降权 = 纯损失）")
    log(f"  中文正确率: {conf['zh'].get('zh',0)/sum(conf['zh'].values())*100:.1f}%"
        f"   泰文正确率: {conf['th'].get('th',0)/sum(conf['th'].values())*100:.1f}%")
    return conf


# ==========================================================================
#  任务 2-3：后置过滤对 dense_score 的依赖 —— 实验分离
#  python _probe_th_dense_weight.py gate
# ==========================================================================
def gate_experiment():
    gold = E.load_gold()
    items = [g for g in gold if g["q_lang"] == "th"]
    mt = (BASE / "rag_llamaindex_storage" / "docstore.json").stat().st_mtime
    log(f"=== 泰文 {len(items)} 条：后置过滤依赖 dense_score 的影响分离 ===")
    log(f"索引 mtime: {time.strftime('%H:%M:%S', time.localtime(mt))}")
    log("\n结构事实：evidence_matches_query 的每一条 return True 路径都要求"
        " score>=0.45（0.55/0.45 若干处），\n"
        "所以 dense_score=0 时它**恒为 False**，门槛退化成只剩 bm25>1.2。")

    variants = [
        ("现状 dense_w=1.0 / asis", {}),
        ("跳过dense / asis(=bm25>1.2)", {"skip_dense": True}),
        ("跳过dense / 显式bm25only", {"skip_dense": True, "gate_mode": "bm25only"}),
        ("跳过dense / 中性分0.50", {"skip_dense": True, "gate_mode": "neutral"}),
    ]
    log(f"\n{'变体':<30}{'raw@3':>8}{'ns@1':>8}{'ns@3':>8}{'MRRns':>9}"
        f"{'空':>7}{'垃圾率':>9}{'均返回':>8}")
    res = {}
    for label, kw in variants:
        rows = [E.evaluate_one(g, hybrid_retrieve_w(g["question"], top_k=5, **kw))
                for g in items]
        n = len(rows)

        def pct(k):
            return sum(1 for r in rows if r.get(k)) / n * 100
        r = {
            "h3": pct("hit@3"), "h1ns": pct("hit@1_ns"), "h3ns": pct("hit@3_ns"),
            "mrrns": statistics.mean(x["mrr_ns"] for x in rows),
            "empty": pct("empty_result"),
            "ind": statistics.mean(x["ind_defect_ratio"] for x in rows) * 100,
            "nret": statistics.mean(x["n_returned"] for x in rows),
        }
        res[label] = r
        log(f"{label:<30}{r['h3']:7.1f}%{r['h1ns']:7.1f}%{r['h3ns']:7.1f}%"
            f"{r['mrrns']:9.3f}{r['empty']:6.1f}%{r['ind']:8.2f}%{r['nret']:8.2f}")

    a = res["跳过dense / asis(=bm25>1.2)"]
    b = res["跳过dense / 显式bm25only"]
    same = (abs(a["h3ns"] - b["h3ns"]) < 1e-9 and abs(a["ind"] - b["ind"]) < 1e-9)
    log(f"\n[验证] asis 与 显式bm25only 是否等价: "
        f"{'✓ 完全一致，证实 dense_score=0 已使 evidence_matches_query 失效' if same else '✗ 不一致'}")
    base = res["现状 dense_w=1.0 / asis"]
    neu = res["跳过dense / 中性分0.50"]
    log(f"[垃圾率归因] 现状 {base['ind']:.2f}% → 跳过(asis) {a['ind']:.2f}%"
        f" → 跳过(中性分) {neu['ind']:.2f}%")
    mt2 = (BASE / "rag_llamaindex_storage" / "docstore.json").stat().st_mtime
    log(f"索引 mtime(跑完): {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mt2))}"
        f"  {'✓ 未变' if mt2 == mt else '✗ 索引被重建，本批作废！'}")
    out = BASE / "_probe_gate_th.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"明细 → {out}")


_CMDS = {"sweep": sweep, "langdetect": langdetect, "gate": gate_experiment}
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in _CMDS:
    fn = _CMDS[sys.argv[1]]
    fn(*sys.argv[2:]) if sys.argv[2:] else fn()
    sys.exit(0)


# ==========================================================================
#  最终确认（python _probe_th_dense_weight.py confirm）
#  两件事：
#  (1) 后置过滤对泰文是否"非绑定"——把中性分推到 0.99（必然放行）看结果是否仍不变。
#      若不变，则证明 top-5 槽位早已被更高 RRF 排名且 bm25>1.2 的候选填满，
#      放宽门槛只会放进排名更低的候选、进不了前 5。垃圾率变化就与门槛无关。
#  (2) 马来文降权的**生产口径**收益：只有 detect_lang=='ms' 的查询才会被降权，
#      故在 139 条（而非标称 152 条）上测 w=1.0 vs w=0.5。
# ==========================================================================
def confirm():
    gold = E.load_gold()
    mt = (BASE / "rag_llamaindex_storage" / "docstore.json").stat().st_mtime
    log(f"索引 docstore mtime: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mt))}")

    def metrics(items, **kw):
        rows = [E.evaluate_one(g, hybrid_retrieve_w(g["question"], top_k=5, **kw))
                for g in items]
        n = len(rows)
        pct = lambda k: sum(1 for r in rows if r.get(k)) / n * 100
        return {"h3": pct("hit@3"), "h1ns": pct("hit@1_ns"), "h3ns": pct("hit@3_ns"),
                "mrrns": statistics.mean(x["mrr_ns"] for x in rows),
                "empty": pct("empty_result"),
                "ind": statistics.mean(x["ind_defect_ratio"] for x in rows) * 100,
                "nret": statistics.mean(x["n_returned"] for x in rows), "n": n}

    th = [g for g in gold if E.detect_lang(g["question"]) == "th"]
    log(f"\n(1) 后置过滤是否非绑定 —— 泰文 {len(th)} 条，跳过 dense，只改中性分:")
    log(f"{'门槛中性分':<22}{'ns@3':>8}{'MRRns':>9}{'垃圾率':>9}{'均返回':>8}")
    ref = None
    for gm, lbl in (("asis", "asis(dense_score=0)"), ("neutral:0.60", "0.60"),
                    ("neutral:0.99", "0.99(必然放行)")):
        r = metrics(th, skip_dense=True, gate_mode=gm)
        log(f"{lbl:<22}{r['h3ns']:7.1f}%{r['mrrns']:9.3f}{r['ind']:8.2f}%{r['nret']:8.2f}")
        ref = ref or r
        if abs(r["h3ns"] - ref["h3ns"]) > 1e-9 or abs(r["ind"] - ref["ind"]) > 1e-9:
            log("   ↑ 与 asis 不同 → 门槛是绑定的")
    log("   全部一致 ⇒ 后置过滤对泰文 top-5 非绑定，垃圾率变化不由它造成")

    ms139 = [g for g in gold if E.detect_lang(g["question"]) == "ms"]
    ms152 = [g for g in gold if g["q_lang"] == "ms"]
    log(f"\n(2) 马来文降权的生产口径 —— detect_lang=='ms' 共 {len(ms139)} 条"
        f"（标称 q_lang=='ms' 是 {len(ms152)} 条）:")
    log(f"{'队列/权重':<28}{'ns@1':>8}{'ns@3':>8}{'MRRns':>9}{'垃圾率':>9}")
    for name, items in (("生产口径 detect=ms(139)", ms139),
                        ("标称口径 q_lang=ms(152)", ms152)):
        for w in (1.0, 0.5):
            r = metrics(items, dense_weight=w)
            log(f"{name+f' w={w}':<28}{r['h1ns']:7.1f}%{r['h3ns']:7.1f}%"
                f"{r['mrrns']:9.3f}{r['ind']:8.2f}%")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "confirm":
    confirm()


# ==========================================================================
#  在新索引上重跑核心对照（python _probe_th_dense_weight.py final）
#  背景：th-normalize 于 2026-09-03 10:29:09 重建了索引。之前所有数字
#  （索引 mtime 2026-09-02 14:55:47）作废，这里在新索引上重取。
#  跑前跑后都打 mtime，若不一致则本批数字同样作废。
# ==========================================================================
def final():
    gold = E.load_gold()
    dp = BASE / "rag_llamaindex_storage" / "docstore.json"
    mt0 = dp.stat().st_mtime
    log(f"开跑 docstore mtime: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mt0))}")

    def metrics(items, **kw):
        rows = [E.evaluate_one(g, hybrid_retrieve_w(g["question"], top_k=5, **kw))
                for g in items]
        n = len(rows)
        pct = lambda k: sum(1 for r in rows if r.get(k)) / n * 100
        return {"h1": pct("hit@1"), "h3": pct("hit@3"), "h5": pct("hit@5"),
                "h1ns": pct("hit@1_ns"), "h3ns": pct("hit@3_ns"),
                "h5ns": pct("hit@5_ns"), "h3kb": pct("hit@3_kb"),
                "mrrns": statistics.mean(x["mrr_ns"] for x in rows),
                "empty": pct("empty_result"),
                "ind": statistics.mean(x["ind_defect_ratio"] for x in rows) * 100,
                "nret": statistics.mean(x["n_returned"] for x in rows),
                "rows": rows}

    th = [g for g in gold if E.detect_lang(g["question"]) == "th"]
    ms = [g for g in gold if E.detect_lang(g["question"]) == "ms"]
    log(f"泰文 {len(th)} 条 / 马来文 {len(ms)} 条（均按 detect_lang 取，生产口径）")
    validate_reimpl(th[:8])

    log(f"\n【泰文】{'变体':<14}{'raw@3':>8}{'ns@1':>8}{'ns@3':>8}{'ns@5':>8}"
        f"{'MRRns':>9}{'空':>7}{'垃圾率':>9}")
    tb = metrics(th, dense_weight=1.0)
    ts = metrics(th, skip_dense=True)
    t3 = metrics(th, dense_weight=0.3)
    for lbl, r in (("现状 w=1.0", tb), ("跳过 dense", ts), ("w=0.3", t3)):
        log(f"      {lbl:<14}{r['h3']:7.1f}%{r['h1ns']:7.1f}%{r['h3ns']:7.1f}%"
            f"{r['h5ns']:7.1f}%{r['mrrns']:9.3f}{r['empty']:6.1f}%{r['ind']:8.2f}%")
    lost = [g["qid"] for g, a, b in zip(th, tb["rows"], ts["rows"])
            if a.get("hit@3_ns") and not b.get("hit@3_ns")]
    gain = [g["qid"] for g, a, b in zip(th, tb["rows"], ts["rows"])
            if not a.get("hit@3_ns") and b.get("hit@3_ns")]
    log(f"      逐题(ns@3): 靠dense才命中 {len(lost)} 条 {lost[:12]}")
    log(f"      逐题(ns@3): 因dense失手 {len(gain)} 条 {gain[:12]}")

    log(f"\n【马来文】权重细扫 n={len(ms)}")
    log(f"{'w':>6}{'raw@3':>8}{'ns@1':>8}{'ns@3':>8}{'MRRns':>9}{'空':>7}{'垃圾率':>9}")
    sw = {}
    for w in (1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2):
        r = metrics(ms, dense_weight=w); sw[w] = r
        log(f"{w:>6.1f}{r['h3']:7.1f}%{r['h1ns']:7.1f}%{r['h3ns']:7.1f}%"
            f"{r['mrrns']:9.3f}{r['empty']:6.1f}%{r['ind']:8.2f}%")
    r0 = metrics(ms, skip_dense=True); sw["skip"] = r0
    log(f"{'skip':>6}{r0['h3']:7.1f}%{r0['h1ns']:7.1f}%{r0['h3ns']:7.1f}%"
        f"{r0['mrrns']:9.3f}{r0['empty']:6.1f}%{r0['ind']:8.2f}%")
    band = [sw[w]["h3ns"] for w in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)]
    base = sw[1.0]["h3ns"]
    log(f"\n[形状] w=1.0 基线 {base:.1f}%；0.2–0.7 六档 "
        f"{min(band):.1f}–{max(band):.1f}（跨度 {max(band)-min(band):.1f}pp）")
    log(f"       全档 ns@3: " + " ".join(f"{sw[w]['h3ns']:.1f}" for w in
                                         (1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)))
    log(f"       判定: {'平台' if max(band)-min(band) <= 3.0 else '尖峰'}"
        f"（六档跨度 {max(band)-min(band):.1f}pp）")

    mt1 = dp.stat().st_mtime
    log(f"\n跑完 docstore mtime: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mt1))}")
    log(f"索引在本批期间{'未变，数字有效' if mt1 == mt0 else '又被重建，本批作废！'}")
    (BASE / "_probe_final.json").write_text(json.dumps(
        {"index_mtime": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mt0)),
         "stable": mt1 == mt0, "n_th": len(th), "n_ms": len(ms),
         "th": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                for k, v in (("base", tb), ("skip", ts), ("w03", t3))},
         "th_lost_ns": lost, "th_gain_ns": gain,
         "ms": {str(k): {kk: vv for kk, vv in v.items() if kk != "rows"}
                for k, v in sw.items()}},
        ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "final":
    final()


# ==========================================================================
#  任务 2-4：净收益精算 + 配对显著性
#  python _probe_th_dense_weight.py netbenefit
#
#  两件事：
#  (1) 前面的权重扫描是「对该语种全部查询降权」，但生产里降权要靠
#      detect_lang 判语种，判错的那些拿不到降权。所以真实净收益要按
#      **逐题 detect_lang 决定是否降权** 来跑，这才是上线后实际会发生的。
#  (2) 「+4.6pp 是不是噪声」要用配对检验答，不能靠看曲线。
#      用 McNemar 精确检验（二项分布），只看两组结论不一致的题。
# ==========================================================================
def _mcnemar(rows_a, rows_b, key) -> tuple:
    """返回 (b_only, a_only, 双尾精确 p)。b_only=a错b对 的题数。"""
    from math import comb
    a_only = sum(1 for x, y in zip(rows_a, rows_b) if x.get(key) and not y.get(key))
    b_only = sum(1 for x, y in zip(rows_a, rows_b) if not x.get(key) and y.get(key))
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 1.0
    k = min(a_only, b_only)
    p = sum(comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
    return a_only, b_only, min(1.0, p)


def netbenefit():
    gold = E.load_gold()
    mt0 = (BASE / "rag_llamaindex_storage" / "docstore.json").stat().st_mtime
    log(f"索引 docstore mtime(开跑): "
        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mt0))}")

    def run(items, mode):
        """mode: 'base' 全 w=1.0；'gated' 按 detect_lang 决定降权。"""
        rows = []
        for g in items:
            if mode == "base":
                kw = {}
            else:
                d = E.detect_lang(g["question"])
                if d == "th":
                    kw = {"skip_dense": True}
                elif d == "ms":
                    kw = {"dense_weight": 0.4}
                else:
                    kw = {}
            rows.append(E.evaluate_one(g, hybrid_retrieve_w(g["question"], 5, **kw)))
        return rows

    def summ(rows):
        n = len(rows)
        return {
            "ns1": sum(1 for r in rows if r.get("hit@1_ns")) / n * 100,
            "ns3": sum(1 for r in rows if r.get("hit@3_ns")) / n * 100,
            "mrrns": statistics.mean(r["mrr_ns"] for r in rows),
            "raw3": sum(1 for r in rows if r.get("hit@3")) / n * 100,
            "ind": statistics.mean(r["ind_defect_ratio"] for r in rows) * 100,
            "empty": sum(1 for r in rows if r.get("empty_result")) / n * 100,
        }

    log("\n按 detect_lang 逐题决定降权（th→跳过, ms→w=0.4, 其余不动），"
        "这是上线后真实会发生的行为:")
    log(f"\n{'语种':<6}{'n':>5}{'变体':<8}{'ns@1':>8}{'ns@3':>8}{'MRRns':>9}"
        f"{'raw@3':>8}{'垃圾率':>8}{'空':>7}")
    out = {}
    # en/zh 不实跑：已用解析法确认 detect_lang 在 346 条 en / 610 条 zh 上
    # 没有任何一条被判成 th 或 ms（en→{en:109, zh:237}、zh→{zh:610}），
    # 故 gated 分支对它们逐题给出 kw={}、与 base 完全等价，跑了也只是烧 CPU。
    for lang in ("th", "ms"):
        items = [g for g in gold if g["q_lang"] == lang]
        b = run(items, "base")
        a = run(items, "gated")
        sb, sa = summ(b), summ(a)
        out[lang] = {"n": len(items), "base": sb, "gated": sa}
        for tag, s in (("base", sb), ("gated", sa)):
            log(f"{lang:<6}{len(items):>5}{tag:<8}{s['ns1']:7.1f}%{s['ns3']:7.1f}%"
                f"{s['mrrns']:9.3f}{s['raw3']:7.1f}%{s['ind']:7.2f}%{s['empty']:6.1f}%")
        for key, name in (("hit@3_ns", "ns@3"), ("hit@1_ns", "ns@1")):
            ao, bo, p = _mcnemar(b, a, key)
            out[lang][f"mcnemar_{key}"] = {"base_only": ao, "gated_only": bo, "p": p}
            log(f"       └ {name} McNemar: 只base对={ao} 只gated对={bo} "
                f"双尾p={p:.4f} {'显著(p<0.05)' if p < 0.05 else '不显著'}")
        log("")

    mt2 = (BASE / "rag_llamaindex_storage" / "docstore.json").stat().st_mtime
    log(f"索引 docstore mtime(跑完): "
        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mt2))}"
        f"  {'✓ 未变' if mt2 == mt0 else '✗ 索引被重建，本批作废！'}")
    p = BASE / "_probe_netbenefit.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"明细 → {p}")


_CMDS["netbenefit"] = netbenefit
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "netbenefit":
    netbenefit()
    sys.exit(0)


# ==========================================================================
#  任务 2-5：垃圾率归因的最后一步 —— 抬高 BM25 门槛能否收回那 0.87pp？
#  已证实 gate 里的 evidence_matches_query 对泰文恒依赖 dense_score>=0.55
#  （extract_query_terms 对泰文返回 []，词覆盖那套判据根本没参与），
#  所以跳过 dense 后垃圾率上升不可能是 gate 造成的，只能是 BM25 尾部候选
#  被顶进 top-5。若如此，唯一有效的杠杆是 BM25 分数下限。
#  python _probe_th_dense_weight.py bm25floor
# ==========================================================================
def bm25floor():
    gold = E.load_gold()
    items = [g for g in gold if g["q_lang"] == "th"]
    mt0 = (BASE / "rag_llamaindex_storage" / "docstore.json").stat().st_mtime
    log(f"索引 mtime(开跑): {time.strftime('%H:%M:%S', time.localtime(mt0))}")
    log(f"\n泰文 {len(items)} 条，跳过 dense，扫 BM25 分数下限:")
    log(f"{'bm25下限':>9}{'ns@1':>8}{'ns@3':>8}{'MRRns':>9}{'raw@3':>8}"
        f"{'垃圾率':>9}{'空':>7}{'均返回':>8}")
    base = None
    for floor in (1.2, 2.0, 3.0, 4.0, 6.0):
        rows = [E.evaluate_one(g, hybrid_retrieve_w(
            g["question"], 5, skip_dense=True, bm25_floor=floor)) for g in items]
        n = len(rows)
        pct = lambda k: sum(1 for r in rows if r.get(k)) / n * 100
        r = dict(ns1=pct("hit@1_ns"), ns3=pct("hit@3_ns"),
                 mrrns=statistics.mean(x["mrr_ns"] for x in rows),
                 raw3=pct("hit@3"), empty=pct("empty_result"),
                 ind=statistics.mean(x["ind_defect_ratio"] for x in rows) * 100,
                 nret=statistics.mean(x["n_returned"] for x in rows))
        if base is None:
            base = r
        log(f"{floor:>9.1f}{r['ns1']:7.1f}%{r['ns3']:7.1f}%{r['mrrns']:9.3f}"
            f"{r['raw3']:7.1f}%{r['ind']:8.2f}%{r['empty']:6.1f}%{r['nret']:8.2f}")
    mt2 = (BASE / "rag_llamaindex_storage" / "docstore.json").stat().st_mtime
    log(f"索引 mtime(跑完): {time.strftime('%H:%M:%S', time.localtime(mt2))}"
        f"  {'OK 未变' if mt2 == mt0 else '!! 索引被重建，本批作废'}")


_CMDS["bm25floor"] = bm25floor
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "bm25floor":
    bm25floor(); sys.exit(0)


# ==========================================================================
#  任务 2-6（补救）：用 known-item 口径重测「泰文跳过 dense」
#  背景：我把 memory/project_dense_weight_per_lang.md 整file覆盖写了，
#  很可能覆盖掉了别人用 known-item 口径重测出的 ~+3pp。而据
#  project_th_bilingual_concat_done.md，泰文 FAQ 金标结构性测不出知识库
#  检索收益、必须用 known-item —— 也就是说 known-item 的数比我的 ns@3
#  更可信。这里把它重测回来，而不是只道歉。
#  复用 _probe_th_knownitem.py 的数据集与判据，跳过 dense 用 monkey-patch
#  R._dense_retrieve→[] 实现（精确，无需重实现）。
#  python _probe_th_dense_weight_thdenseprobe.py ki
# ==========================================================================
def ki():
    import importlib
    KI = importlib.import_module("_probe_th_knownitem_thnormalize")  # 已被改名
    mt = (BASE / "rag_llamaindex_storage" / "docstore.json").stat().st_mtime
    log(f"索引 mtime(开跑): {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mt))}")
    items = KI.load_set("th")
    log(f"known-item 集(side=th): {len(items)} 条")

    orig = R._dense_retrieve

    def measure(skip: bool):
        if skip:
            R._dense_retrieve = lambda index, query, limit: []
        else:
            R._dense_retrieve = orig
        rows = []
        for it in items:
            try:
                ev = R.retrieve(it["query"], top_k=5)
            except Exception as exc:
                log(f"  [WARN] {it['kid']}: {exc}")
                ev = []
            texts = [R.normalize_thai(str(e.get("text") or "")) for e in ev]
            rows.append({"rank": E.hit_rank(it["anchors"], texts),
                         "n": len(ev)})
        R._dense_retrieve = orig
        n = len(rows)
        at = lambda k: sum(1 for r in rows if r["rank"] and r["rank"] <= k) / n * 100
        return {
            "h1": at(1), "h3": at(3), "h5": at(5),
            "mrr": statistics.mean(1.0 / r["rank"] if r["rank"] else 0.0
                                   for r in rows),
            "empty": sum(1 for r in rows if r["n"] == 0) / n * 100,
            "rows": rows,
        }

    base = measure(False)
    skip = measure(True)
    log(f"\n{'变体':<18}{'Hit@1':>8}{'Hit@3':>8}{'Hit@5':>8}{'MRR':>9}{'空':>7}")
    for lbl, r in (("现状 dense 参与", base), ("跳过 dense", skip)):
        log(f"{lbl:<18}{r['h1']:7.1f}%{r['h3']:7.1f}%{r['h5']:7.1f}%"
            f"{r['mrr']:9.4f}{r['empty']:6.1f}%")
    log(f"\n差值: Hit@1 {skip['h1']-base['h1']:+.1f}pp  "
        f"Hit@3 {skip['h3']-base['h3']:+.1f}pp  "
        f"Hit@5 {skip['h5']-base['h5']:+.1f}pp  MRR {skip['mrr']-base['mrr']:+.4f}")

    # McNemar（配对，只看结论不一致的题）
    from math import comb
    for k in (1, 3):
        a = [bool(r["rank"] and r["rank"] <= k) for r in base["rows"]]
        b = [bool(r["rank"] and r["rank"] <= k) for r in skip["rows"]]
        ao = sum(1 for x, y in zip(a, b) if x and not y)
        bo = sum(1 for x, y in zip(a, b) if not x and y)
        nn = ao + bo
        p = (sum(comb(nn, i) for i in range(min(ao, bo) + 1)) / 2 ** nn * 2
             if nn else 1.0)
        log(f"  Hit@{k} McNemar: 只现状对={ao} 只跳过对={bo} "
            f"双尾p={min(1.0,p):.4f} {'显著' if p < 0.05 else '不显著'}")

    mt2 = (BASE / "rag_llamaindex_storage" / "docstore.json").stat().st_mtime
    log(f"\n索引 mtime(跑完): {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mt2))}"
        f"  {'OK 未变' if mt2 == mt else '!! 被重建，作废'}")


_CMDS["ki"] = ki
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "ki":
    ki(); sys.exit(0)


# ==========================================================================
#  任务 2-7（复核 team-lead 的裁决）：+0.0pp 是基线臂被污染的假象
#  生产代码 1174 行已上线跳过分支（dense = [] if _is_thai_query(query)），
#  我此前 ki 子命令的「现状 dense 参与」臂直接调 R.retrieve()，跑的其实
#  也是跳过 dense —— 两臂都是跳过，当然逐题零差异。
#  正确方向：arm A = 生产现状（跳过）；arm B = patch _is_thai_query→False
#  让 dense 真正参与。唯一变量是那个门。
#  python _probe_th_dense_weight_thdenseprobe.py ki2
# ==========================================================================
def ki2():
    import importlib
    KI = importlib.import_module("_probe_th_knownitem_thnormalize")  # 已被改名
    mt = (BASE / "rag_llamaindex_storage" / "docstore.json").stat().st_mtime
    log(f"索引 mtime(开跑): {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mt))}")
    items = KI.load_set("th")
    log(f"known-item 集(side=th): {len(items)} 条；两臂唯一变量 = _is_thai_query 这个门")

    orig_gate = R._is_thai_query

    def measure(dense_on: bool):
        R._is_thai_query = (lambda q: False) if dense_on else orig_gate
        rows = []
        for it in items:
            try:
                ev = R.retrieve(it["query"], top_k=5)
            except Exception as exc:
                log(f"  [WARN] {it['kid']}: {exc}")
                ev = []
            texts = [R.normalize_thai(str(e.get("text") or "")) for e in ev]
            rows.append({"rank": E.hit_rank(it["anchors"], texts),
                         "n": len(ev)})
        R._is_thai_query = orig_gate
        n = len(rows)
        at = lambda k: sum(1 for r in rows if r["rank"] and r["rank"] <= k) / n * 100
        return {"h1": at(1), "h3": at(3), "h5": at(5),
                "mrr": statistics.mean(1.0 / r["rank"] if r["rank"] else 0.0
                                       for r in rows),
                "empty": sum(1 for r in rows if r["n"] == 0) / n * 100,
                "rows": rows}

    skip = measure(False)     # arm A：生产现状（跳过 dense）
    dense = measure(True)     # arm B：显式让 dense 参与（真基线）
    log(f"\n{'臂':<34}{'Hit@1':>8}{'Hit@3':>8}{'Hit@5':>8}{'MRR':>9}{'空':>7}")
    log(f"{'A 生产现状（跳过 dense）':<34}{skip['h1']:7.1f}%{skip['h3']:7.1f}%"
        f"{skip['h5']:7.1f}%{skip['mrr']:9.4f}{skip['empty']:6.1f}%")
    log(f"{'B dense 参与（真基线，patch门）':<34}{dense['h1']:7.1f}%{dense['h3']:7.1f}%"
        f"{dense['h5']:7.1f}%{dense['mrr']:9.4f}{dense['empty']:6.1f}%")
    log(f"\n跳过 dense 的真实收益: Hit@1 {skip['h1']-dense['h1']:+.1f}pp  "
        f"Hit@3 {skip['h3']-dense['h3']:+.1f}pp  Hit@5 {skip['h5']-dense['h5']:+.1f}pp  "
        f"MRR {skip['mrr']-dense['mrr']:+.4f}")

    from math import comb
    for k in (1, 3):
        a = [bool(r["rank"] and r["rank"] <= k) for r in dense["rows"]]
        b = [bool(r["rank"] and r["rank"] <= k) for r in skip["rows"]]
        ao = sum(1 for x, y in zip(a, b) if x and not y)   # 只基线对
        bo = sum(1 for x, y in zip(a, b) if not x and y)   # 只跳过对
        nn = ao + bo
        p = (sum(comb(nn, i) for i in range(min(ao, bo) + 1)) / 2 ** nn * 2
             if nn else 1.0)
        log(f"  Hit@{k} McNemar: 只基线对={ao} 只跳过对={bo} 双尾p={min(1.0,p):.4f} "
            f"{'显著' if p < 0.05 else '不显著'}")

    mt2 = (BASE / "rag_llamaindex_storage" / "docstore.json").stat().st_mtime
    log(f"\n索引 mtime(跑完): {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mt2))}"
        f"  {'OK 未变' if mt2 == mt else '!! 被重建，作废'}")


_CMDS["ki2"] = ki2
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "ki2":
    ki2(); sys.exit(0)

#!/usr/bin/env python3
# ==========================================================================
#  临 时 探 针 文 件 —— 用完即删，不属于项目正式代码
#  task #15「修泰文 dense 侧向量塌缩」的实测脚本
#
#  目的：客观对比「泰文查询转中/英文再跑 dense」的几种候选方案，
#        只测 dense 单通路（绕开 _hybrid_retrieve，避免 BM25 泰文
#        n-gram 的已有收益污染结论）。
#
#  约束：全程本地 CPU，不访问任何远端服务，不修改任何已有文件。
# ==========================================================================
from __future__ import annotations

import itertools
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

TH_RE = re.compile(r"[\u0e00-\u0e7f]")
ASCII_WORD = re.compile(r"^[\x20-\x7e]+$")


def log(*a):
    print(*a, flush=True)


# ─────────────────────── 术语表：动态读取，不硬编码 ───────────────────────
def load_glossary() -> Tuple[Dict[str, str], Dict[str, str]]:
    """返回 (泰文->中文, 中文->英文)。

    泰文->英文 术语表里没有直接给，通过「中文译名」做枢轴反查：
    同一概念的英文条目往往已存在于术语表（可能在别的分段），
    例如 ทุเรียน->榴莲 与 durian->榴莲 共享中文译名，故 ทุเรียน->durian。
    """
    g = json.loads((BASE / "rag_build" / "glossary.json").read_text(encoding="utf-8"))
    th2zh: Dict[str, str] = {}
    zh2en: Dict[str, str] = {}
    for seg, entries in g.items():
        if seg == "_meta" or not isinstance(entries, dict):
            continue
        for src, zh in entries.items():
            if src.startswith("_") or not isinstance(zh, str):
                continue
            if TH_RE.search(src):
                th2zh[src] = zh
            elif ASCII_WORD.match(src):
                # 同一中文译名可能对应多个英文写法，取最短的（最常规的那个）
                if zh not in zh2en or len(src) < len(zh2en[zh]):
                    zh2en[zh] = src
    return th2zh, zh2en


# ───────────────────── 占位符：预先实测存活率后选定 ─────────────────────
# Marian 的 sentencepiece 对 ZZ..ZZ 会吞字符，对 X<n>X 形式能原样透传。
def placeholder(i: int) -> str:
    return f"XT{i}X"


PH_RE = re.compile(r"XT(\d+)X")


def protect(text: str, th2zh: Dict[str, str]) -> Tuple[str, List[str]]:
    """把命中术语表的泰文词替换成占位符。长词优先，避免子串抢占。"""
    hits: List[str] = []
    out = text
    for th in sorted(th2zh, key=len, reverse=True):
        if th in out:
            out = out.replace(th, f" {placeholder(len(hits) + 1)} ")
            hits.append(th)
    return re.sub(r"\s+", " ", out).strip(), hits


def restore(text: str, hits: List[str], mapping: Dict[str, str]) -> str:
    """把占位符还原成 mapping[术语]（中文或英文写法）。"""
    def sub(m):
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(hits):
            return mapping.get(hits[idx], hits[idx])
        return " "
    out = PH_RE.sub(sub, text)
    return re.sub(r"\s+", " ", out).strip()


# ─────────────────────────────── 翻译器 ───────────────────────────────
class Translator:
    def __init__(self):
        import torch
        from transformers import MarianMTModel, MarianTokenizer
        self.torch = torch
        t0 = time.time()
        p = str(BASE / "models" / "opus-mt-th-en")
        self.tok = MarianTokenizer.from_pretrained(p)
        self.model = MarianMTModel.from_pretrained(p).eval()
        self.load_sec = time.time() - t0
        self.latencies: List[float] = []

    def __call__(self, text: str) -> str:
        t0 = time.time()
        batch = self.tok([text], return_tensors="pt", padding=True, truncation=True)
        with self.torch.no_grad():
            out = self.model.generate(**batch, max_new_tokens=128, num_beams=4)
        self.latencies.append((time.time() - t0) * 1000)
        return self.tok.decode(out[0], skip_special_tokens=True).strip()


# ────────────────────── 候选 D：纯术语表映射，零模型 ──────────────────────
def candidate_d(text: str, th2zh: Dict[str, str]) -> str:
    """命中术语替换为中文，其余泰文字符丢弃，保留 ASCII 数字/单位。"""
    out = text
    for th in sorted(th2zh, key=len, reverse=True):
        if th in out:
            out = out.replace(th, f" {th2zh[th]} ")
    out = TH_RE.sub(" ", out)                       # 剩余泰文全丢
    out = re.sub(r"[^\w\u4e00-\u9fff.%\-/]+", " ", out)
    return re.sub(r"\s+", " ", out).strip()


# ─────────────────────── Phase 0：塌缩判据 ───────────────────────
def phase0_collapse(gold: List[Dict[str, Any]]) -> None:
    log("\n" + "=" * 74)
    log("PHASE 0  前置问题：bge-small-zh-v1.5 对英文短查询是否塌缩？")
    log("=" * 74)
    from llama_index.core import Settings
    import rag_llamaindex as R
    R.setup_llamaindex()
    emb = Settings.embed_model

    def pick(lang: str, n: int = 8) -> List[str]:
        seen, out = set(), []
        for g in gold:
            if g["q_lang"] != lang:
                continue
            q = g["question"].strip()
            k = q[:12]
            if k in seen:
                continue
            seen.add(k)
            out.append(q)
            if len(out) >= n:
                break
        return out

    import math
    for lang in ("zh", "en", "th"):
        qs = pick(lang)
        vs = [emb.get_query_embedding(q) for q in qs]

        def cos(a, b):
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            return sum(x * y for x, y in zip(a, b)) / (na * nb + 1e-12)

        sims = [cos(vs[i], vs[j]) for i, j in itertools.combinations(range(len(vs)), 2)]
        avg_len = statistics.mean(len(q) for q in qs)
        log(f"\n[{lang}] n={len(qs)} 条短查询，均长 {avg_len:.0f} 字符，"
            f"两两组合 {len(sims)} 对")
        log(f"     min={min(sims):.4f}  median={statistics.median(sims):.4f}  "
            f"max={max(sims):.4f}  spread={max(sims) - min(sims):.4f}")
        log("     全部数值: " + " ".join(f"{s:.4f}" for s in sims))
        verdict = ("完全塌缩" if min(sims) > 0.999 else
                   "严重退化" if statistics.median(sims) > 0.90 else
                   "正常（有区分度）")
        log(f"     判定: {verdict}")
        for q in qs[:3]:
            log(f"       样例: {q[:60]}")


# ─────────────────────── Phase 1：术语覆盖率 ───────────────────────
def phase1_coverage(th_gold, th2zh) -> None:
    log("\n" + "=" * 74)
    log(f"PHASE 1  术语表覆盖率（当前泰文条目 {len(th2zh)} 条）")
    log("=" * 74)
    log("     泰文条目: " + " ".join(f"{k}->{v}" for k, v in th2zh.items()))
    hit_counts = []
    for g in th_gold:
        n = sum(1 for th in th2zh if th in g["question"])
        hit_counts.append(n)
    zero = sum(1 for n in hit_counts if n == 0)
    log(f"     139 条泰文查询中，至少命中 1 个术语: "
        f"{len(th_gold) - zero}/{len(th_gold)} = "
        f"{(len(th_gold) - zero) / len(th_gold) * 100:.1f}%")
    log(f"     零命中（候选 D 会退化成空串/纯噪声）: "
        f"{zero}/{len(th_gold)} = {zero / len(th_gold) * 100:.1f}%")
    from collections import Counter
    log(f"     命中术语个数分布: {dict(sorted(Counter(hit_counts).items()))}")
    per_term = {th: sum(1 for g in th_gold if th in g["question"]) for th in th2zh}
    log(f"     各术语命中查询数: {per_term}")


# ─────────────────────── Phase 2：生成各候选查询 ───────────────────────
def phase2_build_queries(th_gold, th2zh, zh2en):
    log("\n" + "=" * 74)
    log("PHASE 2  生成各候选的查询文本")
    log("=" * 74)
    tr = Translator()
    log(f"     opus-mt-th-en 加载耗时 {tr.load_sec:.1f}s")

    th2en = {th: zh2en.get(zh, zh) for th, zh in th2zh.items()}
    log(f"     泰文->英文映射（经中文译名反查）: {th2en}")

    variants: Dict[str, List[str]] = {k: [] for k in
                                      ("BASE", "A", "B", "C", "D", "E")}
    ph_survive = 0
    ph_total = 0
    for i, g in enumerate(th_gold, 1):
        q = g["question"].strip()
        variants["BASE"].append(q)
        variants["A"].append(tr(q))
        prot, hits = protect(q, th2zh)
        raw = tr(prot) if hits else variants["A"][-1]
        if hits:
            ph_total += len(hits)
            ph_survive += len(set(PH_RE.findall(raw)))
        variants["B"].append(restore(raw, hits, th2en))
        variants["C"].append(restore(raw, hits, th2zh))
        variants["D"].append(candidate_d(q, th2zh))
        # 候选 E（上界代理）：见报告说明
        variants["E"].append(_upper_bound_proxy(g))
        if i % 20 == 0:
            log(f"     ... {i}/{len(th_gold)}")

    log(f"     占位符存活率: {ph_survive}/{ph_total} = "
        f"{ph_survive / max(ph_total, 1) * 100:.1f}%")
    lat = tr.latencies
    log(f"     翻译推理延迟(CPU, beams=4, n={len(lat)}): "
        f"median={statistics.median(lat):.0f}ms  p90={sorted(lat)[int(len(lat)*.9)]:.0f}ms  "
        f"max={max(lat):.0f}ms")
    empty = {k: sum(1 for s in v if not s.strip()) for k, v in variants.items()}
    log(f"     各候选空查询数: {empty}")
    return variants, tr


def _upper_bound_proxy(g: Dict[str, Any]) -> str:
    """候选 E 上界代理：用金标答案正文的开头作查询。

    理由见报告——泰文金标的证据 chunk 是纯泰文 FAQ 问答块，库内不存在
    与之平行的中文/英文文本，因此无法构造「完美译文」型上界。改用
    同语种 oracle：把答案原文（与证据 chunk 同语同源）当查询，度量
    「dense 在泰文向量空间里最好能做到什么」。这是 A-D 无法逾越的天花板。
    """
    ans = str(g.get("gold_answer") or "")
    ans = re.sub(r"[【】\n]+", " ", ans)
    return ans.strip()[:220]


# ─────────────────────── Phase 3：dense-only 评测 ───────────────────────
def phase3_eval(variants, th_gold):
    log("\n" + "=" * 74)
    log("PHASE 3  dense-only 检索评测（不走 _hybrid_retrieve，无 BM25）")
    log("=" * 74)
    import rag_eval as E
    import rag_llamaindex as R
    t0 = time.time()
    index = R.load_index()
    log(f"     索引加载 {time.time() - t0:.1f}s")

    results: Dict[str, Dict[str, Any]] = {}
    detail: Dict[str, List[Optional[int]]] = {}
    for name in ("BASE", "A", "B", "C", "D", "E"):
        qs = variants[name]
        ranks: List[Optional[int]] = []
        empties = 0
        th_node_hits = 0
        top_total = 0
        lat = []
        for g, q in zip(th_gold, qs):
            if not q.strip():
                empties += 1
                ranks.append(None)
                continue
            t1 = time.time()
            items = R._dense_retrieve(index, q, 5)[:5]
            lat.append((time.time() - t1) * 1000)
            if not items:
                empties += 1
                ranks.append(None)
                continue
            texts = [it["text"] for it in items]
            # 统计 top-5 里有多少是泰文 node —— 跨语言鸿沟的直接证据
            for t in texts:
                top_total += 1
                if len(TH_RE.findall(t)) > 20:
                    th_node_hits += 1
            ranks.append(E.hit_rank(g["anchors"], texts))
        n = len(th_gold)
        results[name] = {
            "h1": sum(1 for r in ranks if r and r <= 1) / n * 100,
            "h3": sum(1 for r in ranks if r and r <= 3) / n * 100,
            "h5": sum(1 for r in ranks if r and r <= 5) / n * 100,
            "empty": empties / n * 100,
            "th_ratio": th_node_hits / max(top_total, 1) * 100,
            "lat": statistics.median(lat) if lat else 0.0,
        }
        detail[name] = ranks
        r = results[name]
        log(f"     {name:<5} Hit@1={r['h1']:5.1f}%  Hit@3={r['h3']:5.1f}%  "
            f"Hit@5={r['h5']:5.1f}%  空结果={r['empty']:5.1f}%  "
            f"top5泰文node占比={r['th_ratio']:5.1f}%  dense延迟={r['lat']:.0f}ms")
    return results, detail


# ────────────────────────────── 样例展示 ──────────────────────────────
def show_samples(variants, th_gold, k=4):
    log("\n" + "=" * 74)
    log("翻译样例")
    log("=" * 74)
    # 优先挑命中术语的，才能看出 B/C/D 与 A 的差别
    idxs = [i for i, q in enumerate(variants["D"]) if q.strip()][:k]
    while len(idxs) < k:
        for i in range(len(th_gold)):
            if i not in idxs:
                idxs.append(i)
                break
    for i in idxs[:k]:
        log(f"\n  [{th_gold[i]['qid']}]")
        log(f"    泰文原文 : {variants['BASE'][i]}")
        for name in ("A", "B", "C", "D"):
            log(f"    候选 {name}   : {variants[name][i]}")
        log(f"    候选 E   : {variants['E'][i][:120]}")


def main():
    import rag_eval as E
    gold = E.load_gold()
    th_gold = [g for g in gold if g["q_lang"] == "th"]
    log(f"泰文金标 {len(th_gold)} 条")

    th2zh, zh2en = load_glossary()
    phase0_collapse(gold)
    phase1_coverage(th_gold, th2zh)
    variants, tr = phase2_build_queries(th_gold, th2zh, zh2en)
    show_samples(variants, th_gold)
    results, detail = phase3_eval(variants, th_gold)

    log("\n" + "=" * 74)
    log("汇总表（dense-only，泰文金标 139 条）")
    log("=" * 74)
    log(f"{'候选':<6}{'Hit@1':>8}{'Hit@3':>8}{'Hit@5':>8}{'空结果率':>10}{'top5泰文占比':>13}")
    for name in ("BASE", "A", "B", "C", "D", "E"):
        r = results[name]
        log(f"{name:<6}{r['h1']:7.1f}%{r['h3']:7.1f}%{r['h5']:7.1f}%"
            f"{r['empty']:9.1f}%{r['th_ratio']:12.1f}%")

    out = BASE / "_probe_th_dense_result.json"
    out.write_text(json.dumps({
        "results": results,
        "ranks": {k: v for k, v in detail.items()},
        "translate_latency_ms": {
            "median": statistics.median(tr.latencies),
            "p90": sorted(tr.latencies)[int(len(tr.latencies) * .9)],
        },
        "variants": {k: v for k, v in variants.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n明细已写入 {out}")


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


# ==========================================================================
#  附加诊断（python _probe_th_dense.py extra）
#  主实验结论反常：所有翻译候选都比基线更差。这里定位根因，
#  区分两种可能——(1) 翻译质量不够 (2) 向量空间跨语言鸿沟。
# ==========================================================================
def extra_diagnostics():
    import math
    import rag_eval as E
    import rag_llamaindex as R
    from llama_index.core import Settings

    gold = E.load_gold()
    th_gold = [g for g in gold if g["q_lang"] == "th"]

    # ── 诊断 1：tokenizer 层面，泰文查询是否被压成同一 token 序列 ──
    log("\n" + "=" * 74)
    log("诊断 1  tokenizer 根因：bge vocab 对泰文字符的覆盖")
    log("=" * 74)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(BASE / "models" / "bge-small-zh-v1.5"))
    unk = tok.unk_token_id
    from collections import Counter
    sigs = Counter()
    unk_ratios = []
    for g in th_gold:
        ids = tok(g["question"], add_special_tokens=False)["input_ids"]
        if ids:
            unk_ratios.append(sum(1 for i in ids if i == unk) / len(ids))
        sigs[tuple(ids)] += 1
    log(f"     139 条泰文查询的 UNK token 占比: "
        f"median={statistics.median(unk_ratios)*100:.1f}%  "
        f"mean={statistics.mean(unk_ratios)*100:.1f}%  "
        f"max={max(unk_ratios)*100:.1f}%")
    full_unk = sum(1 for r in unk_ratios if r >= 0.999)
    log(f"     100% UNK（语义信息全丢，向量必然相同）: "
        f"{full_unk}/{len(th_gold)} = {full_unk/len(th_gold)*100:.1f}%")
    dup = sum(c for s, c in sigs.items() if c > 1)
    log(f"     token 序列完全重复的查询数: {dup}/{len(th_gold)} "
        f"（{len(sigs)} 个不同序列）")
    biggest = sigs.most_common(1)[0]
    log(f"     最大塌缩簇: {biggest[1]} 条查询共享同一 token 序列 "
        f"（长度 {len(biggest[0])}）")
    # 对照：中文/英文
    for lang in ("zh", "en"):
        rs = []
        for g in [x for x in gold if x["q_lang"] == lang][:139]:
            ids = tok(g["question"], add_special_tokens=False)["input_ids"]
            if ids:
                rs.append(sum(1 for i in ids if i == unk) / len(ids))
        log(f"     [对照 {lang}] UNK 占比 median={statistics.median(rs)*100:.1f}%")

    # ── 诊断 2：向量空间簇结构 —— 中文查询能否靠近泰文 node ──
    log("\n" + "=" * 74)
    log("诊断 2  跨语言向量鸿沟：泰文 node 簇 vs 中文 node 簇")
    log("=" * 74)
    R.setup_llamaindex()
    emb = Settings.embed_model
    d = json.loads((BASE / "rag_llamaindex_storage" / "docstore.json")
                   .read_text(encoding="utf-8"))
    th_texts, zh_texts = [], []
    for v in (d.get("docstore/data") or {}).values():
        dd = v.get("__data__") or v
        t = dd.get("text") or ""
        if len(TH_RE.findall(t)) > 20:
            th_texts.append(t)
        elif len(re.findall(r"[\u4e00-\u9fff]", t)) > 50:
            zh_texts.append(t)
    zh_texts = zh_texts[:150]
    log(f"     泰文 node {len(th_texts)} 条 / 中文 node 抽样 {len(zh_texts)} 条"
        f"（全库 {len(d.get('docstore/data') or {})} node）")

    def enc_many(texts):
        return [emb.get_text_embedding(t) for t in texts]

    t0 = time.time()
    th_vecs = enc_many(th_texts)
    zh_vecs = enc_many(zh_texts)
    log(f"     node 编码耗时 {time.time()-t0:.0f}s")

    def cos(a, b):
        na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
        return sum(x * y for x, y in zip(a, b)) / (na * nb + 1e-12)

    def centroid(vs):
        n = len(vs)
        return [sum(v[i] for v in vs) / n for i in range(len(vs[0]))]

    c_th, c_zh = centroid(th_vecs), centroid(zh_vecs)
    log(f"     泰文簇中心 vs 中文簇中心 余弦 = {cos(c_th, c_zh):.4f}")
    log(f"     泰文簇内两两余弦 median = "
        f"{statistics.median([cos(th_vecs[i], th_vecs[j]) for i, j in itertools.combinations(range(0, len(th_vecs), 7), 2)]):.4f}")
    log(f"     中文簇内两两余弦 median = "
        f"{statistics.median([cos(zh_vecs[i], zh_vecs[j]) for i, j in itertools.combinations(range(0, len(zh_vecs), 7), 2)]):.4f}")

    res = json.loads((BASE / "_probe_th_dense_result.json").read_text(encoding="utf-8"))
    log("\n     各候选查询到两簇的平均相似度（决定 top-k 里能否出现泰文 node）:")
    log(f"     {'候选':<6}{'→泰文node':>12}{'→中文node':>12}{'差值':>10}  倾向")
    for name in ("BASE", "A", "B", "C", "D", "E"):
        qs = [q for q in res["variants"][name] if q.strip()][:40]
        qv = [emb.get_query_embedding(q) for q in qs]
        s_th = statistics.mean(statistics.mean(cos(q, t) for t in th_vecs[::4]) for q in qv)
        s_zh = statistics.mean(statistics.mean(cos(q, t) for t in zh_vecs[::4]) for q in qv)
        log(f"     {name:<6}{s_th:>11.4f}{s_zh:>12.4f}{s_th - s_zh:>+10.4f}"
            f"  {'泰文' if s_th > s_zh else '中文'}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "extra":
    extra_diagnostics()


# ==========================================================================
#  验证 3（python _probe_th_dense.py verify）
#  主实验里 A-D 全部低于基线。有两种互斥解释，必须分开：
#    H1「翻译质量不够」—— 译文语义丢太多，dense 无从匹配；
#    H2「跨语言向量鸿沟」—— 证据 chunk 是泰文，译文查询离开了泰文簇，
#        译得再好也召不回。
#  判据：把「语料侧」也翻译成同一语言，做同语 N 选 1 检索。
#    若 英-英 显著优于 泰-泰 → H2 成立，方向应改为翻译语料侧；
#    若 英-英 也很差       → H1 成立，问题在翻译质量。
#  用同一个 opus-mt 译语料，翻译质量是受控变量，故差异只能归因于语言侧。
# ==========================================================================
def verify_corpus_side():
    import math
    import rag_eval as E
    from llama_index.core import Settings
    import rag_llamaindex as R

    gold = E.load_gold()
    th_gold = [g for g in gold if g["q_lang"] == "th"]
    d = json.loads((BASE / "rag_llamaindex_storage" / "docstore.json")
                   .read_text(encoding="utf-8"))
    th_nodes = []
    for v in (d.get("docstore/data") or {}).values():
        dd = v.get("__data__") or v
        t = dd.get("text") or ""
        if len(TH_RE.findall(t)) > 20:
            th_nodes.append(t)
    log(f"泰文 node 池: {len(th_nodes)} 条（这是泰文金标的全部证据来源）")

    # 为每条金标定位其目标 node（锚点匹配），只保留能唯一定位的
    pairs = []
    normed = [E.norm_for_match(t) for t in th_nodes]
    for g in th_gold:
        idxs = [i for i, n in enumerate(normed)
                if any(a in n for a in g["anchors"])]
        if len(idxs) == 1:
            pairs.append((g, idxs[0]))
    log(f"锚点可唯一定位目标 node 的金标: {len(pairs)}/{len(th_gold)} 条")
    pairs = pairs[:60]
    log(f"本次取前 {len(pairs)} 条做同语 N 选 1（N={len(th_nodes)}，全池干扰）")

    tr = Translator()

    def translate_long(text: str) -> str:
        """长泰文按 ~180 字符切块后逐块翻译再拼接（Marian 512 token 上限）。"""
        parts, cur = [], ""
        for seg in re.split(r"(?<=[\n\.\?！？。])|(?<= )", text):
            if len(cur) + len(seg) > 180:
                parts.append(cur); cur = seg
            else:
                cur += seg
        if cur.strip():
            parts.append(cur)
        return " ".join(tr(p) for p in parts if p.strip())[:2000]

    target_idx = sorted({i for _, i in pairs})
    cache_p = BASE / "_probe_en_corpus_cache.json"
    en_corpus: Dict[int, str] = {}
    if cache_p.exists():
        en_corpus = {int(k): v for k, v in
                     json.loads(cache_p.read_text(encoding="utf-8")).items()}
        log(f"语料侧英文译文命中磁盘缓存 {len(en_corpus)} 条")
    todo = [i for i in target_idx if i not in en_corpus]
    log(f"需翻译的目标 node: {len(todo)} 条 —— 语料侧翻译中（CPU，较慢）...")
    t0 = time.time()
    for k, i in enumerate(todo, 1):
        en_corpus[i] = translate_long(th_nodes[i])
        if k % 10 == 0:
            log(f"     ... {k}/{len(todo)}  已用 {time.time()-t0:.0f}s")
    if todo:
        cache_p.write_text(json.dumps({str(k): v for k, v in en_corpus.items()},
                                      ensure_ascii=False), encoding="utf-8")
    if todo:
        log(f"语料侧翻译完成，耗时 {time.time()-t0:.0f}s "
            f"（{len(todo)} 条，均 {(time.time()-t0)/len(todo):.1f}s/条）")

    R.setup_llamaindex()
    emb = Settings.embed_model

    def cos(a, b):
        na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
        return sum(x * y for x, y in zip(a, b)) / (na * nb + 1e-12)

    res = json.loads((BASE / "_probe_th_dense_result.json").read_text(encoding="utf-8"))
    qid2i = {g["qid"]: i for i, g in enumerate(th_gold)}

    pool = target_idx
    log(f"\n候选池 N={len(pool)} 条 node（只含已定位目标，是最有利于 dense 的窄池）")
    zh_vecs = None

    scenarios = [
        ("泰-泰   查询泰文原文 / 语料泰文原文", "BASE", lambda i: th_nodes[i]),
        ("英-泰   查询英文译文 / 语料泰文原文", "B", lambda i: th_nodes[i]),
        ("英-英   查询英文译文 / 语料英文译文", "B", lambda i: en_corpus[i]),
        ("泰-英   查询泰文原文 / 语料英文译文", "BASE", lambda i: en_corpus[i]),
        # 落地形态之争：index_text 若「译文+原文」拼接，可同时喂 dense 与
        # BM25，但泰文原文会把向量拉回泰文簇。这两行量化污染程度。
        ("英-拼接 查询英文译文 / 语料[英译+泰原]", "B",
         lambda i: en_corpus[i] + " " + th_nodes[i]),
        ("泰-拼接 查询泰文原文 / 语料[英译+泰原]", "BASE",
         lambda i: en_corpus[i] + " " + th_nodes[i]),
    ]
    corpus_cache: Dict[int, List[float]] = {}
    log(f"\n{'场景':<38}{'Top1':>8}{'Top3':>8}{'MRR':>8}")
    for label, qsrc, getter in scenarios:
        vecs = []
        for i in pool:
            key = (id(getter), i)
            vecs.append(emb.get_text_embedding(getter(i)))
        top1 = top3 = 0
        mrr = 0.0
        for g, ti in pairs:
            q = res["variants"][qsrc][qid2i[g["qid"]]]
            if not q.strip():
                continue
            qv = emb.get_query_embedding(q)
            sims = sorted(((cos(qv, vecs[k]), pool[k]) for k in range(len(pool))),
                          reverse=True)
            rank = next((r for r, (_, i) in enumerate(sims, 1) if i == ti), None)
            if rank:
                mrr += 1.0 / rank
                top1 += rank <= 1
                top3 += rank <= 3
        n = len(pairs)
        log(f"{label:<38}{top1/n*100:7.1f}%{top3/n*100:7.1f}%{mrr/n:8.3f}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "verify":
    verify_corpus_side()


# ==========================================================================
#  验证 4（python _probe_th_dense.py bm25risk）
#  我推荐的落地形态是 index_text =「中文译文 + 泰文原文」拼接。dense 侧已验证
#  不受污染，但 BM25 侧有个未验证风险：拼接使 index_text 变长，BM25 的长度
#  归一化(b)会压低这些 node 的得分，可能损害泰文 n-gram 已有的 94.2% Hit@3。
#
#  可以精确模拟而无需真的翻译+重建索引：中文译文**不含泰文字符**，对泰文
#  n-gram 的 tf 贡献恒为 0，只贡献 length。所以只需把这 146 条 node 的
#  length 按实测比率放大，其余 node 不动，再跑真实的 _bm25_retrieve。
#  实测比率：中文译文/泰文原文 = 0.378（651 条已译 chunk，p10 0.316 / p90 0.456）
# ==========================================================================
def bm25_dilution_risk():
    import rag_eval as E
    import rag_llamaindex as R

    gold = E.load_gold()
    th_gold = [g for g in gold if g["q_lang"] == "th"]
    index = R.load_index()
    nodes = R._build_lexical_node_cache(index)
    log(f"全库 lexical node: {len(nodes)}")

    th_idx = [i for i, n in enumerate(nodes)
              if len(TH_RE.findall(n.get("text") or "")) > 20]
    log(f"其中泰文 node: {len(th_idx)} 条（将被拼接改造的就是这批）")
    orig_len = {i: nodes[i]["length"] for i in th_idx}

    def run(tag: str) -> Dict[str, float]:
        ranks = []
        for g in th_gold:
            items = R._bm25_retrieve(index, g["question"], 5)[:5]
            ranks.append(E.hit_rank(g["anchors"], [it["text"] for it in items]))
        n = len(th_gold)
        r = {
            "h1": sum(1 for x in ranks if x and x <= 1) / n * 100,
            "h3": sum(1 for x in ranks if x and x <= 3) / n * 100,
            "h5": sum(1 for x in ranks if x and x <= 5) / n * 100,
        }
        log(f"     {tag:<44} Hit@1={r['h1']:5.1f}%  Hit@3={r['h3']:5.1f}%  "
            f"Hit@5={r['h5']:5.1f}%")
        return r

    log("\nBM25-only（泰文金标 139 条，真实 _bm25_retrieve，含别名扩展）:")
    base = run("现状 index_text=泰文原文")
    # 三档：p10 / median / p90 的译文长度比率，看敏感度
    out = {}
    for ratio, label in ((0.316, "p10"), (0.378, "median"), (0.456, "p90")):
        for i in th_idx:
            nodes[i]["length"] = max(1, int(orig_len[i] * (1.0 + ratio)))
        out[label] = run(f"拼接后 length×{1+ratio:.3f}（译文比率 {label}）")
    # 极端对照：若译文长度与原文相当（劣质译文膨胀）
    for i in th_idx:
        nodes[i]["length"] = max(1, int(orig_len[i] * 2.0))
    out["x2"] = run("极端对照 length×2.000（译文与原文等长）")
    for i in th_idx:
        nodes[i]["length"] = orig_len[i]

    log("\n结论:")
    d = out["median"]["h3"] - base["h3"]
    log(f"     中位比率下 Hit@3 变化: {base['h3']:.1f}% → {out['median']['h3']:.1f}% "
        f"（{d:+.1f} 个百分点）")
    log(f"     极端 ×2 下 Hit@3 变化: {base['h3']:.1f}% → {out['x2']['h3']:.1f}% "
        f"（{out['x2']['h3'] - base['h3']:+.1f} 个百分点）")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "bm25risk":
    bm25_dilution_risk()

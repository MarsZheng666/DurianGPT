#!/usr/bin/env python3
# ==========================================================================
#  临 时 探 针 —— owner: th-dense-probe（用完即删）
#  问题：dense 通路的辨识力是否随 chunk 变长而稀释？
#
#  为什么需要单独测：glossary-th 的剂量曲线（0-400 Hit@5 96.1% →
#  700-900 65.5%）跑在**完整混合链路**上，BM25 的长度归一化(b) 也会
#  贡献一部分塌陷，所以它答不了"dense 侧塌多少"。必须拆通路。
#
#  为什么这事关任务 #15：#15 要把泰文 index_text 拼成「中文译文+泰文原文」，
#  长度 687 → 约 946（译文/原文长度比实测 median 0.378）。而 #15 的**全部
#  价值都在 dense 侧**。若长度把 dense 稀释掉，收益模型就得重算。
#
#  设计（受控，隔离"长度稀释"这一个变量）：
#    core   = 某中文 chunk 的前 150 字符（**查询相关内容，全程固定不变**）
#    filler = 追加的无关文本，只有它的长度在变
#    query  = core 本身（known-item 探针；绝对难度不重要，
#             因为我们比较的是**同一查询在不同 chunk 长度下**的排名变化）
#    指标   = 在 M 条候选里找回自己的 Top1/Top3/MRR
#
#  两种 filler 是本实验的关键对照：
#    filler_zh —— 中文无关文本：真实语义稀释
#    filler_th —— 泰文文本：模拟 #15 拼接的实际形态。泰文对 bge 基本是
#                 UNK，若它无害而中文 filler 有害，就说明拼接之所以安全
#                 恰恰因为泰文在 tokenizer 眼里近乎不存在。
#
#  用中文测而不用泰文测：泰文 dense 本身已塌缩（UNK 中位 66.7%、139 条
#  查询只有 13 个不同 token 序列），在泰文上测不出长度效应。#15 拼接后
#  真正承载语义的是**中文那半**，所以中文才是正确的被试语言。
#
#  只读：不改任何已有文件、不重建索引、不访问远端。
# ==========================================================================
from __future__ import annotations

import json
import math
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

TH_RE = re.compile(r"[\u0e00-\u0e7f]")
ZH_RE = re.compile(r"[\u4e00-\u9fff]")
M = 400                      # 候选池大小（加严：竞争者更多）
CORE = 150                   # 固定的"有效信号"长度
EXTRAS = [0, 150, 400, 800, 1200]


def log(*a):
    print(*a, flush=True)


def cos(a, b):
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb + 1e-12)


def main():
    mt0 = (BASE / "rag_llamaindex_storage" / "docstore.json").stat().st_mtime
    log(f"索引 docstore mtime(开跑): "
        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mt0))}")

    d = json.loads((BASE / "rag_llamaindex_storage" / "docstore.json")
                   .read_text(encoding="utf-8"))
    zh_pool, th_pool = [], []
    for v in (d.get("docstore/data") or {}).values():
        dd = v.get("__data__") or v
        t = (dd.get("text") or "").strip()
        if not t:
            continue
        if len(TH_RE.findall(t)) > 20:
            th_pool.append(t)
        elif len(ZH_RE.findall(t)) > 120 and len(t) >= CORE + 60:
            zh_pool.append(t)
    log(f"中文候选 {len(zh_pool)} 条 / 泰文 filler 源 {len(th_pool)} 条")

    rng = random.Random(20260903)
    rng.shuffle(zh_pool)
    docs = zh_pool[:M]
    if len(docs) < M:
        raise SystemExit(f"中文候选不足 {M}")

    cores = [t[:CORE] for t in docs]
    # filler 取自**其它** chunk，保证与 core 语义无关
    zh_filler_src = zh_pool[M:M + M] or zh_pool[:M]
    th_filler_src = th_pool

    import rag_llamaindex as R
    from llama_index.core import Settings
    R.setup_llamaindex()
    emb = Settings.embed_model

    log(f"\n对 {M} 条中文 core 编码查询向量（查询=core 前 60 字符，比用 core 全文更接近真实短查询工况，也更难）...")
    QLEN = 60
    qv = [emb.get_query_embedding(c[:QLEN]) for c in cores]

    def build(i: int, extra: int, kind: str) -> str:
        if extra == 0:
            return cores[i]
        if kind == "zh":
            src = zh_filler_src[(i + 7) % len(zh_filler_src)]
        else:
            src = th_filler_src[(i + 7) % len(th_filler_src)]
        f = (src * (extra // max(len(src), 1) + 1))[:extra]
        return cores[i] + " " + f

    results: Dict[str, Dict[int, dict]] = {"zh": {}, "th": {}}
    for kind, label in (("zh", "中文 filler（真实语义稀释）"),
                        ("th", "泰文 filler（模拟 #15 拼接）")):
        log(f"\n=== filler 类型：{label} ===")
        log(f"{'追加长度':>9}{'总长':>7}{'信号占比':>9}"
            f"{'Top1':>8}{'Top3':>8}{'MRR':>8}")
        for extra in EXTRAS:
            texts = [build(i, extra, kind) for i in range(M)]
            vecs = [emb.get_text_embedding(t) for t in texts]
            top1 = top3 = 0
            mrr = 0.0
            for i in range(M):
                sims = sorted(range(M), key=lambda j: cos(qv[i], vecs[j]),
                              reverse=True)
                rank = sims.index(i) + 1
                mrr += 1.0 / rank
                top1 += rank <= 1
                top3 += rank <= 3
            total = CORE + (extra + 1 if extra else 0)
            r = {"top1": top1 / M * 100, "top3": top3 / M * 100,
                 "mrr": mrr / M, "total": total,
                 "signal": CORE / total * 100}
            results[kind][extra] = r
            log(f"{extra:>9}{total:>7}{r['signal']:8.1f}%"
                f"{r['top1']:7.1f}%{r['top3']:7.1f}%{r['mrr']:8.3f}")

    log("\n" + "=" * 70)
    log("结论")
    log("=" * 70)
    for kind, label in (("zh", "中文 filler"), ("th", "泰文 filler")):
        a = results[kind][0]
        b = results[kind][EXTRAS[-1]]
        log(f"{label}: Top1 {a['top1']:.1f}% → {b['top1']:.1f}% "
            f"({b['top1'] - a['top1']:+.1f}pp)   "
            f"MRR {a['mrr']:.3f} → {b['mrr']:.3f}")

    # #15 的实际工况：中文译文约 260 字符 + 泰文原文约 687 字符
    log("\n#15 实际工况定位：拼接后总长约 946，中文译文约 260（信号占比约 27%）")
    log("  → 对照上表 filler_th 在 追加 800（总长 950、信号占比 15.8%）那一行")

    mt2 = (BASE / "rag_llamaindex_storage" / "docstore.json").stat().st_mtime
    log(f"\n索引 docstore mtime(跑完): "
        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mt2))}"
        f"  {'✓ 未变' if mt2 == mt0 else '✗ 索引被重建，本批作废！'}")
    out = BASE / "_probe_lendilute_hard_thdenseprobe.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    log(f"明细 → {out}")


if __name__ == "__main__":
    main()

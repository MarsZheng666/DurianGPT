"""临时诊断脚本（可删）：马来文 Hit@1 偏低的根因分析。

只读，不修改任何项目文件。走 rag_eval 实际调用的检索链路 R.retrieve()，
命中判定复用 rag_eval.hit_rank / norm_for_match，保证口径可比。
"""
import json
import re
import sys
import collections

sys.path.insert(0, '.')

import rag_llamaindex as R
import rag_eval as E

TOP_K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
OUT = sys.argv[2] if len(sys.argv) > 2 else f'/tmp/ms_probe_top{TOP_K}.json'

gold = [json.loads(l) for l in open('rag_eval/gold.jsonl')]
ms = [g for g in gold if g.get('q_lang') == 'ms']

RE_TH = re.compile(r'[\u0e00-\u0e7f]')
RE_CN = re.compile(r'[\u4e00-\u9fff]')
RE_LAT = re.compile(r'[A-Za-z]')

MS_STOP = {'yang', 'dan', 'untuk', 'pada', 'dengan', 'adalah', 'ini', 'itu',
           'atau', 'tidak', 'dari', 'akan', 'boleh', 'dalam', 'juga'}


def lang_of(t: str) -> str:
    if not t:
        return 'empty'
    if len(RE_TH.findall(t)) > 0.15 * len(t):
        return 'th'
    if len(RE_CN.findall(t)) > 0.15 * len(t):
        return 'zh'
    if RE_LAT.search(t):
        low = t.lower()
        hits = sum(1 for w in MS_STOP if re.search(r'\b' + w + r'\b', low))
        return 'ms' if hits >= 2 else 'en'
    return 'other'


rows = []
for i, g in enumerate(ms, 1):
    q = g['question']
    try:
        ev = R.retrieve(q, top_k=TOP_K)
    except Exception as exc:
        print(f'  [WARN] {g["qid"]}: {exc}')
        ev = []
    texts = [str(e.get('text') or e.get('display_text') or '') for e in ev]
    rank = E.hit_rank(g['anchors'], texts)
    cand = []
    for j, (t, e) in enumerate(zip(texts, ev), 1):
        m = e.get('metadata') or {}
        n = E.norm_for_match(t)
        cand.append({
            'rank': j,
            'lang': lang_of(t),
            'doc': m.get('doc') or m.get('source_file') or '?',
            'block_type': m.get('block_type') or e.get('block_type') or '?',
            'provenance': m.get('provenance') or e.get('provenance') or '?',
            'is_qa': bool(E._is_qa_chunk(e)),
            'anchor_hit': any(a in n for a in g['anchors']),
            'self_hit': bool(E._is_self_hit(q, t)),
            'preview': t[:160],
        })
    rows.append({
        'qid': g['qid'], 'question': q, 'category': g['category'],
        'anchors': g['anchors'], 'rank': rank, 'cand': cand,
    })
    if i % 25 == 0:
        print(f'  ...{i}/{len(ms)}')

json.dump(rows, open(OUT, 'w'), ensure_ascii=False)

n = len(rows)
for k in (1, 3, 5):
    h = sum(1 for r in rows if r['rank'] and r['rank'] <= k)
    print(f'Hit@{k}: {h}/{n} = {h/n*100:.1f}%')
mrr = sum(1.0 / r['rank'] for r in rows if r['rank']) / n
print(f'MRR: {mrr:.4f}')
print(f'空结果: {sum(1 for r in rows if not r["cand"])}/{n}')
print(f'写入 {OUT}')

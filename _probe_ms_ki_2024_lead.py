"""临时脚本（可删）：马来文文档类 known-item 评测集构造 + 实测。

只读。不修改 rag_eval/gold.jsonl，不修改任何代码文件。
命中判定复用 rag_eval.hit_rank / norm_for_match / anchor_is_junk，
检索走 rag_llamaindex.retrieve()，与评估体系同口径。
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, '.')

import rag_llamaindex as R
import rag_eval as E

DOCS = {
    'MY_DOA_Pakej_Teknologi_Durian_2024.pdf': 'pdf2024',
    
}
WINDOW = 24
MAX_ANCHORS = 8
MIN_LEN = 220

MS_WORDS = set(('yang dan untuk pada dengan adalah ini itu atau tidak dari akan '
                'boleh dalam juga pokok buah daun baja tanaman perlu ialah serta '
                'oleh dapat lebih bagi kepada telah').split())


def is_malay(t):
    low = t.lower()
    return sum(1 for w in MS_WORDS if re.search(r'\b' + w + r'\b', low)) >= 4


def pick_query(first_half):
    """从前半取一段实词密集的连续片段（40-80 字符）。

    必须从词边界开始/结束 —— chunk 本身常从半个词切起（如 "batan Pertanian"），
    若不约束会造出 "kok durian" 这种残词查询，对检索不公平。
    """
    best, best_score = None, -1.0
    for m in re.finditer(r'(?<![A-Za-z])[A-Za-z][A-Za-z0-9 ,\.\-%/()]{45,79}(?![A-Za-z])',
                         first_half):
        s = m.group(0).strip()
        if len(s) < 40:
            continue
        letters = sum(1 for c in s if c.isalpha())
        digits = sum(1 for c in s if c.isdigit())
        if letters / len(s) < 0.72 or digits / len(s) > 0.12:
            continue
        if s.count('.') > 6 or '....' in s:      # 目录点导线
            continue
        words = [w for w in re.findall(r'[A-Za-z]+', s) if len(w) > 2]
        if len(words) < 6:
            continue
        score = letters / len(s) + 0.02 * len(words)
        if is_malay(s):
            score += 0.5
        if score > best_score:
            best, best_score = s, score
    return best


def make_anchors(second_half):
    """从后半不同位置取 24 字符归一化窗口。"""
    n = E.norm_for_match(second_half)
    if len(n) < WINDOW * 2:
        return []
    span = len(n) - WINDOW
    step = max(1, span // MAX_ANCHORS)
    out, seen = [], set()
    for i in range(0, span + 1, step):
        a = n[i:i + WINDOW]
        if len(a) < WINDOW or a in seen:
            continue
        seen.add(a)
        out.append(a)
        if len(out) >= MAX_ANCHORS:
            break
    return out


def main():
    ds = 'rag_llamaindex_storage/docstore.json'
    print(f'[index] docstore mtime (before) = {time.ctime(os.path.getmtime(ds))}')

    rows = [json.loads(l) for l in open('rag_build/chunks.jsonl')]
    pool = [r for r in rows if (r.get('doc') or '') in DOCS]
    print(f'[pool] 两份解析共 {len(pool)} chunk')

    stat = {'too_short': 0, 'not_malay': 0, 'no_query': 0, 'no_anchor': 0,
            'junk_all': 0, 'table': 0}
    items = []
    for r in pool:
        t = r.get('index_text') or ''
        if r.get('block_type') == 'table':
            stat['table'] += 1
            continue
        if len(t) < MIN_LEN:
            stat['too_short'] += 1
            continue
        if not is_malay(t):
            stat['not_malay'] += 1
            continue
        mid = len(t) // 2
        q = pick_query(t[:mid])
        if not q:
            stat['no_query'] += 1
            continue
        anchors = make_anchors(t[mid:])
        if not anchors:
            stat['no_anchor'] += 1
            continue
        kept = [a for a in anchors if not E.anchor_is_junk(a)]
        n_junk = len(anchors) - len(kept)
        if not kept:
            stat['junk_all'] += 1
            continue
        # 查询与锚点不重叠（查询取自前半、锚点取自后半，再显式校验）
        qn = E.norm_for_match(q)
        kept = [a for a in kept if a not in qn]
        if not kept:
            stat['junk_all'] += 1
            continue
        items.append({
            'qid': f"ms_ki_{len(items):04d}",
            'doc': r.get('doc'), 'parse': DOCS[r.get('doc')],
            'page': r.get('page'), 'chunk_id': r.get('id'),
            'question': q, 'anchors': kept, 'n_junk_dropped': n_junk,
            'src_len': len(t),
        })
    print(f'[build] 构造 {len(items)} 条；剔除统计 {stat}')
    print(f"[build] anchor_is_junk 共剔除锚点 {sum(i['n_junk_dropped'] for i in items)} 个"
          f"（另有 {stat['junk_all']} 条题因锚点全被剔或与查询重叠而丢弃）")

    # ── 可达性审计 ──
    index = R.load_index()
    cache = R._build_lexical_node_cache(index)
    corpus = [E.norm_for_match(c.get('text') or '') for c in cache]
    print(f'[audit] 索引内 node {len(cache)} 个')
    reachable, unreachable = [], []
    for it in items:
        ok = [a for a in it['anchors'] if any(a in c for c in corpus)]
        if ok:
            it['anchors'] = ok
            reachable.append(it)
        else:
            unreachable.append(it)
    print(f'[audit] 可达 {len(reachable)}；不可达剔除 {len(unreachable)}')

    json.dump(reachable, open('/tmp/lead_ms_ki_2024_set.json', 'w'), ensure_ascii=False)

    # ── 实测 ──
    res = []
    for i, it in enumerate(reachable, 1):
        try:
            ev = R.retrieve(it['question'], top_k=5)
        except Exception as exc:
            print(f"  [WARN] {it['qid']}: {exc}")
            ev = []
        texts = [str(e.get('text') or e.get('display_text') or '') for e in ev]
        rank = E.hit_rank(it['anchors'], texts)
        res.append({**it, 'rank': rank, 'n_ev': len(ev),
                    'top': [{'doc': (e.get('metadata') or {}).get('doc'),
                             'prev': t[:110]} for e, t in zip(ev, texts)][:5]})
        if i % 50 == 0:
            print(f'  ...{i}/{len(reachable)}')

    json.dump(res, open('/tmp/lead_ms_ki_2024_result.json', 'w'), ensure_ascii=False)
    n = len(res)
    print(f'\n===== 马来文文档类 known-item 实测 (n={n}) =====')
    for k in (1, 3, 5):
        h = sum(1 for r in res if r['rank'] and r['rank'] <= k)
        print(f'  Hit@{k}: {h}/{n} = {h/n*100:.1f}%')
    print(f"  MRR: {sum(1.0/r['rank'] for r in res if r['rank'])/n:.4f}")
    print(f"  空结果率: {sum(1 for r in res if r['n_ev']==0)/n*100:.1f}%")
    for p in set(DOCS.values()):
        sub = [r for r in res if r['parse'] == p]
        if sub:
            h1 = sum(1 for r in sub if r['rank'] == 1)
            h5 = sum(1 for r in sub if r['rank'] and r['rank'] <= 5)
            print(f'  [{p}] n={len(sub)} Hit@1 {h1/len(sub)*100:.1f}% Hit@5 {h5/len(sub)*100:.1f}%')
    print(f'[index] docstore mtime (after) = {time.ctime(os.path.getmtime(ds))}')


if __name__ == '__main__':
    main()

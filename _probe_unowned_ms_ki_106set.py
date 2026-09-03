# ⚠️ 归属已确认：glossary-th 本人所建（会话日志取证；其否认系上下文摘要在 10:44-10:51 窗口丢失所致）
"""临时脚本（可删）— 归属 glossary-th。马来文文档类 known-item 构造+实测。
唯一文件名/输出路径，避免与其他 agent 的同名脚本冲突。只读，不改项目文件。
"""
import json, re, sys, collections
sys.path.insert(0,'.')
import rag_llamaindex as R, rag_eval as E

DOC_KEY='MY_DOA_Pakej_Teknologi_Durian'; WINDOW=24; TOP_K=5
chunks=[json.loads(l) for l in open('rag_build/chunks.jsonl')]
docs=[c for c in chunks if DOC_KEY in str(c.get('doc') or '') or DOC_KEY in str(c.get('source_file') or '')]
print(f'目标文档 chunk: {len(docs)}')
_RUN=re.compile(r'[A-Za-z][A-Za-z0-9\-\'/,\.\s%°]{30,}')
def pick_query(head):
    best=None
    for m in _RUN.finditer(head):
        s=m.group(0).strip()
        if len(s)<40: continue
        seg=s[:80]
        if sum(1 for c in seg if c.isalpha())/len(seg)<0.65: continue
        if sum(1 for c in seg if c.isdigit())/len(seg)>0.15: continue
        if len([w for w in re.findall(r'[A-Za-z]+',seg) if len(w)>2])<6: continue
        if best is None or len(seg)>len(best): best=seg
    return best
items=[]; skip=collections.Counter()
for c in docs:
    t=E.strip_parser_residue(c.get('index_text') or '')
    if len(t)<220: skip['太短']+=1; continue
    mid=len(t)//2
    q=pick_query(t[:mid])
    if not q: skip['前半无合格查询']+=1; continue
    a=[x for x in E.extract_anchors(t[mid:],window=WINDOW,max_anchors=8) if not E.anchor_is_junk(x)]
    nq=E.norm_for_match(q); a=[x for x in a if x not in nq]
    if len(a)<4: skip['后半锚点不足4个']+=1; continue
    items.append({'qid':f'gth_ms_{len(items):04d}','doc_id':c.get('id'),'page':c.get('page'),
                  'question':q,'anchors':a})
print(f'构造 {len(items)} 条; 剔除 {dict(skip)}')
cache=R._build_lexical_node_cache(R.load_index())
blob='\n'.join(E.norm_for_match(n['text']) for n in cache)
reach=[]; nreach=0
for it in items:
    ok=[a for a in it['anchors'] if a in blob]
    if len(ok)>=4: it['anchors']=ok; reach.append(it)
    else: nreach+=1
print(f'可达性审计: 保留 {len(reach)}, 剔除 {nreach}')
def tok(s): return set(w for w in re.findall(r'[a-z]{4,}',(s or '').lower()))
out=[]
for i,it in enumerate(reach,1):
    ev=R.retrieve(it['question'],top_k=TOP_K)
    texts=[str(e.get('text') or e.get('display_text') or '') for e in ev]
    nq=E.norm_for_match(it['question']); tq=tok(it['question'])
    strict=E.hit_rank(it['anchors'],texts); loose=None; fuzzy=None
    for j,t in enumerate(texts,1):
        nt=E.norm_for_match(t)
        if loose is None and (any(x in nt for x in it['anchors']) or (len(nq)>=30 and nq[:30] in nt)): loose=j
        if fuzzy is None and tq and len(tq&tok(t))/len(tq)>=0.75: fuzzy=j
    out.append({**it,'strict':strict,'loose':loose,'fuzzy':fuzzy,
                'docs':[(e.get('metadata') or {}).get('doc','?') for e in ev],
                'previews':[t[:200] for t in texts]})
    if i%50==0: print(f'  ...{i}/{len(reach)}')
json.dump(out,open('/tmp/gth_ms_ki.json','w'),ensure_ascii=False)
n=len(out)
print(f"\n=== glossary-th 口径 (n={n}) ===")
print(f"{'口径':<38}{'Hit@1':>8}{'Hit@3':>8}{'Hit@5':>8}{'MRR':>9}")
for name,k in [('A 严格锚点','strict'),('B 段落级逐字','loose'),('C 段落级模糊(容OCR)','fuzzy')]:
    rk=[r[k] for r in out]
    h=[sum(1 for x in rk if x and x<=q)/n for q in (1,3,5)]
    print(f"{name:<34}{h[0]*100:>8.1f}%{h[1]*100:>7.1f}%{h[2]*100:>7.1f}%{sum(1/x for x in rk if x)/n:>9.4f}")
print('空结果:', sum(1 for r in out if not r['docs']))

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""人工/助手翻译的导入工具。

用途：不经由远端 LLM，把人工（或助手在对话中）产出的译文写入
`rag_build/translations.jsonl`，与 rag_translate.py 共用同一套缓存格式与校验。

为什么单独做这个：远端 vLLM 走 tunnel 转发，调用它等于占用远端算力，
已明确禁止（见 memory: 永不改动远端）。因此需要一条不碰远端的导入通道。

用法：
    # 1) 导出待译批次（含原文与 key）
    python rag_translate_manual.py export --lang th --limit 20 --out batch.json

    # 2) 人工/助手填好 translation 字段后导入（会跑与自动管道相同的校验）
    python rag_translate_manual.py import --file batch.json

    # 3) 查看进度
    python rag_translate_manual.py status
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import rag_translate as T  # 复用语种判定、校验、缓存路径

BUILD_DIR = T.BUILD_DIR
CHUNKS_PATH = T.CHUNKS_PATH
CACHE_PATH = T.CACHE_PATH


def _pending(lang: str | None = None) -> List[Dict[str, Any]]:
    chunks = T.load_chunks()
    cache = T.load_cache()
    out = []
    for c in chunks:
        if not T.needs_translation(c):
            continue
        key = (c.get("metadata") or {}).get("text_sha1")
        if not key or key in cache:
            continue
        lg = T.detect_lang(c["index_text"])
        if lang and lg != lang:
            continue
        out.append({"key": key, "src_lang": lg, "doc": c.get("doc"),
                    "source": c["index_text"]})
    # 同 key 去重
    seen, uniq = set(), []
    for r in out:
        if r["key"] not in seen:
            seen.add(r["key"])
            uniq.append(r)
    return uniq


def export(lang: str | None, limit: int, out_path: str) -> None:
    items = _pending(lang)[:limit]
    if not items:
        print("没有待译内容")
        return
    for it in items:
        it["translation"] = ""          # 待填
        it["terms_hint"] = [f"{s}={d}" for s, d in
                            T.relevant_terms(it["source"], T.load_glossary())]
    Path(out_path).write_text(
        json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(len(i["source"]) for i in items)
    print(f"已导出 {len(items)} 条（{total:,} 字符）到 {out_path}")


def do_import(file_path: str, strict: bool = False) -> None:
    data = json.loads(Path(file_path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]

    chunks = {(c.get("metadata") or {}).get("text_sha1"): c
              for c in T.load_chunks()}
    cache = T.load_cache()
    stats: Counter = Counter()

    for item in data:
        key = item.get("key")
        tr = (item.get("translation") or "").strip()
        if not key:
            stats["no_key"] += 1
            continue
        if not tr:
            stats["empty_skipped"] += 1
            continue
        if key in cache:
            stats["already_cached"] += 1
            continue

        src = (chunks.get(key) or {}).get("index_text", "")
        if not src:
            stats["key_not_found"] += 1
            continue

        # 跑与自动管道完全相同的校验
        unhealthy = T.output_is_unhealthy(tr, src)
        if unhealthy:
            stats[f"rejected_{unhealthy.split(':')[0]}"] += 1
            print(f"  ✗ {key[:10]} 被拒: {unhealthy}")
            if strict:
                continue
        num_issue = T.check_numbers(src, tr)
        subj_issue = T.check_subject(src, tr)
        if num_issue:
            stats["number_issue"] += 1
            print(f"  ⚠ {key[:10]} 数值: {num_issue[:70]}")
        if subj_issue:
            stats["subject_issue"] += 1
            print(f"  ⚠ {key[:10]} 主体: {subj_issue[:70]}")

        T.append_cache({
            "key": key,
            "src_lang": item.get("src_lang") or T.detect_lang(src),
            "ok": True,
            "translation": tr,
            "number_issue": num_issue,
            "subject_issue": subj_issue,
            "translated_by": "assistant_manual",
            "attempts": 1,
        })
        stats["imported"] += 1

    print(f"\n导入结果: {dict(stats)}")
    status()


def status() -> None:
    chunks = T.load_chunks()
    cache = T.load_cache()
    need = [c for c in chunks if T.needs_translation(c)]
    done_keys = {c for c in cache}
    done = sum(1 for c in need
               if (c.get("metadata") or {}).get("text_sha1") in done_keys)
    by_lang_todo: Counter = Counter()
    by_lang_chars: Counter = Counter()
    for c in need:
        k = (c.get("metadata") or {}).get("text_sha1")
        if k in done_keys:
            continue
        lg = T.detect_lang(c["index_text"])
        by_lang_todo[lg] += 1
        by_lang_chars[lg] += len(c["index_text"])

    src_stat = Counter(r.get("translated_by") or "vllm" for r in cache.values())
    print(f"\n需翻译 {len(need)}  已完成 {done}  剩余 {len(need)-done}")
    for lg in sorted(by_lang_todo):
        print(f"  剩余 {lg}: {by_lang_todo[lg]} 条 / {by_lang_chars[lg]:,} 字符")
    print(f"译文来源: {dict(src_stat)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="人工翻译导入工具（不碰远端）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--lang", default=None, choices=["th", "en"])
    e.add_argument("--limit", type=int, default=20)
    e.add_argument("--out", default="rag_build/batch.json")
    i = sub.add_parser("import")
    i.add_argument("--file", required=True)
    i.add_argument("--strict", action="store_true", help="校验不过则跳过而非仅告警")
    sub.add_parser("status")
    a = ap.parse_args()

    if a.cmd == "export":
        export(a.lang, a.limit, a.out)
    elif a.cmd == "import":
        do_import(a.file, strict=a.strict)
    else:
        status()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Automatic PDF ingestion for the local RAG store."""

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_pdf_reader():
    try:
        from pypdf import PdfReader  # type: ignore

        return PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore

            return PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "PDF parsing requires pypdf. Install it with: pip install pypdf"
            ) from exc


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _clean_text(text: str) -> str:
    text = (text or "").replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<![.!?:;。！？；：])\n(?!\n)", " ", text)
    return text.strip()


def _safe_filename(filename: str) -> str:
    name = os.path.basename(filename or "").strip()
    if not name:
        name = f"document-{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    stem, ext = os.path.splitext(name)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "document"
    if ext.lower() != ".pdf":
        ext = ".pdf"
    return f"{stem}{ext}"


def unique_pdf_path(pdf_dir: Path, filename: str) -> Path:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_filename(filename)
    target = pdf_dir / safe_name
    if not target.exists():
        return target

    stem = target.stem
    suffix = target.suffix
    marker = datetime.now().strftime("%Y%m%d%H%M%S")
    for i in range(1, 1000):
        candidate = pdf_dir / f"{stem}-{marker}-{i}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Unable to allocate a unique PDF filename")


@dataclass
class AutoRAGIngestor:
    pdf_dir: Path
    chunks_path: Path
    manifest_path: Path
    chunk_size: int = 900
    chunk_overlap: int = 160

    def __post_init__(self):
        self.pdf_dir = Path(self.pdf_dir)
        self.chunks_path = Path(self.chunks_path)
        self.manifest_path = Path(self.manifest_path)
        self.chunk_size = max(200, int(self.chunk_size or 900))
        self.chunk_overlap = max(0, min(int(self.chunk_overlap or 0), self.chunk_size // 2))

    def sync(self, force: bool = False) -> Dict[str, Any]:
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        pdfs = sorted(self.pdf_dir.glob("*.pdf"))
        fingerprint = self._fingerprint(pdfs)
        manifest = self._read_manifest()

        if (
            not force
            and self.chunks_path.exists()
            and manifest.get("fingerprint") == fingerprint
        ):
            return {
                "changed": False,
                "pdf_dir": str(self.pdf_dir),
                "chunks_path": str(self.chunks_path),
                "pdf_count": len(pdfs),
                "chunks_count": int(manifest.get("chunks_count") or 0),
                "files": manifest.get("files") or [],
                "last_sync": manifest.get("last_sync"),
            }

        chunks, files = self._build_chunks(pdfs)
        self._write_jsonl(chunks)
        last_sync = datetime.now().isoformat()
        manifest = {
            "fingerprint": fingerprint,
            "pdf_dir": str(self.pdf_dir),
            "chunks_path": str(self.chunks_path),
            "pdf_count": len(pdfs),
            "chunks_count": len(chunks),
            "files": files,
            "last_sync": last_sync,
        }
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {"changed": True, **manifest}

    def status(self) -> Dict[str, Any]:
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        manifest = self._read_manifest()
        pdfs = sorted(self.pdf_dir.glob("*.pdf"))
        return {
            "pdf_dir": str(self.pdf_dir),
            "chunks_path": str(self.chunks_path),
            "manifest_path": str(self.manifest_path),
            "pdf_count": len(pdfs),
            "chunks_count": int(manifest.get("chunks_count") or 0),
            "files": manifest.get("files") or [
                {"name": p.name, "size": p.stat().st_size, "chunks": 0} for p in pdfs
            ],
            "last_sync": manifest.get("last_sync"),
        }

    def _read_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _fingerprint(self, pdfs: List[Path]) -> str:
        parts = []
        for path in pdfs:
            stat = path.stat()
            parts.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
        return _sha1("\n".join(parts))

    def _build_chunks(self, pdfs: List[Path]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        chunks: List[Dict[str, Any]] = []
        files: List[Dict[str, Any]] = []
        PdfReader = _load_pdf_reader()

        for pdf_path in pdfs:
            before = len(chunks)
            page_count = 0
            try:
                reader = PdfReader(str(pdf_path))
                pages = getattr(reader, "pages", [])
                page_count = len(pages)
                for page_index, page in enumerate(pages, start=1):
                    try:
                        text = _clean_text(page.extract_text() or "")
                    except Exception:
                        text = ""
                    for chunk_index, chunk_text in enumerate(self._chunk_text(text)):
                        text_hash = _sha1(chunk_text)
                        chunks.append(
                            {
                                "id": _sha1(f"{pdf_path.name}:{page_index}:{chunk_index}:{text_hash}"),
                                "source_type": "pdf",
                                "source_file": pdf_path.name,
                                "doc": pdf_path.name,
                                "page": page_index,
                                "chunk_index": chunk_index,
                                "text": chunk_text,
                                "text_sha1": text_hash,
                            }
                        )
            except Exception as exc:
                files.append(
                    {
                        "name": pdf_path.name,
                        "size": pdf_path.stat().st_size,
                        "pages": page_count,
                        "chunks": 0,
                        "error": str(exc),
                    }
                )
                continue

            files.append(
                {
                    "name": pdf_path.name,
                    "size": pdf_path.stat().st_size,
                    "pages": page_count,
                    "chunks": len(chunks) - before,
                }
            )

        return chunks, files

    def _chunk_text(self, text: str) -> List[str]:
        text = _clean_text(text)
        if len(text) < 80:
            return []

        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(text_len, start + self.chunk_size)
            if end < text_len:
                boundary = max(
                    text.rfind("\n\n", start, end),
                    text.rfind("。", start, end),
                    text.rfind(".", start, end),
                    text.rfind(";", start, end),
                )
                if boundary > start + int(self.chunk_size * 0.55):
                    end = boundary + 1

            chunk = text[start:end].strip()
            if len(chunk) >= 80:
                chunks.append(chunk)

            if end >= text_len:
                break
            start = max(end - self.chunk_overlap, start + 1)

        return chunks

    def _write_jsonl(self, chunks: List[Dict[str, Any]]) -> None:
        tmp_path = self.chunks_path.with_suffix(self.chunks_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        tmp_path.replace(self.chunks_path)

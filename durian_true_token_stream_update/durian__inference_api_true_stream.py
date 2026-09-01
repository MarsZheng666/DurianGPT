#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
榴莲GPT - 推理服务 API
基于微调的 Qwen3-14B LoRA 模型 + RAG 检索 + SQLite 对话存储 + Active Topic Router v3

功能：
1. 本地模型推理（Merged / LoRA）
2. API 鉴权
3. RAG 检索
4. 非流式 / SSE 流式聊天接口
5. SQLite 对话存储
"""

import os
import sys

# 添加当前目录到 Python 路径，以便导入 qwen_vl_max_client
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ===== 在任何 vLLM 导入前设置环境变量 =====
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "triton")
os.environ.setdefault("VLLM_DISABLE_CUSTOM_ALL_REDUCE", "1")
os.environ.setdefault("VLLM_DISABLE_VLLM_WORKER_MULTIPROCESSING_INIT", "1")
# os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")  # 注释掉以提升 GPU 并发性能
os.environ.setdefault("VLLM_TARGET_DEVICE", "cuda")
# ==========================================

import torch
import json
import uuid
import sqlite3
import logging
import numpy as np
import time
import queue
import re
import asyncio
from typing import List, Dict, Optional, Iterator, Tuple, Any
from datetime import datetime
from threading import Lock, RLock, Thread
from io import BytesIO

import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50

from fastapi import FastAPI, HTTPException, Header, Depends, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

try:
    from auto_rag import AutoRAGIngestor, unique_pdf_path
    AUTO_RAG_AVAILABLE = True
except ImportError:
    AutoRAGIngestor = None
    unique_pdf_path = None
    AUTO_RAG_AVAILABLE = False

# 导入 Qwen-VL-Max 客户端
try:
    from qwen_vl_max_client import QwenVLMaxClient, get_qwen_vl_max_client
    QWEN_VL_MAX_AVAILABLE = True
except ImportError:
    QwenVLMaxClient = Any
    get_qwen_vl_max_client = None
    QWEN_VL_MAX_AVAILABLE = False

# =========================
# SentenceTransformer for embedding
# =========================
try:
    from sentence_transformers import SentenceTransformer

    ST_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    ST_AVAILABLE = False

# =========================
# FAISS
# =========================
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

# =========================
# 日志配置
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("durian--api")

# =========================
# 配置参数
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = Path(__file__).resolve().parent


def env_bool(name: str, default: bool = False) -> bool:
    """从环境变量读取布尔值。"""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def first_existing_path(paths: List[str]) -> str:
    """返回第一个存在的路径；如果都不存在，返回第一个非空路径。"""
    for path in paths:
        if path and os.path.exists(path):
            return path
    for path in paths:
        if path:
            return path
    return ""


DEFAULT_LORA_MODEL_PATH = first_existing_path([
    os.getenv("DURIAN_LORA_MODEL_PATH", ""),

    "/home/admin01/下载/ToDesk/outputs/durian_qwen3_14b_qlora_v2_industry_final",
])

MODEL_CONFIG = {
    # 推理优先使用 AWQ base，显存占用低；LoRA adapter 由 vLLM 动态加载。
    # 如果 AWQ + LoRA 在你的 vLLM 版本报错，可通过环境变量切到 28G merged 模型：
    # export DURIAN_BASE_MODEL=/home/admin01/durian_qwen3_14b_merged
    "base_model": os.getenv("DURIAN_BASE_MODEL", "/home/admin01/durian_qwen3_14b_awq"),
    "lora_model_path": DEFAULT_LORA_MODEL_PATH,
    "enable_lora": env_bool("DURIAN_ENABLE_LORA", True),
    "use_merged": env_bool("DURIAN_USE_MERGED", False),
    "merged_model_path": os.getenv("DURIAN_MERGED_MODEL_PATH", ""),
    "use_quantization": True,
    "quantization_bits": 4,
    "max_lora_rank": int(os.getenv("DURIAN_MAX_LORA_RANK", "64")),
}


# =========================
# vLLM OpenAI Proxy / True Streaming 配置
# =========================
# local_vllm：当前 FastAPI 进程内直接加载 vLLM LLM（兼容旧模式）。
# openai_proxy：FastAPI 不再加载大模型，只把最终 messages 转发到独立 vLLM OpenAI Server。
#               这是推荐的真正 token-by-token 流式模式。
ENGINE_MODE = os.getenv("DURIAN_ENGINE_MODE", "local_vllm").strip().lower()
OPENAI_PROXY_MODES = {"openai_proxy", "proxy", "vllm_server", "remote_vllm"}
DURIAN_OPENAI_BASE_URL = os.getenv("DURIAN_OPENAI_BASE_URL", "http://127.0.0.1:8010/v1").rstrip("/")
DURIAN_OPENAI_MODEL = os.getenv("DURIAN_OPENAI_MODEL", "durian-lora")
DURIAN_OPENAI_API_KEY = os.getenv("DURIAN_OPENAI_API_KEY", "")
DURIAN_OPENAI_TIMEOUT = float(os.getenv("DURIAN_OPENAI_TIMEOUT", "600"))
DURIAN_PROXY_DISABLE_THINKING = env_bool("DURIAN_PROXY_DISABLE_THINKING", True)


RAG_CONFIG = {
    "chunks_path": os.getenv(
        "DURIAN_RAG_CHUNKS_PATH",
        first_existing_path([
            str(BASE_DIR / "clean_chunks.jsonl"),
            "/home/admin01/桌面/Desktop/durian-training/clean_chunks.jsonl",
        ]),
    ),
    "extra_chunks_paths": [
        p
        for p in os.getenv(
            "DURIAN_RAG_EXTRA_CHUNKS_PATHS",
            str(BASE_DIR / "rag_auto" / "auto_chunks.jsonl"),
        ).split(os.pathsep)
        if p
    ],
    "embed_model_path": os.getenv(
        "DURIAN_RAG_EMBED_MODEL_PATH",
        first_existing_path([
            str(BASE_DIR / "model_e5"),
            "/home/admin01/桌面/Desktop/durian-training/model_e5",
        ]),
    ),
    "enable_rag": env_bool("DURIAN_ENABLE_RAG", True),
    "always_on": env_bool("DURIAN_RAG_ALWAYS_ON", True),
    "auto_pdf_enabled": env_bool("DURIAN_AUTO_RAG_PDF", True),
    "pdf_dir": os.getenv("DURIAN_RAG_PDF_DIR", str(BASE_DIR / "rag_pdfs")),
    "auto_chunks_path": os.getenv("DURIAN_RAG_AUTO_CHUNKS_PATH", str(BASE_DIR / "rag_auto" / "auto_chunks.jsonl")),
    "auto_manifest_path": os.getenv("DURIAN_RAG_AUTO_MANIFEST_PATH", str(BASE_DIR / "rag_auto" / "manifest.json")),
    "index_cache_path": os.getenv("DURIAN_RAG_INDEX_CACHE_PATH", str(BASE_DIR / "rag_auto" / "faiss_index.bin")),
    "chunks_cache_path": os.getenv("DURIAN_RAG_CHUNKS_CACHE_PATH", str(BASE_DIR / "rag_auto" / "chunks_cache.json")),
    "top_k": int(os.getenv("DURIAN_RAG_TOP_K", "3")),
    "min_score": float(os.getenv("DURIAN_RAG_MIN_SCORE", "0.35")),
    "fallback_min_score": float(os.getenv("DURIAN_RAG_FALLBACK_MIN_SCORE", "-1.0")),
    "max_evidence_chars": int(os.getenv("DURIAN_RAG_MAX_EVIDENCE_CHARS", "700")),
    "max_history_messages": 2,
    "pdf_chunk_size": int(os.getenv("DURIAN_RAG_PDF_CHUNK_SIZE", "900")),
    "pdf_chunk_overlap": int(os.getenv("DURIAN_RAG_PDF_CHUNK_OVERLAP", "160")),
    "max_pdf_upload_mb": int(os.getenv("DURIAN_RAG_MAX_PDF_UPLOAD_MB", "80")),
}
if RAG_CONFIG["auto_chunks_path"] not in RAG_CONFIG["extra_chunks_paths"]:
    RAG_CONFIG["extra_chunks_paths"].append(RAG_CONFIG["auto_chunks_path"])

INFERENCE_CONFIG = {
    "max_new_tokens": int(os.getenv("DURIAN_DEFAULT_MAX_NEW_TOKENS", "2048")),
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.08,
    "do_sample": True,
    "use_cache": True,
    "max_history_messages": 2,
}


# =========================
# 输出格式配置
# =========================
# DURIAN_TABLE_POLICY:
#   adaptive / prefer / auto / true：结构化内容优先使用 Markdown 表格；简单问题不强制。
#   off / false / none：不做表格偏好提示，也不追加兜底表格。
TABLE_POLICY = os.getenv("DURIAN_TABLE_POLICY", os.getenv("DURIAN_PREFER_MARKDOWN_TABLE", "off")).strip().lower()
PREFER_MARKDOWN_TABLE_OUTPUT = TABLE_POLICY not in {"0", "false", "no", "off", "none", "disable", "disabled"}

API_KEY = os.getenv("DURIAN_API_KEY", os.getenv("API_KEY", "change-me"))
APP_USERS = {
    user.strip()
    for user in os.getenv("DURIAN_APP_USERS", "admin,admin2").split(",")
    if user.strip()
}
DB_PATH = os.getenv("DB_PATH", "./durian__conversations.db")

# ==================== 图片上传配置 ====================
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
logger.info("图片上传目录: %s", UPLOAD_DIR)

# ==================== 病虫害分类模型配置 ====================
PEST_MODEL_PATH = os.getenv("PEST_MODEL_PATH", "/home/admin01/桌面/Desktop/durian-training/best_resnet50.pth")
PEST_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("病虫害模型路径: %s", PEST_MODEL_PATH)
logger.info("病虫害模型设备: %s", PEST_DEVICE)

PEST_CLASSES = {
    0: {"zh": "背景或其他", "en": "Background or other", "ms": "Latar belakang atau lain-lain", "th": "พื้นหลังหรืออื่น ๆ"},
    1: {"zh": "健康叶片", "en": "Healthy leaves", "ms": "Daun sihat", "th": "ใบที่แข็งแรง"},
    2: {"zh": "健康树干/果实", "en": "Healthy stem/fruit", "ms": "Batang/buah sihat", "th": "ลำต้น/ผลที่แข็งแรง"},
    3: {"zh": "茎部疫病", "en": "Stem blight", "ms": "Penyakit batang", "th": "โรคใบไหม้ลำต้น"},
    4: {"zh": "茎部裂纹/胶质化", "en": "Stem cracks/gummosis", "ms": "Retak batang/gummosis", "th": "ลำต้นแตกร้าว/ยางไหล"},
    5: {"zh": "立枯丝核菌叶枯病", "en": "Rhizoctonia leaf blight", "ms": "Penyakit daun Rhizoctonia", "th": "โรคใบไหม้ไรโซคโทเนีย"},
    6: {"zh": "叶部炭疽病", "en": "Leaf anthracnose", "ms": "Antraknosa daun", "th": "โรคแอนแทรคโนสที่ใบ"},
    7: {"zh": "溃疡病", "en": "Canker", "ms": "Penyakit kanker", "th": "โรคแคงเกอร์"},
    8: {"zh": "粉红病", "en": "Pink disease", "ms": "Penyakit merah jambu", "th": "โรคราสีชมพู"},
    9: {"zh": "白根病", "en": "White root rot", "ms": "Penyakit akar putih", "th": "โรครากขาว"},
    10: {"zh": "藻斑病", "en": "Algal leaf spot", "ms": "Bintik daun alga", "th": "โรคจุดใบสาหร่าย"},
    11: {"zh": "Phomopsis叶斑病", "en": "Phomopsis leaf spot", "ms": "Bintik daun Phomopsis", "th": "โรคจุดใบโฟมอปซิส"},
    12: {"zh": "果腐病", "en": "Fruit rot", "ms": "Penyakit busuk buah", "th": "โรคผลเน่า"},
    13: {"zh": "煤烟病", "en": "Sooty mold", "ms": "Penyakit jelaga", "th": "ราดำ"},
    14: {"zh": "白蚁/蚂蚁", "en": "Termites/ants", "ms": "Rayap/semut", "th": "ปลวก/มด"},
    15: {"zh": "蝉", "en": "Cicada", "ms": "Cicada", "th": "จักจั่น"},
    16: {"zh": "木虱", "en": "Psyllid", "ms": "Psyllid", "th": "เพลี้ยไก่แจ้"},
    17: {"zh": "粉蚧", "en": "Mealybug", "ms": "Kutu putih", "th": "เพลี้ยแป้ง"},
    18: {"zh": "叶蝉", "en": "Leafhopper", "ms": "Wereng daun", "th": "เพลี้ยจักจั่น"},
    19: {"zh": "蓟马", "en": "Thrips", "ms": "Thrips", "th": "เพลี้ยไฟ"},
    20: {"zh": "红蜘蛛", "en": "Spider mite", "ms": "Tungau merah", "th": "ไรแดง"},
    21: {"zh": "盔蚧", "en": "Armored scale", "ms": "Sisik bersenjata", "th": "เพลี้ยหอย"},
    22: {"zh": "茎钻虫", "en": "Stem borer", "ms": "Penggerek batang", "th": "หนอนเจาะลำต้น"},
    23: {"zh": "小蠹虫", "en": "Bark beetle", "ms": "Kumbang kulit kayu", "th": "ด้วงเปลือกไม้"},
    24: {"zh": "果钻虫", "en": "Fruit borer", "ms": "Penggerek buah", "th": "หนอนเจาะผล"},
    25: {"zh": "黄叶", "en": "Yellow leaves", "ms": "Daun kuning", "th": "ใบเหลือง"},
}


def get_pest_class_name(class_id: int, language: str = "zh") -> str:
    if class_id not in PEST_CLASSES:
        return "Unknown"
    return PEST_CLASSES[class_id].get(language, PEST_CLASSES[class_id]["zh"])


SYSTEM_PROMPTS = {
    "zh": (
        "你是榴莲领域顾问，熟悉榴莲品种、口感、食用、保存、购买、市场、种植管理和病虫害。"
        "请直接回答用户当前问题，不要套固定模板。根据问题自然决定回答长短、结构和是否使用表格。"
        "消费者问题只谈品种、口感、品质、食用、保存、挑选；不要强行加入果园管理、病虫害、防治、喷药、复查时间。"
        "种植或病虫害问题可以给判断、原因、检查点和处理建议。不要输出思考过程。"
    ),
    "en": (
        "You are a durian-domain advisor familiar with varieties, taste, eating quality, storage, buying, market, cultivation, pests and diseases. "
        "Answer the user's current question directly and naturally. Do not force a fixed template. Choose length, structure and tables only when useful. "
        "For consumer questions, discuss variety, taste, quality, eating, storage or selection only; do not force orchard management, disease control, spraying or recheck schedules. "
        "For cultivation or pest/disease questions, give a judgment, reasons, checks and practical actions. Do not reveal reasoning or drafts."
    ),
    "ms": (
        "Anda ialah penasihat domain durian yang memahami varieti, rasa, kualiti makan, simpanan, pembelian, pasaran, penanaman, penyakit dan perosak. "
        "Jawab soalan semasa pengguna secara langsung dan semula jadi. Jangan paksa templat tetap. Pilih panjang jawapan, struktur dan jadual hanya apabila berguna. "
        "Untuk soalan pengguna biasa, fokus pada varieti, rasa, kualiti, cara makan, simpanan atau pemilihan; jangan paksa pengurusan kebun, kawalan penyakit, semburan atau jadual semakan. "
        "Untuk soalan penanaman atau penyakit/perosak, beri penilaian, sebab, semakan dan tindakan praktikal. Jangan paparkan proses berfikir atau draf."
    ),
    "th": (
        "คุณเป็นที่ปรึกษาเรื่องทุเรียน เข้าใจสายพันธุ์ รสชาติ คุณภาพการกิน การเก็บรักษา การซื้อ ตลาด การปลูก โรคและแมลง. "
        "ตอบคำถามปัจจุบันของผู้ใช้โดยตรงและเป็นธรรมชาติ ไม่บังคับใช้แม่แบบตายตัว เลือกความยาว โครงสร้าง และตารางเมื่อมีประโยชน์จริง. "
        "สำหรับคำถามผู้บริโภค ให้พูดเรื่องสายพันธุ์ รสชาติ คุณภาพ การกิน การเก็บ หรือการเลือกซื้อเท่านั้น ไม่ดึงไปเรื่องสวน โรค การพ่น หรือเวลาตรวจซ้ำ. "
        "สำหรับคำถามการปลูกหรือโรค/แมลง ให้ให้ข้อสรุป เหตุผล จุดตรวจ และการจัดการที่ทำได้จริง ห้ามแสดงกระบวนการคิดหรือร่างคำตอบ."
    ),
}

SYSTEM_PROMPT = SYSTEM_PROMPTS["zh"]

IMAGE_ANALYSIS_PROMPTS = {
    "zh": (
        "你是榴莲图片问答顾问。你会收到视觉模型给出的分类名称和可见依据。"
        "先回答用户真正的问题，不要把所有图片都转成农业治理报告。"
        "如果是品种或果肉品质，只谈可能品种、口感、成熟度、能否食用、保存或挑选；不要讲喷药、防治、冷库分选、商业渠道。"
        "如果是病虫害或种植问题，再谈诊断、严重程度、现场复核和处理建议。"
        "回答自然，短问题可以短答；只有用户要求对比或信息很多时才用表格。不要输出思考过程。"
    ),
    "en": (
        "You are a durian image Q&A advisor. You receive a visual model's classification label and visible evidence. "
        "Answer the user's actual question first; do not turn every image into an orchard-treatment report. "
        "For variety or fruit-quality images, discuss likely variety, taste, ripeness, edibility, storage or selection only; avoid spraying, disease control, cold-room grading or commercial channels. "
        "For disease, pest or cultivation images, discuss diagnosis, severity, field verification and actions. "
        "Keep the answer natural. Short questions may receive concise answers. Use tables only for comparisons or genuinely complex information. Do not reveal reasoning."
    ),
    "ms": (
        "Anda ialah penasihat soal jawab imej durian. Anda menerima label klasifikasi dan bukti visual daripada model penglihatan. "
        "Jawab soalan sebenar pengguna dahulu; jangan tukar semua imej menjadi laporan rawatan kebun. "
        "Untuk imej varieti atau kualiti isi, bincang varieti kemungkinan, rasa, kematangan, boleh dimakan, simpanan atau pemilihan sahaja; elakkan semburan, kawalan penyakit, penggredan bilik sejuk atau saluran komersial. "
        "Untuk imej penyakit, perosak atau penanaman, bincang diagnosis, tahap serius, pengesahan lapangan dan tindakan. "
        "Jawab secara semula jadi. Soalan pendek boleh dijawab ringkas. Gunakan jadual hanya untuk perbandingan atau maklumat yang benar-benar kompleks."
    ),
    "th": (
        "คุณเป็นที่ปรึกษาถามตอบภาพทุเรียน คุณจะได้รับชื่อการจำแนกและหลักฐานภาพจากโมเดลมองภาพ. "
        "ตอบคำถามจริงของผู้ใช้ก่อน อย่าเปลี่ยนทุกภาพเป็นรายงานการจัดการสวน. "
        "ถ้าเป็นภาพสายพันธุ์หรือคุณภาพเนื้อ ให้พูดเรื่องสายพันธุ์ที่เป็นไปได้ รสชาติ ความสุก กินได้หรือไม่ การเก็บ หรือการเลือกซื้อเท่านั้น ไม่พูดเรื่องพ่นยา ควบคุมโรค คัดเกรดห้องเย็น หรือช่องทางค้า. "
        "ถ้าเป็นภาพโรค แมลง หรือการปลูก ให้พูดเรื่องการวินิจฉัย ความรุนแรง การยืนยันภาคสนาม และการจัดการ. "
        "ตอบอย่างเป็นธรรมชาติ คำถามสั้นตอบสั้นได้ ใช้ตารางเฉพาะเมื่อเป็นการเปรียบเทียบหรือข้อมูลซับซ้อนจริง."
    ),
}

# =========================
# Pydantic 模型
# =========================
class Message(BaseModel):
    role: str = Field(..., description="消息角色: user/assistant/system")
    content: str = Field(default="", description="消息内容")
    image_url: Optional[str] = Field(default=None, description="图片访问地址")
    active_context_card: Optional[Dict[str, Any]] = Field(default=None, description="后端生成的 active topic context card；只给上下文路由使用，不参与前端可见渲染")


class InferenceRequest(BaseModel):
    messages: List[Message] = Field(..., description="对话消息列表")
    max_tokens: int = Field(default=2048, description="最大生成 token 数")
    max_new_tokens: int = Field(default=2048, description="最大生成 token 数（兼容）")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="温度参数")
    top_p: float = Field(default=0.85, ge=0.0, le=1.0, description="top_p 采样")
    top_k: int = Field(default=40, ge=0, description="top_k 采样")
    repetition_penalty: float = Field(default=1.08, ge=1.0, description="重复惩罚")
    stream: bool = Field(default=False, description="是否流式输出")
    use_rag: bool = Field(default=True, description="是否使用 RAG")
    response_language: str = Field(default="zh", description="响应语言: zh/en/ms/th")


class Evidence(BaseModel):
    text: str = Field(..., description="证据文本")
    score: float = Field(..., description="相似度分数")
    doc: Optional[str] = Field(None, description="文档来源")


class InferenceResponse(BaseModel):
    response: str = Field(..., description="模型生成的回复")
    tokens_generated: int = Field(..., description="生成的 token 数")
    model: str = Field(..., description="使用的模型")
    timestamp: str = Field(..., description="生成时间戳")
    evidence: Optional[List[Evidence]] = Field(default_factory=list, description="检索到的证据")
    evidence_quality: Optional[str] = Field(default="none", description="证据质量评估")
    image_url: Optional[str] = Field(default=None, description="图片访问地址")
    visual_observation: Optional[str] = Field(default=None, description="Vision model context")
    active_context_card: Optional[Dict[str, Any]] = Field(default=None, description="本轮回答绑定的 active topic context card；前端需随 assistant 消息保存并在后续请求中传回")


class HealthResponse(BaseModel):
    status: str = Field(..., description="服务状态")
    device: str = Field(..., description="计算设备")
    model_loaded: bool = Field(..., description="模型是否已加载")
    model_path: str = Field(..., description="模型路径")
    memory_usage: Optional[Dict] = Field(None, description="显存使用情况")
    rag_enabled: bool = Field(..., description="RAG 是否启用")
    rag_chunks: int = Field(default=0, description="RAG 数据块数量")
    rag_pdf_dir: Optional[str] = Field(default=None, description="PDF RAG 上传目录")
    rag_pdf_files: int = Field(default=0, description="PDF RAG 文件数量")
    rag_auto_chunks: int = Field(default=0, description="PDF RAG 自动数据块数量")
    rag_auto_last_sync: Optional[str] = Field(default=None, description="PDF RAG 最近同步时间")
    embed_model_loaded: bool = Field(default=False, description="嵌入模型是否加载成功")
    faiss_available: bool = Field(default=False, description="FAISS 是否可用")
    sentence_transformers_available: bool = Field(default=False, description="SentenceTransformers 是否可用")
    db_path: str = Field(..., description="SQLite 数据库路径")
    timestamp: str = Field(..., description="时间戳")
    engine_mode: str = Field(default="local_vllm", description="推理引擎模式: local_vllm/openai_proxy")
    openai_proxy_base_url: Optional[str] = Field(default=None, description="openai_proxy 模式下的 vLLM OpenAI Server 地址")
    openai_proxy_model: Optional[str] = Field(default=None, description="openai_proxy 模式下的模型名")


class PestClassificationResult(BaseModel):
    class_id: int = Field(..., description="类别ID")
    class_name: str = Field(..., description="类别名称")
    class_name_en: str = Field(..., description="英文类别名称")
    class_name_ms: str = Field(..., description="马来语类别名称")
    confidence: float = Field(..., description="置信度")
    all_probs: Dict[str, float] = Field(..., description="所有类别的概率")


class ConversationCreateResponse(BaseModel):
    id: str = Field(..., description="对话 ID")
    title: str = Field(..., description="对话标题")
    timestamp: str = Field(..., description="创建时间")
    owner: str = Field(default="admin", description="对话所属账号")


# =========================
# 鉴权
# =========================
def verify_api_key(
    x_api_key: str = Header(default=""),
    authorization: str = Header(default=""),
):
    if API_KEY == "change-me":
        logger.warning("当前 API_KEY 仍为默认值 change-me，请在生产环境中修改")

    if x_api_key and x_api_key == API_KEY:
        return

    if authorization and authorization.startswith("Bearer "):
        token = authorization[len("Bearer ") :]
        if token == API_KEY:
            return

    raise HTTPException(status_code=401, detail="Invalid API key")


def get_current_user(x_user_id: str = Header(default="admin")) -> str:
    user = (x_user_id or "admin").strip()
    if user not in APP_USERS:
        raise HTTPException(status_code=403, detail="Invalid user")
    return user


# =========================
# SQLite 存储
# =========================
class ConversationStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    owner TEXT NOT NULL DEFAULT 'admin',
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    image_url TEXT,
                    active_context_card_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id)
                )
                """
            )
            conn.commit()
            
            # 迁移：为旧表添加 image_url 列
            try:
                conn.execute("ALTER TABLE messages ADD COLUMN image_url TEXT")
                conn.commit()
                logger.info("✓ 已为 messages 表添加 image_url 列")
            except sqlite3.OperationalError:
                # 列已存在，忽略错误
                pass

            # 迁移：active context card 不再塞进 content，单独存 JSON。
            try:
                conn.execute("ALTER TABLE messages ADD COLUMN active_context_card_json TEXT")
                conn.commit()
                logger.info("✓ 已为 messages 表添加 active_context_card_json 列")
            except sqlite3.OperationalError:
                pass

            # 迁移：多账号登录后，对话需要按账号隔离。旧数据默认归 admin。
            try:
                conn.execute("ALTER TABLE conversations ADD COLUMN owner TEXT NOT NULL DEFAULT 'admin'")
                conn.commit()
                logger.info("✓ 已为 conversations 表添加 owner 列")
            except sqlite3.OperationalError:
                pass

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_owner_updated ON conversations(owner, updated_at DESC)"
            )
            conn.commit()

    def create_conversation(self, owner: str, title: str = "新对话") -> Dict:
        now = datetime.now().isoformat()
        conv_id = str(uuid.uuid4())

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations(id, owner, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conv_id, owner, title, now, now),
            )
            conn.commit()

        return {"id": conv_id, "title": title, "timestamp": now, "owner": owner}

    def list_conversations(self, owner: str) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.title, c.created_at, c.updated_at, COUNT(m.id) AS message_count, c.owner
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                WHERE c.owner = ?
                GROUP BY c.id, c.owner
                ORDER BY c.updated_at DESC
                """,
                (owner,),
            ).fetchall()

        return [
            {
                "id": row[0],
                "title": row[1],
                "created_at": row[2],
                "updated_at": row[3],
                "timestamp": row[3],
                "message_count": row[4],
                "owner": row[5],
            }
            for row in rows
        ]

    def get_conversation(self, conv_id: str, owner: str) -> Dict:
        with self._connect() as conn:
            conv = conn.execute(
                """
                SELECT id, title, created_at, updated_at, owner
                FROM conversations
                WHERE id = ? AND owner = ?
                """,
                (conv_id, owner),
            ).fetchone()

            if not conv:
                raise HTTPException(status_code=404, detail="对话不存在")

            messages = conn.execute(
                """
                SELECT role, content, image_url, active_context_card_json, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
                """,
                (conv_id,),
            ).fetchall()

        return {
            "id": conv[0],
            "title": conv[1],
            "created_at": conv[2],
            "updated_at": conv[3],
            "timestamp": conv[3],
            "owner": conv[4],
            "messages": [
                {
                    "role": m[0],
                    "content": self._strip_hidden_context_from_content(m[1] or ""),
                    "image_url": m[2],
                    "active_context_card": self._parse_context_card_json(m[3]),
                    "created_at": m[4],
                }
                for m in messages
            ],
        }

    @staticmethod
    def _strip_hidden_context_from_content(content: str) -> str:
        """兼容清理旧版本塞进 content 的隐藏上下文，保证前端只看到干净回答。"""
        if not content:
            return ""
        cleaned = re.sub(r"<!--\s*\[(?:Image context|Durian context card|视觉识别上下文)\][\s\S]*?(?:-->|$)", "", content, flags=re.I)
        cleaned = re.sub(r"\[Durian context card\][\s\S]*$", "", cleaned, flags=re.I)
        # 兜底清理旧版本泄漏的 JSON 尾巴。
        cleaned = re.sub(r"[\r\n]*[\"“]?created_at[\"”]?\s*[:：]\s*[\"“][^\n{}]+[\"”]\s*}\s*(?:-->)?\s*$", "", cleaned, flags=re.I)
        return cleaned.strip()

    @staticmethod
    def _parse_context_card_json(raw: Optional[str]) -> Optional[Dict[str, Any]]:
        if not raw:
            return None
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def add_message(self, conv_id: str, owner: str, role: str, content: str, image_url: Optional[str] = None, active_context_card: Optional[Dict[str, Any]] = None):
        now = datetime.now().isoformat()
        msg_id = str(uuid.uuid4())
        visible_content = self._strip_hidden_context_from_content(content or "")
        card_json = json.dumps(active_context_card, ensure_ascii=False) if isinstance(active_context_card, dict) else None

        with self._connect() as conn:
            exists = conn.execute(
                "SELECT id FROM conversations WHERE id = ? AND owner = ?",
                (conv_id, owner),
            ).fetchone()

            if not exists:
                raise HTTPException(status_code=404, detail="对话不存在")

            conn.execute(
                """
                INSERT INTO messages(id, conversation_id, role, content, image_url, active_context_card_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (msg_id, conv_id, role, visible_content, image_url, card_json, now),
            )

            if role == "user":
                count = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM messages
                    WHERE conversation_id = ? AND role = 'user'
                    """,
                    (conv_id,),
                ).fetchone()[0]

                if count == 1:
                    title = content[:50] if content else "新对话"
                    conn.execute(
                        "UPDATE conversations SET title = ? WHERE id = ?",
                        (title, conv_id),
                    )

            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conv_id),
            )
            conn.commit()

    def delete_conversation(self, conv_id: str, owner: str):
        with self._connect() as conn:
            exists = conn.execute(
                "SELECT id FROM conversations WHERE id = ? AND owner = ?",
                (conv_id, owner),
            ).fetchone()

            if not exists:
                raise HTTPException(status_code=404, detail="对话不存在")

            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
            conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
            conn.commit()


# =========================
# RAG 检索类
# =========================
class RAGRetriever:
    """FAISS 向量检索系统"""

    def __init__(self, config: Dict):
        self.config = config
        self.chunks: List[Dict] = []
        self.model = None
        self.index = None
        self.enabled = False
        self.index_cache_path = config.get("index_cache_path", "./faiss_index.bin")
        self.chunks_cache_path = config.get("chunks_cache_path", "./chunks_cache.json")

        if not config.get("enable_rag", False):
            logger.info("RAG 未启用")
            return

        if not FAISS_AVAILABLE:
            logger.warning("✗ FAISS 不可用，RAG 已禁用")
            return

        if not ST_AVAILABLE:
            logger.warning("✗ sentence-transformers 不可用，RAG 已禁用")
            return

        self._load_chunks()
        self._load_embed_model()
        self._load_or_build_index()
        self.enabled = self.index is not None and self.model is not None

        if self.enabled:
            logger.info("✓ RAG 检索器初始化成功")
        else:
            logger.warning("✗ RAG 检索器初始化失败")

    def _configured_chunk_paths(self) -> List[str]:
        paths: List[str] = []
        main_path = self.config.get("chunks_path")
        if main_path:
            paths.append(str(main_path))
        for path in self.config.get("extra_chunks_paths", []) or []:
            if path:
                paths.append(str(path))
        return paths

    def _load_chunks(self):
        try:
            chunks = []
            seen_ids = set()
            for path in self._configured_chunk_paths():
                if not os.path.exists(path):
                    logger.info("数据块文件不存在，跳过: %s", path)
                    continue

                loaded = 0
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        chunk = json.loads(line)
                        chunk_id = chunk.get("id") or chunk.get("text_sha1") or chunk.get("text")
                        if chunk_id in seen_ids:
                            continue
                        seen_ids.add(chunk_id)
                        chunks.append(chunk)
                        loaded += 1
                logger.info("✓ 加载数据块文件 %s: %d 条", path, loaded)

            self.chunks = chunks
            logger.info("✓ RAG 总数据块: %d 条", len(self.chunks))
        except Exception as e:
            logger.warning("✗ 加载数据块失败: %s", e)
            self.chunks = []

    def _load_embed_model(self):
        try:
            path = self.config["embed_model_path"]
            if not os.path.exists(path):
                logger.warning("嵌入模型不存在: %s", path)
                return

            logger.info("加载嵌入模型: %s", path)
            embed_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer(path, device=embed_device)
            logger.info("✓ 嵌入模型加载成功 (device=%s)", embed_device)
        except Exception as e:
            logger.warning("✗ 加载嵌入模型失败: %s", e)
            self.model = None

    def _encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        if self.model is None or not texts:
            return np.array([], dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        return embeddings.astype(np.float32)

    def _load_or_build_index(self):
        """优先加载缓存索引，如果不存在则构建新索引"""
        if self._try_load_cache():
            logger.info("✓ 从缓存加载 FAISS 索引成功")
            return
        
        logger.info("缓存不存在或加载失败，重新构建索引...")
        self._build_index()
        self._save_cache()

    def _try_load_cache(self) -> bool:
        """尝试从缓存加载索引和数据块"""
        try:
            if not os.path.exists(self.index_cache_path) or not os.path.exists(self.chunks_cache_path):
                logger.info("缓存文件不存在")
                return False

            cache_mtime = os.path.getmtime(self.index_cache_path)
            for chunks_path in self._configured_chunk_paths():
                if os.path.exists(chunks_path) and os.path.getmtime(chunks_path) > cache_mtime:
                    logger.info("数据块文件比索引缓存新，重新构建索引: %s", chunks_path)
                    return False

            # 加载数据块缓存
            with open(self.chunks_cache_path, "r", encoding="utf-8") as f:
                cached_chunks = json.load(f)
            
            # 验证数据块是否一致
            if len(cached_chunks) != len(self.chunks):
                logger.warning("缓存数据块数量不匹配 (%d vs %d)", len(cached_chunks), len(self.chunks))
                return False
            
            # 加载索引
            self.index = faiss.read_index(self.index_cache_path)
            logger.info("✓ 从缓存加载索引成功 (%d 条记录)", self.index.ntotal)
            return True
        except Exception as e:
            logger.warning("✗ 加载缓存失败: %s", e)
            return False

    def _save_cache(self):
        """保存索引和数据块到缓存"""
        try:
            if self.index is None:
                return
            
            # 保存索引
            faiss.write_index(self.index, self.index_cache_path)
            
            # 保存数据块
            with open(self.chunks_cache_path, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, ensure_ascii=False, indent=2)
            
            logger.info("✓ 索引缓存已保存到 %s", self.index_cache_path)
        except Exception as e:
            logger.warning("✗ 保存缓存失败: %s", e)

    def _build_index(self):
        if not self.chunks:
            logger.warning("✗ 无数据块，无法构建索引")
            return

        if self.model is None:
            logger.warning("✗ 嵌入模型未加载，无法构建索引")
            return

        try:
            logger.info("构建 FAISS 索引...")
            texts = [c.get("text", "") for c in self.chunks]
            vecs = self._encode(texts, batch_size=64)

            if vecs.size == 0:
                logger.warning("✗ 向量为空，索引构建失败")
                return

            self.index = faiss.IndexFlatIP(vecs.shape[1])
            self.index.add(vecs)
            logger.info("✓ FAISS 索引构建成功 (%d 条记录)", len(vecs))
        except Exception as e:
            logger.warning("✗ 构建索引失败: %s", e)
            self.index = None

    def retrieve(self, query: str, top_k: Optional[int] = None) -> Tuple[List[Dict], str]:
        if not self.enabled or self.index is None or self.model is None or not query.strip():
            return [], "none"

        top_k = top_k or self.config.get("top_k", 3)
        min_score = float(self.config.get("min_score", 0.45))
        fallback_min_score = float(self.config.get("fallback_min_score", -1.0))

        try:
            qv = self._encode([query], batch_size=1)
            if qv.size == 0:
                return [], "none"

            scores, ids = self.index.search(qv, max(top_k * 4, top_k))

            candidates = []
            for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
                if idx < 0:
                    continue
                chunk = self.chunks[idx].copy()
                chunk["score"] = float(score)
                candidates.append(chunk)

            results = [chunk for chunk in candidates if float(chunk.get("score", 0.0)) >= min_score]
            results = results[:top_k]

            if not results and candidates:
                results = [
                    chunk
                    for chunk in candidates
                    if float(chunk.get("score", 0.0)) >= fallback_min_score
                ][:top_k]

            if not results:
                quality = "none"
            elif results[0]["score"] >= 0.7:
                quality = "high"
            elif results[0]["score"] >= 0.45:
                quality = "medium"
            else:
                quality = "low"

            logger.info(
                "RAG retrieve: query=%s results=%d quality=%s top_score=%.4f chunks=%d",
                query[:120].replace("\n", " "),
                len(results),
                quality,
                float(results[0].get("score", 0.0)) if results else 0.0,
                len(self.chunks),
            )
            return results, quality
        except Exception as e:
            logger.warning("检索失败: %s", e)
            return [], "none"


# =========================
# 病虫害分类器
# =========================
class PestClassifier:
    def __init__(self):
        self.model = None
        self.transform = None
        self._load_model()
    
    def _load_model(self):
        """加载ResNet50分类模型"""
        try:
            self.model = resnet50(pretrained=False)
            num_classes = 26  # 26类病虫害分类
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
            
            checkpoint = torch.load(PEST_MODEL_PATH, map_location=PEST_DEVICE)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(PEST_DEVICE)
            self.model.eval()
            
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            logger.info("✓ 病虫害分类模型加载成功")
        except Exception as e:
            logger.error("✗ 病虫害分类模型加载失败: %s", e)
            raise
    
    def classify(self, image: Image.Image) -> PestClassificationResult:
        """分类图片"""
        try:
            img_tensor = self.transform(image).unsqueeze(0).to(PEST_DEVICE)

            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, pred_class = torch.max(probabilities, 1)

            class_id = pred_class.item()
            confidence_score = confidence.item()

            return PestClassificationResult(
                class_id=class_id,
            class_name=get_pest_class_name(class_id, "zh"),
            class_name_en=get_pest_class_name(class_id, "en"),
            class_name_ms=get_pest_class_name(class_id, "ms"),
            confidence=round(confidence_score, 4),
                all_probs={
                    get_pest_class_name(i, "zh"): round(probabilities[0][i].item(), 4)
                    for i in range(len(PEST_CLASSES))
                },
            )
        except Exception as e:
            logger.error("分类失败: %s", e)
            raise



# =========================
# OpenAI-compatible vLLM proxy helpers
# =========================
class ApproxTokenizer:
    """轻量 token 估算器。

    openai_proxy 模式下，FastAPI 业务层不直接持有 vLLM tokenizer；
    这里仅用于 health、metadata、token budget 的近似计数，不参与真正生成。
    """

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        # 中文约 1 字/token，英文约 3-4 chars/token；这里取保守近似。
        estimated = max(1, int(len(str(text)) / 2.2))
        return [0] * estimated

    def apply_chat_template(self, messages: List[Dict], tokenize: bool = False, add_generation_prompt: bool = True, **kwargs):
        parts = []
        for msg in messages or []:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<{role}>\n{content}")
        if add_generation_prompt:
            parts.append("<assistant>\n")
        rendered = "\n\n".join(parts)
        if tokenize:
            return self.encode(rendered)
        return rendered


def is_openai_proxy_mode() -> bool:
    return ENGINE_MODE in OPENAI_PROXY_MODES


def import_httpx_for_proxy():
    try:
        import httpx  # type: ignore
        return httpx
    except Exception as exc:
        raise RuntimeError(
            "openai_proxy 模式需要安装 httpx：pip install httpx"
        ) from exc

# =========================
# 模型管理类
# =========================
class DurianGPTModel:
    def __init__(self, config: Dict):
        self.config = config
        self.device = DEVICE
        self.model = None
        self.tokenizer = None
        self.model_name = "DurianGPT-Qwen3-14B-vLLM"
        self.use_lora_adapter = False
        self.active_lora_path = None
        self.engine_mode = ENGINE_MODE
        self.max_model_len = int(os.getenv("DURIAN_MAX_MODEL_LEN", "6144"))
        self.gpu_memory_utilization = float(os.getenv("DURIAN_GPU_MEMORY_UTILIZATION", "0.90"))

    def load_model(self):
        logger.info("=" * 60)
        logger.info("正在加载榴莲GPT模型（engine_mode=%s）...", self.engine_mode)
        logger.info("=" * 60)

        if self.engine_mode in OPENAI_PROXY_MODES:
            # 真正流式推荐模式：模型由独立 vLLM OpenAI Server 常驻加载。
            # 当前 FastAPI 进程只做 RAG、对话管理、图片逻辑和 SSE 转发，不再占用 14B 模型显存。
            self.model = "openai_proxy"
            self.tokenizer = ApproxTokenizer()
            self.model_name = f"DurianGPT-OpenAIProxy({DURIAN_OPENAI_MODEL})"
            self.use_lora_adapter = False
            self.active_lora_path = None
            logger.info("✓ openai_proxy 模式已启用；FastAPI 不在本进程加载 vLLM LLM")
            logger.info("vLLM OpenAI base_url: %s", DURIAN_OPENAI_BASE_URL)
            logger.info("vLLM OpenAI model: %s", DURIAN_OPENAI_MODEL)
            logger.info("=" * 60)
            return

        try:
            merged_model_path = self.config.get("merged_model_path", "")
            lora_model_path = self.config.get("lora_model_path", "")
            base_model_path = self.config["base_model"]
            enable_lora = bool(self.config.get("enable_lora", False))

            if self.config.get("use_merged") and merged_model_path and os.path.exists(merged_model_path):
                model_path = merged_model_path
                self.use_lora_adapter = False
                self.active_lora_path = None
                self.model_name = "DurianGPT-Qwen3-14B-Merged-vLLM"
                logger.info("使用 merged 模型: %s", model_path)
            else:
                model_path = base_model_path
                if enable_lora and lora_model_path and os.path.exists(lora_model_path):
                    self.use_lora_adapter = True
                    self.active_lora_path = lora_model_path
                    self.model_name = "DurianGPT-Qwen3-14B-LoRA-vLLM"
                    logger.info("使用 base 模型: %s", model_path)
                    logger.info("启用 LoRA adapter: %s", self.active_lora_path)
                else:
                    self.use_lora_adapter = False
                    self.active_lora_path = None
                    self.model_name = "DurianGPT-Qwen3-14B-Base-vLLM"
                    logger.info("使用 base 模型: %s", model_path)
                    if enable_lora:
                        logger.warning("LoRA 路径不存在或为空，已回退为 base 模型。lora_model_path=%s", lora_model_path)
                    else:
                        logger.info("LoRA 已禁用，将只使用 base 模型")

            logger.info(
                "初始化 vLLM LLM（attention backend: %s）...",
                os.environ.get("VLLM_ATTENTION_BACKEND", "default"),
            )

            # 强制 GPU 模式，不支持 CPU
            if not torch.cuda.is_available():
                raise RuntimeError("未检测到 CUDA GPU，vLLM GPU 推理服务无法启动。请检查 CUDA_VISIBLE_DEVICES、NVIDIA 驱动和 PyTorch CUDA。")

            self.max_model_len = int(os.getenv("DURIAN_MAX_MODEL_LEN", "6144"))
            self.gpu_memory_utilization = float(os.getenv("DURIAN_GPU_MEMORY_UTILIZATION", "0.90"))
            logger.info(
                "vLLM 参数: max_model_len=%d, gpu_memory_utilization=%.2f",
                self.max_model_len,
                self.gpu_memory_utilization,
            )

            llm_kwargs = dict(
                model=model_path,
                tensor_parallel_size=1,
                dtype="float16",
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                max_num_seqs=1,
                enforce_eager=True,
                trust_remote_code=True,
                enable_lora=self.use_lora_adapter,
                disable_custom_all_reduce=True,
                disable_log_stats=True,
            )

            if self.use_lora_adapter:
                # vLLM 默认 max_lora_rank 可能偏小；你的 adapter rank=16，这里留足余量。
                llm_kwargs["max_loras"] = 1
                llm_kwargs["max_lora_rank"] = int(self.config.get("max_lora_rank", 64))

            try:
                self.model = LLM(**llm_kwargs)
            except TypeError as e:
                # 兼容旧版 vLLM：如果不支持 max_loras / max_lora_rank，就移除后重试。
                logger.warning("当前 vLLM 不支持部分 LoRA 参数，准备回退重试: %s", e)
                llm_kwargs.pop("max_loras", None)
                llm_kwargs.pop("max_lora_rank", None)
                self.model = LLM(**llm_kwargs)

            self.tokenizer = self.model.get_tokenizer()
            logger.info("✓ vLLM 模型加载成功！")
            logger.info("=" * 60)
            self._print_model_info()

        except Exception as e:
            logger.error("模型加载失败: %s", e)
            raise

    def _proxy_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if DURIAN_OPENAI_API_KEY:
            headers["Authorization"] = f"Bearer {DURIAN_OPENAI_API_KEY}"
        return headers

    def _normalize_proxy_max_tokens(self, requested_tokens: Any) -> int:
        try:
            value = int(requested_tokens or INFERENCE_CONFIG["max_new_tokens"])
        except Exception:
            value = int(INFERENCE_CONFIG["max_new_tokens"])
        hard_output_cap = int(os.getenv("DURIAN_HARD_OUTPUT_CAP", "2048"))
        return max(1, min(value, hard_output_cap))

    def _build_proxy_payload(self, messages: List[Dict], stream: bool, **kwargs) -> Dict[str, Any]:
        max_tokens = self._normalize_proxy_max_tokens(kwargs.get("max_new_tokens", kwargs.get("max_tokens")))
        payload: Dict[str, Any] = {
            "model": DURIAN_OPENAI_MODEL,
            "messages": self._trim_messages(messages),
            "temperature": float(kwargs.get("temperature", INFERENCE_CONFIG["temperature"])),
            "top_p": float(kwargs.get("top_p", INFERENCE_CONFIG["top_p"])),
            "max_tokens": max_tokens,
            "stream": bool(stream),
        }

        top_k = kwargs.get("top_k", INFERENCE_CONFIG.get("top_k"))
        if top_k is not None:
            try:
                payload["top_k"] = int(top_k)
            except Exception:
                pass

        repetition_penalty = kwargs.get("repetition_penalty", INFERENCE_CONFIG.get("repetition_penalty"))
        if repetition_penalty is not None:
            try:
                payload["repetition_penalty"] = float(repetition_penalty)
            except Exception:
                pass

        # vLLM OpenAI server 支持额外字段；Qwen3 通过 chat_template_kwargs 关闭 thinking。
        if DURIAN_PROXY_DISABLE_THINKING:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        return payload

    def _proxy_chat_completion(self, messages: List[Dict], **kwargs) -> str:
        httpx = import_httpx_for_proxy()
        payload = self._build_proxy_payload(messages, stream=False, **kwargs)
        url = f"{DURIAN_OPENAI_BASE_URL}/chat/completions"
        try:
            with httpx.Client(timeout=DURIAN_OPENAI_TIMEOUT) as client:
                resp = client.post(url, headers=self._proxy_headers(), json=payload)
                resp.raise_for_status()
                data = resp.json()
            return str(data.get("choices", [{}])[0].get("message", {}).get("content", "") or "")
        except Exception as exc:
            logger.error("OpenAI proxy 非流式请求失败: %s", exc)
            raise

    def _proxy_chat_stream(self, messages: List[Dict], **kwargs) -> Iterator[str]:
        httpx = import_httpx_for_proxy()
        payload = self._build_proxy_payload(messages, stream=True, **kwargs)
        url = f"{DURIAN_OPENAI_BASE_URL}/chat/completions"
        try:
            with httpx.Client(timeout=None) as client:
                with client.stream("POST", url, headers=self._proxy_headers(), json=payload) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        if isinstance(line, bytes):
                            line = line.decode("utf-8", errors="ignore")
                        line = str(line).strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
                        if not data:
                            continue
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                            choice = (obj.get("choices") or [{}])[0]
                            delta = choice.get("delta") or {}
                            content = delta.get("content") or ""
                            # 忽略 reasoning_content，避免前端显示思考过程。
                            if content:
                                yield str(content)
                        except Exception as parse_exc:
                            logger.debug("忽略无法解析的 OpenAI stream 行: %s; error=%s", data[:200], parse_exc)
                            continue
        except Exception as exc:
            logger.error("OpenAI proxy 流式请求失败: %s", exc)
            raise

    def _print_model_info(self):
        if self.model is None:
            return

        logger.info("模型名称: %s", self.model_name)
        logger.info("计算设备: %s", self.device)

        if torch.cuda.is_available():
            logger.info("GPU 显卡: %s", torch.cuda.get_device_name(0))
            logger.info(
                "显存总量: %.2f GB",
                torch.cuda.get_device_properties(0).total_memory / 1e9,
            )

    def get_memory_usage(self) -> Dict:
        if not torch.cuda.is_available():
            return {"status": "CPU mode"}

        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        }

    def _limit_generation_tokens(self, prompt_text: str, requested_tokens: int) -> int:
        """动态限制输出 token，避免 input + output 超过 vLLM max_model_len。

        关键点：
        1. 不再写死 3072 / 768；
        2. 默认上下文 6144，默认输出上限 2048；
        3. 若前端传得太小，可通过 DURIAN_MIN_REQUESTED_OUTPUT_TOKENS 设置最低输出预算。
        """
        max_model_len = getattr(
            self,
            "max_model_len",
            int(os.getenv("DURIAN_MAX_MODEL_LEN", "6144")),
        )
        safety_margin = int(os.getenv("DURIAN_TOKEN_SAFETY_MARGIN", "64"))
        hard_output_cap = int(os.getenv("DURIAN_HARD_OUTPUT_CAP", "2048"))
        min_requested_output = int(os.getenv("DURIAN_MIN_REQUESTED_OUTPUT_TOKENS", "0"))

        try:
            requested_tokens = int(requested_tokens or INFERENCE_CONFIG["max_new_tokens"])
        except Exception:
            requested_tokens = int(INFERENCE_CONFIG["max_new_tokens"])

        # 防止前端仍然传 512，导致长回答被截断；如果确实想短答，可把该环境变量设为 0。
        if min_requested_output > 0:
            requested_tokens = max(requested_tokens, min_requested_output)

        input_token_count = len(self.tokenizer.encode(prompt_text))
        available_output_tokens = max_model_len - input_token_count - safety_margin

        logger.info(
            "Token budget: input_tokens=%d, requested_output=%d, hard_output_cap=%d, available_output=%d, max_model_len=%d",
            input_token_count,
            requested_tokens,
            hard_output_cap,
            available_output_tokens,
            max_model_len,
        )

        if available_output_tokens <= 0:
            raise RuntimeError(
                f"输入内容过长，input_tokens={input_token_count}, max_model_len={max_model_len}。"
                "请减少历史消息、RAG资料或缩短问题。"
            )

        if available_output_tokens < 64:
            raise RuntimeError(
                f"输入内容接近上下文上限，input_tokens={input_token_count}, "
                f"available_output_tokens={available_output_tokens}。"
                "剩余输出空间不足，请减少历史消息、RAG资料或缩短问题。"
            )

        final_tokens = min(requested_tokens, hard_output_cap, available_output_tokens)

        if final_tokens < 128:
            logger.warning(
                "可用输出 token 较少：input_tokens=%d, final_output_tokens=%d",
                input_token_count,
                final_tokens,
            )

        logger.info("Final generation token limit: final_tokens=%d", final_tokens)
        return int(final_tokens)

    def _trim_messages(self, messages: List[Dict]) -> List[Dict]:
        max_history = INFERENCE_CONFIG.get("max_history_messages", 5)
        if len(messages) <= max_history:
            return messages

        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]

        trimmed = system_msgs + other_msgs[-max_history:]
        logger.info("历史消息已裁剪: %d -> %d", len(messages), len(trimmed))
        return trimmed

    def _apply_chat_template(self, messages: List[Dict]) -> str:
        """应用 Qwen chat template，并尽量关闭 Qwen3 thinking 输出。"""
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def _review_answer_quality(self, user_query: str, response: str, response_language: str = "zh", **kwargs) -> Dict[str, Any]:
        """轻量质量审查器：判断回答是否需要二次改写。

        审查维度只看三件事：
        1. relevance：有没有回答用户问题，是否跑题；
        2. detail：复杂问题是否足够展开；
        3. readability：结构是否好读，是否一上来就是大表格/坏表格。

        这一步输出短 JSON，成本远低于完整 editor；只有 needs_rewrite=true 时才进入二次编辑。
        """
        lang = (response_language or "zh").lower()
        default_review = {
            "relevance_score": 0.8,
            "detail_score": 0.8,
            "readability_score": 0.8,
            "needs_rewrite": False,
            "rewrite_focus": [],
            "reason": "heuristic pass",
        }

        if not env_bool("DURIAN_ENABLE_QUALITY_CHECKER", True):
            return default_review

        q = (user_query or "").strip()
        r = (response or "").strip()
        if not q or not r or is_greeting_only(q):
            return default_review

        # 快速硬规则：明显坏格式/明显没答题时，不必再让审查器判断，直接触发 editor。
        focus = []
        if response_starts_with_table_or_list(r):
            focus.append("improve_readability_start_with_conclusion")
        if has_malformed_pipe_table(r) or has_slash_style_table(r):
            focus.append("fix_markdown_table")
        if is_binary_choice_question(q) and not has_direct_answer_signal(r):
            focus.append("answer_choice_question_directly")
        if is_detail_worthy_query(q, lang) and response_seems_too_short(r, lang):
            focus.append("add_practical_details")
        if asks_treatment_or_pesticide(q) and not has_treatment_or_pesticide_content(r):
            focus.append("add_treatment_and_pesticide_direction")

        # 如果硬规则命中两个及以上，直接认为需要改写，省一次审查器调用。
        if len(set(focus)) >= 2:
            return {
                "relevance_score": 0.75,
                "detail_score": 0.55 if "add_practical_details" in focus else 0.75,
                "readability_score": 0.55 if any(x in focus for x in ["improve_readability_start_with_conclusion", "fix_markdown_table"]) else 0.75,
                "needs_rewrite": True,
                "rewrite_focus": sorted(set(focus)),
                "reason": "hard rules triggered",
            }

        # 简单非农业闲聊/普通知识短答，一般不跑审查器，避免慢。
        if not is_detail_worthy_query(q, lang) and len(q) <= 20 and len(r) >= 60 and not focus:
            return default_review

        reviewer_prompt = build_quality_checker_instruction(q, r, lang)
        review_messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict answer quality judge. Output JSON only. "
                    "Do not rewrite the answer. Do not add explanations outside JSON."
                ),
            },
            {"role": "user", "content": reviewer_prompt},
        ]

        try:
            prompt = self._apply_chat_template(review_messages)
            review_tokens = int(os.getenv("DURIAN_QUALITY_CHECKER_MAX_TOKENS", "160"))
            review_tokens = min(review_tokens, self._limit_generation_tokens(prompt, review_tokens))
            params = SamplingParams(
                max_tokens=review_tokens,
                temperature=float(os.getenv("DURIAN_QUALITY_CHECKER_TEMPERATURE", "0.0")),
                top_p=float(os.getenv("DURIAN_QUALITY_CHECKER_TOP_P", "0.8")),
                top_k=int(os.getenv("DURIAN_QUALITY_CHECKER_TOP_K", "20")),
                repetition_penalty=1.0,
            )
            lora_request = None
            if self.use_lora_adapter and self.active_lora_path:
                lora_request = LoRARequest("durian-lora", 1, self.active_lora_path)
            outputs = self.model.generate([prompt], params, lora_request=lora_request)
            raw = clean_model_output(outputs[0].outputs[0].text.strip())
            review = parse_quality_review_json(raw)

            # 合并硬规则 focus，避免审查器漏掉明显格式问题。
            merged_focus = list(dict.fromkeys((review.get("rewrite_focus") or []) + focus))
            review["rewrite_focus"] = merged_focus

            threshold_rel = float(os.getenv("DURIAN_RELEVANCE_THRESHOLD", "0.72"))
            threshold_detail = float(os.getenv("DURIAN_DETAIL_THRESHOLD", "0.62"))
            threshold_readability = float(os.getenv("DURIAN_READABILITY_THRESHOLD", "0.62"))
            needs = bool(review.get("needs_rewrite")) or \
                float(review.get("relevance_score", 1.0)) < threshold_rel or \
                float(review.get("detail_score", 1.0)) < threshold_detail or \
                float(review.get("readability_score", 1.0)) < threshold_readability or \
                bool(merged_focus)
            review["needs_rewrite"] = needs
            logger.info(
                "质量审查: needs=%s rel=%.2f detail=%.2f read=%.2f focus=%s reason=%s",
                review.get("needs_rewrite"),
                float(review.get("relevance_score", 0.0)),
                float(review.get("detail_score", 0.0)),
                float(review.get("readability_score", 0.0)),
                review.get("rewrite_focus"),
                review.get("reason", ""),
            )
            return review
        except Exception as e:
            logger.warning("质量审查器失败，回退硬规则: %s", e)
            return {
                "relevance_score": 0.8,
                "detail_score": 0.6 if "add_practical_details" in focus else 0.8,
                "readability_score": 0.6 if focus else 0.8,
                "needs_rewrite": bool(focus),
                "rewrite_focus": sorted(set(focus)),
                "reason": "quality checker failed, heuristic fallback",
            }

    def _post_edit_response(self, user_query: str, response: str, response_language: str = "zh", **kwargs) -> str:
        """条件二次编辑器：先做质量审查，只在相关性/详细度/可读性不达标时改写。"""
        if not env_bool("DURIAN_ENABLE_POST_EDITOR", False):
            return response

        mode = os.getenv("DURIAN_EDITOR_MODE", "quality").strip().lower()
        quality_review = None

        if mode in {"off", "false", "none", "disable", "disabled"}:
            return response
        if mode in {"always", "force"}:
            needs_edit = True
            quality_review = {"rewrite_focus": ["general_polish"], "reason": "forced editor"}
        elif mode in {"quality", "smart", "auto"}:
            quality_review = self._review_answer_quality(user_query, response, response_language, **kwargs)
            needs_edit = bool(quality_review.get("needs_rewrite"))
        else:
            # 兼容旧逻辑。
            needs_edit = should_post_edit_response(user_query, response, response_language)
            quality_review = {"rewrite_focus": [], "reason": "legacy heuristic"}

        if not needs_edit:
            return response

        lang = (response_language or "zh").lower()
        editor_instruction = build_post_editor_instruction(user_query, response, lang, quality_review=quality_review)
        editor_messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict answer editor. You rewrite only when needed. "
                    "Preserve the original diagnosis, facts, and safety meaning unless they are off-topic. "
                    "Return only the final edited answer."
                ),
            },
            {"role": "user", "content": editor_instruction},
        ]

        try:
            editor_prompt = self._apply_chat_template(editor_messages)
            editor_max_tokens = int(os.getenv("DURIAN_EDITOR_MAX_TOKENS", "1200"))
            editor_max_tokens = min(editor_max_tokens, self._limit_generation_tokens(editor_prompt, editor_max_tokens))

            params = SamplingParams(
                max_tokens=editor_max_tokens,
                temperature=float(os.getenv("DURIAN_EDITOR_TEMPERATURE", "0.2")),
                top_p=float(os.getenv("DURIAN_EDITOR_TOP_P", "0.85")),
                top_k=int(os.getenv("DURIAN_EDITOR_TOP_K", "40")),
                repetition_penalty=float(os.getenv("DURIAN_EDITOR_REPETITION_PENALTY", "1.03")),
            )
            lora_request = None
            if self.use_lora_adapter and self.active_lora_path:
                lora_request = LoRARequest("durian-lora", 1, self.active_lora_path)

            outputs = self.model.generate([editor_prompt], params, lora_request=lora_request)
            edited = clean_model_output(outputs[0].outputs[0].text.strip())
            edited = ensure_adaptive_markdown_table(edited, user_query, lang)

            if accept_edited_response(original=response, edited=edited, response_language=lang, quality_review=quality_review):
                logger.info(
                    "回答已按质量审查结果改写: before_chars=%d, after_chars=%d, focus=%s",
                    len(response), len(edited), quality_review.get("rewrite_focus") if quality_review else [],
                )
                return edited
            logger.info("质量改写结果未通过校验，保留原回答: before_chars=%d, edited_chars=%d", len(response), len(edited or ""))
            return response
        except Exception as e:
            logger.warning("二次编辑器失败，保留原回答: %s", e)
            return response

    def generate(self, messages: List[Dict], **kwargs) -> str:
        if self.model is None:
            raise RuntimeError("模型未加载")

        if self.engine_mode in OPENAI_PROXY_MODES:
            raw_response = self._proxy_chat_completion(messages, **kwargs)
            response = clean_model_output(raw_response.strip())
            user_query = extract_last_user_content_from_dict_messages(messages)
            response_lang = kwargs.get("response_language", "zh")
            scenario = detect_query_scenario(user_query)
            if scenario != "consumer" and not kwargs.get("skip_post_editor", False):
                response = ensure_adaptive_markdown_table(response, user_query, response_lang)
            logger.info(
                "OpenAI proxy 生成完成: chars=%d, approx_tokens=%d",
                len(response),
                len(self.tokenizer.encode(response)) if self.tokenizer else 0,
            )
            return response

        messages = self._trim_messages(messages)
        text = self._apply_chat_template(messages)

        requested_max_tokens = kwargs.get("max_new_tokens", INFERENCE_CONFIG["max_new_tokens"])
        max_tokens = self._limit_generation_tokens(text, requested_max_tokens)

        params_dict = {
            "max_tokens": max_tokens,
            "temperature": kwargs.get("temperature", INFERENCE_CONFIG["temperature"]),
            "top_p": kwargs.get("top_p", INFERENCE_CONFIG["top_p"]),
            "top_k": kwargs.get("top_k", INFERENCE_CONFIG["top_k"]),
            "repetition_penalty": kwargs.get("repetition_penalty", INFERENCE_CONFIG["repetition_penalty"]),
        }

        min_tokens = kwargs.get("min_tokens")
        if min_tokens is not None:
            params_dict["min_tokens"] = min(min_tokens, max_tokens)

        ignore_eos = kwargs.get("ignore_eos")
        if ignore_eos is not None:
            params_dict["ignore_eos"] = ignore_eos

        try:
            sampling_params = SamplingParams(**params_dict)
        except TypeError:
            params_dict.pop("min_tokens", None)
            params_dict.pop("ignore_eos", None)
            sampling_params = SamplingParams(**params_dict)

        lora_request = None
        if self.use_lora_adapter and self.active_lora_path:
            lora_request = LoRARequest("durian-lora", 1, self.active_lora_path)

        outputs = self.model.generate([text], sampling_params, lora_request=lora_request)
        response = clean_model_output(outputs[0].outputs[0].text.strip())
        user_query = extract_last_user_content_from_dict_messages(messages)
        response_lang = kwargs.get("response_language", "zh")
        scenario = detect_query_scenario(user_query)
        if scenario != "consumer":
            response = ensure_adaptive_markdown_table(
                response,
                user_query,
                response_lang,
            )

        # 可选二次扩写：不是继续堆 prompt，而是后端质量控制。
        # 当回答明显偏短且问题值得展开时，追加一次“扩写/补全”生成。
        if (scenario != "consumer") and (not kwargs.get("skip_detail_expand", False)) and env_bool("DURIAN_ENABLE_DETAIL_EXPAND", False) and is_detail_worthy_query(user_query, response_lang) and response_seems_too_short(response, response_lang):
            try:
                expand_messages = list(messages) + [
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": build_expansion_instruction(response_lang)},
                ]
                expand_text = self._apply_chat_template(expand_messages)
                expand_tokens = min(
                    int(os.getenv("DURIAN_EXPAND_MAX_TOKENS", "900")),
                    self._limit_generation_tokens(expand_text, int(os.getenv("DURIAN_EXPAND_MAX_TOKENS", "900"))),
                )
                expand_params = SamplingParams(
                    max_tokens=expand_tokens,
                    temperature=min(float(kwargs.get("temperature", INFERENCE_CONFIG["temperature"])), 0.45),
                    top_p=kwargs.get("top_p", INFERENCE_CONFIG["top_p"]),
                    top_k=kwargs.get("top_k", INFERENCE_CONFIG["top_k"]),
                    repetition_penalty=kwargs.get("repetition_penalty", INFERENCE_CONFIG["repetition_penalty"]),
                )
                expanded_outputs = self.model.generate([expand_text], expand_params, lora_request=lora_request)
                expanded = clean_model_output(expanded_outputs[0].outputs[0].text.strip())
                if len(expanded) > len(response) * 1.15:
                    response = expanded
                    response = ensure_adaptive_markdown_table(response, user_query, response_lang)
                    logger.info("回答已二次扩写: chars=%d", len(response))
            except Exception as e:
                logger.warning("二次扩写失败，使用原始回答: %s", e)

        # 避免 kwargs 中已包含 response_language，导致 Python 报 multiple values for keyword argument
        edit_kwargs = dict(kwargs)
        edit_kwargs.pop("response_language", None)
        if (scenario != "consumer") and (not kwargs.get("skip_post_editor", False)):
            response = self._post_edit_response(
                user_query=user_query,
                response=response,
                response_language=response_lang,
                **edit_kwargs,
            )

        logger.info(
            "模型生成完成: chars=%d, tokens=%d, min_tokens=%s, max_tokens=%d",
            len(response),
            len(self.tokenizer.encode(response)),
            kwargs.get("min_tokens"),
            max_tokens,
        )
        return response

    def generate_stream(self, messages: List[Dict], **kwargs) -> Iterator[str]:
        """流式生成。

        local_vllm 模式：兼容旧逻辑，vLLM LLM.generate 一次性生成后再分块。
        openai_proxy 模式：转发独立 vLLM OpenAI Server 的 stream=true，是真正 token/chunk 流式。
        """
        if self.model is None:
            raise RuntimeError("模型未加载")

        if self.engine_mode in OPENAI_PROXY_MODES:
            for chunk in self._proxy_chat_stream(messages, **kwargs):
                if chunk:
                    yield chunk
            return

        messages = self._trim_messages(messages)
        text = self._apply_chat_template(messages)

        requested_max_tokens = kwargs.get("max_new_tokens", INFERENCE_CONFIG["max_new_tokens"])
        max_tokens = self._limit_generation_tokens(text, requested_max_tokens)

        params_dict = {
            "max_tokens": max_tokens,
            "temperature": kwargs.get("temperature", INFERENCE_CONFIG["temperature"]),
            "top_p": kwargs.get("top_p", INFERENCE_CONFIG["top_p"]),
            "top_k": kwargs.get("top_k", INFERENCE_CONFIG["top_k"]),
            "repetition_penalty": kwargs.get("repetition_penalty", INFERENCE_CONFIG["repetition_penalty"]),
        }

        min_tokens = kwargs.get("min_tokens")
        if min_tokens is not None:
            params_dict["min_tokens"] = min(min_tokens, max_tokens)

        ignore_eos = kwargs.get("ignore_eos")
        if ignore_eos is not None:
            params_dict["ignore_eos"] = ignore_eos

        try:
            sampling_params = SamplingParams(**params_dict)
        except TypeError:
            params_dict.pop("min_tokens", None)
            params_dict.pop("ignore_eos", None)
            sampling_params = SamplingParams(**params_dict)

        lora_request = None
        if self.use_lora_adapter and self.active_lora_path:
            lora_request = LoRARequest("durian-lora", 1, self.active_lora_path)

        try:
            # vLLM 一次性生成完整结果
            outputs = self.model.generate([text], sampling_params, lora_request=lora_request)
            full_text = clean_model_output(outputs[0].outputs[0].text.strip())
            user_query = extract_last_user_content_from_dict_messages(messages)
            response_lang = kwargs.get("response_language", "zh")
            full_text = ensure_adaptive_markdown_table(
                full_text,
                user_query,
                response_lang,
            )
            if env_bool("DURIAN_ENABLE_DETAIL_EXPAND", False) and is_detail_worthy_query(user_query, response_lang) and response_seems_too_short(full_text, response_lang):
                try:
                    expand_messages = list(messages) + [
                        {"role": "assistant", "content": full_text},
                        {"role": "user", "content": build_expansion_instruction(response_lang)},
                    ]
                    expand_text = self._apply_chat_template(expand_messages)
                    expand_tokens = min(
                        int(os.getenv("DURIAN_EXPAND_MAX_TOKENS", "900")),
                        self._limit_generation_tokens(expand_text, int(os.getenv("DURIAN_EXPAND_MAX_TOKENS", "900"))),
                    )
                    expand_params = SamplingParams(
                        max_tokens=expand_tokens,
                        temperature=min(float(kwargs.get("temperature", INFERENCE_CONFIG["temperature"])), 0.45),
                        top_p=kwargs.get("top_p", INFERENCE_CONFIG["top_p"]),
                        top_k=kwargs.get("top_k", INFERENCE_CONFIG["top_k"]),
                        repetition_penalty=kwargs.get("repetition_penalty", INFERENCE_CONFIG["repetition_penalty"]),
                    )
                    expanded_outputs = self.model.generate([expand_text], expand_params, lora_request=lora_request)
                    expanded = clean_model_output(expanded_outputs[0].outputs[0].text.strip())
                    if len(expanded) > len(full_text) * 1.15:
                        full_text = ensure_adaptive_markdown_table(expanded, user_query, response_lang)
                except Exception as e:
                    logger.warning("流式二次扩写失败，使用原始回答: %s", e)
            
            # 避免 kwargs 中已包含 response_language，导致 Python 报 multiple values for keyword argument
            edit_kwargs = dict(kwargs)
            edit_kwargs.pop("response_language", None)
            full_text = self._post_edit_response(
                user_query=user_query,
                response=full_text,
                response_language=response_lang,
                **edit_kwargs,
            )

            # 按字符数分块（每块 15 字符）
            chunk_size = 15
            for i in range(0, len(full_text), chunk_size):
                chunk = full_text[i:i + chunk_size]
                if chunk:
                    yield chunk
                
        except Exception as e:
            logger.error("流式生成异常: %s", e)
            raise


# =========================
# 模型输出清理
# =========================
def clean_model_output(text: str) -> str:
    """清理模型输出中的思考标签、内部分析和常见开场废话"""
    if not text:
        return ""

    # 移除完整闭合的思考标签
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<analysis>[\s\S]*?</analysis>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<reasoning>[\s\S]*?</reasoning>", "", text, flags=re.IGNORECASE)

    # 移除未闭合的标签本身，但不删除后续内容
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?analysis>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?reasoning>", "", text, flags=re.IGNORECASE)

    # 删除常见开场废话
    text = re.sub(r"^\s*(好的|首先|让我|我来|需要先).*?\n", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(Okay|Let me|First|I need to|I should).*?\n", "", text, flags=re.IGNORECASE)

    # 修复模型常见的伪表格：斜杠表格、制表符表格、只有表头的坏表格。
    # 这里不是追加兜底模板，只是把模型已经输出的结构化内容转成合法 Markdown。
    text = convert_pseudo_tables_to_markdown(text)
    text = repair_fake_markdown_tables(text)

    # 清理多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()




def enforce_classification_consistency(text: str, class_name: str, confidence: float, response_language: str = "zh") -> str:
    """防止图像分析回答直接否定分类模型给出的类别。

    允许表达“置信度低、需要复核”，但不允许出现“更可能不是该类别”等前后打架表述。
    """
    if not text or not class_name:
        return text or ""

    escaped = re.escape(class_name)

    # 先删除或替换最常见的冲突句式。覆盖中文和部分英文/马来/泰文否定结构。
    conflict_patterns = [
        rf"当前图像更可能是[^。；;\n]*而非{escaped}[^。；;\n]*[。；;]?",
        rf"当前图片更可能是[^。；;\n]*而非{escaped}[^。；;\n]*[。；;]?",
        rf"更可能不是{escaped}[^。；;\n]*[。；;]?",
        rf"不像是{escaped}[^。；;\n]*[。；;]?",
        rf"不是典型的{escaped}[^。；;\n]*[。；;]?",
        rf"并非{escaped}[^。；;\n]*[。；;]?",
        rf"而非{escaped}[^。；;\n]*[。；;]?",
        rf"more likely not (?:to be )?{escaped}[^.\n]*\. ?",
        rf"not typical(?:ly)? (?:of )?{escaped}[^.\n]*\. ?",
        rf"rather than {escaped}[^.\n]*\. ?",
        rf"bukan {escaped}[^.\n]*\. ?",
        rf"ไม่ใช่{escaped}[^.\n]*\. ?",
    ]
    for pattern in conflict_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    lang = (response_language or "zh").lower()
    pct = confidence * 100
    if lang == "en":
        replacement = (
            f"The image classification result is {class_name} ({pct:.1f}%). "
            "Because the confidence is low, treat it as a suspected diagnosis and confirm it with field symptoms."
            if confidence < 0.55 else
            f"The image classification result is {class_name} ({pct:.1f}%). Use this category as the main basis for field checking and management."
        )
        heading_re = r"(##\s*Problem Assessment\s*\n)(.*?)(\n##|\n\n##|$)"
    elif lang == "ms":
        replacement = (
            f"Hasil klasifikasi imej ialah {class_name} ({pct:.1f}%). Oleh sebab tahap keyakinan rendah, anggap ini sebagai diagnosis disyaki dan sahkan dengan gejala di lapangan."
            if confidence < 0.55 else
            f"Hasil klasifikasi imej ialah {class_name} ({pct:.1f}%). Gunakan kategori ini sebagai asas utama untuk pemeriksaan dan pengurusan di lapangan."
        )
        heading_re = r"(##\s*Penilaian Masalah\s*\n)(.*?)(\n##|\n\n##|$)"
    elif lang == "th":
        replacement = (
            f"ผลการจำแนกภาพคือ {class_name} ({pct:.1f}%) เนื่องจากค่าความมั่นใจต่ำ ให้ถือเป็นข้อสงสัยและยืนยันร่วมกับอาการในแปลงอีกครั้ง"
            if confidence < 0.55 else
            f"ผลการจำแนกภาพคือ {class_name} ({pct:.1f}%) ให้ใช้หมวดหมู่นี้เป็นหลักในการตรวจสอบและจัดการในแปลง"
        )
        heading_re = r"(##\s*การประเมินปัญหา\s*\n)(.*?)(\n##|\n\n##|$)"
    else:
        replacement = (
            f"当前图像识别结果为{class_name}（{pct:.1f}%）。由于置信度较低，建议先按疑似问题处理，并结合现场症状进一步复核。"
            if confidence < 0.55 else
            f"当前图像识别结果为{class_name}（{pct:.1f}%），建议围绕该类别进行现场复核和处理。"
        )
        heading_re = r"(##\s*问题判断\s*\n)(.*?)(\n##|\n\n##|$)"

    # 如果回答有“问题判断/Problem Assessment”等标题，直接替换该段，确保不再打架。
    new_text, n = re.subn(
        heading_re,
        lambda m: m.group(1) + replacement + (m.group(3) if m.group(3).strip() else ""),
        text,
        count=1,
        flags=re.S | re.I,
    )
    if n:
        text = new_text
    else:
        # 没有对应标题时，在分类结果后补一句一致性说明。
        text = replacement + "\n\n" + text.strip()

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()




def _split_pseudo_table_row(line: str) -> List[str]:
    """把模型生成的伪表格行拆成单元格。

    支持：
    1. A / B / C
    2. A\tB\tC
    3. A    B    C

    重点修复：
    - 跳过 ---- / ---- / ---- 这种分隔行；
    - 跳过 — / — / — 这种占位行；
    - 不把普通句子误判成表格。
    """
    value = (line or "").strip()
    if not value:
        return []

    if value.startswith("|") and value.endswith("|"):
        return []

    if _is_pseudo_separator_row(value):
        return []

    if "\t" in value:
        cells = [c.strip() for c in value.split("\t")]
    elif "/" in value:
        # 允许 “A/B/C” 和 “A / B / C”，但要求至少 3 列，避免误伤日期、路径。
        cells = [c.strip() for c in re.split(r"\s*/\s*", value)]
    elif re.search(r"\S\s{2,}\S", value):
        cells = [c.strip() for c in re.split(r"\s{2,}", value)]
    else:
        return []

    cells = [re.sub(r"\s+", " ", c).strip() for c in cells if c.strip()]

    if len(cells) < 3:
        return []

    # 全是横杠/破折号/占位符，不是数据行。
    if all(re.fullmatch(r"[-—–_]+", c) or c in {"—", "-", "--", "——"} for c in cells):
        return []

    return cells


def _is_pseudo_separator_row(line: str) -> bool:
    """判断伪表格分隔行，如 ---- / ---- / ----。"""
    value = (line or "").strip()
    if not value:
        return False

    # Markdown pipe 分隔线
    if re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", value):
        return True

    # slash 分隔线：---- / ---- / ---- 或 —— / —— / ——
    if "/" in value:
        parts = [p.strip() for p in re.split(r"\s*/\s*", value) if p.strip()]
        if len(parts) >= 2 and all(re.fullmatch(r"[-—–_]{2,}", p) for p in parts):
            return True

    # 纯横线
    if re.fullmatch(r"[-—–_\s]{3,}", value):
        return True

    return False


def _looks_like_pseudo_table_header(cells: List[str]) -> bool:
    """判断拆出来的行是否像表头。"""
    if len(cells) < 3:
        return False
    joined = " ".join(cells).lower()
    header_keywords = [
        "品种", "外形", "特征", "风味", "口感", "用途", "保存", "冷链", "产地", "糖度", "货架", "香气", "颜色", "适配",
        "判断", "操作", "复查", "处理", "项目", "症状", "药剂", "检查", "措施", "风险", "严重", "时间",
        "variety", "feature", "flavour", "flavor", "taste", "use", "storage", "cold", "origin", "brix", "aroma", "color", "colour",
        "check", "action", "timing", "treatment", "symptom", "risk", "severity",
        "varieti", "ciri", "rasa", "penyimpanan", "tindakan", "semak", "rawatan",
        "พันธุ์", "ลักษณะ", "รส", "เก็บ", "ตรวจ", "จัดการ",
    ]
    return any(k in joined for k in header_keywords)


def _normalize_table_cells(cells: List[str], expected_cols: int) -> Optional[List[str]]:
    """把一行单元格修正到固定列数。"""
    if not cells:
        return None
    if all(re.fullmatch(r"[-—–_]+", c) or c in {"—", "-", "--", "——"} for c in cells):
        return None
    if len(cells) < max(2, expected_cols - 1):
        return None
    if len(cells) < expected_cols:
        cells = cells + [""] * (expected_cols - len(cells))
    elif len(cells) > expected_cols:
        # 多出的内容合并到最后一列，避免 Markdown 列数错乱。
        cells = cells[: expected_cols - 1] + [" / ".join(cells[expected_cols - 1:])]
    return cells


def convert_pseudo_tables_to_markdown(text: str) -> str:
    """把 LLM 常见伪表格转换为合法 Markdown 表格。

    重点修复：
    - A / B / C 斜杠表格；
    - 制表符/多空格表格；
    - 跳过横杠分隔行，不再出现 “— — —” 数据行；
    - 最后一行列数不一致时补齐或合并，避免掉到表格外。
    """
    if not text:
        return text

    lines = text.splitlines()
    out: List[str] = []
    i = 0

    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        cells = _split_pseudo_table_row(line)

        if not cells or not _looks_like_pseudo_table_header(cells):
            out.append(raw)
            i += 1
            continue

        header = cells
        expected_cols = len(header)
        j = i + 1

        rows: List[List[str]] = []
        while j < len(lines):
            candidate = lines[j].strip()
            if not candidate:
                break

            if _is_pseudo_separator_row(candidate):
                j += 1
                continue

            row_cells = _split_pseudo_table_row(candidate)
            if not row_cells:
                break

            normalized = _normalize_table_cells(row_cells, expected_cols)
            if normalized is None:
                break
            rows.append(normalized)
            j += 1

        # 有至少 1 行数据就转换。用户对比品种时可能只有少数行，也应正确渲染。
        if rows:
            if out and out[-1].strip():
                out.append("")
            out.append("| " + " | ".join(header) + " |")
            out.append("|" + "|".join(["---"] * expected_cols) + "|")
            for row in rows:
                out.append("| " + " | ".join(row) + " |")
            i = j
        else:
            out.append(raw)
            i += 1

    return "\n".join(out)


def repair_fake_markdown_tables(text: str) -> str:
    """修复模型生成的坏 Markdown 表格。

    - 修复分隔行；
    - 自动补缺失分隔行；
    - 删除全是横杠/破折号的占位行；
    - 保证每行列数一致。
    """
    if not text:
        return text

    lines = text.splitlines()
    out: List[str] = []
    i = 0

    def parse_pipe(line: str) -> List[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    def is_pipe_line(line: str) -> bool:
        s = line.strip()
        return s.startswith("|") and s.endswith("|") and s.count("|") >= 2

    def is_sep_cells(cells: List[str]) -> bool:
        return bool(cells) and all(re.fullmatch(r"[:：]?[-—–]{2,}[:：]?", c.replace(" ", "")) for c in cells)

    while i < len(lines):
        raw = lines[i]
        if not is_pipe_line(raw):
            out.append(raw)
            i += 1
            continue

        header = parse_pipe(raw)
        if len(header) < 2:
            out.append(raw)
            i += 1
            continue

        j = i + 1
        has_separator = False
        if j < len(lines) and is_pipe_line(lines[j]):
            maybe_sep = parse_pipe(lines[j])
            if is_sep_cells(maybe_sep):
                has_separator = True
                j += 1

        rows: List[List[str]] = []
        expected_cols = len(header)
        while j < len(lines) and is_pipe_line(lines[j]):
            row = parse_pipe(lines[j])
            j += 1
            if is_sep_cells(row):
                continue
            normalized = _normalize_table_cells(row, expected_cols)
            if normalized is not None:
                rows.append(normalized)

        # 合法表格或可修复表格
        if has_separator or rows:
            out.append("| " + " | ".join(header) + " |")
            out.append("|" + "|".join(["---"] * expected_cols) + "|")
            for row in rows:
                out.append("| " + " | ".join(row) + " |")
            i = j
            continue

        # 只有一个孤立 pipe 表头：转成普通加粗行，避免破表格。
        out.append("**" + " / ".join(header) + "**")
        i += 1

    return "\n".join(out)

def has_markdown_table(text: str) -> bool:
    """判断文本中是否包含标准 Markdown 表格。"""
    if not text:
        return False

    lines = [line.strip() for line in text.splitlines()]
    for i in range(len(lines) - 1):
        header = lines[i]
        sep = lines[i + 1]
        if header.startswith("|") and header.endswith("|") and sep.startswith("|") and sep.endswith("|"):
            cells = [c.strip() for c in sep.strip("|").split("|")]
            if cells and all(re.fullmatch(r":?-{3,}:?", c.replace(" ", "")) for c in cells):
                return True
    return False





def detect_query_scenario(user_query: str, visual_label: Optional[Dict[str, Any]] = None) -> str:
    """场景路由：用于决定回答长度、是否 RAG、是否 editor/扩写。

    返回值：consumer / agriculture / general
    consumer：品种、口感、果肉品质、能否食用、保存、挑选、价格等。
    agriculture：病虫害、黄叶、根腐、流胶、喷药、防治、施肥、排水、修剪等。
    """
    q = (user_query or "").strip().lower()
    label = visual_label or {}
    category_type = str(label.get("category_type") or "").strip().lower()

    if category_type in {"variety", "quality"}:
        return "consumer"
    if category_type in {"disease", "pest", "cultivation_issue"}:
        return "agriculture"

    consumer_keywords = [
        "品种", "口感", "味道", "甜度", "香气", "好吃", "苦不苦", "甜不甜", "能吃", "可以吃", "怎么吃",
        "果肉", "成熟", "熟了", "坏了", "发霉", "异味", "酒味", "保存", "冷冻", "挑选", "怎么选", "价格", "市场",
        "猫山王", "黑刺", "金枕", "d24", "红虾", "musang", "monthong", "black thorn", "chanee",
        "taste", "flavor", "flavour", "sweet", "bitter", "aroma", "ripe", "ripeness", "edible", "eat", "storage", "buy", "price", "variety",
        "rasa", "manis", "makan", "simpan", "beli", "varieti",
        "รส", "หวาน", "กิน", "เก็บ", "ซื้อ", "พันธุ์",
    ]
    agri_keywords = [
        "病", "虫", "黄叶", "叶片", "根腐", "果腐", "炭疽", "叶斑", "流胶", "疫病", "树干", "根系", "喷药", "防治", "药剂", "施肥", "排水", "修剪", "落花", "落果", "积水", "雨季",
        "disease", "pest", "leaf", "yellow", "rot", "gummosis", "canker", "spray", "fungicide", "insecticide", "fertilizer", "drainage", "pruning",
        "penyakit", "perosak", "daun", "akar", "reput", "sembur", "racun", "baja", "saliran",
        "โรค", "แมลง", "ใบ", "ราก", "พ่น", "สาร", "ปุ๋ย", "ระบายน้ำ",
    ]
    has_agri = any(k in q for k in agri_keywords)
    has_consumer = any(k in q for k in consumer_keywords)
    if has_agri:
        return "agriculture"
    if has_consumer:
        return "consumer"
    return "general"


def is_consumer_query(user_query: str) -> bool:
    return detect_query_scenario(user_query) == "consumer"


def build_expansion_instruction(response_language: str = "zh") -> str:
    """二次扩写指令：让模型补全细节，而不是改结论或强套模板。"""
    lang = (response_language or "zh").lower()
    if lang == "en":
        return (
            "Please improve the previous answer. Keep the same conclusion, but make it more useful and complete. "
            "Add practical details, decision priorities, checking points, treatment or chemical direction when relevant, recheck timing, and one concise safety reminder. "
            "Keep it natural and readable. Do not use a fixed template. Use a Markdown table only if it truly improves readability, and make sure it is valid Markdown."
        )
    if lang == "ms":
        return (
            "Sila perbaiki jawapan sebelumnya. Kekalkan kesimpulan yang sama, tetapi jadikan lebih lengkap dan praktikal. "
            "Tambah keutamaan tindakan, perkara pemeriksaan, arah rawatan atau bahan kimia jika berkaitan, masa semakan semula, dan satu peringatan keselamatan ringkas. "
            "Susun dengan kemas tetapi jangan guna templat tetap. Gunakan jadual Markdown hanya jika benar-benar membantu, dan pastikan jadual sah."
        )
    if lang == "th":
        return (
            "ช่วยปรับปรุงคำตอบก่อนหน้า โดยคงข้อสรุปเดิมไว้ แต่เพิ่มรายละเอียดที่นำไปใช้ได้จริง "
            "ระบุลำดับความสำคัญ จุดตรวจ แนวทางจัดการหรือสารเคมีเมื่อเกี่ยวข้อง เวลาตรวจซ้ำ และคำเตือนความปลอดภัยสั้น ๆ หนึ่งประโยค "
            "จัดให้อ่านง่ายแต่ไม่ใช้แม่แบบตายตัว ใช้ตาราง Markdown เฉพาะเมื่อช่วยให้อ่านง่ายจริง และต้องเป็นตารางที่ถูกต้อง"
        )
    return (
        "请改进上一条回答。保持原结论不变，但把内容补充得更完整、更实用。"
        "补充处理优先级、现场检查点、治理或药剂方向、复查时间，以及最后一句简短安全提醒。"
        "要求自然、清晰、好读，不要固定模板；只有确实有助于阅读时才用标准 Markdown 表格，表格列数必须一致。"
    )


def is_detail_worthy_query(user_query: str, response_language: str = "zh") -> bool:
    """判断是否值得走二次扩写。不是 prompt，而是后端质量控制。"""
    q = (user_query or "").strip().lower()
    if not q:
        return False
    if detect_query_scenario(q) == "consumer":
        return False
    if len(q) <= 8 and q in {"你好", "您好", "hi", "hello", "hey", "嗨", "哈喽"}:
        return False
    keywords = [
        "病", "虫", "黄叶", "炭疽", "叶斑", "根腐", "果腐", "流胶", "防治", "喷药", "药剂", "处理", "管理", "施肥", "排水", "修剪", "对比", "比较", "品种", "保存", "冷链", "市场",
        "disease", "pest", "yellow", "rot", "fungicide", "spray", "treatment", "manage", "fertilizer", "drainage", "compare", "variety", "storage",
        "penyakit", "perosak", "rawatan", "racun", "sembur", "saliran", "varieti",
        "โรค", "แมลง", "สาร", "พ่น", "จัดการ", "ระบายน้ำ", "พันธุ์",
    ]
    return len(q) >= 10 or any(k in q for k in keywords)


def response_seems_too_short(response: str, response_language: str = "zh") -> bool:
    """粗略判断回答是否偏短。"""
    text = (response or "").strip()
    if not text:
        return True
    lang = (response_language or "zh").lower()
    if lang == "zh":
        return len(text) < int(os.getenv("DURIAN_DETAIL_MIN_CHARS_ZH", "520"))
    return len(text.split()) < int(os.getenv("DURIAN_DETAIL_MIN_WORDS_OTHER", "180"))



def is_binary_choice_question(user_query: str) -> bool:
    """判断是否是 A/B 选择型问题，用于编辑器把直接结论放在第一段。"""
    q = (user_query or "").strip().lower()
    if not q:
        return False
    patterns = [
        "还是", "是不是", "要不要", "该不该", "能不能", "会不会", "先", "更像", "更可能",
        "or", "should i", "should we", "more likely", "is it", "whether",
        "atau", "patut", "lebih cenderung",
        "หรือ", "ควร", "น่าจะ",
    ]
    return any(p in q for p in patterns)


def response_starts_with_table_or_list(response: str) -> bool:
    """判断回答是否一上来就是表格/列表，通常可读性较差。"""
    lines = [ln.strip() for ln in (response or "").splitlines() if ln.strip()]
    if not lines:
        return False
    first = lines[0]
    return first.startswith("|") or bool(re.match(r"^(\d+[.、)]|[-•*])\s+", first))


def should_post_edit_response(user_query: str, response: str, response_language: str = "zh") -> bool:
    """是否启动二次编辑器。

    启动条件尽量克制，避免所有问候/短答都被重写。
    """
    if not env_bool("DURIAN_ENABLE_POST_EDITOR", False):
        return False
    q = (user_query or "").strip()
    r = (response or "").strip()
    if not q or not r:
        return False

    greetings = {"你好", "您好", "嗨", "哈喽", "hello", "hi", "hey", "helo", "hai", "สวัสดี"}
    if q.lower() in greetings:
        return False

    lang = (response_language or "zh").lower()
    long_enough = len(r) >= (260 if lang == "zh" else 90)
    has_table = has_markdown_table(r) or response_starts_with_table_or_list(r)
    needs_decision_order = is_binary_choice_question(q)
    detail_worthy = is_detail_worthy_query(q, lang)

    return detail_worthy and (long_enough or has_table or needs_decision_order)


def build_post_editor_instruction(user_query: str, response: str, response_language: str = "zh", quality_review: Optional[Dict[str, Any]] = None) -> str:
    """构造二次编辑器输入。编辑器只负责整理，不负责新推理。"""
    lang = (response_language or "zh").lower()
    binary_hint_zh = "如果用户问题是二选一或判断题，第一段必须先给出明确倾向。" if is_binary_choice_question(user_query) else "第一段先给简洁结论或核心建议。"

    if lang == "en":
        return f"""Please reorganize the answer below without changing its facts, diagnosis, or recommendations.

User question:
{user_query}

Current answer:
{response}

Editing requirements:
- Start with a direct conclusion or priority judgment, especially if the user asks A vs B or whether to do something.
- Do not start with a table. Put any table after the short conclusion and explanation.
- Keep the answer more detailed and useful, but remove repetition and unsupported claims.
- Keep treatment or pesticide directions only if they are relevant to the user's question. Do not add disease control, fertilizer, or pesticide advice to variety, eating, market, or general consumer questions. If pesticides are relevant, leave only one concise label/compliance reminder at the end.
- Use clean Markdown: short paragraphs, bullets, and valid tables only. Do not use slash-separated pseudo tables.
- Return only the edited answer in English."""

    if lang == "ms":
        return f"""Sila susun semula jawapan di bawah tanpa mengubah fakta, diagnosis atau cadangan asal.

Soalan pengguna:
{user_query}

Jawapan semasa:
{response}

Keperluan suntingan:
- Mulakan dengan kesimpulan atau keutamaan tindakan yang jelas.
- Jangan mulakan terus dengan jadual; letakkan jadual selepas kesimpulan ringkas dan penerangan.
- Jadikan jawapan lebih kemas, terperinci dan praktikal, tetapi buang pengulangan.
- Kekalkan arah rawatan atau racun hanya jika benar-benar berkaitan dengan soalan pengguna. Jangan tambah nasihat penyakit, baja atau racun untuk soalan varieti, makanan, pasaran atau pengguna umum. Jika racun berkaitan, letakkan hanya satu peringatan pematuhan ringkas di akhir.
- Gunakan Markdown yang bersih: perenggan pendek, poin, dan jadual yang sah sahaja. Jangan gunakan jadual palsu berpisah dengan garis miring.
- Jawab hanya dalam Bahasa Melayu."""

    if lang == "th":
        return f"""ช่วยจัดเรียงคำตอบด้านล่างใหม่ โดยไม่เปลี่ยนข้อเท็จจริง การวินิจฉัย หรือคำแนะนำเดิม

คำถามผู้ใช้:
{user_query}

คำตอบปัจจุบัน:
{response}

ข้อกำหนดการแก้ไข:
- เริ่มด้วยข้อสรุปหรือสิ่งที่ควรทำก่อนอย่างชัดเจน
- ห้ามเริ่มด้วยตารางทันที ให้วางตารางหลังข้อสรุปและคำอธิบายสั้น ๆ
- ทำให้คำตอบอ่านง่ายขึ้น มีรายละเอียดและนำไปใช้ได้จริง แต่ตัดความซ้ำซ้อนออก
- คงแนวทางการจัดการหรือสารเคมีไว้เฉพาะเมื่อเกี่ยวข้องกับคำถามจริง ๆ อย่าเพิ่มคำแนะนำเรื่องโรค ปุ๋ย หรือสารเคมีในคำถามเรื่องสายพันธุ์ การกิน ตลาด หรือผู้บริโภคทั่วไป หากเกี่ยวข้องกับสารเคมี ให้มีคำเตือนเรื่องฉลาก/ข้อกำหนดเพียงสั้น ๆ ตอนท้าย
- ใช้ Markdown ที่ถูกต้อง: ย่อหน้าสั้น bullet และตารางที่ถูกต้องเท่านั้น ห้ามใช้ตารางปลอมที่คั่นด้วย /
- ตอบเป็นภาษาไทยเท่านั้น"""

    return f"""请只整理和润色下面的回答，不要改变原结论、诊断方向和建议事实。

用户问题：
{user_query}

当前回答：
{response}

编辑要求：
- {binary_hint_zh}
- 不要一开头就放表格；表格应放在简短结论和原因解释之后。
- 让回答更详细、更好读、更有现场决策顺序，但删除重复和空话。
- 只有当用户问题确实涉及病虫害、喷药、防治、根区处理或农业管理时，才保留治理、药剂/处理方向和复查时间。若用户问的是品种、口感、食用、市场、购买、保存或普通常识，不要额外加入治理、施肥、喷药内容。如果确实涉及农药，最后只保留一句简短合规提醒。
- 使用干净 Markdown：短段落、项目符号、必要时使用合法表格。禁止使用斜杠 / 分隔的伪表格。
- 只输出整理后的中文最终答案。"""


def accept_edited_response(
    original: str,
    edited: str,
    response_language: str = "zh",
    quality_review: Optional[Dict[str, Any]] = None,
) -> bool:
    """校验编辑器输出，避免把答案改坏。"""
    if not edited:
        return False
    o = (original or "").strip()
    e = (edited or "").strip()
    if not e:
        return False
    # 不接受明显过短的编辑结果。
    rewrite_focus = set((quality_review or {}).get("rewrite_focus") or [])
    min_ratio = 0.45 if rewrite_focus.intersection({"improve_structure", "fix_table"}) else 0.55
    if len(e) < max(80, int(len(o) * min_ratio)):
        return False
    # 不接受包含编辑器元话语。
    bad_phrases = ["整理后的", "编辑后的", "Here is", "Edited answer", "以下是", "我将"]
    if any(p.lower() in e.lower() for p in bad_phrases):
        return False
    return True

def has_slash_style_table(text: str) -> bool:
    """检测模型输出的斜杠伪表格。

    典型形式：
    A / B / C
    --- / --- / ---
    x / y / z
    """
    if not text:
        return False

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    slash_rows = 0
    sep_rows = 0

    for ln in lines:
        if "|" in ln:
            continue
        if ln.count("/") < 2:
            continue
        parts = [p.strip() for p in ln.split("/")]
        non_empty = [p for p in parts if p]
        if len(non_empty) < 3:
            continue
        if all(re.fullmatch(r"[-—–]{2,}", p) for p in non_empty):
            sep_rows += 1
        else:
            slash_rows += 1

    return slash_rows >= 2 or (slash_rows >= 1 and sep_rows >= 1)


def has_malformed_pipe_table(text: str) -> bool:
    """检测破损 Markdown pipe 表格。

    典型问题：
    1. 有多行竖线内容，但缺少标准分隔行；
    2. 表头/内容列数不一致；
    3. 模型输出了单边竖线或横杠占位行。
    该函数只负责检测，不负责修复；修复交给 repair_* 系列函数。
    """
    if not text:
        return False

    if has_markdown_table(text):
        return False

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False

    pipe_like = [
        ln for ln in lines
        if ln.startswith("|") or ln.endswith("|") or ln.count("|") >= 2
    ]

    dash_like = [
        ln for ln in lines
        if re.fullmatch(r"[|\s:：—–-]+", ln)
    ]

    # 多行像表格，但没有合法 Markdown 分隔行。
    if len(pipe_like) >= 2 and len(dash_like) >= 1:
        return True

    # 出现大量 pipe 行，但不是合法 Markdown 表格。
    if len(pipe_like) >= 4:
        return True

    # 单行列标题样式，但缺少分隔行，也认为是潜在坏表格。
    for i, ln in enumerate(lines[:-1]):
        if ln.startswith("|") and ln.endswith("|") and ln.count("|") >= 3:
            nxt = lines[i + 1]
            if not re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", nxt):
                return True

    return False


def extract_last_user_content_from_dict_messages(messages: List[Dict]) -> str:
    """从 dict 消息列表中取最后一条 user 内容。"""
    for msg in reversed(messages or []):
        if msg.get("role") == "user":
            return str(msg.get("content") or "")
    return ""


def looks_like_chinese(text: str) -> bool:
    """简单判断文本是否以中文为主或包含中文问题。"""
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def detect_response_language(text: str, fallback: str = "zh") -> str:
    """根据用户最后一条文本自动判断回答语言。

    优先级：泰文字符 > 中文字符 > 马来语关键词 > 英文拉丁字符 > fallback。
    这样即使前端漏传 response_language，输入 Hello 也不会再回中文。
    """
    value = (text or "").strip()
    if not value:
        return fallback if fallback in {"zh", "en", "ms", "th"} else "zh"

    if re.search(r"[\u0E00-\u0E7F]", value):
        return "th"
    if re.search(r"[\u4e00-\u9fff]", value):
        return "zh"

    lower = value.lower()
    ms_keywords = {
        "apa", "apakah", "bagaimana", "kenapa", "mengapa", "boleh", "tidak",
        "pokok", "daun", "buah", "akar", "batang", "tanah", "hujan", "saliran",
        "penyakit", "perosak", "baja", "racun", "kuning", "reput", "durian",
    }
    ms_hits = sum(1 for word in ms_keywords if re.search(rf"\b{re.escape(word)}\b", lower))
    if ms_hits >= 2:
        return "ms"

    if re.search(r"[A-Za-z]", value):
        return "en"

    return fallback if fallback in {"zh", "en", "ms", "th"} else "zh"


def resolve_response_language(request_language: str, user_query: str) -> str:
    """最终回答语言：以前端传入为主，但 zh 默认值可能只是 Pydantic 默认值。

    如果前端传 zh 但用户最后一条明显是英文/马来文/泰文，则自动覆盖。
    """
    requested = (request_language or "").strip().lower()
    detected = detect_response_language(user_query, fallback=requested or "zh")
    if requested in {"en", "ms", "th"}:
        return requested
    if requested == "zh" and detected != "zh":
        return detected
    return requested if requested in {"zh", "en", "ms", "th"} else detected


# =========================
# 对话意图与上下文隔离
# =========================
def normalize_short_text(text: str) -> str:
    """用于短文本意图判断的轻量归一化。"""
    return re.sub(r"[\s。！!？?，,；;：:\.]+", "", (text or "").strip().lower())


def is_greeting_only(text: str) -> bool:
    """判断是否只是问候。问候不进入 RAG，也不继承上一轮病害/图片上下文。"""
    q_raw = (text or "").strip().lower()
    q = normalize_short_text(q_raw)
    greetings = {
        "你好", "您好", "嗨", "哈喽", "哈罗", "早上好", "上午好", "中午好", "下午好", "晚上好",
        "hello", "hi", "hey", "helo", "hai",
        "selamatpagi", "selamattengahhari", "selamatpetang", "selamatmalam",
        "สวัสดี", "สวัสดีครับ", "สวัสดีค่ะ",
    }
    return q in greetings


def greeting_reply(response_language: str) -> str:
    """多语言问候快速回复。"""
    lang = (response_language or "zh").lower()
    replies = {
        "zh": "你好，我在。你可以问我榴莲种植、病虫害、品种、购买、保存、食用安全、加工或市场相关问题。",
        "en": "Hello. You can ask me about durian cultivation, pests and diseases, varieties, buying, storage, food safety, processing, or market issues.",
        "ms": "Helo, saya di sini. Anda boleh tanya tentang penanaman durian, penyakit, perosak, varieti, pembelian, penyimpanan, keselamatan makanan, pemprosesan atau pasaran.",
        "th": "สวัสดีครับ ผมอยู่ตรงนี้ คุณสามารถถามเรื่องการปลูกทุเรียน โรค แมลง พันธุ์ การเลือกซื้อ การเก็บรักษา ความปลอดภัยอาหาร การแปรรูป หรือเรื่องตลาดได้",
    }
    return replies.get(lang, replies["zh"])


def has_recent_image_context(messages: List[Message], lookback: int = 12) -> bool:
    """最近几轮是否包含图片消息，用于判断“这个情况/要不要喷药”等追问是否应继承图片上下文。"""
    for msg in reversed(messages[-lookback:] if messages else []):
        if getattr(msg, "image_url", None):
            return True
        content = getattr(msg, "content", "") or ""
        if "[Image context]" in content or "[视觉识别上下文]" in content or "Image identification" in content:
            return True
        if any(k in content for k in ["分类结果", "图像分类", "图片分类", "Classification", "Klasifikasi", "การจำแนก"]):
            return True
    return False


def classify_user_intent(text: str, has_recent_image: bool = False) -> str:
    """快速强规则：只处理确定性很高的情况。

    注意：这里不再用"短句 + 最近有图片 = follow_up"的粗暴规则。
    不确定时交给 LLM 上下文路由器判断。
    返回：greeting / explicit_follow_up / unknown
    """
    q_raw = (text or "").strip()
    q = q_raw.lower()

    if is_greeting_only(q_raw):
        return "greeting"

    # 强指代关键词：明确指代上一轮的内容
    explicit_follow_up_keywords = [
        # 中文强指代
        "刚才", "上面", "上一张", "这张图", "这个图", "图片里", "图里", "这棵", "这株",
        "这个情况", "这种情况", "这个病", "这片叶", "这棵树", "这个果", "继续刚才",
        "还要喷药", "要不要喷药", "还需要喷", "喷不喷", "怎么处理这个", "这个严重吗",
        "严重吗", "严不严重", "怎么办", "怎么处理", "要怎么做", "下一步", "怎么救",
        "需要处理吗", "需要剪掉吗", "需要施肥吗", "需要补肥吗", "需要浇水吗", "需要排水吗",
        "可以吗", "行不行", "能不能", "会不会死", "还有救吗",
        "它", "它们", "那个", "那些", "前面", "之前",
        # English strong references
        "this image", "that image", "the image", "this case", "that case", "above", "previous",
        "should i spray it", "spray it", "is this serious", "how serious", "what should i do next",
        "what should i do", "how to treat it", "can i save it", "will it die", "next step",
        "it", "them", "that", "those", "before",
        # Malay
        "gambar ini", "keadaan ini", "kes ini", "pokok ini", "perlu sembur", "serius", "apa perlu buat", "ia", "mereka",
        # Thai
        "รูปนี้", "กรณีนี้", "แบบนี้", "ต้นนี้", "ต้องฉีด", "รุนแรงไหม", "ควรทำอย่างไร", "มัน", "พวกมัน",
    ]
    if any(k in q for k in explicit_follow_up_keywords):
        return "explicit_follow_up"

    short_follow_up_keywords = [
        "严重", "处理", "喷药", "施肥", "浇水", "排水", "剪掉", "修剪", "补救", "救", "下一步",
        "serious", "spray", "treat", "save", "next",
        "serius", "sembur", "rawat",
        "รุนแรง", "ฉีด", "รักษา",
    ]
    if has_recent_image and len(normalize_short_text(q_raw)) <= 18 and any(k in q for k in short_follow_up_keywords):
        return "explicit_follow_up"

    # 如果是完整的新问题（有明确的农业或消费者关键词），即使短也是 new_topic
    if is_agricultural_diagnosis_query(text) or any(k in q for k in [
        "品种", "口感", "购买", "保存", "价格", "市场",
        "variety", "taste", "buy", "storage", "price", "market",
    ]):
        return "new_topic"

    return "unknown"


def build_recent_context_summary(raw_messages: List[Message], max_items: int = 6, max_chars_each: int = 180) -> str:
    """给上下文路由器看的短摘要，避免把长历史直接塞给 router。"""
    if not raw_messages:
        return "(no previous context)"

    items = []
    recent = raw_messages[-max_items:]
    for msg in recent:
        role = getattr(msg, "role", "") or ""
        content = (getattr(msg, "content", "") or "").replace("\n", " ").strip()
        image_url = getattr(msg, "image_url", None)
        marker = " [image]" if image_url else ""
        if len(content) > max_chars_each:
            content = content[:max_chars_each] + "..."
        if content or marker:
            items.append(f"{role}{marker}: {content}")
    return "\n".join(items) if items else "(no useful previous context)"


def parse_router_json(text: str) -> Dict[str, Any]:
    """解析 router 输出。模型偶尔会包 code fence 或多说几字，这里做容错。"""
    if not text:
        return {}
    value = text.strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
    value = re.sub(r"\s*```$", "", value)
    try:
        return json.loads(value)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", value)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}


def route_context_with_llm(
    current_query: str,
    raw_messages: List[Message],
    response_language: str = "zh",
) -> Dict[str, Any]:
    """使用同一个大模型做轻量上下文路由。

    输出字段：
    - intent: greeting / new_topic / follow_up
    - use_history: bool
    - use_image_context: bool
    - reason: str

    设计原则：
    1. 先走确定性强规则，避免问候走大模型；
    2. 其余交给 LLM 判断是否依赖上下文；
    3. router 只输出 JSON，max_tokens 很小，temperature=0。
    4. 关键：基于语义判断是否需要上下文，不是基于问题长度或最近有无图片。
    5. 对图片分析的追问：如果最近有图片上下文且是追问，应该保留历史以便基于图片分析回答。
    6. 追问判断不依赖关键词，而是用 LLM 理解语义。
    """
    quick_intent = classify_user_intent(current_query, has_recent_image=has_recent_image_context(raw_messages))
    if quick_intent == "greeting":
        return {
            "intent": "greeting",
            "use_history": False,
            "use_image_context": False,
            "reason": "greeting only",
            "source": "rule",
        }
    if quick_intent == "explicit_follow_up":
        return {
            "intent": "follow_up",
            "use_history": True,
            "use_image_context": has_recent_image_context(raw_messages),
            "reason": "explicit reference to previous context or image analysis",
            "source": "rule",
        }
    if quick_intent == "new_topic":
        return {
            "intent": "new_topic",
            "use_history": False,
            "use_image_context": False,
            "reason": "complete new question with clear keywords",
            "source": "rule",
        }

    # 如果模型还没加载，降级为 new_topic，宁可少带历史，也不要乱带历史污染。
    if model_instance is None or getattr(model_instance, "model", None) is None:
        return {
            "intent": "new_topic",
            "use_history": False,
            "use_image_context": False,
            "reason": "model unavailable, safe default",
            "source": "fallback",
        }

    if not env_bool("DURIAN_ENABLE_LLM_CONTEXT_ROUTER", False):
        return {
            "intent": "new_topic",
            "use_history": False,
            "use_image_context": False,
            "reason": "fast rule router default; set DURIAN_ENABLE_LLM_CONTEXT_ROUTER=1 for semantic routing",
            "source": "fast_rule_router",
        }

    context_summary = build_recent_context_summary(raw_messages)
    router_system = (
        "你是一个对话上下文路由器，只判断当前用户输入是否需要引用上一轮上下文。"
        "你不是回答助手，不要回答用户问题。"
        "判断的关键是语义：当前问题是否在语义上依赖上一轮内容才能理解。"
        "不要依赖关键词匹配，而是理解问题的真实意图。"
        "特别注意：如果上一轮是图片分析，当前问题是对图片分析的追问（即使没有明确指代词），也应该保留历史以便基于图片分析回答。"
        "只输出严格 JSON，不要 Markdown，不要解释。"
    )
    router_user = f"""
判断当前用户输入是否在语义上依赖最近上下文。

关键规则：
1. 问候（"你好""Hi"）→ greeting，use_history=false。
2. 完整的新问题（有明确主语、谓语、宾语，或明确的农业/消费者关键词）→ new_topic，use_history=false。
   例如："榴莲品种""黄叶怎么办""猫山王和金枕区别"都是完整问题。
3. 不完整的问题（缺少主语、宾语或关键信息，必须依赖上一轮才能理解）→ follow_up，use_history=true。
   例如："怎么处理""要不要喷""严重吗"这类短句，如果没有上下文就无法理解。
4. 明确指代上一轮（"刚才""这个""那个""上面的""它""那个"）→ follow_up，use_history=true。
5. 【重要】如果上一轮是图片分析，当前问题是对图片的追问（即使没有明确指代词），也应该是 follow_up，use_history=true。
   例如：上一轮分析了图片中的病害，当前问"严重吗""要不要喷药""怎么处理"，都应该保留历史。
   即使问题很短或没有"这个""它"这样的指代词，只要语义上是在追问图片分析的结果，就应该是 follow_up。
6. 如果问题是独立的新问题（"黄叶怎么办""什么品种"），即使最近有图片，也是 new_topic，use_history=false。

当前用户输入：
{current_query}

最近上下文摘要：
{context_summary}

只输出 JSON，格式如下：
{{"intent":"greeting|new_topic|follow_up","use_history":false,"use_image_context":false,"reason":"short reason"}}
""".strip()

    try:
        router_text = run_with_model_lock(
            model_instance.generate,
            [
                {"role": "system", "content": router_system},
                {"role": "user", "content": router_user},
            ],
            max_new_tokens=int(os.getenv("DURIAN_ROUTER_MAX_TOKENS", "96")),
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            repetition_penalty=1.0,
            response_language=response_language,
        )
        data = parse_router_json(router_text)
        intent = str(data.get("intent", "new_topic")).strip().lower()
        if intent not in {"greeting", "new_topic", "follow_up"}:
            intent = "new_topic"

        use_history = bool(data.get("use_history", intent == "follow_up"))
        use_image_context = bool(data.get("use_image_context", use_history and has_recent_image_context(raw_messages)))

        # 安全约束：greeting/new_topic 强制不带历史；follow_up 才允许带历史。
        if intent in {"greeting", "new_topic"}:
            use_history = False
            use_image_context = False
        if intent == "follow_up":
            use_history = True
            use_image_context = use_image_context or has_recent_image_context(raw_messages)

        return {
            "intent": intent,
            "use_history": use_history,
            "use_image_context": use_image_context,
            "reason": str(data.get("reason", "llm router"))[:160],
            "source": "llm",
            "raw": router_text[:300],
        }
    except Exception as e:
        logger.warning("上下文路由器失败，降级为 new_topic: %s", e)
        return {
            "intent": "new_topic",
            "use_history": False,
            "use_image_context": False,
            "reason": f"router error: {e}",
            "source": "fallback",
        }


def build_context_aware_raw_messages(raw_messages: List[Message], route_decision: Any) -> List[Message]:
    """根据 LLM 路由结果选择是否保留历史。

    - follow_up/use_history=true：保留原消息，供 build_messages 再裁剪；
    - greeting/new_topic/use_history=false：只保留最后一条 user，避免上一轮图片诊断污染；
    
    特别处理：
    1. 如果是对图片分析的追问（follow_up + 有图片上下文），保留完整历史包括图片分析结果
    2. 图片追问（new_topic 但有图片上下文）只保留当前问题，不带历史
    """
    if isinstance(route_decision, dict):
        use_history = bool(route_decision.get("use_history", False))
        intent = str(route_decision.get("intent", "new_topic"))
        use_image_context = bool(route_decision.get("use_image_context", False))
    else:
        intent = str(route_decision or "new_topic")
        use_history = intent == "follow_up"
        use_image_context = False

    # follow_up 且需要图片上下文：保留完整历史（包括图片分析结果）
    if intent == "follow_up" and use_history and use_image_context:
        return raw_messages

    # follow_up 但不需要图片上下文：保留历史但可能裁剪
    if intent == "follow_up" and use_history:
        return raw_messages

    # new_topic 或 greeting：只保留最后一条 user 消息，不带历史
    for msg in reversed(raw_messages or []):
        if msg.role == "user":
            return [msg]
    return raw_messages[-1:] if raw_messages else []

def extract_image_context_messages(raw_messages: List[Message], max_items: int = 3, max_chars_each: int = 1200) -> str:
    contexts = []
    for msg in raw_messages or []:
        content = (getattr(msg, "content", "") or "").strip()
        if not content:
            continue
        if "[Image context]" not in content and "[视觉识别上下文]" not in content:
            continue
        if len(content) > max_chars_each:
            content = content[:max_chars_each] + "..."
        contexts.append(content)
    return "\n\n".join(contexts[-max_items:])



# =========================
# 统一上下文卡片选择器：embedding 召回 + LLM 语义选择 + 规则护栏
# =========================
_CONTEXT_CARD_MARKERS = ["[Image context]", "[Durian context card]", "[视觉识别上下文]"]


def strip_html_comments(text: str) -> str:
    """去掉 HTML 注释，避免隐藏上下文污染可见摘要。"""
    return re.sub(r"<!--([\s\S]*?)-->", "", text or "").strip()


def extract_json_blocks_from_context_comments(text: str) -> List[Dict[str, Any]]:
    """从前端/后端保存的 HTML 注释中抽取 JSON 上下文。"""
    cards = []
    if not text:
        return cards
    for m in re.finditer(r"<!--\s*\[(?:Image context|Durian context card|视觉识别上下文)\]\s*([\s\S]*?)\s*-->", text, flags=re.I):
        raw = (m.group(1) or "").strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                cards.append(obj)
        except Exception:
            jm = re.search(r"\{[\s\S]*\}", raw)
            if jm:
                try:
                    obj = json.loads(jm.group(0))
                    if isinstance(obj, dict):
                        cards.append(obj)
                except Exception:
                    pass
    return cards


def _json_loads_maybe(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    value = value.strip()
    if not value:
        return value
    try:
        return json.loads(value)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", value)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return value
    return value


def infer_followup_intent(query: str) -> str:
    """追问意图粗识别；不是最终选择器，只给 embedding/LLM selector 提示。"""
    q = (query or "").lower().strip()
    if not q:
        return "general_followup"
    if any(k in q for k in ["药", "药物", "用药", "农药", "杀菌剂", "杀虫剂", "杀螨剂", "喷药", "药剂", "fungicide", "pesticide", "spray", "chemical"]):
        return "chemical_treatment"
    if any(k in q for k in ["防治", "治理", "怎么处理", "怎么办", "下一步", "处理", "控制", "management", "treat", "control"]):
        return "treatment"
    if any(k in q for k in ["严重", "危险", "会扩散", "扩散", "还能救", "要紧", "严重吗", "risk", "serious", "spread"]):
        return "severity"
    if any(k in q for k in ["多久", "复查", "观察", "几天", "什么时候", "recheck", "monitor", "how long"]):
        return "recheck"
    if any(k in q for k in ["能吃", "可以吃", "还能吃", "口感", "味道", "保存", "怎么吃", "熟", "坏", "makan", "eat", "taste", "ripe", "storage"]):
        return "consumer_quality"
    if any(k in q for k in ["哪个", "区别", "对比", "更像", "还是", "compare", "difference", "which"]):
        return "comparison"
    if len(q) <= 40 and any(k in q for k in ["这个", "这种", "那", "它", "刚才", "前面", "上面", "this", "that", "it"]):
        return "general_followup"
    return "unknown"


def infer_card_scenario(text: str, default: str = "general") -> str:
    t = (text or "").lower()
    consumer = ["品种", "口感", "甜", "苦", "香", "果肉", "能吃", "保存", "购买", "价格", "猫山王", "黑刺", "金枕", "d24", "musang", "monthong", "taste", "eat", "ripe", "variety", "storage"]
    agri = ["病", "虫", "叶", "树干", "根", "黄叶", "流胶", "炭疽", "叶斑", "腐", "喷药", "防治", "施肥", "排水", "leaf", "disease", "pest", "root", "trunk", "fungicide"]
    if any(k in t for k in agri):
        return "agriculture"
    if any(k in t for k in consumer):
        return "consumer"
    return default


def infer_card_category(text: str, scenario: str = "general") -> str:
    t = (text or "").lower()
    if any(k in t for k in ["炭疽", "叶斑", "病斑", "疫病", "根腐", "果腐", "白根", "粉红病", "溃疡", "canker", "anthracnose", "rot", "disease"]):
        return "disease"
    if any(k in t for k in ["虫", "粉蚧", "蓟马", "红蜘蛛", "白蚁", "borer", "pest", "insect", "mite", "thrips"]):
        return "pest"
    if any(k in t for k in ["黄叶", "积水", "缺肥", "施肥", "排水", "蒸腾", "根系", "雨季", "落果", "开花", "fertilizer", "drainage", "waterlogging"]):
        return "cultivation_issue"
    if any(k in t for k in ["猫山王", "黑刺", "金枕", "d24", "musang", "monthong", "black thorn", "品种", "variety"]):
        return "variety"
    if any(k in t for k in ["果肉", "成熟", "能吃", "发霉", "异味", "酒味", "保存", "ripe", "edible", "mold", "storage", "quality"]):
        return "quality"
    if scenario == "agriculture":
        return "cultivation_issue"
    if scenario == "consumer":
        return "quality"
    return "general"


def extract_main_subject_heuristic(text: str, fallback: str = "上一轮话题") -> str:
    """从历史问答里提取一个可读主题。"""
    if not text:
        return fallback
    patterns = [
        r"(叶部炭疽病|炭疽病|Colletotrichum[^，。\n]*)",
        r"(茎部裂纹/胶质化|茎部裂纹|流胶|gummosis)",
        r"(果腐病|Fruit rot|fruit rot)",
        r"(根腐|白根病|root rot)",
        r"(叶斑病|Phomopsis叶斑病|leaf spot)",
        r"(猫山王|Musang King|黑刺|Black Thorn|金枕|Monthong|D24|红虾|Chanee)",
        r"(黄叶|雨后黄叶|短暂蒸腾失衡|根系缺氧)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            return m.group(1).strip()
    plain = strip_html_comments(text)
    plain = re.sub(r"\s+", " ", plain).strip()
    # 优先取问句或第一句
    for sep in ["。", "？", "?", "!", "！", "\n"]:
        if sep in plain:
            part = plain.split(sep)[0].strip()
            if 4 <= len(part) <= 40:
                return part
    return plain[:40] if plain else fallback


def followup_topics_for_card(scenario: str, category_type: str, subject: str = "") -> List[str]:
    if category_type in {"disease", "pest"}:
        return ["防治", "药物", "用药", "杀菌剂", "喷药", "农药", "处理", "严重程度", "复查", "扩散风险"]
    if category_type == "cultivation_issue":
        return ["处理", "排水", "施肥", "恢复", "严重程度", "复查", "下一步", "是否喷药"]
    if scenario in {"consumer", "variety"} or category_type in {"variety", "quality"}:
        return ["口感", "甜度", "苦味", "香气", "成熟度", "能不能吃", "保存", "挑选", "对比"]
    return ["继续解释", "原因", "下一步", "注意事项"]


def normalize_context_card(raw: Dict[str, Any], fallback_id: str, recency_index: int = 0) -> Dict[str, Any]:
    """把图片上下文/文本上下文统一成 Context Card。

    v2 修正点：图片链路里，VL label 可能误判成品种/果肉，但最终可见回答已经诊断为病害。
    上下文卡片必须优先尊重上一轮可见回答，否则追问“药物方面”会被错误绑定到猫山王口感。
    """
    raw = raw or {}
    visual = _json_loads_maybe(raw.get("visual_label") or raw.get("vl_label") or raw.get("visual_observation") or raw.get("label_context"))
    visual_dict = visual if isinstance(visual, dict) else {}

    assistant_text = str(raw.get("assistant_summary") or raw.get("last_answer_summary") or raw.get("summary") or "").strip()
    user_text = str(raw.get("user_question") or raw.get("last_user_query") or "").strip()
    visual_text = visual if isinstance(visual, str) else json.dumps(visual_dict, ensure_ascii=False)

    raw_category = str(raw.get("category_type") or visual_dict.get("category_type") or "").strip().lower()
    raw_label = str(raw.get("main_subject") or raw.get("label") or visual_dict.get("label_zh") or visual_dict.get("label") or "").strip()
    raw_scenario = str(raw.get("scenario") or "").strip().lower()

    visible_blob = "\n".join(x for x in [user_text, assistant_text] if x)
    full_blob = "\n".join(str(x) for x in [user_text, assistant_text, visual_text] if x)

    visible_scenario = infer_card_scenario(visible_blob, default="") if visible_blob else ""
    visible_category = infer_card_category(visible_blob, visible_scenario or "general") if visible_blob else ""
    visible_subject = extract_main_subject_heuristic(visible_blob, fallback="") if visible_blob else ""

    scenario = raw_scenario or visible_scenario or infer_card_scenario(full_blob)
    category_type = raw_category or visible_category or infer_card_category(full_blob + "\n" + raw_label, scenario)
    label = raw_label or visible_subject or extract_main_subject_heuristic(full_blob)

    # 强冲突修正：可见回答含明确病虫害/栽培问题，但 VL/raw label 是品种/品质时，以可见回答为准。
    # 这是修复“上一轮明明是叶部炭疽病，追问药物却选到猫山王”的关键。
    agri_categories = {"disease", "pest", "cultivation_issue"}
    consumer_categories = {"variety", "quality"}
    if visible_category in agri_categories and category_type in consumer_categories.union({"general", ""}):
        category_type = visible_category
        scenario = "agriculture"
        if visible_subject:
            label = visible_subject
    elif visible_category in {"disease", "pest"} and any(k in assistant_text for k in ["病害", "病斑", "炭疽", "叶斑", "虫害", "杀菌", "防治"]):
        category_type = visible_category
        scenario = "agriculture"
        if visible_subject:
            label = visible_subject

    # 如果 subject 仍然是明显消费类，但可见文本存在更强病害主体，也强制替换。
    if label and any(k in label.lower() for k in ["猫山王", "musang", "黑刺", "金枕", "monthong", "d24"]):
        if visible_category in agri_categories and visible_subject:
            label = visible_subject
            category_type = visible_category
            scenario = "agriculture"

    text_blob = full_blob

    key_facts = raw.get("key_facts") or visual_dict.get("key_facts") or []
    if isinstance(key_facts, str):
        key_facts = [x.strip() for x in re.split(r"[；;\n]", key_facts) if x.strip()]
    if not isinstance(key_facts, list):
        key_facts = []
    visual_evidence = visual_dict.get("visual_evidence") or raw.get("visual_evidence")
    if visual_evidence and len(key_facts) < 4:
        if isinstance(visual_evidence, list):
            key_facts.extend([str(x) for x in visual_evidence[:3]])
        else:
            key_facts.append(str(visual_evidence)[:220])

    summary = str(raw.get("last_answer_summary") or raw.get("assistant_summary") or raw.get("summary") or "").strip()
    if not summary:
        summary = strip_html_comments(text_blob)[:260]

    card = {
        "context_id": str(raw.get("context_id") or raw.get("id") or fallback_id),
        "source": str(raw.get("source") or ("image" if raw.get("image_url") or raw.get("type") == "durian_image_context" else "text")),
        "scenario": scenario or "general",
        "category_type": category_type or "general",
        "main_subject": label or "上一轮话题",
        "latin_name": str(raw.get("latin_name") or visual_dict.get("latin_name") or ""),
        "key_facts": key_facts[:6],
        "last_answer_summary": summary[:600],
        "image_url": str(raw.get("image_url") or ""),
        "followup_topics": raw.get("followup_topics") or followup_topics_for_card(scenario, category_type, label),
        "created_at": str(raw.get("created_at") or ""),
        "recency_index": recency_index,
    }
    return card


def context_card_to_match_text(card: Dict[str, Any]) -> str:
    return "\n".join([
        f"主题: {card.get('main_subject', '')}",
        f"场景: {card.get('scenario', '')}",
        f"类型: {card.get('category_type', '')}",
        f"来源: {card.get('source', '')}",
        f"关键信息: {'；'.join(map(str, card.get('key_facts') or []))}",
        f"摘要: {card.get('last_answer_summary', '')}",
        f"适合追问: {'；'.join(map(str, card.get('followup_topics') or []))}",
    ]).strip()


def build_context_cards_from_messages(raw_messages: List[Message], max_cards: int = 10) -> List[Dict[str, Any]]:
    """从当前前端传来的聊天历史构建 Context Cards。支持图片卡和文本卡。"""
    if not raw_messages:
        return []
    # 排除最后一条 user 当前输入，只用其之前的上下文
    msgs = list(raw_messages)
    if msgs and getattr(msgs[-1], "role", "") == "user":
        msgs = msgs[:-1]

    cards: List[Dict[str, Any]] = []
    recency = 0
    # 先使用结构化 active_context_card；这是 v4 主路径，不再依赖 content 里的隐藏注释。
    for msg in reversed(msgs):
        structured_card = getattr(msg, "active_context_card", None)
        if isinstance(structured_card, dict) and structured_card:
            card = normalize_context_card(structured_card, fallback_id=f"structured_{len(cards)+1}", recency_index=recency)
            cards.append(card)
            recency += 1
            if len(cards) >= max_cards:
                return cards

    # 兼容旧历史：再解析隐藏上下文卡，越靠后越新
    for idx, msg in enumerate(reversed(msgs)):
        content = getattr(msg, "content", "") or ""
        for raw in extract_json_blocks_from_context_comments(content):
            card = normalize_context_card(raw, fallback_id=f"hidden_{len(cards)+1}", recency_index=recency)
            cards.append(card)
            recency += 1
            if len(cards) >= max_cards:
                return cards

    # 再从最近普通 user-assistant 对话生成文本卡，给没有隐藏卡的文本问题使用
    pairs = []
    last_user = None
    for msg in msgs:
        role = getattr(msg, "role", "")
        content = strip_html_comments(getattr(msg, "content", "") or "")
        if not content:
            continue
        if role == "user":
            last_user = content
        elif role == "assistant" and last_user:
            pairs.append((last_user, content))
            last_user = None
    for u, a in reversed(pairs[-max_cards:]):
        blob = f"用户问题：{u}\n助手回答：{a}"
        scenario = infer_card_scenario(blob)
        category = infer_card_category(blob, scenario)
        subject = extract_main_subject_heuristic(blob)
        card = normalize_context_card({
            "type": "durian_text_context",
            "source": "text",
            "scenario": scenario,
            "category_type": category,
            "main_subject": subject,
            "last_user_query": u[:300],
            "last_answer_summary": a[:500],
            "key_facts": [u[:160], a[:220]],
        }, fallback_id=f"text_{len(cards)+1}", recency_index=recency)
        # 避免和隐藏卡主题完全重复
        if not any(c.get("main_subject") == card.get("main_subject") and c.get("category_type") == card.get("category_type") for c in cards):
            cards.append(card)
            recency += 1
        if len(cards) >= max_cards:
            break
    return cards[:max_cards]


def build_embedding_query(query: str, intent: str) -> str:
    expansions = {
        "chemical_treatment": "药物 用药 杀菌剂 农药 喷药 防治 fungicide pesticide chemical treatment disease pest",
        "treatment": "处理 防治 治理 下一步 控制 management treatment control",
        "severity": "严重程度 扩散 风险 危险 能否恢复 severity spread risk",
        "recheck": "复查 观察 几天 判断标准 monitor recheck timing",
        "consumer_quality": "口感 品质 成熟 能不能吃 保存 taste quality edible ripe storage",
        "comparison": "对比 区别 哪个 更像 compare difference which",
    }
    return f"{query}\n{expansions.get(intent, '')}".strip()


def embedding_recall_context_cards(query: str, intent: str, cards: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """用 RAG embedding 模型召回候选卡；不可用时按新近度返回。"""
    if not cards:
        return []
    top_k = max(1, min(top_k, len(cards)))
    if not (rag_retriever and getattr(rag_retriever, "model", None)):
        return cards[:top_k]
    try:
        texts = [context_card_to_match_text(c) for c in cards]
        qv = rag_retriever._encode([build_embedding_query(query, intent)], batch_size=1)
        cv = rag_retriever._encode(texts, batch_size=min(16, len(texts)))
        if qv.size == 0 or cv.size == 0:
            return cards[:top_k]
        scores = (cv @ qv[0]).astype(float).tolist()
        ranked = sorted(zip(scores, cards), key=lambda x: x[0], reverse=True)
        result = []
        for score, card in ranked[:top_k]:
            cc = dict(card)
            cc["embedding_score"] = float(score)
            result.append(cc)

        # 上一条可见助手回答是短追问的强候选，不能因为 embedding 轻微偏差被 top_k 挤掉。
        immediate = next((c for c in cards if c.get("context_id") == "ctx_immediate_previous"), None)
        if immediate and not any(c.get("context_id") == "ctx_immediate_previous" for c in result):
            cc = dict(immediate)
            cc["embedding_score"] = 1.0
            result = [cc] + result[:max(0, top_k - 1)]
        return result
    except Exception as e:
        logger.warning("上下文卡 embedding 召回失败，回退新近度: %s", e)
        return cards[:top_k]


def parse_context_selector_json(text: str) -> Dict[str, Any]:
    return parse_router_json(text)


def llm_select_context_card(query: str, cards: List[Dict[str, Any]], response_language: str = "zh") -> Dict[str, Any]:
    """让本地 LLM 从候选 Context Cards 中选择最相关的一张，并生成 standalone query。"""
    default = {
        "selected_context_id": None,
        "confidence": 0.0,
        "followup_intent": infer_followup_intent(query),
        "should_use_rag": False,
        "standalone_query": query,
        "answer_mode": "new_topic",
        "reason": "no context selected",
        "source": "fallback",
    }
    if not cards or not query or model_instance is None or getattr(model_instance, "model", None) is None:
        return default

    intent_hint = infer_followup_intent(query)
    candidate_lines = []
    for i, card in enumerate(cards, 1):
        candidate_lines.append(
            f"[{card.get('context_id')}]\n"
            f"主题: {card.get('main_subject')}\n"
            f"场景/类型: {card.get('scenario')} / {card.get('category_type')}\n"
            f"来源: {card.get('source')}\n"
            f"摘要: {card.get('last_answer_summary')}\n"
            f"适合追问: {'；'.join(map(str, card.get('followup_topics') or []))}\n"
            f"embedding_score: {card.get('embedding_score', '')}\n"
        )
    selector_system = (
        "你是对话上下文选择器。你的任务不是回答问题，而是判断当前用户问题最可能追问哪一张上下文卡片。"
        "你必须灵活理解语义，不要只按最近一条选择。"
        "如果用户问药物/用药/杀菌剂/喷药，应优先选择 disease/pest/cultivation_issue 上下文，不要选择品种口感上下文。"
        "如果用户问口感/能不能吃/保存，应优先选择 consumer/variety/quality 上下文。"
        "如果没有任何卡片相关，selected_context_id=null。"
        "只输出严格 JSON。"
    )
    selector_user = f"""
当前用户问题：
{query}

初步追问意图提示：{intent_hint}

候选上下文卡片：
{chr(10).join(candidate_lines)}

请输出 JSON：
{{
  "selected_context_id": "卡片ID或null",
  "confidence": 0.0,
  "followup_intent": "chemical_treatment|treatment|severity|recheck|consumer_quality|comparison|general_followup|new_topic",
  "should_use_rag": false,
  "standalone_query": "把当前短追问补全后的完整问题；如果不是追问则原样返回",
  "answer_mode": "chemical_treatment|treatment|judgment_explanation|diagnostic_guidance|consumer_natural|general",
  "reason": "一句话原因"
}}
""".strip()
    try:
        raw = run_with_model_lock(
            model_instance.generate,
            [
                {"role": "system", "content": selector_system},
                {"role": "user", "content": selector_user},
            ],
            max_new_tokens=int(os.getenv("DURIAN_CONTEXT_SELECTOR_MAX_TOKENS", "220")),
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            repetition_penalty=1.0,
            response_language=response_language,
            skip_detail_expand=True,
            skip_post_editor=True,
        )
        data = parse_context_selector_json(raw)
        if not isinstance(data, dict):
            return default
        selected_id = data.get("selected_context_id")
        if selected_id in {"null", "None", "none", ""}:
            selected_id = None
        confidence = float(data.get("confidence") or 0.0)
        data.update({
            "selected_context_id": selected_id,
            "confidence": confidence,
            "followup_intent": str(data.get("followup_intent") or intent_hint or "general_followup"),
            "should_use_rag": bool(data.get("should_use_rag", False)),
            "standalone_query": str(data.get("standalone_query") or query),
            "answer_mode": str(data.get("answer_mode") or "general"),
            "reason": str(data.get("reason") or "llm selected context")[:200],
            "source": "llm_context_selector",
        })
        return data
    except Exception as e:
        logger.warning("LLM 上下文卡选择失败: %s", e)
        return default


def guard_context_selection(query: str, selection: Dict[str, Any], cards: List[Dict[str, Any]]) -> Dict[str, Any]:
    """规则只做护栏：防止明显离谱选择。"""
    if not selection or not cards:
        return selection
    selected_id = selection.get("selected_context_id")
    if not selected_id:
        return selection
    card = next((c for c in cards if c.get("context_id") == selected_id), None)
    if not card:
        selection["selected_context_id"] = None
        selection["confidence"] = 0.0
        selection["reason"] = "selected card id not found"
        return selection
    intent = selection.get("followup_intent") or infer_followup_intent(query)
    scenario = card.get("scenario", "")
    category = card.get("category_type", "")
    # 药物/防治/严重程度追问不能绑定到猫山王口感这类 consumer/variety/quality 卡。
    # 注意：旧版只在 scenario 是 consumer 时才拦截；如果一张冲突卡被归成 agriculture + variety，仍会漏掉。
    treatment_like = intent in {"chemical_treatment", "treatment", "severity", "recheck"}
    selected_is_consumerish = (
        category in {"variety", "quality"}
        or scenario in {"consumer", "variety", "quality"}
        or any(k in str(card.get("main_subject", "")).lower() for k in ["猫山王", "musang", "黑刺", "金枕", "monthong", "d24", "口感", "果肉"])
    )
    if treatment_like and selected_is_consumerish:
        better = next((
            c for c in cards
            if c.get("category_type") in {"disease", "pest", "cultivation_issue"}
            and not any(k in str(c.get("main_subject", "")).lower() for k in ["猫山王", "musang", "黑刺", "金枕", "monthong", "d24", "口感", "果肉"])
        ), None)
        if better:
            selection["selected_context_id"] = better.get("context_id")
            selection["confidence"] = max(float(selection.get("confidence") or 0.0), 0.78)
            selection["reason"] = "guardrail switched to disease/pest/cultivation context for treatment-like follow-up"
        else:
            selection["selected_context_id"] = None
            selection["confidence"] = 0.0
            selection["reason"] = "guardrail rejected consumer/variety context for treatment-like follow-up"
    return selection


def build_followup_context_instruction(selection: Dict[str, Any], card: Dict[str, Any], response_language: str = "zh") -> str:
    subject = card.get("main_subject", "上一轮话题")
    latin = card.get("latin_name", "")
    facts = "；".join(map(str, (card.get("key_facts") or [])[:5]))
    summary = card.get("last_answer_summary", "")
    intent = selection.get("followup_intent") or infer_followup_intent(selection.get("standalone_query", ""))
    standalone = selection.get("standalone_query") or ""

    if intent == "chemical_treatment":
        task = "当前用户在追问上一轮主题的药剂/用药方向。请直接说明适用药剂类别、使用条件、轮换原则和注意事项；不要重新重复完整诊断过程，不要转向品种/口感/保存。"
    elif intent == "treatment":
        task = "当前用户在追问处理/防治。请直接说明处理优先级和操作步骤；不要从头重复上一轮诊断。"
    elif intent == "severity":
        task = "当前用户在追问严重程度。请判断风险、扩展可能和需要升级处理的信号。"
    elif intent == "recheck":
        task = "当前用户在追问复查和观察。请说明观察周期、判断标准和下一步分支。"
    elif intent == "consumer_quality":
        task = "当前用户在追问消费/品质/食用相关内容。请只围绕口感、品质、成熟度、保存或能否食用回答。"
    elif intent == "comparison":
        task = "当前用户在追问比较判断。请直接比较，不要展开成通用方案。"
    else:
        task = "请基于选中的上一轮主题回答当前追问。"

    return f"""
[Selected conversation context]
上一轮绑定主题：{subject}{f" ({latin})" if latin else ""}
上下文类型：{card.get('scenario')} / {card.get('category_type')} / source={card.get('source')}
上一轮关键信息：{facts or '无'}
上一轮摘要：{summary or '无'}

当前用户原始追问：{standalone or ''}
回答任务：{task}

优先级：当前用户追问 > 选中的上下文卡片 > RAG资料。若RAG资料与该主题无关，不要使用。
""".strip()


def build_immediate_previous_assistant_card(raw_messages: List[Message], intent: str) -> Optional[Dict[str, Any]]:
    """
    从“当前用户问题之前的最后一条助手可见回答”直接构建高优先级卡片。

    用途：短追问通常指向上一条可见回答。即使隐藏 VL 卡片被误判、embedding 召回偏移，
    也要让上一条明确的病害诊断（如叶部炭疽病）优先参与选择。
    """
    if not raw_messages:
        return None
    msgs = list(raw_messages)
    if msgs and getattr(msgs[-1], "role", "") == "user":
        msgs = msgs[:-1]
    last_assistant = None
    last_user = ""
    for msg in reversed(msgs):
        role = getattr(msg, "role", "")
        content = strip_html_comments(getattr(msg, "content", "") or "").strip()
        if not content:
            continue
        if last_assistant is None and role == "assistant":
            last_assistant = content
            continue
        if last_assistant is not None and role == "user":
            last_user = content
            break
    if not last_assistant:
        return None

    blob = f"用户问题：{last_user}\n助手回答：{last_assistant}"
    scenario = infer_card_scenario(blob)
    category = infer_card_category(blob, scenario)

    # 只对追问类，尤其 treatment-like/consumer-like，构造上一轮卡；完整新问题不强行绑定。
    if intent in {"unknown", "new_topic"}:
        return None
    if intent in {"chemical_treatment", "treatment", "severity", "recheck"} and category not in {"disease", "pest", "cultivation_issue"}:
        return None
    if intent == "consumer_quality" and category not in {"variety", "quality"}:
        return None

    subject = extract_main_subject_heuristic(blob)
    return normalize_context_card({
        "type": "immediate_previous_assistant_context",
        "context_id": "ctx_immediate_previous",
        "source": "last_assistant",
        "scenario": scenario,
        "category_type": category,
        "main_subject": subject,
        "last_user_query": last_user[:300],
        "last_answer_summary": last_assistant[:800],
        "key_facts": [last_user[:160], last_assistant[:320]],
        "followup_topics": followup_topics_for_card(scenario, category, subject),
    }, fallback_id="ctx_immediate_previous", recency_index=-1)



def extract_context_comment_objects_with_marker(text: str) -> List[Dict[str, Any]]:
    """Extract context JSON comments and keep their marker.

    v3 uses message-level active-topic binding. The last assistant message may contain
    more than one hidden comment. We prefer the backend-generated [Durian context card]
    because it is generated from the final visible assistant answer, while [Image context]
    is raw frontend/VL context and may contain a lower-confidence visual label.
    """
    items: List[Dict[str, Any]] = []
    if not text:
        return items
    pattern = r"<!--\s*\[(Image context|Durian context card|视觉识别上下文)\]\s*([\s\S]*?)\s*-->"
    for m in re.finditer(pattern, text, flags=re.I):
        marker = (m.group(1) or "").strip()
        raw = (m.group(2) or "").strip()
        if not raw:
            continue
        obj = None
        try:
            obj = json.loads(raw)
        except Exception:
            jm = re.search(r"\{[\s\S]*\}", raw)
            if jm:
                try:
                    obj = json.loads(jm.group(0))
                except Exception:
                    obj = None
        if isinstance(obj, dict):
            obj = dict(obj)
            obj["__marker"] = marker
            items.append(obj)
    return items


def get_last_assistant_message_content(raw_messages: List[Message]) -> str:
    """Return the stored content of the assistant message immediately before the current user message."""
    if not raw_messages:
        return ""
    msgs = list(raw_messages)
    if msgs and getattr(msgs[-1], "role", "") == "user":
        msgs = msgs[:-1]
    for msg in reversed(msgs):
        if getattr(msg, "role", "") == "assistant":
            return getattr(msg, "content", "") or ""
    return ""


def get_last_assistant_active_card(raw_messages: List[Message]) -> Optional[Dict[str, Any]]:
    """Resolve the active context card bound to the previous assistant answer.

    Primary source: hidden [Durian context card] appended to that exact assistant message.
    Secondary source: [Image context] on the same assistant message, used only for legacy
    image answers that do not yet have a backend-generated context card.
    """
    for msg in reversed(raw_messages or []):
        if getattr(msg, "role", "") != "assistant":
            continue
        structured_card = getattr(msg, "active_context_card", None)
        if isinstance(structured_card, dict) and structured_card:
            return normalize_context_card(structured_card, fallback_id="ctx_last_assistant_active", recency_index=-1)
        break

    content = get_last_assistant_message_content(raw_messages)
    if not content:
        return None

    items = extract_context_comment_objects_with_marker(content)
    if not items:
        return None

    preferred = [x for x in items if str(x.get("__marker", "")).lower() == "durian context card"]
    if not preferred:
        preferred = [x for x in items if str(x.get("__marker", "")).lower() in {"image context", "视觉识别上下文"}]
    if not preferred:
        return None

    raw = dict(preferred[-1])
    raw.pop("__marker", None)
    return normalize_context_card(raw, fallback_id="ctx_last_assistant_active", recency_index=-1)


def _router_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "是", "对", "需要", "自然"}
    return False


def normalize_topic_transition_route(route: Dict[str, Any], query: str, active_card: Dict[str, Any]) -> Dict[str, Any]:
    """Use semantic self-check fields from the router to remove contradictory routes.

    This is not the primary decision maker. The LLM still decides the semantic relation,
    but if it outputs contradictory fields such as transition=new_topic while also saying
    the question requires the active topic and fits it, we normalize the route before
    final generation.
    """
    if not route or not active_card:
        return route

    transition = str(route.get("transition") or "").lower()
    requires_prev = _router_bool(route.get("requires_previous_context"))
    fits_active = _router_bool(route.get("fits_active_topic"))
    introduces_new = _router_bool(route.get("introduces_new_topic"))
    referenced = str(route.get("referenced_context_id") or "")
    active_id = str(active_card.get("context_id") or "")
    standalone = str(route.get("standalone_query") or "")
    subject = str(active_card.get("main_subject") or "")

    # LLM 自检：它认为当前问题依赖上一轮且接着上一轮自然，就不应判 new_topic。
    if transition == "new_topic" and requires_prev and fits_active and not introduces_new:
        route["transition"] = "continue"
        route["confidence"] = max(float(route.get("confidence") or 0.0), float(os.getenv("DURIAN_TOPIC_ROUTER_MIN_CONF", "0.65")))
        route["referenced_context_id"] = active_id or referenced
        route["need_context_search"] = False
        route["reason"] = f"router consistency normalized to continue; original reason: {route.get('reason', '')}"[:240]
        return route

    # 如果 standalone_query 已经明确补全成 active subject，且 referenced_context_id 也指向 active card，
    # new_topic 同样属于结构化输出矛盾。
    if transition == "new_topic" and active_id and referenced == active_id and subject and subject in standalone:
        route["transition"] = "continue"
        route["confidence"] = max(float(route.get("confidence") or 0.0), float(os.getenv("DURIAN_TOPIC_ROUTER_MIN_CONF", "0.65")))
        route["need_context_search"] = False
        route["reason"] = f"router consistency normalized to continue because standalone_query binds active subject; original reason: {route.get('reason', '')}"[:240]
        return route

    # 反向纠偏：如果模型判 continue，但同时承认引入全新主题且不适配 active topic，则切断。
    if transition == "continue" and introduces_new and not fits_active:
        route["transition"] = "new_topic"
        route["referenced_context_id"] = None
        route["need_context_search"] = False
        route["reason"] = f"router consistency normalized to new_topic; original reason: {route.get('reason', '')}"[:240]

    return route

def llm_topic_transition_router(query: str, active_card: Optional[Dict[str, Any]], response_language: str = "zh") -> Dict[str, Any]:
    """LLM-based semantic router: continue / new_topic / switch_old_topic / clarify.

    It does not answer the user's question. It only decides whether the current query
    should inherit the previous assistant message's active context card.
    """
    default = {
        "transition": "new_topic" if not active_card else "clarify",
        "confidence": 0.0,
        "referenced_context_id": active_card.get("context_id") if active_card else None,
        "intent": "general",
        "standalone_query": query,
        "need_context_search": bool(active_card),
        "should_use_rag": False,
        "reason": "router default",
        "source": "topic_transition_default",
    }
    if not query or is_greeting_only(query):
        default.update({"transition": "new_topic", "reason": "greeting or empty query", "need_context_search": False})
        return default
    if not active_card:
        default.update({"transition": "new_topic", "reason": "no previous active context card", "need_context_search": True})
        return default
    quick_intent = classify_user_intent(query, has_recent_image=str(active_card.get("source") or "").startswith("image"))
    if quick_intent == "explicit_follow_up":
        route = {
            "transition": "continue",
            "confidence": 0.9,
            "referenced_context_id": active_card.get("context_id"),
            "intent": "general",
            "standalone_query": query,
            "requires_previous_context": True,
            "fits_active_topic": True,
            "introduces_new_topic": False,
            "need_context_search": False,
            "should_use_rag": bool(is_agricultural_diagnosis_query(query) or asks_treatment_or_pesticide(query)),
            "reason": "explicit follow-up matched by fast router",
            "source": "fast_topic_router",
        }
        return normalize_topic_transition_route(route, query, active_card)
    if quick_intent == "new_topic":
        default.update({
            "transition": "new_topic",
            "confidence": 0.9,
            "referenced_context_id": None,
            "need_context_search": False,
            "reason": "complete new question matched by fast router",
            "source": "fast_topic_router",
        })
        return default
    if not env_bool("DURIAN_ENABLE_LLM_TOPIC_ROUTER", False):
        default.update({
            "transition": "new_topic",
            "confidence": 0.75,
            "referenced_context_id": None,
            "need_context_search": False,
            "reason": "fast topic router default; set DURIAN_ENABLE_LLM_TOPIC_ROUTER=1 for semantic routing",
            "source": "fast_topic_router",
        })
        return default
    if model_instance is None or model_instance.model is None:
        default.update({"transition": "clarify", "confidence": 0.0, "reason": "model unavailable"})
        return default

    card_brief = json.dumps(
        {
            "context_id": active_card.get("context_id"),
            "main_subject": active_card.get("main_subject"),
            "category_type": active_card.get("category_type"),
            "scenario": active_card.get("scenario"),
            "source": active_card.get("source"),
            "summary": active_card.get("last_answer_summary"),
            "key_facts": active_card.get("key_facts") or [],
        },
        ensure_ascii=False,
        indent=2,
    )

    system = (
        "你是对话主题转移判断器。你不回答用户问题，只判断当前用户问题和上一轮 active topic 的关系。"
        "必须根据语义关系判断，不要机械依赖关键词。只输出 JSON。"
    )
    user = f"""
上一轮 active topic card:
{card_brief}

当前用户问题:
{query}

请判断当前问题属于哪一种：
1. continue：继续追问上一轮 active topic。
2. new_topic：开启全新主题，不继承上一轮。
3. switch_old_topic：不是上一轮，但可能在追问更早的历史主题，需要从历史卡片池中重新选择。
4. clarify：无法确定，应向用户澄清。

判断标准：
- 如果当前问题缺少明确对象，离开上一轮主题无法完整理解，并且接着上一轮回答会自然，选择 continue。
- 如果当前问题自身已经有明确对象/任务，并且明显讨论新的对象、领域或任务，选择 new_topic。
- 如果当前问题明确指向更早内容，例如“前面那个、上一张、另一个、不是这个”，但不是上一轮主题，选择 switch_old_topic。
- 不要因为单个词就机械判断；请综合“能否独立理解、是否引入新对象、接着上一轮是否自然”。
- 如果 continue，请把当前问题补全成 standalone_query，明确继承的主题。

只输出 JSON：
{{
  "transition": "continue|new_topic|switch_old_topic|clarify",
  "confidence": 0.0,
  "referenced_context_id": "{active_card.get('context_id')}",
  "intent": "diagnosis|treatment|chemical_treatment|severity|recheck|comparison|consumer_quality|implementation|general",
  "standalone_query": "补全后的独立问题；new_topic 时可等于原问题",
  "requires_previous_context": true,
  "fits_active_topic": true,
  "introduces_new_topic": false,
  "need_context_search": true,
  "should_use_rag": false,
  "reason": "一句话说明"
}}
""".strip()
    try:
        raw = run_with_model_lock(
            model_instance.generate,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_new_tokens=int(os.getenv("DURIAN_TOPIC_ROUTER_MAX_TOKENS", "240")),
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            repetition_penalty=1.0,
            response_language=response_language,
            skip_detail_expand=True,
            skip_post_editor=True,
        )
        data = parse_context_selector_json(raw)
        if not isinstance(data, dict):
            return default
        transition = str(data.get("transition") or "clarify").strip().lower()
        if transition not in {"continue", "new_topic", "switch_old_topic", "clarify"}:
            transition = "clarify"
        confidence = float(data.get("confidence") or 0.0)
        route = {
            "transition": transition,
            "confidence": confidence,
            "referenced_context_id": data.get("referenced_context_id") or active_card.get("context_id"),
            "intent": str(data.get("intent") or "general"),
            "standalone_query": str(data.get("standalone_query") or query),
            "requires_previous_context": _router_bool(data.get("requires_previous_context")),
            "fits_active_topic": _router_bool(data.get("fits_active_topic")),
            "introduces_new_topic": _router_bool(data.get("introduces_new_topic")),
            "need_context_search": _router_bool(data.get("need_context_search", transition in {"switch_old_topic", "clarify"})),
            "should_use_rag": _router_bool(data.get("should_use_rag", False)),
            "reason": str(data.get("reason") or "semantic topic transition")[:240],
            "source": "llm_topic_transition_router",
        }
        return normalize_topic_transition_route(route, query, active_card)
    except Exception as e:
        logger.warning("LLM topic transition router failed: %s", e)
        return default


def build_active_topic_instruction(route: Dict[str, Any], card: Dict[str, Any], response_language: str = "zh") -> str:
    """Build final-model instruction for a continued active topic."""
    selection = {
        "followup_intent": route.get("intent") or "general",
        "standalone_query": route.get("standalone_query") or "",
        "should_use_rag": route.get("should_use_rag", False),
        "reason": route.get("reason") or "active topic continued",
    }
    return build_followup_context_instruction(selection, card, response_language)


def should_use_historical_selector_after_route(route: Dict[str, Any]) -> bool:
    transition = route.get("transition")
    confidence = float(route.get("confidence") or 0.0)
    min_conf = float(os.getenv("DURIAN_TOPIC_ROUTER_MIN_CONF", "0.65"))
    if transition == "switch_old_topic":
        return True
    if transition == "clarify":
        return True
    if confidence < min_conf and bool(route.get("need_context_search", True)):
        return True
    return False


def resolve_conversation_context(query: str, raw_messages: List[Message], response_language: str = "zh") -> Dict[str, Any]:
    """v3 unified context resolver.

    Priority:
    1. Previous assistant message's active context card, bound inside that message.
    2. LLM TopicTransitionRouter decides continue/new_topic/switch_old_topic/clarify.
    3. Only for switch_old_topic/unclear do we run embedding recall + LLM context selector.
    """
    result = {
        "selected": False,
        "selection": None,
        "card": None,
        "cards_count": 0,
        "standalone_query": query,
        "should_use_rag": None,
        "system_instruction": "",
        "transition": "none",
        "topic_route": None,
        "source": "active_topic_router_v3",
    }
    if not query or is_greeting_only(query):
        return result

    active_card = get_last_assistant_active_card(raw_messages)
    route = llm_topic_transition_router(query, active_card, response_language) if active_card else {
        "transition": "new_topic",
        "confidence": 1.0,
        "referenced_context_id": None,
        "intent": "general",
        "standalone_query": query,
        "need_context_search": True,
        "should_use_rag": False,
        "reason": "no active card",
        "source": "no_active_card",
    }
    result["topic_route"] = route
    result["transition"] = route.get("transition", "none")

    min_conf = float(os.getenv("DURIAN_TOPIC_ROUTER_MIN_CONF", "0.65"))

    if active_card and route.get("transition") == "continue" and float(route.get("confidence") or 0.0) >= min_conf:
        instruction = build_active_topic_instruction(route, active_card, response_language)
        result.update({
            "selected": True,
            "selection": {
                "selected_context_id": active_card.get("context_id"),
                "confidence": float(route.get("confidence") or 0.0),
                "followup_intent": route.get("intent") or "general",
                "standalone_query": route.get("standalone_query") or query,
                "should_use_rag": bool(route.get("should_use_rag", False)),
                "answer_mode": route.get("intent") or "general",
                "reason": route.get("reason") or "continue active topic",
                "source": route.get("source") or "llm_topic_transition_router",
            },
            "card": active_card,
            "standalone_query": route.get("standalone_query") or query,
            "should_use_rag": bool(route.get("should_use_rag", False)),
            "system_instruction": instruction,
        })
        logger.info(
            "active topic continued: subject=%s category=%s intent=%s confidence=%.2f rag=%s reason=%s",
            active_card.get("main_subject"), active_card.get("category_type"), route.get("intent"),
            float(route.get("confidence") or 0.0), bool(route.get("should_use_rag", False)), route.get("reason"),
        )
        return result

    if route.get("transition") == "new_topic" and float(route.get("confidence") or 0.0) >= min_conf:
        result.update({"selected": False, "standalone_query": query, "should_use_rag": None, "system_instruction": ""})
        logger.info("topic router chose new_topic: confidence=%.2f reason=%s", float(route.get("confidence") or 0.0), route.get("reason"))
        return result

    if not should_use_historical_selector_after_route(route):
        return result

    cards = build_context_cards_from_messages(raw_messages, max_cards=int(os.getenv("DURIAN_CONTEXT_CARD_MAX", "10")))
    # Do not add a visible-answer heuristic pseudo card here. v3's active binding is the
    # previous assistant's hidden card; historical search should operate on real cards.
    result["cards_count"] = len(cards)
    if not cards:
        return result

    intent_hint = str(route.get("intent") or "general")
    candidates = embedding_recall_context_cards(
        route.get("standalone_query") or query,
        intent_hint,
        cards,
        top_k=int(os.getenv("DURIAN_CONTEXT_RECALL_TOP_K", "5")),
    )
    selection = llm_select_context_card(query, candidates, response_language)
    selection = guard_context_selection(query, selection, candidates)
    selected_id = selection.get("selected_context_id")
    confidence = float(selection.get("confidence") or 0.0)
    min_context_conf = float(os.getenv("DURIAN_CONTEXT_SELECTOR_MIN_CONF", "0.60"))
    if not selected_id or confidence < min_context_conf:
        logger.info("historical context not selected: cards=%d confidence=%.2f reason=%s", len(cards), confidence, selection.get("reason"))
        return result

    card = next((c for c in candidates if c.get("context_id") == selected_id), None) or next((c for c in cards if c.get("context_id") == selected_id), None)
    if not card:
        return result
    instruction = build_followup_context_instruction(selection, card, response_language)
    result.update({
        "selected": True,
        "selection": selection,
        "card": card,
        "standalone_query": selection.get("standalone_query") or route.get("standalone_query") or query,
        "should_use_rag": bool(selection.get("should_use_rag", False)),
        "system_instruction": instruction,
        "transition": route.get("transition") or "switch_old_topic",
    })
    logger.info(
        "historical context selected: subject=%s category=%s intent=%s confidence=%.2f rag=%s reason=%s",
        card.get("main_subject"), card.get("category_type"), selection.get("followup_intent"), confidence,
        result["should_use_rag"], selection.get("reason"),
    )
    return result


def build_text_context_comment(user_query: str, response: str, response_language: str = "zh") -> str:
    """把普通文本问答保存成隐藏上下文卡，供后续追问选择。"""
    if not user_query or not response or is_greeting_only(user_query):
        return ""
    # 避免把已包含上下文注释的内容再次套娃
    visible_response = strip_html_comments(response)
    scenario = infer_card_scenario(user_query + "\n" + visible_response)
    category = infer_card_category(user_query + "\n" + visible_response, scenario)
    subject = extract_main_subject_heuristic(user_query + "\n" + visible_response)
    card = {
        "type": "durian_text_context",
        "context_id": f"ctx_{uuid.uuid4().hex[:12]}",
        "source": "text",
        "scenario": scenario,
        "category_type": category,
        "main_subject": subject,
        "last_user_query": user_query[:500],
        "last_answer_summary": visible_response[:700],
        "key_facts": [user_query[:180], visible_response[:260]],
        "followup_topics": followup_topics_for_card(scenario, category, subject),
        "created_at": datetime.now().isoformat(),
    }
    return "\n\n<!-- [Durian context card]\n" + json.dumps(card, ensure_ascii=False, indent=2) + "\n-->"




def build_text_context_card(
    user_query: str,
    response: str,
    response_language: str = "zh",
    base_card: Optional[Dict[str, Any]] = None,
    source: Optional[str] = None,
    image_url: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Build a structured context card for the current visible assistant answer.

    v4 stores this card separately from visible content. It must never be appended
    into assistant content.
    """
    if not user_query or not response or is_greeting_only(user_query):
        return None

    visible_response = strip_html_comments(response)

    if isinstance(base_card, dict) and base_card:
        card = dict(base_card)
        card.setdefault("context_id", f"ctx_{uuid.uuid4().hex[:12]}")
        card["source"] = source or card.get("source") or "text_followup"
        card["last_user_query"] = user_query[:500]
        card["last_answer_summary"] = visible_response[:800]
        old_facts = [str(x) for x in (card.get("key_facts") or []) if str(x).strip()]
        card["key_facts"] = (old_facts + [user_query[:180], visible_response[:280]])[-8:]
        if image_url:
            card["image_url"] = image_url
        card["created_at"] = datetime.now().isoformat()
        return card

    blob = user_query + "\n" + visible_response
    scenario = infer_card_scenario(blob)
    category = infer_card_category(blob, scenario)
    subject = extract_main_subject_heuristic(blob)
    card = {
        "type": "durian_context_card",
        "context_id": f"ctx_{uuid.uuid4().hex[:12]}",
        "source": source or "text",
        "scenario": scenario,
        "category_type": category,
        "main_subject": subject,
        "latin_name": "Colletotrichum spp." if "炭疽" in blob and not "Colletotrichum" in subject else "",
        "last_user_query": user_query[:500],
        "last_answer_summary": visible_response[:800],
        "key_facts": [user_query[:180], visible_response[:280]],
        "followup_topics": followup_topics_for_card(scenario, category, subject),
        "image_url": image_url or "",
        "confidence": 0.80,
        "created_at": datetime.now().isoformat(),
    }
    return card

def visual_observation_suggests_agricultural_problem(text: str) -> bool:
    """Return True only when the visual notes show clear crop-health symptoms."""
    t = (text or "").lower()
    if not t:
        return False

    problem_keywords = [
        "病斑", "叶斑", "黄叶", "坏死", "腐烂", "流胶", "虫体", "虫卵", "霉层", "萎蔫", "裂口", "根腐",
        "炭疽", "疫病", "粉蚧", "蓟马", "红蜘蛛", "叶片异常", "果腐",
        "lesion", "spot", "yellowing", "necrosis", "rot", "gummosis", "insect", "eggs",
        "mold", "wilting", "canker", "root rot", "fruit rot", "pest", "disease",
        "tompok", "kuning", "reput", "serangga", "penyakit", "perosak",
        "โรค", "แมลง", "ใบเหลือง", "เน่า",
    ]
    if any(k in t for k in problem_keywords):
        negative_phrases = [
            "无病害", "无虫害", "没有病害", "没有虫害", "未见病害", "未见虫害",
            "no visible disease", "no disease", "no pest", "no signs of disease", "no signs of pest",
            "healthy fruit", "healthy leaves",
        ]
        if any(p in t for p in negative_phrases) and not any(
            k in t for k in ["病斑", "腐烂", "流胶", "虫体", "坏死", "lesion", "rot", "gummosis", "insect"]
        ):
            return False
        return True
    return False


def asks_treatment_or_pesticide(text: str) -> bool:
    """判断用户是否在询问治理、防治、喷药、用药或处理方案。"""
    q = (text or "").strip().lower()
    if not q:
        return False
    keywords = [
        # 中文
        "喷药", "打药", "用药", "农药", "药剂", "杀菌剂", "杀虫剂", "杀螨剂", "铜制剂", "代森锰锌", "三唑", "甲氧基丙烯酸酯",
        "防治", "治理", "处理", "怎么治", "怎么处理", "要不要喷", "能不能喷", "需要喷", "根区处理", "伤口处理", "复查", "轮换用药",
        # English
        "spray", "pesticide", "fungicide", "insecticide", "miticide", "treatment", "treat", "control", "manage", "chemical", "active ingredient",
        # Malay
        "sembur", "racun", "rawatan", "kawalan", "fungisid", "insektisid", "bahan aktif",
        # Thai
        "พ่น", "สารเคมี", "ยาฆ่าเชื้อรา", "ยาฆ่าแมลง", "กำจัด", "ควบคุม", "รักษา", "จัดการ",
    ]
    return any(k in q for k in keywords)


def has_treatment_or_pesticide_content(text: str) -> bool:
    """判断回答里是否已经包含治理/药剂/复查等实际处理内容。"""
    r = (text or "").strip().lower()
    if not r:
        return False
    keywords = [
        # 中文
        "处理", "治理", "防治", "喷", "药", "杀菌剂", "杀虫剂", "药剂", "铜制剂", "代森锰锌", "三唑", "复查", "排水", "修剪", "清除", "清园", "登记", "标签",
        # English
        "treat", "treatment", "control", "spray", "fungicide", "insecticide", "pesticide", "active ingredient", "recheck", "drainage", "prune", "remove", "label",
        # Malay
        "rawatan", "kawalan", "sembur", "racun", "semak semula", "saliran", "pangkas", "label",
        # Thai
        "รักษา", "จัดการ", "พ่น", "สาร", "ตรวจซ้ำ", "ระบายน้ำ", "ตัดแต่ง", "ฉลาก",
    ]
    return any(k in r for k in keywords)

def is_agricultural_diagnosis_query(text: str) -> bool:
    """判断是否属于农业诊断/种植管理场景。用于控制是否追加治理/药剂规则。
    
    关键：这个函数决定是否走农业诊断路线。必须严格区分消费者问题和农业问题。
    """
    q = (text or "").strip().lower()
    if not q:
        return False
    
    # 明确的消费者问题关键词 - 如果匹配这些，就不是农业诊断
    consumer_keywords = [
        "品种", "口感", "味道", "甜度", "香气", "好吃", "不好吃", "苦", "腥",
        "购买", "挑选", "怎么选", "怎么挑", "保存", "冷冻", "冷链", "怎么吃", "能不能吃", "可以吃吗",
        "价格", "市场", "出口", "甜品", "搭配", "牛奶", "榴莲冰淇淋", "榴莲蛋糕",
        "variety", "taste", "flavor", "flavour", "sweet", "aroma", "buy", "select", "storage", "frozen", "eat", "price", "market",
        "varieti", "rasa", "manis", "beli", "pilih", "simpan", "makan", "harga", "pasaran",
        "พันธุ์", "รสชาติ", "หวาน", "กลิ่น", "ซื้อ", "เลือก", "เก็บ", "กิน", "ราคา", "ตลาด",
    ]
    
    # 明确的农业诊断关键词 - 如果匹配这些，就是农业诊断
    agri_keywords = [
        "病", "虫", "黄叶", "叶斑", "炭疽", "根腐", "果腐", "流胶", "疫病", "白根", "粉蚧", "蓟马", "红蜘蛛",
        "喷药", "防治", "药剂", "施肥", "排水", "修剪", "落花", "落果", "雨季", "积水", "根系", "树干", "树冠",
        "disease", "pest", "yellow", "leaf", "rot", "gummosis", "fungicide", "spray", "fertilizer", "drainage", "pruning",
        "penyakit", "perosak", "daun", "reput", "racun", "sembur", "baja", "saliran", "akar",
        "โรค", "แมลง", "ใบ", "รากเน่า", "พ่น", "สาร", "ปุ๋ย", "ระบายน้ำ", "ราก",
    ]
    
    has_consumer = any(k in q for k in consumer_keywords)
    has_agri = any(k in q for k in agri_keywords)
    
    # 优先级：如果明确有农业关键词，就是农业诊断
    if has_agri:
        return True
    
    # 如果只有消费者关键词，就不是农业诊断
    if has_consumer:
        return False
    
    # 都没有明确关键词时，保守判断为非农业诊断
    return False


def chemical_guidance_rules(response_language: str) -> str:
    """农药/药剂输出规则：允许给具体方案方向，但最终只保留一句合规提醒。"""
    lang = (response_language or "zh").lower()
    if lang == "en":
        return (
            "\n\n[Agrochemical guidance] When the problem involves disease or pests, you may provide practical pesticide/fungicide方案方向: "
            "active-ingredient groups, protection vs curative role, target site, rotation logic, rainfastness/re-application timing, and when to upgrade treatment. "
            "Do not over-refuse. Avoid inventing exact dilution rates or local brand names unless given by reference. "
            "Put only one concise final compliance reminder at the end."
        )
    if lang == "ms":
        return (
            "\n\n[Panduan racun pertanian] Jika masalah melibatkan penyakit atau perosak, anda boleh memberikan arah rawatan racun/kulat yang praktikal: "
            "kumpulan bahan aktif, fungsi perlindungan atau rawatan, bahagian sasaran, putaran mekanisme tindakan, masa ulang semburan selepas hujan, dan bila rawatan perlu dinaik taraf. "
            "Jangan terlalu menolak. Elakkan mereka cipta kadar bancuhan tepat atau jenama tempatan jika tiada rujukan. "
            "Letakkan hanya satu peringatan pematuhan ringkas pada akhir jawapan."
        )
    if lang == "th":
        return (
            "\n\n[แนวทางสารเคมีเกษตร] หากปัญหาเกี่ยวข้องกับโรคหรือแมลง สามารถให้แนวทางการใช้สารป้องกันกำจัดโรค/แมลงได้อย่างเป็นรูปธรรม: "
            "กลุ่มสารออกฤทธิ์ บทบาทเชิงป้องกันหรือรักษา จุดที่ควรใช้ การสลับกลุ่มสาร ช่วงเวลาพ่นซ้ำหลังฝน และเงื่อนไขที่ควรยกระดับการจัดการ. "
            "ไม่ต้องปฏิเสธมากเกินไป แต่หลีกเลี่ยงการแต่งอัตราผสมที่แน่นอนหรือชื่อการค้าที่ไม่มีข้อมูลอ้างอิง. "
            "ใส่คำเตือนเรื่องการปฏิบัติตามฉลากเพียงประโยคสั้น ๆ ท้ายคำตอบ."
        )
    return (
        "\n\n【农药/药剂方案规则】如果问题涉及病害或虫害，可以大胆给出实用的农药/杀菌剂/杀虫剂方向："
        "包括有效成分类别、保护性或治疗性定位、使用部位、轮换逻辑、雨后补防时机、什么时候需要升级处理。"
        "不要过度拒答。不要凭空编造精确兑水比例、当地商品名或未给出的登记信息。"
        "答案最后只保留一句简短的合规提醒即可。"
    )


def table_is_recommended_for_query(user_query: str, response: str = "", response_language: str = "zh") -> bool:
    """判断当前问题/回答是否适合用 Markdown 表格展示。

    策略：表格优先，但不是所有问题都强制表格。
    适合：病虫害诊断、图片诊断、水肥/土壤/修剪/花果管理、多步骤处理、风险分级、方案比较。
    不适合：问候、单句事实、非常简单的短问答，或用户明确说不要表格。
    """
    if detect_query_scenario(user_query) == "consumer":
        return False
    combined = f"{user_query or ''}\n{response or ''}".strip()
    if not combined:
        return False

    no_table_patterns = [
        r"不要表格", r"别用表格", r"不用表格", r"不要图表", r"纯文字",
        r"no table", r"without table", r"plain text",
        r"jangan jadual", r"tanpa jadual",
        r"ไม่ต้องใช้ตาราง", r"ไม่เอาตาราง",
    ]
    if any(re.search(p, combined, flags=re.IGNORECASE) for p in no_table_patterns):
        return False

    explicit_table_patterns = [
        r"表格", r"图表", r"对比", r"比较", r"清单", r"列表", r"方案", r"步骤", r"流程", r"结构",
        r"table", r"chart", r"compare", r"comparison", r"checklist", r"steps", r"plan", r"structure",
        r"jadual", r"banding", r"langkah", r"struktur",
        r"ตาราง", r"เปรียบเทียบ", r"ขั้นตอน", r"โครงสร้าง",
    ]
    if any(re.search(p, combined, flags=re.IGNORECASE) for p in explicit_table_patterns):
        return True

    complex_keywords = [
        # 中文
        "病", "虫", "炭疽", "叶斑", "果腐", "根腐", "疫病", "流胶", "黄叶", "积水", "雨季", "连续下雨",
        "缺钾", "缺镁", "缺钙", "肥", "药", "喷", "防治", "处理", "管理", "排水", "修剪", "授粉", "落花", "落果",
        "检查", "判断", "复查", "症状", "风险", "严重", "怎么办", "怎么处理", "注意事项", "药剂",
        # English
        "disease", "pest", "anthracnose", "leaf spot", "fruit rot", "root rot", "yellow leaves", "waterlogging",
        "rain", "fertilizer", "pesticide", "spray", "management", "drainage", "pruning", "flower drop", "symptom", "risk", "severity", "recheck", "fungicide",
        # Malay
        "penyakit", "perosak", "antraknosa", "reput", "daun kuning", "air bertakung", "hujan", "racun", "baja", "saliran", "gejala", "semak semula",
        # Thai
        "โรค", "แมลง", "แอนแทรคโนส", "ผลเน่า", "รากเน่า", "ใบเหลือง", "น้ำขัง", "ฝน", "สารเคมี", "ปุ๋ย", "ระบายน้ำ", "อาการ", "ตรวจซ้ำ",
    ]
    lowered = combined.lower()
    keyword_hits = 0
    for kw in complex_keywords:
        if kw.lower() in lowered:
            keyword_hits += 1
            if keyword_hits >= 1:
                break

    numbered_count = len(re.findall(r"(?m)^\s*\d+[.、)]", response or ""))
    bullet_count = len(re.findall(r"(?m)^\s*[-•]", response or ""))
    long_answer = len(response or "") >= 450

    short_simple_query = len(user_query or "") <= 24 and keyword_hits == 0 and numbered_count < 3 and bullet_count < 4
    if short_simple_query:
        return False

    return keyword_hits >= 1 or numbered_count >= 4 or bullet_count >= 5 or long_answer


def build_adaptive_fallback_table(response_language: str = "zh") -> str:
    """当回答适合结构化但模型没有表格时，追加一个简洁的通用检查/处理表。"""
    lang = (response_language or "zh").lower()
    if lang == "en":
        return (
            "\n\n## Field Checklist\n\n"
            "| Check Item | What to Observe | Decision Criteria | Recommended Action | Recheck Timing |\n"
            "|---|---|---|---|---|\n"
            "| Symptoms | Leaves, fruit, trunk, root collar or soil condition | New lesions, yellowing, rot, wilting or rapid spread | Record affected parts and compare with likely causes | Recheck in 3-7 days |\n"
            "| Severity | Affected area, spread speed and tree vigor | Mild, moderate or severe based on spread and canopy impact | Prioritize drainage, sanitation, pruning or registered treatment as needed | Recheck after rain or treatment |\n"
            "| Treatment safety | Product label, local registration and harvest interval | Avoid off-label use or increasing dose without guidance | Follow local label, PPE, re-entry interval and pre-harvest interval | Review before every spray |\n"
        )
    if lang == "ms":
        return (
            "\n\n## Senarai Semak Lapangan\n\n"
            "| Perkara Diperiksa | Apa Yang Dilihat | Kriteria Penilaian | Tindakan Cadangan | Masa Semak Semula |\n"
            "|---|---|---|---|---|\n"
            "| Gejala | Daun, buah, batang, pangkal pokok atau keadaan tanah | Lesi baru, kuning, reput, layu atau merebak cepat | Catat bahagian terjejas dan bandingkan dengan punca yang mungkin | Semak semula 3-7 hari |\n"
            "| Tahap masalah | Keluasan terjejas, kelajuan merebak dan vigor pokok | Ringan, sederhana atau serius mengikut kesan pada kanopi | Utamakan saliran, sanitasi, pemangkasan atau rawatan berdaftar jika perlu | Semak selepas hujan atau rawatan |\n"
            "| Keselamatan rawatan | Label produk, pendaftaran tempatan dan tempoh pra-tuai | Elakkan penggunaan luar label atau menambah dos sendiri | Ikut label, PPE, tempoh masuk semula dan tempoh pra-tuai | Semak sebelum setiap semburan |\n"
        )
    if lang == "th":
        return (
            "\n\n## ตารางตรวจสอบภาคสนาม\n\n"
            "| รายการตรวจ | สิ่งที่ต้องสังเกต | เกณฑ์ประเมิน | การดำเนินการที่แนะนำ | เวลาตรวจซ้ำ |\n"
            "|---|---|---|---|---|\n"
            "| อาการ | ใบ ผล ลำต้น โคนต้น หรือสภาพดิน | มีแผลใหม่ ใบเหลือง ผลเน่า เหี่ยว หรือระบาดเร็ว | บันทึกส่วนที่ได้รับผลกระทบและเทียบกับสาเหตุที่เป็นไปได้ | ตรวจซ้ำใน 3-7 วัน |\n"
            "| ระดับความรุนแรง | พื้นที่เสียหาย ความเร็วในการลุกลาม และความแข็งแรงของต้น | แยกเป็นเบา ปานกลาง หรือรุนแรงตามผลต่อทรงพุ่ม | ให้ความสำคัญกับการระบายน้ำ สุขอนามัย การตัดแต่ง หรือผลิตภัณฑ์ที่ขึ้นทะเบียนเมื่อจำเป็น | ตรวจหลังฝนหรือหลังจัดการ |\n"
            "| ความปลอดภัย | ฉลากผลิตภัณฑ์ การขึ้นทะเบียน และระยะก่อนเก็บเกี่ยว | หลีกเลี่ยงการใช้ผิดฉลากหรือเพิ่มอัตราเอง | ปฏิบัติตามฉลาก PPE ระยะกลับเข้าแปลง และระยะก่อนเก็บเกี่ยว | ตรวจสอบก่อนฉีดพ่นทุกครั้ง |\n"
        )
    return (
        "\n\n## 现场检查与处理表\n\n"
        "| 检查项目 | 观察重点 | 判断标准 | 建议操作 | 复查时间 |\n"
        "|---|---|---|---|---|\n"
        "| 症状表现 | 叶片、果实、枝干、根颈或土壤异常 | 是否有新病斑、黄化、腐烂、萎蔫或快速扩展 | 记录受害部位，先区分病害、虫害、积水或营养问题 | 3-7 天复查 |\n"
        "| 严重程度 | 受害比例、扩展速度、树势变化 | 按轻度、中度、重度分层处理 | 优先处理排水、清园、修剪和必要的合规防治 | 雨后或处理后复查 |\n"
        "| 处理安全 | 产品标签、当地登记、安全间隔期 | 避免擅自加量、混配或连续重喷 | 按标签剂量、PPE、再入间隔和采收安全间隔执行 | 每次用药前复核 |\n"
    )


def ensure_adaptive_markdown_table(response: str, user_query: str = "", response_language: str = "zh") -> str:
    """只做输出清理，不再追加任何兜底表格或通用模板。

    原逻辑会在回答适合结构化但模型没有表格时调用 build_adaptive_fallback_table() 追加通用检查表。
    现在按需求关闭：模型怎么回答就怎么返回，只保留 clean_model_output() 的基础清理。
    """
    return clean_model_output(response or "").strip()

def has_core_zh_pest_structure(response: str) -> bool:
    """检查中文病虫害分析是否包含核心结构（问题判断、症状、防治建议）"""
    text = response or ""
    has_problem = "## 问题判断" in text
    has_symptom = "## 可能症状" in text or "## 症状表现" in text
    has_action = "## 防治建议" in text
    return has_problem and has_symptom and has_action


def is_generic_pest_response(response: str) -> bool:
    """判断回答是否太泛化、太短"""
    text = response or ""

    generic_phrases = [
        "加强管理",
        "注意观察",
        "及时防治",
        "合理施肥",
        "加强水肥管理",
        "选择合适药剂",
        "咨询当地农业技术人员",
        "提高抗病能力",
        "防止病情蔓延",
        "定期巡查",
    ]

    hit_count = sum(1 for phrase in generic_phrases if phrase in text)

    # 内容太短也视为泛化
    if len(text) < 700:
        return True

    # 泛化短语过多也视为泛化
    if hit_count >= 3:
        return True

    # 防治建议部分没有具体动作，也视为泛化
    concrete_actions = [
        "剪除",
        "刮除",
        "清理",
        "深埋",
        "烧毁",
        "疏通",
        "排水",
        "修剪",
        "检查叶背",
        "检查树干",
        "检查根区",
        "复查",
        "叶片正反面",
        "主干基部",
        "嫁接口",
        "树盘",
    ]

    action_count = sum(1 for action in concrete_actions if action in text)
    if action_count < 3:
        return True

    return False


def complete_zh_pest_response(response: str, class_name: str) -> str:
    """补全中文病虫害分析缺失的'药剂/处理参考'和'注意事项'章节"""
    text = (response or "").strip()
    name = class_name or ""

    # 检查是否包含所有必需章节
    required_sections = [
        "## 问题判断",
        "## 可能症状",
        "## 防治建议",
        "## 药剂/处理参考",
        "## 注意事项",
    ]

    missing_sections = [section for section in required_sections if section not in text]

    if not missing_sections and len(text) >= 700:
        return text

    # 如果缺失这两个章节，让大模型自己生成
    if "## 药剂/处理参考" not in text or "## 注意事项" not in text:
        if model_instance is None or model_instance.model is None:
            logger.warning("模型未加载，无法补全缺失章节")
            return text

        try:
            system_content = (
                "你是榴莲病虫害防治专家。"
                "下面的回答缺少'## 药剂/处理参考'和/或'## 注意事项'章节。"
                "请根据识别类别和已有内容，补全这两个缺失的章节。"
                "必须保留原有的所有内容，只在末尾添加缺失的章节。"
                "\n【关键要求】\n"
                "- 药剂/处理参考必须根据识别类别生成差异化内容，不要使用通用模板。\n"
                "- 叶斑类病害：重点围绕叶片正反面、叶缘、嫩梢、雨后湿度、叶面保护。\n"
                "- 树干流胶类：重点围绕主干基部、嫁接口、裂口边缘、坏死树皮、伤口干燥。\n"
                "- 根部病害：重点围绕根颈、根区积水、病根、土壤通气、排水路径。\n"
                "- 虫害：重点围绕叶背、嫩梢、虫体聚集处、蚂蚁、蜜露、虫口密度。\n"
                "- 注意事项也必须根据类别生成，不要输出所有病害通用的提醒。\n"
                "- 不要输出思考过程、<think> 或内部分析。\n"
                "- 直接输出补全后的完整回答。"
            )

            messages = [
                {"role": "system", "content": system_content},
                {
                    "role": "user",
                    "content": (
                        f"识别类别：{class_name}\n\n"
                        f"当前回答：\n{text}\n\n"
                        f"请补全缺失的'## 药剂/处理参考'和'## 注意事项'章节。"
                    ),
                },
            ]

            completed = run_with_model_lock(
                model_instance.generate,
                messages,
                max_new_tokens=1024,
                temperature=0.35,
                top_p=0.75,
                top_k=25,
                repetition_penalty=1.1,
            )

            completed = clean_model_output(completed)
            logger.info("病虫害分析已补全缺失章节，长度: %d", len(completed))
            return completed
        except Exception as e:
            logger.warning("补全缺失章节失败: %s，使用原始回答", e)
            return text

    return text.strip()


def expand_zh_pest_response(original_response: str, class_name: str, pest_info: str, evidence_text: str) -> str:
    """对泛化回答进行二次扩写，使其更具体、可执行"""
    if model_instance is None or model_instance.model is None:
        logger.warning("模型未加载，无法扩写")
        return original_response

    try:
        system_content = (
            "你是榴莲病虫害防治专家。"
            "下面的回答太短或太泛，请在不改变诊断类别的前提下扩写为更具体、可执行的版本。"
            "必须保留 Markdown 标题：## 问题判断、## 可能症状、## 防治建议、## 药剂/处理参考、## 注意事项。"
            "每条防治建议必须包含：检查对象、判断标准、具体操作、复查时间。"
            "药剂/处理参考必须包含：有效成分类别、使用部位、适用条件、标签合规提醒。"
            "禁止只写：加强管理、注意观察、合理施肥、及时防治、选择合适药剂。"
            "\n【关键要求】\n"
            "扩写时必须根据识别类别生成差异化内容，不要复用固定模板。"
            "必须点名当前类别对应的重点部位和处理重点：\n"
            "- 叶部病害（叶斑、炭疽、藻斑、Phomopsis）：重点围绕叶片正反面、叶缘、嫩梢、新病斑数量、雨后湿度。\n"
            "- 树干流胶类（茎部裂纹、胶质化、流胶）：重点围绕主干基部、嫁接口、裂口边缘、坏死树皮、伤口干燥。\n"
            "- 根部病害（根腐、白根、疫病）：重点围绕根颈、根区积水、病根、土壤通气、排水路径。\n"
            "- 虫害（粉蚧、蚧壳虫、蓟马、红蜘蛛、木虱、叶蝉）：重点围绕叶背、嫩梢、虫体聚集处、蚂蚁、蜜露、虫口密度。\n"
            "禁止输出所有病害通用的防治建议。\n"
            "不要输出思考过程、<think> 或内部分析。"
        )

        if evidence_text:
            system_content += f"\n\n参考资料：{evidence_text}"

        messages = [
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": (
                    f"识别类别：{class_name}\n\n"
                    f"原始回答：\n{original_response}\n\n"
                    f"请扩写为更具体的最终回答。"
                ),
            },
        ]

        expanded = run_with_model_lock(
            model_instance.generate,
            messages,
            max_new_tokens=768,
            temperature=0.45,
            top_p=0.85,
            top_k=20,
            repetition_penalty=1.18,
        )

        expanded = clean_model_output(expanded)
        logger.info("病虫害分析已扩写，原长度: %d, 扩写后: %d", len(original_response), len(expanded))
        return expanded
    except Exception as e:
        logger.warning("扩写失败: %s，使用原始回答", e)
        return original_response


def score_pest_answer(text: str, response_language: str = "zh") -> int:
    """评分病虫害分析回答，用于选择最佳版本"""
    if not text:
        return 0

    score = 0
    clean = text.lower()

    # 没有思考标签加分
    if "<think>" not in clean and "</think>" not in clean:
        score += 10

    # 长度加分（每 100 字符 1 分，最多 20 分）
    score += min(len(text) // 100, 20)

    # 章节加分
    if response_language == "zh":
        sections = [
            "## 问题判断",
            "## 可能症状",
            "## 防治建议",
            "## 药剂/处理参考",
            "## 注意事项",
        ]
    elif response_language == "en":
        sections = [
            "## Problem Assessment",
            "## Possible Symptoms",
            "## Prevention Measures",
            "## Precautions",
        ]
    elif response_language == "ms":
        sections = [
            "## Penilaian Masalah",
            "## Maklumat Penting",
            "## Langkah Cadangan",
            "## Langkah Berjaga-jaga",
        ]
    elif response_language == "th":
        sections = [
            "## การประเมินปัญหา",
            "## ข้อมูลสำคัญ",
            "## มาตรการที่แนะนำ",
            "## ข้อควรระวัง",
        ]
    else:
        sections = []

    for section in sections:
        if section in text:
            score += 10

    # 编号项加分（最多 6 项，每项 3 分）
    numbered_count = len(re.findall(r"(?m)^\s*\d+[.、)]", text))
    score += min(numbered_count, 6) * 3

    # 要点加分（最多 8 项，每项 2 分）
    bullet_count = len(re.findall(r"(?m)^\s*[-•]", text))
    score += min(bullet_count, 8) * 2

    return score


def is_valid_zh_pest_answer(text: str) -> bool:
    """校验中文病虫害分析是否有效（放宽标准）"""
    if not text or not text.strip():
        return False

    required_sections = [
        "## 问题判断",
        "## 可能症状",
        "## 防治建议",
    ]

    # 中文必须至少有这三个核心章节
    if any(section not in text for section in required_sections):
        return False

    # 长度从 700 放宽到 350
    if len(text) < 350:
        return False

    # 有至少 3 条编号建议即可
    numbered_count = len(re.findall(r"(?m)^\s*\d+[.、)]", text))
    if numbered_count < 3:
        return False

    return True


def validate_pest_response(response: str, class_name: str, response_language: str = "zh") -> bool:
    """校验图片分析回答是否结构完整、没有思考过程、没有明显空输出"""
    if not response or len(response.strip()) < 80:
        return False

    lowered = response.lower()
    if "<think>" in lowered or "</think>" in lowered:
        return False

    if "<analysis>" in lowered or "</analysis>" in lowered:
        return False

    if "<reasoning>" in lowered or "</reasoning>" in lowered:
        return False

    if response_language == "en":
        required_headings = [
            "## Problem Assessment",
            "## Possible Symptoms",
            "## Prevention Measures",
        ]
        min_len = 600
    elif response_language == "ms":
        required_headings = [
            "## Penilaian Masalah",
            "## Maklumat Penting",
            "## Langkah Cadangan",
        ]
        min_len = 500
    elif response_language == "th":
        required_headings = [
            "## การประเมินปัญหา",
            "## ข้อมูลสำคัญ",
            "## มาตรการที่แนะนำ",
        ]
        min_len = 500
    else:
        return is_valid_zh_pest_answer(response)

    matched_count = sum(1 for heading in required_headings if heading in response)

    if matched_count < 3:
        return False

    if len(response) < min_len:
        return False

    return True


def regenerate_pest_response_with_llm(
    bad_response: str,
    class_name: str,
    confidence: float,
    pest_info: str,
    evidence_text: str,
    response_language: str = "zh",
    max_attempts: int = 2,
) -> str:
    """使用大模型重新生成病虫害分析，保存最佳版本而不是轻易失败"""
    if model_instance is None or model_instance.model is None:
        logger.warning("模型未加载，无法重新生成")
        if response_language == "en":
            return "Image analysis generation failed: model not loaded. Please try again later."
        elif response_language == "ms":
            return "Analisis imej gagal dijana: model tidak dimuatkan. Sila cuba lagi kemudian."
        elif response_language == "th":
            return "การสร้างผลวิเคราะห์รูปภาพล้มเหลว: โมเดลไม่ได้โหลด โปรดลองใหม่ภายหลัง"
        else:
            return "图片分析生成失败：模型未加载，请稍后重试。"

    best_response = clean_model_output(bad_response or "")
    best_score = score_pest_answer(best_response, response_language)

    for attempt in range(max_attempts):
        try:
            logger.info("大模型重写病虫害分析 - 尝试 %d/%d", attempt + 1, max_attempts)

            if response_language == "en":
                system_content = (
                    "You are a durian pest and disease management expert. "
                    "Output only the final answer. Do not output thinking, <think>, internal analysis, or drafts. "
                    "Use exactly these Markdown headings: "
                    "## Problem Assessment, ## Possible Symptoms, ## Prevention Measures, ## Precautions. "
                    "Problem Assessment must be 2-3 sentences. "
                    "Possible Symptoms must contain exactly 3 bullet points. "
                    "Prevention Measures must contain exactly 5 numbered recommendations. "
                    "Each recommendation must include inspection target, severity criteria, specific action, and recheck timing. "
                    "Precautions must contain exactly 3 bullet points only. "
                    "Do not add extra paragraphs after Precautions. "
                    "Total length should be around 250-350 English words. "
                    "Do not use a generic disease template. "
                )
                user_content = (
                    f"Identified category: {class_name}\n"
                    f"Classification confidence: {confidence * 100:.1f}%\n\n"
                    f"Classification info:\n{pest_info}\n\n"
                    f"Reference materials:\n{evidence_text or 'No valid reference materials'}\n\n"
                    f"Previous unqualified output:\n{bad_response or 'Empty'}\n\n"
                    "Please regenerate the final answer. Output only the final answer."
                )
            elif response_language == "ms":
                system_content = (
                    "Anda ialah pakar penyakit dan perosak durian. "
                    "Anda mesti menjana jawapan akhir yang lengkap, berbeza dan boleh dilaksanakan berdasarkan kategori yang dikenal pasti. "
                    "Larang mengeluarkan proses pemikiran, <think>, analisis dalaman atau draf. "
                    "Larang menggunakan templat generik. "
                    "Mesti mengeluarkan tajuk Markdown berikut dengan ketat: "
                    "## Penilaian Masalah, ## Gejala Mungkin, ## Langkah Cadangan, ## Langkah Berjaga-jaga. "
                    "## Gejala Mungkin mesti mempunyai sekurang-kurangnya 3 item, berkait rapat dengan kategori semasa. "
                    "## Langkah Cadangan mesti mempunyai 5 cadangan bernombor, setiap satu mengandungi: sasaran pemeriksaan, kriteria penghakiman, tindakan khusus, masa pemeriksaan semula. "
                    "Jumlah aksara tidak boleh kurang daripada 500 aksara Melayu. "
                    "Kategori berbeza mesti jelas berbeza. "
                    "Keluarkan hanya jawapan akhir."
                )
                user_content = (
                    f"Kategori yang dikenal pasti: {class_name}\n"
                    f"Keyakinan klasifikasi: {confidence * 100:.1f}%\n\n"
                    f"Maklumat klasifikasi:\n{pest_info}\n\n"
                    f"Bahan rujukan:\n{evidence_text or 'Tiada bahan rujukan yang sah'}\n\n"
                    f"Output tidak layak sebelumnya:\n{bad_response or 'Kosong'}\n\n"
                    "Sila janakan semula jawapan akhir. Keluarkan hanya jawapan akhir."
                )
            elif response_language == "th":
                system_content = (
                    "คุณเป็นผู้เชี่ยวชาญด้านโรคและแมลงศัตรูทุเรียน "
                    "คุณต้องสร้างคำตอบสุดท้ายที่สมบูรณ์ แตกต่าง และสามารถปฏิบัติได้ตามหมวดหมู่ที่ระบุ "
                    "ห้ามแสดงกระบวนการคิด <think> การวิเคราะห์ภายใน หรือร่างคำตอบ "
                    "ห้ามใช้เทมเพลตทั่วไป "
                    "ต้องแสดงหัวข้อ Markdown ต่อไปนี้อย่างเคร่งครัด: "
                    "## การประเมินปัญหา ## อาการที่อาจเกิดขึ้น ## มาตรการที่แนะนำ ## ข้อควรระวัง "
                    "## อาการที่อาจเกิดขึ้น ต้องมีอย่างน้อย 3 รายการ เกี่ยวข้องอย่างใกล้ชิดกับหมวดหมู่ปัจจุบัน "
                    "## มาตรการที่แนะนำ ต้องมี 5 คำแนะนำที่มีหมายเลข แต่ละรายการต้องมี: เป้าหมายการตรวจสอบ เกณฑ์การตัดสินใจ การดำเนินการเฉพาะ เวลาการตรวจสอบซ้ำ "
                    "จำนวนอักขระทั้งหมดต้องไม่น้อยกว่า 500 อักขระภาษาไทย "
                    "หมวดหมู่ต่างๆ ต้องแตกต่างกันอย่างชัดเจน "
                    "แสดงเฉพาะคำตอบสุดท้าย"
                )
                user_content = (
                    f"หมวดหมู่ที่ระบุ: {class_name}\n"
                    f"ความมั่นใจในการจำแนก: {confidence * 100:.1f}%\n\n"
                    f"ข้อมูลการจำแนก:\n{pest_info}\n\n"
                    f"เอกสารอ้างอิง:\n{evidence_text or 'ไม่มีเอกสารอ้างอิงที่ถูกต้อง'}\n\n"
                    f"ผลลัพธ์ที่ไม่ถูกต้องก่อนหน้า:\n{bad_response or 'ว่าง'}\n\n"
                    "โปรดสร้างคำตอบสุดท้ายใหม่ แสดงเฉพาะคำตอบสุดท้าย"
                )
            else:
                system_content = (
                    "你是榴莲病虫害防治专家。"
                    "请只输出最终答案，不要输出思考过程、<think>、内部分析或草稿。"
                    "必须严格包含以下 Markdown 标题："
                    "## 问题判断、## 可能症状、## 防治建议、## 药剂/处理参考、## 注意事项。"
                    "必须根据识别类别生成差异化内容，禁止通用模板。"
                    "## 可能症状 至少 3 条，必须紧扣当前类别和对应部位。"
                    "## 防治建议 必须有 5 条编号建议，每条包含：检查对象、判断标准、具体操作、复查时间。"
                    "## 药剂/处理参考 至少 4 条，包含：处理方向、使用部位、适用条件、合规提醒、抗药性提醒。"
                    "总字数不少于 700 个中文字符。"
                )
                user_content = (
                    f"识别类别：{class_name}\n"
                    f"分类置信度：{confidence * 100:.1f}%\n\n"
                    f"分类信息和类别重点：\n{pest_info}\n\n"
                    f"上一次不合格输出：\n{bad_response or '空'}\n\n"
                    "请重新生成最终回答。不要复用上一次的错误内容。"
                )

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]

            if response_language == "en":
                response = run_with_model_lock(
                    model_instance.generate,
                    messages,
                    max_new_tokens=800,
                    min_tokens=220,
                    temperature=0.35,
                    top_p=0.85,
                    top_k=40,
                    repetition_penalty=1.05,
                )
            else:
                response = run_with_model_lock(
                    model_instance.generate,
                    messages,
                    max_new_tokens=1024,
                    temperature=0.35,
                    top_p=0.8,
                    top_k=30,
                    repetition_penalty=1.18,
                )

            candidate = clean_model_output(response)

            if candidate:
                candidate_score = score_pest_answer(candidate, response_language)
                if candidate_score > best_score:
                    best_response = candidate
                    best_score = candidate_score
                    logger.info("更新最佳版本 - 尝试 %d/%d，分数: %d", attempt + 1, max_attempts, best_score)

            # 如果完全合格，立即返回
            if response_language == "zh":
                if is_valid_zh_pest_answer(candidate):
                    logger.info("大模型重写成功 - 尝试 %d/%d，长度: %d", attempt + 1, max_attempts, len(candidate))
                    return candidate
            else:
                if validate_pest_response(candidate, class_name, response_language):
                    logger.info("大模型重写成功 - 尝试 %d/%d", attempt + 1, max_attempts)
                    return candidate

        except Exception as e:
            logger.warning("大模型重写异常 - 尝试 %d/%d: %s", attempt + 1, max_attempts, e)

    # 所有尝试完成，返回最佳版本或失败提示
    if best_response.strip():
        logger.warning("模型多次生成未完全合格，返回当前最好版本，长度=%d，分数=%d", len(best_response), best_score)
        return best_response

    logger.error("大模型重写失败，无有效输出")
    if response_language == "en":
        return "Image analysis generation failed: model did not return valid content. Please upload a clearer image or try again later."
    elif response_language == "ms":
        return "Analisis imej gagal dijana: model tidak mengembalikan kandungan yang sah. Sila muat naik imej yang lebih jelas atau cuba lagi kemudian."
    elif response_language == "th":
        return "การสร้างผลวิเคราะห์รูปภาพล้มเหลว: โมเดลไม่ได้ส่งคืนเนื้อหาที่ถูกต้อง โปรดอัปโหลดรูปภาพที่ชัดเจนขึ้นหรือลองใหม่ภายหลัง"
    else:
        return "图片分析生成失败：模型没有返回有效内容，请重新上传更清晰的图片或稍后重试。"


def build_fallback_pest_response(class_name: str, confidence: float, response_language: str = "zh") -> str:
    """当大模型输出不合格时，返回稳定的兜底病虫害建议模板"""
    confidence_percent = confidence * 100

    if response_language == "en":
        return f"""## Problem Assessment
The image classification model identified the issue as {class_name}, with a confidence of {confidence_percent:.1f}%. This result should be treated as a preliminary reference and confirmed with field observation.

## Possible Symptoms
- Leaves may show spots, blight, yellowing, or local necrosis.
- The affected area may expand under humid or poorly ventilated conditions.
- Plant vigor may decline if the issue continues to spread.

## Prevention Measures
1. Mark and observe the suspected plant area to confirm whether symptoms expand.
2. Remove severely affected leaves or plant residues where appropriate.
3. Improve field drainage and ventilation to reduce prolonged humidity.
4. If pesticides are needed, consult local agricultural technicians and follow label instructions.

## Precautions
- This result is generated from image classification and does not replace field diagnosis.
- If symptoms spread quickly, seek professional on-site confirmation."""

    if response_language == "ms":
        return f"""## Penilaian Masalah
Model klasifikasi imej mengenal pasti masalah ini sebagai {class_name}, dengan keyakinan {confidence_percent:.1f}%. Keputusan ini harus dianggap sebagai rujukan awal dan perlu disahkan melalui pemerhatian di lapangan.

## Gejala
- Daun mungkin menunjukkan bintik, kekeringan, kekuningan atau nekrosis setempat.
- Kawasan terjejas mungkin merebak dalam keadaan lembap atau pengudaraan lemah.
- Pertumbuhan pokok boleh menurun jika masalah terus merebak.

## Langkah Cadangan
1. Tandakan dan pantau bahagian pokok yang disyaki untuk melihat sama ada gejala merebak.
2. Buang daun atau sisa tanaman yang terjejas dengan jelas jika sesuai.
3. Perbaiki saliran dan pengudaraan kebun untuk mengurangkan kelembapan berpanjangan.
4. Jika racun diperlukan, rujuk pegawai pertanian tempatan dan ikut arahan label.

## Langkah Berjaga-jaga
- Keputusan ini berasal daripada model klasifikasi imej dan tidak menggantikan diagnosis lapangan.
- Jika gejala merebak dengan cepat, dapatkan pengesahan pakar di lokasi."""

    if response_language == "th":
        return f"""## การประเมินปัญหา
โมเดลจำแนกรูปภาพระบุปัญหานี้เป็น {class_name} โดยมีความมั่นใจ {confidence_percent:.1f}% ผลลัพธ์นี้ควรใช้เป็นข้อมูลอ้างอิงเบื้องต้นและควรยืนยันด้วยการสังเกตในพื้นที่จริง

## อาการที่อาจเกิดขึ้น
- ใบอาจมีจุด โรคใบไหม้ ใบเหลือง หรือเนื้อเยื่อตายเฉพาะจุด
- พื้นที่ที่ได้รับผลกระทบอาจขยายตัวในสภาพแวดล้อมที่ชื้นหรือระบายอากาศไม่ดี
- ความแข็งแรงของต้นอาจลดลงหากปัญหายังคงลุกลาม

## มาตรการที่แนะนำ
1. ทำเครื่องหมายและติดตามบริเวณที่สงสัยเพื่อดูว่าอาการขยายตัวหรือไม่
2. กำจัดใบหรือเศษพืชที่ได้รับผลกระทบอย่างชัดเจนตามความเหมาะสม
3. ปรับปรุงการระบายน้ำและการระบายอากาศในสวนเพื่อลดความชื้นสะสม
4. หากจำเป็นต้องใช้สารเคมี ควรปรึกษาเจ้าหน้าที่เกษตรในพื้นที่และปฏิบัติตามฉลากอย่างเคร่งครัด

## ข้อควรระวัง
- ผลลัพธ์นี้มาจากโมเดลจำแนกรูปภาพ ไม่สามารถแทนที่การวินิจฉัยภาคสนามได้
- หากอาการแพร่กระจายเร็ว ควรขอการยืนยันจากผู้เชี่ยวชาญในพื้นที่"""

    return f"""## 问题判断
图像分类模型识别结果为 {class_name}，置信度为 {confidence_percent:.1f}%。该结果应作为初步参考，建议结合现场症状进一步确认。

## 可能症状
- 叶片可能出现病斑、枯斑、黄化或局部坏死。
- 高湿、排水不良或通风不足时，症状可能进一步扩展。
- 如果病害持续发展，可能影响植株长势和叶片光合作用。

## 防治建议
1. 现场复核：先检查病斑是否集中在老叶、嫩叶、叶缘或叶脉附近，并观察是否有扩大趋势。
2. 环境管理：检查园区是否长期潮湿、排水不畅或通风不足，雨季应优先改善排水和降低叶面长期湿润时间。
3. 修剪清园：剪除明显受害严重的叶片或枝条，清理落叶和病残体，避免病原继续积累。
4. 营养管理：避免偏施氮肥，适当补充钾肥、有机肥和中微量元素，提高植株抗逆能力。
5. 防治措施：如果症状持续扩散，可结合当地农业技术人员建议选择合规防治措施；涉及药剂时必须遵守登记范围、标签剂量和安全间隔期。
6. 复查周期：处理后连续观察 3-7 天，如果病斑扩大、叶片脱落加重或周边植株出现类似症状，应尽快进行现场诊断。

## 药剂/处理参考
- 可先以清园、修剪、排水、降低湿度和改善通风为基础措施。
- 如果病斑继续扩展或多株发病，可咨询当地农业技术人员，选择当地登记允许用于榴莲或相应病害的药剂。
- 对叶斑、炭疽类问题，可参考保护性杀菌剂和内吸性杀菌剂的防治方向，但不得自行混配或超量使用。
- 涉及药剂时必须按标签剂量、施药间隔、安全间隔期和再入间隔执行，并注意轮换不同作用机制药剂。

## 注意事项
- 当前结果来自图像分类模型，不能替代现场人工诊断。
- 如果症状快速扩散，建议尽快联系当地农业技术人员确认。"""


# =========================
# Prompt 工具
# =========================
def enhance_retrieval_query(user_query: str, response_language: str = "zh") -> str:
    """增强检索查询，针对土壤等特定问题添加相关关键词。
    
    关键：消费者问题不应该检索农业资料。
    """
    query = user_query or ""
    lowered = query.lower()
    
    # 如果是消费者问题，不增强查询，直接返回原问题
    if not is_agricultural_diagnosis_query(query):
        return query

    soil_keywords = [
        "土壤", "土质", "酸碱", "pH", "排水",
        "soil", "drainage", "organic matter",
        "tanah", "saliran", "kelembapan",
        "ดิน", "ระบายน้ำ", "อินทรียวัตถุ"
    ]

    if any(keyword.lower() in lowered for keyword in soil_keywords):
        if response_language == "en":
            return query + " durian soil pH drainage organic matter planting"
        if response_language == "ms":
            return query + " durian tanah pH saliran bahan organik penanaman"
        if response_language == "th":
            return query + " ทุเรียน ดิน ค่า pH การระบายน้ำ อินทรียวัตถุ การปลูก"
        return query + " 榴莲 土壤 pH 排水 有机质 种植"

    return query


def extract_last_user_message(messages: List[Message]) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    return ""


def build_evidence_text(evidence: List[Dict], max_chars: int) -> str:
    parts = []

    for i, ev in enumerate(evidence, 1):
        text = (ev.get("text") or "")[:max_chars]
        doc = ev.get("doc") or ev.get("source") or ""
        score = float(ev.get("score", 0.0))

        if doc:
            parts.append(f"[资料{i} | 来源: {doc} | 相似度: {score:.3f}]\n{text}")
        else:
            parts.append(f"[资料{i} | 相似度: {score:.3f}]\n{text}")

    return "\n\n".join(parts)


def build_messages(
    raw_messages: List[Message],
    evidence_text: str = "",
    max_history_messages: int = 10,
    response_language: str = "zh",
) -> List[Dict]:
    response_language = response_language or "zh"
    system_prompt = SYSTEM_PROMPTS.get(response_language, SYSTEM_PROMPTS["zh"])

    # 添加强制语言规则
    language_rules = {
        "zh": (
            "你必须只使用中文回答。"
            "即使参考资料或历史消息包含其他语言，最终回答也必须是中文。"
        ),
        "en": (
            "You must answer only in English. "
            "Even if the references or previous messages contain Chinese, Malay, or Thai, the final answer must be in English."
        ),
        "ms": (
            "Anda mesti menjawab hanya dalam Bahasa Melayu. "
            "Walaupun bahan rujukan atau sejarah perbualan mengandungi bahasa Cina, Inggeris atau Thai, jawapan akhir mesti dalam Bahasa Melayu. "
            "Jangan gunakan tajuk bahasa Cina seperti '土壤要求', '防治建议', '症状表现' atau '注意事项'. "
            "Gunakan hanya tajuk Bahasa Melayu."
        ),
        "th": (
            "คุณต้องตอบเป็นภาษาไทยเท่านั้น "
            "แม้ว่าเอกสารอ้างอิงหรือประวัติการสนทนาจะมีภาษาจีน อังกฤษ หรือมลายู คำตอบสุดท้ายต้องเป็นภาษาไทยเท่านั้น"
        ),
    }

    system_prompt += "\n\n" + language_rules.get(response_language, language_rules["zh"])

    # 合并由上下文选择器插入的临时 system 指令。
    extra_system_prompts = [
        str(getattr(m, "content", "") or "").strip()
        for m in (raw_messages or [])
        if getattr(m, "role", "") == "system" and str(getattr(m, "content", "") or "").strip()
    ]
    if extra_system_prompts:
        system_prompt += "\n\n" + "\n\n".join(extra_system_prompts[-3:])

    # ===== 场景自由度规则：不要把所有问题都导向治理/病虫害/药剂 =====
    latest_user_query = ""
    for _msg in reversed(raw_messages or []):
        if getattr(_msg, "role", None) == "user":
            latest_user_query = str(getattr(_msg, "content", "") or "")
            break

    scenario_rules = {
        "zh": (
            "\n\n【场景自由度规则】"
            "先判断用户真正要问的场景，再决定回答重点。"
            "如果用户问品种、口感、购买、保存、食用、营养、价格、市场、冷链、甜品或普通常识，就直接回答这些内容；不要强行加入病虫害、治理、施肥、喷药或果园管理。"
            "只有当用户明确问病虫害、黄叶、根腐、流胶、喷药、防治、施肥、排水、修剪、花果管理，或图片识别结果确实属于病虫害时，才展开治理和药剂方向。"
            "如果问题是泛泛的‘榴莲品种’、‘猫山王为什么苦’、‘榴莲和牛奶’，应像产业/消费顾问一样回答，而不是像植保医生一样回答。"
        ),
        "en": (
            "\n\n[Scenario freedom rule] First identify what the user is really asking. "
            "If the question is about varieties, taste, buying, storage, eating, nutrition, price, market, cold chain, desserts, or general knowledge, answer that scenario directly. Do not force disease control, fertilizer, pesticide, or orchard management into the answer. "
            "Only discuss treatment or pesticide direction when the user clearly asks about pests, disease, yellowing, root rot, gummosis, spraying, control, fertilization, drainage, pruning, fruit management, or when an image diagnosis is actually pest/disease-related."
        ),
        "ms": (
            "\n\n[Peraturan kebebasan senario] Kenal pasti dahulu apa sebenarnya yang ditanya pengguna. "
            "Jika soalan tentang varieti, rasa, pembelian, penyimpanan, cara makan, nutrisi, harga, pasaran, rantaian sejuk, pencuci mulut atau pengetahuan umum, jawab terus topik itu. Jangan paksa nasihat penyakit, baja, racun atau pengurusan kebun. "
            "Hanya bincang rawatan atau racun apabila pengguna jelas bertanya tentang penyakit, perosak, daun kuning, reput akar, gummosis, semburan, kawalan, baja, saliran, pemangkasan atau pengurusan buah."
        ),
        "th": (
            "\n\n[กฎอิสระตามสถานการณ์] ให้ระบุก่อนว่าผู้ใช้ถามเรื่องอะไรจริง ๆ "
            "ถ้าถามเรื่องสายพันธุ์ รสชาติ การซื้อ การเก็บรักษา การกิน โภชนาการ ราคา ตลาด โซ่ความเย็น ของหวาน หรือความรู้ทั่วไป ให้ตอบเรื่องนั้นโดยตรง อย่าบังคับไปเรื่องโรค ปุ๋ย สารเคมี หรือการจัดการสวน "
            "พูดเรื่องการจัดการโรคหรือสารเคมีเฉพาะเมื่อผู้ใช้ถามชัดเจนเรื่องโรค แมลง ใบเหลือง รากเน่า ยางไหล การพ่น การควบคุม ปุ๋ย การระบายน้ำ การตัดแต่ง หรือการจัดการผลเท่านั้น"
        ),
    }
    system_prompt += scenario_rules.get(response_language, scenario_rules["zh"])

    # 农药/药剂规则只在问题确实相关时追加，避免“啥都往治理靠”。
    if asks_treatment_or_pesticide(latest_user_query) or is_agricultural_diagnosis_query(latest_user_query):
        system_prompt += chemical_guidance_rules(response_language)

    if PREFER_MARKDOWN_TABLE_OUTPUT:
        adaptive_table_rules = {
            "zh": (
                "\n\n【表格优先规则】"
                "如果本次问题涉及诊断、检查项、风险分级、处理方案、药剂/处理方向、步骤安排或方案比较，优先输出一个标准 Markdown 表格。"
                "表格必须有表头和分隔行，格式类似：\n"
                "| 项目 | 判断标准 | 具体操作 | 复查时间 |\n"
                "|---|---|---|---|\n"
                "如果问题很简单、只需一句结论或少量说明，不要强行使用表格。"
                "需要表格时，优先用 4-6 列展示检查、判断、处理和复查时间。"
            ),
            "en": (
                "\n\n[Adaptive table rule] Use a Markdown table when the answer involves diagnosis, inspection items, severity grading, treatment options, chemical/handling references, timing, or comparison. "
                "Do not force a table for simple questions. If using a table, include a header and separator row, for example:\n"
                "| Check item | Criteria | Action | Recheck timing |\n"
                "|---|---|---|---|\n"
            ),
            "ms": (
                "\n\n[Peraturan jadual adaptif] Gunakan jadual Markdown apabila jawapan melibatkan diagnosis, perkara pemeriksaan, tahap keterukan, pilihan rawatan, rujukan bahan/kaedah, masa semakan atau perbandingan. "
                "Jika soalan mudah, jangan paksa jadual. Jadual mesti mempunyai baris tajuk dan pemisah.\n"
            ),
            "th": (
                "\n\n[กฎตารางแบบปรับตามเนื้อหา] ใช้ตาราง Markdown เมื่อคำตอบเกี่ยวข้องกับการวินิจฉัย รายการตรวจ ระดับความรุนแรง ทางเลือกการจัดการ แนวทางการใช้สาร/การจัดการ เวลาในการตรวจซ้ำ หรือการเปรียบเทียบ "
                "หากคำถามเรียบง่าย ไม่จำเป็นต้องบังคับใช้ตาราง และตารางต้องมีหัวตารางกับเส้นแบ่ง Markdown.\n"
            ),
        }
        system_prompt += adaptive_table_rules.get(response_language, adaptive_table_rules["zh"])

    if evidence_text:
        if response_language == "en":
            system_prompt += (
                "\n\nThe following are retrieved reference materials. Prioritize them when relevant. "
                "If the materials are insufficient or irrelevant, do not describe irrelevant materials in detail; "
                "briefly state that references are insufficient and provide cautious general advice.\n\n"
                f"{evidence_text}"
            )
        elif response_language == "ms":
            system_prompt += (
                "\n\nBerikut ialah bahan rujukan yang ditemui. Utamakan bahan ini hanya jika berkaitan. "
                "Jika bahan tidak mencukupi atau tidak berkaitan, jangan huraikan bahan yang tidak berkaitan secara panjang lebar; "
                "nyatakan secara ringkas bahawa bahan rujukan tidak mencukupi dan berikan cadangan umum secara berhati-hati.\n\n"
                f"{evidence_text}"
            )
        elif response_language == "th":
            system_prompt += (
                "\n\nต่อไปนี้คือข้อมูลอ้างอิงที่ค้นคืนมาได้ ให้ใช้ข้อมูลเหล่านี้เฉพาะเมื่อเกี่ยวข้องเท่านั้น "
                "หากข้อมูลไม่เพียงพอหรือไม่ตรงกับคำถาม ห้ามอธิบายข้อมูลที่ไม่เกี่ยวข้องอย่างยืดยาว "
                "ให้ระบุสั้น ๆ ว่าข้อมูลอ้างอิงไม่เพียงพอ แล้วให้คำแนะนำทั่วไปอย่างระมัดระวัง\n\n"
                f"{evidence_text}"
            )
        else:
            system_prompt += (
                "\n\n以下是检索到的参考资料。回答时优先依据这些资料；如果资料与问题不相关或不足，请不要详细描述不相关资料，"
                "只需说明资料不足，并给出谨慎的一般建议。\n\n"
                f"{evidence_text}"
            )

    image_context_text = extract_image_context_messages(raw_messages)
    if image_context_text:
        system_prompt += (
            "\n\n[Saved image context]\n"
            "The following hidden context was produced by the vision model in earlier image turns. "
            "Use it only when the current user message is a follow-up to that image; do not let it affect unrelated new topics.\n\n"
            f"{image_context_text}"
        )

    non_system = [
        {"role": m.role, "content": m.content}
        for m in raw_messages
        if m.role != "system"
    ]
    non_system = non_system[-max_history_messages:]

    logger.info("构建消息 - 响应语言: %s, 系统提示词长度: %d", response_language, len(system_prompt))
    return [{"role": "system", "content": system_prompt}] + non_system


# =========================
# FastAPI 应用
# =========================
app = FastAPI(
    title="榴莲GPT - 推理服务",
    description="基于微调 Qwen3-14B LoRA 的榴莲种植专家系统",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件服务
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# =========================
# 图片分析任务存储（两段式任务模式）
# =========================
ANALYZE_JOBS: Dict[str, Dict[str, Any]] = {}

def save_uploaded_image_bytes(file_bytes: bytes, filename: str = "") -> str:
    file_ext = os.path.splitext(filename or "")[1].lower()
    if file_ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        file_ext = ".jpg"
    image_filename = f"{uuid.uuid4().hex}{file_ext}"
    image_path = UPLOAD_DIR / image_filename
    with open(image_path, "wb") as f:
        f.write(file_bytes)
    return f"/uploads/{image_filename}"

def run_analyze_pest_core_from_bytes(
    file_bytes: bytes,
    filename: str,
    content_type: str,
    query: str,
    response_language: str,
) -> dict:
    """从字节数据分析病虫害，返回完整分析结果字典"""
    if pest_classifier is None:
        raise RuntimeError("病虫害分类模型未加载")
    if not is_model_ready():
        raise RuntimeError("大模型未加载")
    
    response_language = response_language or "zh"
    image = Image.open(BytesIO(file_bytes)).convert('RGB')
    image_url = save_uploaded_image_bytes(file_bytes, filename)
    image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    classification = pest_classifier.classify(image)
    if response_language == "en":
        class_name = get_pest_class_name(classification.class_id, "en")
    elif response_language == "ms":
        class_name = get_pest_class_name(classification.class_id, "ms")
    elif response_language == "th":
        class_name = get_pest_class_name(classification.class_id, "th")
    else:
        class_name = get_pest_class_name(classification.class_id, "zh")
    
    confidence_warning = ""
    if classification.confidence < 0.8:
        if response_language == "en":
            confidence_warning = "\nNote: Current classification confidence is below 80%. This result is for reference only."
        elif response_language == "ms":
            confidence_warning = "\nNota: Keyakinan klasifikasi semasa di bawah 80%. Hasil ini hanya untuk rujukan."
        elif response_language == "th":
            confidence_warning = "\nหมายเหตุ: ความมั่นใจในการจำแนกปัจจุบันต่ำกว่า 80%"
        else:
            confidence_warning = "\n注意：当前分类置信度不足 80%，该结果仅供参考。"
    
    if response_language == "en":
        pest_info = f"Image classification: {class_name} ({classification.confidence * 100:.1f}%){confidence_warning}\n\nProvide analysis based on this classification."
    elif response_language == "ms":
        pest_info = f"Klasifikasi imej: {class_name} ({classification.confidence * 100:.1f}%){confidence_warning}\n\nBerikan analisis berdasarkan klasifikasi ini."
    elif response_language == "th":
        pest_info = f"การจำแนกรูปภาพ: {class_name} ({classification.confidence * 100:.1f}%){confidence_warning}\n\nให้การวิเคราะห์ตามการจำแนกนี้"
    else:
        pest_info = f"图像分类：{class_name}（{classification.confidence * 100:.1f}%）{confidence_warning}\n\n请基于此分类结果进行分析。"
    
    # 一致性约束：允许低置信度复核，但禁止否定图像分类结果。
    if response_language == "en":
        consistency_instruction = (
            f"\n\nConsistency requirement: The identified category is {class_name} with confidence {classification.confidence * 100:.1f}%. "
            f"Base the answer on {class_name}. Do not write that it is not {class_name}, more likely not {class_name}, or rather than {class_name}. "
            "If uncertain, state that the classification has low confidence and needs field confirmation."
        )
    elif response_language == "ms":
        consistency_instruction = (
            f"\n\nKeperluan konsistensi: Kategori yang dikenal pasti ialah {class_name} dengan keyakinan {classification.confidence * 100:.1f}%. "
            f"Jawapan mesti berpaksikan {class_name}. Jangan tulis bahawa ia bukan {class_name} atau lebih mungkin bukan {class_name}. "
            "Jika tidak pasti, nyatakan bahawa keyakinan klasifikasi rendah dan perlu disahkan di lapangan."
        )
    elif response_language == "th":
        consistency_instruction = (
            f"\n\nข้อกำหนดด้านความสอดคล้อง: หมวดหมู่ที่จำแนกได้คือ {class_name} โดยมีความมั่นใจ {classification.confidence * 100:.1f}% "
            f"คำตอบต้องอิง {class_name} ห้ามเขียนว่าไม่ใช่ {class_name} หรือมีแนวโน้มไม่ใช่ {class_name} "
            "หากไม่แน่ใจ ให้ระบุว่าค่าความมั่นใจต่ำและต้องยืนยันภาคสนาม"
        )
    else:
        consistency_instruction = (
            f"\n\n一致性要求：本次图像分类结果是 {class_name}，置信度为 {classification.confidence * 100:.1f}%。"
            f"回答必须围绕“{class_name}”展开。禁止写“不是{class_name}”、“更可能不是{class_name}”、“而非{class_name}”、“不像{class_name}”这类否定分类结果的句子。"
            "如果置信度不足，只能说“疑似”或“需要结合现场症状复核”。"
        )
    pest_info += consistency_instruction

    evidence = []
    evidence_quality = "none"
    evidence_text = ""
    
    if rag_retriever and rag_retriever.enabled:
        if response_language == "en":
            search_query = f"{class_name} prevention symptoms"
        elif response_language == "ms":
            search_query = f"{class_name} pencegahan gejala"
        elif response_language == "th":
            search_query = f"{class_name} ป้องกัน อาการ"
        else:
            search_query = f"{class_name} 防治 症状"
        evidence, evidence_quality = rag_retriever.retrieve(search_query, top_k=RAG_CONFIG.get("top_k", 3))
        if evidence:
            evidence_text = build_evidence_text(evidence, max_chars=RAG_CONFIG.get("max_evidence_chars", 700))
    
    system_prompt = IMAGE_ANALYSIS_PROMPTS.get(response_language, IMAGE_ANALYSIS_PROMPTS["zh"])
    system_prompt += chemical_guidance_rules(response_language)
    if evidence_text:
        system_prompt += f"\n\n参考资料：{evidence_text}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": pest_info}
    ]
    
    if response_language == "zh":
        generation_kwargs = {
            "max_new_tokens": int(os.getenv("DURIAN_IMAGE_ZH_MAX_NEW_TOKENS", "1600")),
            "temperature": 0.35,
            "top_p": 0.85,
            "top_k": 40,
            "repetition_penalty": 1.05,
        }
    else:
        generation_kwargs = {
            "max_new_tokens": int(os.getenv("DURIAN_IMAGE_OTHER_MAX_NEW_TOKENS", "1600")),
            "temperature": 0.35,
            "top_p": 0.85,
            "top_k": 40,
            "repetition_penalty": 1.05,
        }
    
    generation_kwargs["response_language"] = response_language

    response = run_with_model_lock(
        model_instance.generate,
        messages,
        **generation_kwargs,
    )
    
    logger.warning("====== 当前运行的是任务模式新质量控制版本 V2 ======")
    
    response = clean_model_output(response)
    response = enforce_classification_consistency(
        response,
        class_name,
        classification.confidence,
        response_language,
    )

    # 不再做强制结构校验、重写或模板兜底。
    # 模型按对应语言的自然逻辑回答：简单问题简短答，复杂问题自然分段，必要时才用表格。

    if response_language == "en":
        response = f"## Classification\n{class_name} ({classification.confidence * 100:.1f}%)\n\n{response}"
    elif response_language == "ms":
        response = f"## Klasifikasi\n{class_name} ({classification.confidence * 100:.1f}%)\n\n{response}"
    elif response_language == "th":
        response = f"## การจำแนก\n{class_name} ({classification.confidence * 100:.1f}%)\n\n{response}"
    else:
        response = f"## 分类结果\n{class_name}（{classification.confidence * 100:.1f}%）\n\n{response}"

    # 不追加兜底表格或模板；是否使用表格完全由模型自然决定。
    tokens_generated = approx_token_count(response)
    
    return {
        "response": response,
        "tokens_generated": tokens_generated,
        "model": model_instance.model_name,
        "timestamp": datetime.now().isoformat(),
        "evidence": [{"text": ev.get("text", ""), "score": float(ev.get("score", 0.0)), "doc": ev.get("doc")} for ev in evidence],
        "evidence_quality": evidence_quality,
        "image_url": image_url,
    }

def run_analyze_pest_job(job_id: str, file_bytes: bytes, filename: str, content_type: str, query: str, response_language: str):
    """后台任务：执行病虫害分析"""
    try:
        ANALYZE_JOBS[job_id]["status"] = "running"
        result = run_analyze_pest_core_from_bytes(file_bytes, filename, content_type, query, response_language)
        ANALYZE_JOBS[job_id]["status"] = "done"
        ANALYZE_JOBS[job_id]["result"] = result
    except Exception as e:
        logger.exception("图片分析任务失败: %s", job_id)
        ANALYZE_JOBS[job_id]["status"] = "error"
        ANALYZE_JOBS[job_id]["error"] = str(e)

def run_analyze_pest_core_from_vl_label(
    file_bytes: bytes,
    filename: str,
    content_type: str,
    query: str,
    response_language: str,
    vl_label_result: Dict[str, Any],
    vision_model: str = "qwen-vl-max",
    vision_tokens: int = 0,
) -> dict:
    """Use local LoRA as the final advisor after Qwen-VL returns a strict image label."""
    if model_instance is None or model_instance.model is None:
        raise RuntimeError("Local advisor model is not loaded")

    response_language = response_language or "zh"
    image_url = save_uploaded_image_bytes(file_bytes, filename)
    query = query or ""
    vl_label_result = vl_label_result or {}
    category_type = str(vl_label_result.get("category_type") or "unknown").strip().lower()
    label = str(vl_label_result.get("label") or "Unknown").strip()
    label_zh = str(vl_label_result.get("label_zh") or label).strip()
    confidence = float(vl_label_result.get("confidence") or 0.0)
    visual_evidence_text = str(vl_label_result.get("visual_evidence") or "").strip()
    alternatives = vl_label_result.get("alternatives") or []
    if not isinstance(alternatives, list):
        alternatives = [str(alternatives)]
    cannot_confirm = str(vl_label_result.get("cannot_confirm") or "").strip()

    vl_label_context = json.dumps(
        {
            "category_type": category_type,
            "label": label,
            "label_zh": label_zh,
            "confidence": confidence,
            "visual_evidence": visual_evidence_text,
            "alternatives": alternatives,
            "cannot_confirm": cannot_confirm,
        },
        ensure_ascii=False,
        indent=2,
    )

    evidence = []
    evidence_quality = "none"
    evidence_text = ""
    if rag_retriever and rag_retriever.enabled and (category_type in {"disease", "pest"} or is_agricultural_diagnosis_query(query) or asks_treatment_or_pesticide(query)):
        search_query = (query + "\n" + label_zh + " " + label + "\n" + visual_evidence_text[:300]).strip()
        evidence, evidence_quality = rag_retriever.retrieve(search_query, top_k=RAG_CONFIG.get("top_k", 3))
        if evidence:
            evidence_text = build_evidence_text(evidence, max_chars=RAG_CONFIG.get("max_evidence_chars", 700))

    # ===== 场景档位：减法版 prompt，但图片病虫害不能短到像一句摘要 =====
    agri_image_types = {"disease", "pest", "cultivation_issue"}
    consumer_image_types = {"variety", "quality"}

    if category_type in agri_image_types:
        image_profile = "image_agri_medium"
    elif category_type in consumer_image_types:
        image_profile = "image_consumer_natural"
    else:
        image_profile = "image_general_natural"

    system_prompt = IMAGE_ANALYSIS_PROMPTS.get(response_language, IMAGE_ANALYSIS_PROMPTS["zh"])
    system_prompt += (
        "\n\n[Image label]\n"
        "Qwen-VL-Max has returned a structured image label. Use it as image evidence, not as a final answer. "
        "Answer the user's actual question. Do not claim details that are not present in the label/evidence."
    )

    if image_profile == "image_agri_medium":
        profile_rules = {
            "zh": (
                "\n\n【图片病虫害/种植问题回答档位】"
                "本次是榴莲图片诊断问题，回答要中等详细，不要只给一句结论，也不要套固定表格。"
                "自然组织为：先给判断；再说明图像依据；然后给现场复核点；最后给低风险处理建议。"
                "如果不能百分百确认病名，要说明不确定性，但仍围绕分类名称给出合理处理方向。"
                "不要输出商业分选、冷链、采后渠道等无关内容。除非用户明确问喷药，否则药剂只给方向，不要长篇用药方案。"
            ),
            "en": (
                "\n\n[Image crop-health answer profile] This is a durian image diagnosis. "
                "Give a medium-detail answer: conclusion, visual basis, field checks, and practical low-risk actions. "
                "Do not reduce it to one sentence, but do not force a table or a rigid report. "
                "If the label is uncertain, say so while still giving a reasonable direction. Avoid commercial grading or cold-chain advice."
            ),
            "ms": (
                "\n\n[Profil jawapan imej kesihatan tanaman] Ini ialah diagnosis imej durian. "
                "Berikan jawapan sederhana terperinci: kesimpulan, asas visual, semakan lapangan dan tindakan praktikal berisiko rendah. "
                "Jangan jadikan satu ayat sahaja dan jangan paksa jadual."
            ),
            "th": (
                "\n\n[รูปแบบคำตอบภาพสุขภาพพืช] นี่คือการวินิจฉัยภาพทุเรียน. "
                "ตอบระดับปานกลาง: ข้อสรุป หลักฐานจากภาพ จุดตรวจภาคสนาม และการจัดการเบื้องต้นที่เสี่ยงต่ำ. "
                "อย่าตอบสั้นแค่ประโยคเดียว และไม่ต้องบังคับตาราง."
            ),
        }
        system_prompt += profile_rules.get(response_language, profile_rules["zh"])
    elif image_profile == "image_consumer_natural":
        profile_rules = {
            "zh": (
                "\n\n【消费/品种/品质图片回答档位】"
                "本次是榴莲品种、口感、品质、成熟度、食用或保存问题。回答自然、简洁但有信息量。"
                "只谈品种可能性、口感、成熟度、能否食用、保存或挑选。不要输出病虫害、防治、喷药、施肥、复查时间、果园管理、商业分选或冷链 SOP。"
            ),
            "en": (
                "\n\n[Consumer image answer profile] This is a variety, taste, quality, ripeness, eating, or storage question. "
                "Answer naturally and concisely. Discuss only likely variety, taste, ripeness, edibility, storage, or selection. "
                "Do not add pest control, spraying, fertilization, orchard management, commercial grading, or cold-chain SOP."
            ),
            "ms": "\n\n[Profil imej pengguna] Jawab secara semula jadi tentang varieti, rasa, kematangan, boleh dimakan, simpanan atau pemilihan sahaja.",
            "th": "\n\n[รูปแบบภาพผู้บริโภค] ตอบอย่างเป็นธรรมชาติ เฉพาะเรื่องสายพันธุ์ รสชาติ ความสุก การกิน การเก็บ หรือการเลือกซื้อเท่านั้น.",
        }
        system_prompt += profile_rules.get(response_language, profile_rules["zh"])
    else:
        system_prompt += (
            "\n\n[General image answer profile]\n"
            "Answer naturally from the image label and user question. If the label is uncertain, state what cannot be confirmed and what extra image/context is needed."
        )

    if (
        asks_treatment_or_pesticide(query)
        or is_agricultural_diagnosis_query(query)
        or category_type in {"disease", "pest", "cultivation_issue"}
    ):
        system_prompt += chemical_guidance_rules(response_language)

    system_prompt += (
        "\n\n[Two-stage image workflow]\n"
        "Qwen-VL-Max provides only the image classification label and visual basis. "
        "You are the final durian advisor. Use the label, visual basis, user question and any relevant references to answer."
    )
    if evidence_text:
        system_prompt += f"\n\nRetrieved references:\n{evidence_text}"

    if image_profile == "image_agri_medium":
        task_hint = (
            "Answer with medium detail. Include: likely problem name, visible basis, what to check on-site, and what to do first. "
            "Avoid a rigid table unless it is truly clearer."
        )
    elif image_profile == "image_consumer_natural":
        task_hint = (
            "Answer naturally about variety/quality/eating/storage only. No orchard treatment, commercial grading or pesticide advice."
        )
    else:
        task_hint = "Answer naturally based on the label and the user's actual question."

    user_content = f"""
User question:
{query or "(none)"}

Qwen-VL-Max structured image classification:
{vl_label_context}

Answer profile:
{image_profile}

Task:
{task_hint}
""".strip()

    if image_profile == "image_agri_medium":
        image_final_tokens = int(os.getenv("DURIAN_IMAGE_AGRI_MAX_NEW_TOKENS", "1000"))
    elif image_profile == "image_consumer_natural":
        image_final_tokens = int(os.getenv("DURIAN_IMAGE_CONSUMER_MAX_NEW_TOKENS", "520"))
    else:
        image_final_tokens = int(os.getenv("DURIAN_IMAGE_GENERAL_MAX_NEW_TOKENS", "700"))

    response = run_with_model_lock(
        model_instance.generate,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_new_tokens=image_final_tokens,
        temperature=0.35,
        top_p=0.85,
        top_k=40,
        repetition_penalty=1.05,
        response_language=response_language,
        # 图片回答由场景档位 prompt 控制，不再交给全局 editor/expand，避免重新变成强制报告。
        skip_detail_expand=True,
        skip_post_editor=True,
    )

    response = clean_model_output(response)
    active_context_card = build_text_context_card(
        query or "图片诊断",
        response,
        response_language,
        source="image_diagnosis",
        image_url=image_url,
    )
    tokens_generated = approx_token_count(response) if model_instance.tokenizer else 0

    return {
        "response": response,
        "tokens_generated": tokens_generated + int(vision_tokens or 0),
        "model": f"{model_instance.model_name} + vl_label({vision_model})",
        "timestamp": datetime.now().isoformat(),
        "evidence": [{"text": ev.get("text", ""), "score": float(ev.get("score", 0.0)), "doc": ev.get("doc")} for ev in evidence],
        "evidence_quality": evidence_quality,
        "image_url": image_url,
        "visual_observation": vl_label_context,
        "active_context_card": active_context_card,
    }

def run_analyze_pest_job_qwen_vl_max(job_id: str, file_bytes: bytes, filename: str, content_type: str, query: str, response_language: str):
    """Background task for Qwen-VL-Max image+text analysis."""
    try:
        ANALYZE_JOBS[job_id]["status"] = "running"
        if not qwen_vl_max_client or not qwen_vl_max_client.api_key:
            raise RuntimeError("Qwen-VL-Max client is not available")

        image = Image.open(BytesIO(file_bytes)).convert("RGB")
        visual_result = qwen_vl_max_client.classify_durian_image_label(
            image=image,
            user_query=query or "",
            response_language=response_language or "zh",
        )
        result = run_analyze_pest_core_from_vl_label(
            file_bytes=file_bytes,
            filename=filename,
            content_type=content_type,
            query=query,
            response_language=response_language,
            vl_label_result=visual_result,
            vision_model=visual_result.get("model", "qwen-vl-max"),
            vision_tokens=int(visual_result.get("tokens_generated", 0) or 0),
        )
        ANALYZE_JOBS[job_id]["status"] = "done"
        ANALYZE_JOBS[job_id]["result"] = result
    except Exception as e:
        logger.exception("Qwen-VL-Max image analysis job failed: %s", job_id)
        ANALYZE_JOBS[job_id]["status"] = "error"
        ANALYZE_JOBS[job_id]["error"] = str(e)

model_instance: Optional[DurianGPTModel] = None
rag_retriever: Optional[RAGRetriever] = None
pest_classifier: Optional[PestClassifier] = None
qwen_vl_max_client: Optional[QwenVLMaxClient] = None
store = ConversationStore(DB_PATH)
RAG_REBUILD_LOCK = Lock()
MODEL_GENERATION_LOCK = RLock()


def is_model_ready() -> bool:
    return model_instance is not None and model_instance.model is not None


def approx_token_count(text: str) -> int:
    if model_instance is not None and model_instance.tokenizer is not None:
        try:
            return len(model_instance.tokenizer.encode(text or ""))
        except Exception:
            pass
    return max(0, int(len(text or "") / 2.2))
auto_rag_status: Dict[str, Any] = {}


def get_auto_rag_ingestor() -> Optional[Any]:
    if not RAG_CONFIG.get("auto_pdf_enabled", True):
        return None
    if not AUTO_RAG_AVAILABLE or AutoRAGIngestor is None:
        logger.warning("Auto RAG module is not available")
        return None
    return AutoRAGIngestor(
        pdf_dir=Path(RAG_CONFIG["pdf_dir"]),
        chunks_path=Path(RAG_CONFIG["auto_chunks_path"]),
        manifest_path=Path(RAG_CONFIG["auto_manifest_path"]),
        chunk_size=int(RAG_CONFIG.get("pdf_chunk_size", 900)),
        chunk_overlap=int(RAG_CONFIG.get("pdf_chunk_overlap", 160)),
    )


def sync_auto_rag_pdfs(force: bool = False) -> Dict[str, Any]:
    global auto_rag_status
    ingestor = get_auto_rag_ingestor()
    if ingestor is None:
        auto_rag_status = {
            "enabled": False,
            "pdf_dir": RAG_CONFIG.get("pdf_dir"),
            "chunks_path": RAG_CONFIG.get("auto_chunks_path"),
            "pdf_count": 0,
            "chunks_count": 0,
            "files": [],
            "last_sync": None,
        }
        return auto_rag_status

    try:
        auto_rag_status = {"enabled": True, **ingestor.sync(force=force)}
    except Exception as e:
        logger.warning("Auto PDF RAG sync failed: %s", e)
        auto_rag_status = {"enabled": True, "error": str(e), **ingestor.status()}
    return auto_rag_status


def get_rag_runtime_status() -> Dict[str, Any]:
    ingestor = get_auto_rag_ingestor()
    status = ingestor.status() if ingestor is not None else dict(auto_rag_status or {})
    return {
        "enabled": bool(RAG_CONFIG.get("auto_pdf_enabled", True) and AUTO_RAG_AVAILABLE),
        "rag_enabled": rag_retriever is not None and rag_retriever.enabled,
        "rag_chunks": len(rag_retriever.chunks) if rag_retriever else 0,
        **status,
    }


def rebuild_rag_retriever(force_auto_sync: bool = False) -> Dict[str, Any]:
    global rag_retriever
    with RAG_REBUILD_LOCK:
        status = sync_auto_rag_pdfs(force=force_auto_sync)
        if not RAG_CONFIG.get("enable_rag"):
            rag_retriever = None
            return {"rag_enabled": False, **status}

        rag_retriever = RAGRetriever(RAG_CONFIG)
        return {
            "rag_enabled": rag_retriever is not None and rag_retriever.enabled,
            "rag_chunks": len(rag_retriever.chunks) if rag_retriever else 0,
            "embed_model_loaded": rag_retriever is not None and rag_retriever.model is not None,
            **status,
        }


@app.on_event("startup")
async def startup_event():
    global model_instance, rag_retriever, pest_classifier, qwen_vl_max_client

    try:
        model_instance = DurianGPTModel(MODEL_CONFIG)
        model_instance.load_model()

        if RAG_CONFIG.get("enable_rag"):
            rag_status = rebuild_rag_retriever(force_auto_sync=False)
            logger.info(
                "RAG ready: enabled=%s chunks=%s pdf_chunks=%s",
                rag_status.get("rag_enabled"),
                rag_status.get("rag_chunks"),
                rag_status.get("chunks_count"),
            )

        # 尝试初始化 Qwen-VL-Max 客户端
        if QWEN_VL_MAX_AVAILABLE:
            try:
                qwen_vl_max_client = get_qwen_vl_max_client()
                if qwen_vl_max_client and qwen_vl_max_client.api_key:
                    logger.info("✓ Qwen-VL-Max 客户端已初始化")
                else:
                    logger.warning("⚠️ Qwen-VL-Max API 密钥未设置，相关功能将不可用")
            except Exception as e:
                logger.warning("Qwen-VL-Max 客户端初始化失败: %s", e)
                qwen_vl_max_client = None
        else:
            logger.warning("⚠️ Qwen-VL-Max 模块不可用")

        # 如果 Qwen-VL-Max 可用，则不加载 ResNet50
        if qwen_vl_max_client and qwen_vl_max_client.api_key:
            logger.info("✓ 使用 Qwen-VL-Max 替代 ResNet50 分类模型")
            pest_classifier = None
        else:
            try:
                pest_classifier = PestClassifier()
            except Exception as e:
                logger.warning("病虫害分类模型加载失败，相关功能将不可用: %s", e)
                pest_classifier = None

        logger.info("✓ 应用启动完成")
    except Exception as e:
        logger.error("✗ 应用启动失败: %s", e)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    global model_instance

    if model_instance and model_instance.model is not None:
        del model_instance.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("应用已关闭")


@app.get("/", tags=["Info"])
async def root():
    return {
        "name": "榴莲GPT - 推理服务",
        "version": "2.0.0",
        "description": "基于微调 Qwen3-14B LoRA 的榴莲种植专家系统",
        "endpoints": {
            "health": "/health",
            "infer": "/infer (POST)",
            "infer_stream": "/infer/stream (POST)",
            "chat": "/chat (POST)",
            "chat_stream": "/chat/stream (POST)",
            "rag_status": "/rag/status (GET)",
            "rag_upload_pdf": "/rag/upload-pdf (POST)",
            "rag_rebuild": "/rag/rebuild (POST)",
            "conversations": "/conversations",
            "docs": "/docs",
            "redoc": "/redoc",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    if model_instance is None:
        raise HTTPException(status_code=503, detail="模型未初始化")

    rag_chunks = len(rag_retriever.chunks) if rag_retriever else 0
    rag_status = get_rag_runtime_status()
    
    # 确定当前使用的模型路径
    if model_instance and model_instance.active_lora_path:
        model_path = f"{MODEL_CONFIG.get('base_model', 'unknown')} + LoRA({model_instance.active_lora_path})"
    elif MODEL_CONFIG.get("use_merged") and MODEL_CONFIG.get("merged_model_path"):
        model_path = MODEL_CONFIG["merged_model_path"]
    else:
        model_path = MODEL_CONFIG.get("base_model", "unknown")

    return HealthResponse(
        status="healthy",
        device=DEVICE,
        model_loaded=model_instance.model is not None,
        model_path=model_path,
        memory_usage=model_instance.get_memory_usage(),
        rag_enabled=rag_retriever is not None and rag_retriever.enabled,
        rag_chunks=rag_chunks,
        rag_pdf_dir=rag_status.get("pdf_dir"),
        rag_pdf_files=int(rag_status.get("pdf_count") or 0),
        rag_auto_chunks=int(rag_status.get("chunks_count") or 0),
        rag_auto_last_sync=rag_status.get("last_sync"),
        embed_model_loaded=rag_retriever is not None and rag_retriever.model is not None,
        faiss_available=FAISS_AVAILABLE,
        sentence_transformers_available=ST_AVAILABLE,
        db_path=DB_PATH,
        timestamp=datetime.now().isoformat(),
        engine_mode=ENGINE_MODE,
        openai_proxy_base_url=DURIAN_OPENAI_BASE_URL if ENGINE_MODE in OPENAI_PROXY_MODES else None,
        openai_proxy_model=DURIAN_OPENAI_MODEL if ENGINE_MODE in OPENAI_PROXY_MODES else None,
    )


@app.get("/rag/status", tags=["RAG"], dependencies=[Depends(verify_api_key)])
async def rag_status():
    return get_rag_runtime_status()


@app.post("/rag/rebuild", tags=["RAG"], dependencies=[Depends(verify_api_key)])
async def rag_rebuild():
    status = await asyncio.to_thread(rebuild_rag_retriever, force_auto_sync=True)
    if status.get("error"):
        raise HTTPException(status_code=500, detail=status["error"])
    return status


@app.post("/rag/upload-pdf", tags=["RAG"], dependencies=[Depends(verify_api_key)])
async def rag_upload_pdf(file: UploadFile = File(...)):
    if not RAG_CONFIG.get("auto_pdf_enabled", True):
        raise HTTPException(status_code=400, detail="Auto PDF RAG is disabled")
    if unique_pdf_path is None:
        raise HTTPException(status_code=500, detail="Auto RAG module is not available")

    filename = file.filename or ""
    content_type = (file.content_type or "").lower()
    if not filename.lower().endswith(".pdf") and content_type not in {"application/pdf", "application/x-pdf"}:
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty")

    max_bytes = int(RAG_CONFIG.get("max_pdf_upload_mb", 80)) * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"PDF is larger than {RAG_CONFIG.get('max_pdf_upload_mb', 80)} MB")

    pdf_dir = Path(RAG_CONFIG["pdf_dir"])
    target_path = unique_pdf_path(pdf_dir, filename)
    await asyncio.to_thread(target_path.write_bytes, content)

    status = await asyncio.to_thread(rebuild_rag_retriever, force_auto_sync=True)
    if status.get("error"):
        raise HTTPException(status_code=500, detail=status["error"])

    return {
        "uploaded": True,
        "filename": target_path.name,
        "saved_path": str(target_path),
        **status,
    }




def resolve_request_max_tokens(request: InferenceRequest) -> int:
    """统一解析接口请求的最大输出 token。

    前端旧版本可能仍传 512/1024。这里不强制写死，最终仍会经过
    DurianGPTModel._limit_generation_tokens 的上下文预算限制。
    """
    value = request.max_tokens or request.max_new_tokens or INFERENCE_CONFIG["max_new_tokens"]
    try:
        return int(value)
    except Exception:
        return int(INFERENCE_CONFIG["max_new_tokens"])

def run_with_model_lock(fn, *args, **kwargs):
    """Serialize access to the local vLLM engine."""
    with MODEL_GENERATION_LOCK:
        return fn(*args, **kwargs)


def collect_model_stream(fn, *args, **kwargs):
    """Consume the model stream while holding the model lock.

    Do not materialize all chunks into a list here. Keeping this as a generator
    lets the SSE endpoint flush chunks as soon as generate_stream yields them.
    Note: vLLM LLM.generate is still synchronous in this file, so this is SSE
    chunk streaming rather than true token-by-token AsyncLLMEngine streaming.
    """
    with MODEL_GENERATION_LOCK:
        for chunk in fn(*args, **kwargs):
            yield chunk


async def run_model_blocking(fn, *args, **kwargs):
    """Run blocking model work outside FastAPI's event loop."""
    return await asyncio.to_thread(run_with_model_lock, fn, *args, **kwargs)


@app.post("/infer", response_model=InferenceResponse, tags=["Inference"], dependencies=[Depends(verify_api_key)])
async def infer(request: InferenceRequest):
    if not is_model_ready():
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        user_query = extract_last_user_message(request.messages)
        response_language = resolve_response_language(request.response_language, user_query)
        max_tokens = resolve_request_max_tokens(request)

        response = await run_model_blocking(
            model_instance.generate,
            messages,
            max_new_tokens=max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            response_language=response_language,
        )

        tokens_generated = approx_token_count(response)

        return InferenceResponse(
            response=response,
            tokens_generated=tokens_generated,
            model=model_instance.model_name,
            timestamp=datetime.now().isoformat(),
            evidence=[],
            evidence_quality="none",
        )
    except Exception as e:
        logger.error("推理失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/infer/stream", tags=["Inference"], dependencies=[Depends(verify_api_key)])
async def infer_stream(request: InferenceRequest):
    if not is_model_ready():
        raise HTTPException(status_code=503, detail="模型未加载")

    if not request.stream:
        raise HTTPException(status_code=400, detail="请设置 stream=true")

    try:
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        user_query = extract_last_user_message(request.messages)
        response_language = resolve_response_language(request.response_language, user_query)
        max_tokens = resolve_request_max_tokens(request)

        def generate():
            full_response = ""
            try:
                for chunk in collect_model_stream(
                    model_instance.generate_stream,
                    messages,
                    max_new_tokens=max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repetition_penalty=request.repetition_penalty,
                    response_language=response_language,
                ):
                    if chunk:
                        full_response += chunk
                        yield f"data: {json.dumps({'type': 'content', 'content': chunk}, ensure_ascii=False)}\n\n"

                hidden_context = build_text_context_comment(user_query, full_response, response_language)
                if hidden_context:
                    yield f"data: {json.dumps({'type': 'content', 'content': hidden_context}, ensure_ascii=False)}\n\n"

                yield f"data: {json.dumps({'type': 'metadata', 'evidence': [], 'evidence_quality': 'none'}, ensure_ascii=False)}\n\n"
            except Exception as e:
                logger.error("流式推理失败: %s", e)
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"
            finally:
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as e:
        logger.error("流式推理失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=InferenceResponse, tags=["Chat"], dependencies=[Depends(verify_api_key)])
async def chat(request: InferenceRequest):
    if not is_model_ready():
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        user_query = extract_last_user_message(request.messages)
        response_language = resolve_response_language(request.response_language, user_query)
        has_image_context = has_recent_image_context(request.messages)
        route_decision = await run_model_blocking(
            route_context_with_llm,
            user_query,
            request.messages,
            response_language,
        )
        user_intent = route_decision.get("intent", "new_topic")
        logger.info(
            "最终响应语言 response_language=%s, intent=%s, route_source=%s, use_history=%s, use_image_context=%s, reason=%s, user_query=%s",
            response_language,
            user_intent,
            route_decision.get("source"),
            route_decision.get("use_history"),
            route_decision.get("use_image_context"),
            route_decision.get("reason"),
            user_query[:80],
        )

        # 问候类输入直接短答：不进 RAG、不带历史、不触发农业诊断。
        if user_intent == "greeting":
            reply = greeting_reply(response_language)
            return InferenceResponse(
                response=reply,
                tokens_generated=len(model_instance.tokenizer.encode(reply)) if model_instance and model_instance.tokenizer else 0,
                model=model_instance.model_name,
                timestamp=datetime.now().isoformat(),
                evidence=[],
                evidence_quality="none",
            )

        context_resolution = await run_model_blocking(
            resolve_conversation_context,
            user_query,
            request.messages,
            response_language,
        )
        if context_resolution.get("selected"):
            route_decision["intent"] = "follow_up"
            route_decision["use_history"] = True
            route_decision["use_image_context"] = context_resolution.get("card", {}).get("source") == "image"
            route_decision["reason"] = context_resolution.get("selection", {}).get("reason", "context card selected")
            # 只给最终模型最相关上下文 + 当前问题，避免整段历史污染。
            context_messages = [
                Message(role="system", content=context_resolution.get("system_instruction", "")),
                Message(role="user", content=user_query),
            ]
        elif context_resolution.get("transition") == "new_topic":
            # v3: TopicTransitionRouter 已确认是新题时，不再让旧 route_context_with_llm 带历史。
            route_decision["intent"] = "new_topic"
            route_decision["use_history"] = False
            route_decision["use_image_context"] = False
            route_decision["reason"] = context_resolution.get("topic_route", {}).get("reason", "topic router chose new_topic")
            context_messages = [Message(role="user", content=user_query)]
        else:
            context_messages = build_context_aware_raw_messages(request.messages, route_decision)

        evidence = []
        evidence_quality = "none"
        evidence_text = ""

        # RAG is always attempted by default; retrieval quality still decides whether evidence is used.
        rag_allowed = bool(request.use_rag and rag_retriever and rag_retriever.enabled and user_query)
        if context_resolution.get("selected"):
            rag_query = context_resolution.get("standalone_query") or user_query
            if not RAG_CONFIG.get("always_on", True):
                rag_allowed = rag_allowed and bool(context_resolution.get("should_use_rag", False))
        else:
            rag_query = enhance_retrieval_query(user_query, response_language)
            if not RAG_CONFIG.get("always_on", True):
                rag_allowed = rag_allowed and is_agricultural_diagnosis_query(user_query)

        if rag_allowed:
            evidence, evidence_quality = await asyncio.to_thread(rag_retriever.retrieve, rag_query)
            evidence_text = build_evidence_text(
                evidence,
                max_chars=RAG_CONFIG.get("max_evidence_chars", 700),
            )

        messages = build_messages(
            context_messages,
            evidence_text=evidence_text,
            max_history_messages=RAG_CONFIG.get("max_history_messages", 4),
            response_language=response_language,
        )

        max_tokens = resolve_request_max_tokens(request)
        response = await run_model_blocking(
            model_instance.generate,
            messages,
            max_new_tokens=max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            response_language=response_language,
        )

        response = clean_model_output(response)
        response = ensure_adaptive_markdown_table(response, user_query, response_language)
        active_context_card = build_text_context_card(
            user_query,
            response,
            response_language,
            base_card=context_resolution.get("card") if context_resolution.get("selected") else None,
            source="text_followup" if context_resolution.get("selected") else "text",
        )
        tokens_generated = approx_token_count(response)

        return InferenceResponse(
            response=response,
            tokens_generated=tokens_generated,
            model=model_instance.model_name,
            timestamp=datetime.now().isoformat(),
            evidence=[
                Evidence(
                    text=ev.get("text", ""),
                    score=float(ev.get("score", 0.0)),
                    doc=ev.get("doc") or ev.get("source"),
                )
                for ev in evidence
            ],
            evidence_quality=evidence_quality,
            active_context_card=active_context_card,
        )
    except Exception as e:
        logger.error("聊天失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream", tags=["Chat"], dependencies=[Depends(verify_api_key)])
async def chat_stream(request: InferenceRequest):
    if not is_model_ready():
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        user_query = extract_last_user_message(request.messages)
        response_language = resolve_response_language(request.response_language, user_query)
        has_image_context = has_recent_image_context(request.messages)
        route_decision = await run_model_blocking(
            route_context_with_llm,
            user_query,
            request.messages,
            response_language,
        )
        user_intent = route_decision.get("intent", "new_topic")
        logger.info(
            "流式聊天请求 - 响应语言: %s, intent=%s, route_source=%s, use_history=%s, use_image_context=%s, reason=%s, user_query=%s",
            response_language,
            user_intent,
            route_decision.get("source"),
            route_decision.get("use_history"),
            route_decision.get("use_image_context"),
            route_decision.get("reason"),
            user_query[:80],
        )

        # 问候类输入直接走 SSE 快速回复，避免继承上一轮上下文。
        if user_intent == "greeting":
            reply = greeting_reply(response_language)

            def quick_generate():
                yield f"data: {json.dumps({'type': 'content', 'content': reply}, ensure_ascii=False)}\n\n"
                meta = {
                    "type": "metadata",
                    "tokens_generated": approx_token_count(reply),
                    "evidence_quality": "none",
                    "evidence": [],
                }
                yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                quick_generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        context_resolution = await run_model_blocking(
            resolve_conversation_context,
            user_query,
            request.messages,
            response_language,
        )
        if context_resolution.get("selected"):
            route_decision["intent"] = "follow_up"
            route_decision["use_history"] = True
            route_decision["use_image_context"] = context_resolution.get("card", {}).get("source") == "image"
            route_decision["reason"] = context_resolution.get("selection", {}).get("reason", "context card selected")
            context_messages = [
                Message(role="system", content=context_resolution.get("system_instruction", "")),
                Message(role="user", content=user_query),
            ]
        elif context_resolution.get("transition") == "new_topic":
            # v3: TopicTransitionRouter 已确认是新题时，不再让旧 route_context_with_llm 带历史。
            route_decision["intent"] = "new_topic"
            route_decision["use_history"] = False
            route_decision["use_image_context"] = False
            route_decision["reason"] = context_resolution.get("topic_route", {}).get("reason", "topic router chose new_topic")
            context_messages = [Message(role="user", content=user_query)]
        else:
            context_messages = build_context_aware_raw_messages(request.messages, route_decision)

        evidence = []
        evidence_quality = "none"
        evidence_text = ""

        rag_allowed = bool(request.use_rag and rag_retriever and rag_retriever.enabled and user_query)
        if context_resolution.get("selected"):
            rag_query = context_resolution.get("standalone_query") or user_query
            if not RAG_CONFIG.get("always_on", True):
                rag_allowed = rag_allowed and bool(context_resolution.get("should_use_rag", False))
        else:
            rag_query = enhance_retrieval_query(user_query, response_language)
            if not RAG_CONFIG.get("always_on", True):
                rag_allowed = rag_allowed and is_agricultural_diagnosis_query(user_query)

        if rag_allowed:
            evidence, evidence_quality = await asyncio.to_thread(rag_retriever.retrieve, rag_query)
            evidence_text = build_evidence_text(
                evidence,
                max_chars=RAG_CONFIG.get("max_evidence_chars", 700),
            )

        messages = build_messages(
            context_messages,
            evidence_text=evidence_text,
            max_history_messages=RAG_CONFIG.get("max_history_messages", 4),
            response_language=response_language,
        )

        max_tokens = resolve_request_max_tokens(request)

        def generate():
            full_response = ""

            try:
                early_meta = {
                    "type": "metadata",
                    "evidence_quality": evidence_quality,
                    "evidence": [
                        {
                            "text": ev.get("text", ""),
                            "score": float(ev.get("score", 0.0)),
                            "doc": ev.get("doc") or ev.get("source"),
                        }
                        for ev in evidence
                    ],
                }
                yield f"data: {json.dumps(early_meta, ensure_ascii=False)}\n\n"

                for chunk in collect_model_stream(
                    model_instance.generate_stream,
                    messages,
                    max_new_tokens=max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repetition_penalty=request.repetition_penalty,
                    response_language=response_language,
                ):
                    if chunk:
                        full_response += chunk
                        # openai_proxy 模式下这里会随着 vLLM OpenAI Server 的 token/chunk 立即 flush；
                        # local_vllm 模式仍然只是旧的兼容性分块。
                        yield f"data: {json.dumps({'type': 'content', 'content': chunk}, ensure_ascii=False)}\n\n"

                clean_full_response = clean_model_output(full_response)
                active_context_card = build_text_context_card(
                    user_query,
                    clean_full_response,
                    response_language,
                    base_card=context_resolution.get("card") if context_resolution.get("selected") else None,
                    source="text_followup" if context_resolution.get("selected") else "text",
                )

                tokens_generated = approx_token_count(clean_full_response)

                meta = {
                    "type": "metadata",
                    "tokens_generated": tokens_generated,
                    "active_context_card": active_context_card,
                    "evidence_quality": evidence_quality,
                    "evidence": [
                        {
                            "text": ev.get("text", ""),
                            "score": float(ev.get("score", 0.0)),
                            "doc": ev.get("doc") or ev.get("source"),
                        }
                        for ev in evidence
                    ],
                }
                yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

            except Exception as e:
                logger.exception("流式生成失败")
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"

            finally:
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as e:
        logger.error("流式聊天失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# 对话接口
# =========================
@app.post("/conversations", response_model=ConversationCreateResponse, tags=["Conversations"], dependencies=[Depends(verify_api_key)])
async def create_conversation(current_user: str = Depends(get_current_user)):
    return store.create_conversation(owner=current_user)


@app.get("/conversations", tags=["Conversations"], dependencies=[Depends(verify_api_key)])
async def list_conversations(current_user: str = Depends(get_current_user)):
    return {"conversations": store.list_conversations(owner=current_user)}


@app.get("/conversations/{conv_id}", tags=["Conversations"], dependencies=[Depends(verify_api_key)])
async def get_conversation(conv_id: str, current_user: str = Depends(get_current_user)):
    return store.get_conversation(conv_id, owner=current_user)


@app.post("/conversations/{conv_id}/messages", tags=["Conversations"], dependencies=[Depends(verify_api_key)])
async def add_message_to_conversation(conv_id: str, message: Message, current_user: str = Depends(get_current_user)):
    store.add_message(conv_id, current_user, message.role, message.content, message.image_url, message.active_context_card)
    return {"status": "success"}


@app.delete("/conversations/{conv_id}", tags=["Conversations"], dependencies=[Depends(verify_api_key)])
async def delete_conversation(conv_id: str, current_user: str = Depends(get_current_user)):
    store.delete_conversation(conv_id, owner=current_user)
    return {"status": "success"}


# =========================
# 病虫害分类接口
# =========================
@app.post("/classify_pest", response_model=PestClassificationResult, tags=["Pest Classification"], dependencies=[Depends(verify_api_key)])
async def classify_pest(
    file: UploadFile = File(...),
    response_language: str = Form("zh"),
):
    """
    仅分类图片中的病虫害，不调用大模型
    """
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')
        if qwen_vl_max_client and qwen_vl_max_client.api_key:
            label_result = qwen_vl_max_client.classify_durian_image_label(
                image=image,
                user_query="classify this durian image",
                response_language=response_language or "zh",
            )
            return PestClassificationResult(
                class_id=0,
                class_name=label_result.get("label_zh") or label_result.get("label") or "Unknown",
                class_name_en=label_result.get("label") or "Unknown",
                class_name_ms=label_result.get("label") or "Unknown",
                confidence=float(label_result.get("confidence") or 0.0),
                all_probs={label_result.get("label_zh") or label_result.get("label") or "Unknown": float(label_result.get("confidence") or 0.0)},
            )

        if pest_classifier is None:
            raise HTTPException(status_code=503, detail="病虫害分类模型未加载")

        classification = pest_classifier.classify(image)
        return classification
    except HTTPException:
        raise
    except Exception as e:
        logger.error("病虫害分类失败: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/analyze_pest", response_model=InferenceResponse, tags=["Pest Classification"], dependencies=[Depends(verify_api_key)])
async def analyze_pest(
    file: UploadFile = File(...),
    query: Optional[str] = Form(None),
    response_language: str = Form("zh"),
):
    """
    分析图片中的病虫害，返回分类结果和大模型分析 + RAG 参考文献
    支持多语言: zh/en/ms/th
    
    优先使用 Qwen-VL-Max，如果不可用则使用 ResNet50 + 大模型
    """
    response_language = response_language or "zh"
    
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')
        
        # 优先使用 Qwen-VL-Max
        if qwen_vl_max_client and qwen_vl_max_client.api_key:
            logger.info("使用 Qwen-VL-Max 进行图文混合分析")
            try:
                # Qwen-VL-Max only produces visual observation; local LoRA gives final advice.
                visual_result = qwen_vl_max_client.classify_durian_image_label(
                    image=image,
                    user_query=query or "",
                    response_language=response_language,
                )
                result = run_analyze_pest_core_from_vl_label(
                    file_bytes=contents,
                    filename=file.filename or "upload.jpg",
                    content_type=file.content_type or "image/jpeg",
                    query=query or "",
                    response_language=response_language,
                    vl_label_result=visual_result,
                    vision_model=visual_result.get("model", "qwen-vl-max"),
                    vision_tokens=int(visual_result.get("tokens_generated", 0) or 0),
                )
                return InferenceResponse(**result)
            except Exception as e:
                logger.error("Qwen-VL-Max 分析失败，回退到 ResNet50: %s", e)
                # 回退到 ResNet50
                if pest_classifier is None:
                    raise HTTPException(status_code=503, detail="Qwen-VL-Max 和 ResNet50 都不可用")
        
        # 使用 ResNet50 + 大模型
        if pest_classifier is None:
            logger.error("病虫害分类模型未加载")
            raise HTTPException(status_code=503, detail="病虫害分类模型未加载")
        
        if model_instance is None or model_instance.model is None:
            logger.error("大模型未加载")
            raise HTTPException(status_code=503, detail="大模型未加载")
        
        result = run_analyze_pest_core_from_bytes(
            file_bytes=contents,
            filename=file.filename or "upload.jpg",
            content_type=file.content_type or "image/jpeg",
            query=query or "",
            response_language=response_language,
        )
        return InferenceResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("病虫害分析失败: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/analyze_pest_job", tags=["Pest Classification"], dependencies=[Depends(verify_api_key)])
async def analyze_pest_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    query: str = Form(""),
    response_language: str = Form("zh"),
):
    """
    创建病虫害分析任务（两段式模式）
    立即返回 job_id，后台异步执行分析
    支持用户输入的查询文本
    
    优先使用 Qwen-VL-Max，如果不可用则使用 ResNet50 + 大模型
    """
    job_id = str(uuid.uuid4())
    response_language = response_language or "zh"
    
    try:
        file_bytes = await file.read()
        filename = file.filename or "upload.jpg"
        content_type = file.content_type or "image/jpeg"
        
        logger.info("创建分析任务: job_id=%s, query=%s, language=%s", job_id, query[:50] if query else "(empty)", response_language)
        
        ANALYZE_JOBS[job_id] = {
            "status": "pending",
            "result": None,
            "error": None,
            "created_at": datetime.now().isoformat(),
        }
        
        # 优先使用 Qwen-VL-Max
        if qwen_vl_max_client and qwen_vl_max_client.api_key:
            logger.info("创建 Qwen-VL-Max 分析任务: %s", job_id)
            background_tasks.add_task(
                run_analyze_pest_job_qwen_vl_max,
                job_id,
                file_bytes,
                filename,
                content_type,
                query,
                response_language,
            )
        else:
            # 回退到 ResNet50
            if pest_classifier is None:
                raise HTTPException(status_code=503, detail="Qwen-VL-Max 和 ResNet50 都不可用")
            if model_instance is None or model_instance.model is None:
                raise HTTPException(status_code=503, detail="大模型未加载")
            
            logger.info("创建 ResNet50 分析任务: %s", job_id)
            background_tasks.add_task(
                run_analyze_pest_job,
                job_id,
                file_bytes,
                filename,
                content_type,
                query,
                response_language,
            )

        return {
            "job_id": job_id,
            "status": ANALYZE_JOBS[job_id]["status"],
            "created_at": ANALYZE_JOBS[job_id]["created_at"],
        }
        
    except Exception as e:
        logger.error("创建任务失败: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/analyze_pest_job/{job_id}", tags=["Pest Classification"], dependencies=[Depends(verify_api_key)])
async def get_analyze_pest_job(job_id: str):
    """
    查询病虫害分析任务状态和结果
    """
    job = ANALYZE_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return job




# =========================
# 质量审查与表格检测补丁函数
# =========================
def has_slash_style_table(text: str) -> bool:
    """检测模型输出的斜杠伪表格。

    典型形式：
    A / B / C
    --- / --- / ---
    x / y / z
    """
    if not text:
        return False

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    slash_rows = 0
    sep_rows = 0

    for ln in lines:
        if "|" in ln:
            continue
        if ln.count("/") < 2:
            continue
        parts = [p.strip() for p in ln.split("/")]
        non_empty = [p for p in parts if p]
        if len(non_empty) < 3:
            continue
        if all(re.fullmatch(r"[-—–]{2,}", p) for p in non_empty):
            sep_rows += 1
        else:
            slash_rows += 1

    return slash_rows >= 2 or (slash_rows >= 1 and sep_rows >= 1)


def has_direct_answer_signal(response: str) -> bool:
    """判断回答开头是否有直接结论信号。

    用于 A/B 问题：例如“是缺肥还是根系问题”。
    """
    if not response:
        return False
    head = response.strip()[:260].lower()
    signals = [
        # 中文
        "更像", "更偏向", "优先", "先", "不是单纯", "不一定", "大概率", "主要是", "核心是",
        "倾向", "建议先", "我会先", "两者都可能", "不能直接", "不是", "一般不是",
        # English
        "more likely", "less likely", "first", "priority", "not simply", "not necessarily",
        "i would", "the key", "mainly", "both are possible", "start by",
        # Malay
        "lebih cenderung", "keutamaan", "bukan semata", "mula dengan",
        # Thai
        "มีแนวโน้ม", "ควรเริ่ม", "ไม่ใช่แค่", "อย่างแรก",
    ]
    return any(s in head for s in signals)


def build_quality_checker_instruction(user_query: str, response: str, lang: str = "zh") -> str:
    """构造质量审查器提示词。只审查，不改写。"""
    language_hint = {
        "zh": "中文",
        "en": "English",
        "ms": "Bahasa Melayu",
        "th": "Thai",
    }.get((lang or "zh").lower(), "中文")

    return f"""
你是答案质量审查器，只判断下面回答是否需要改写，不要自己回答用户问题。

审查维度：
1. relevance_score：回答是否对应用户问题，是否跑题。用户问品种/口感/购买时，不应强行转到病虫害治理；用户问病虫害/防治时，应回应处理重点。
2. detail_score：复杂问题是否有足够细节；简单问候/简单事实不要求长。
3. readability_score：结构是否清楚、顺序是否自然、Markdown 是否可读；不要因为没有表格就扣分。

重要规则：
- 只看回答是否适合用户问题，不要强迫所有回答都讲治理或农药。
- 如果用户问 A 还是 B，回答必须先给倾向或排查优先级。
- 如果用户问品种、口感、市场、购买、食用安全，回答应围绕这些主题，不要跑到果园管理。
- 如果回答本身已对应问题、细节够、可读，就 needs_rewrite=false。

用户问题：
{user_query}

模型回答：
{response}

请只输出 JSON，语言无关，格式如下：
{{
  "relevance_score": 0.0到1.0,
  "detail_score": 0.0到1.0,
  "readability_score": 0.0到1.0,
  "needs_rewrite": true或false,
  "rewrite_focus": ["answer_question", "add_details", "improve_structure", "fix_table", "avoid_wrong_scenario"],
  "reason": "一句话原因，用{language_hint}简短说明"
}}
""".strip()


def parse_quality_review_json(raw: str) -> Dict[str, Any]:
    """解析质量审查器 JSON，失败时返回保守默认值。"""
    default = {
        "relevance_score": 0.8,
        "detail_score": 0.8,
        "readability_score": 0.8,
        "needs_rewrite": False,
        "rewrite_focus": [],
        "reason": "parse fallback",
    }
    if not raw:
        return default

    text = raw.strip()
    # 去掉 markdown fence
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    # 提取第一个 JSON 对象
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        text = m.group(0)

    try:
        obj = json.loads(text)
        result = default.copy()
        for key in ["relevance_score", "detail_score", "readability_score"]:
            try:
                result[key] = max(0.0, min(1.0, float(obj.get(key, result[key]))))
            except Exception:
                pass
        result["needs_rewrite"] = bool(obj.get("needs_rewrite", result["needs_rewrite"]))
        focus = obj.get("rewrite_focus", [])
        if isinstance(focus, str):
            focus = [focus]
        if not isinstance(focus, list):
            focus = []
        result["rewrite_focus"] = [str(x) for x in focus if str(x).strip()]
        result["reason"] = str(obj.get("reason", result["reason"]))[:160]
        return result
    except Exception:
        return default


# =========================
# 主程序
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="榴莲GPT 推理服务")
    parser.add_argument("--host", default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=8001, help="服务器端口")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数")
    parser.add_argument("--reload", action="store_true", help="启用自动重载")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("启动榴莲GPT推理服务")
    
    if MODEL_CONFIG.get("use_merged") and MODEL_CONFIG.get("merged_model_path"):
        display_model_path = MODEL_CONFIG["merged_model_path"]
    elif MODEL_CONFIG.get("enable_lora") and MODEL_CONFIG.get("lora_model_path"):
        display_model_path = f"{MODEL_CONFIG['base_model']} + LoRA({MODEL_CONFIG['lora_model_path']})"
    else:
        display_model_path = MODEL_CONFIG["base_model"]
    
    logger.info("模型路径: %s", display_model_path)
    logger.info("数据库路径: %s", DB_PATH)
    logger.info("API 文档: http://%s:%s/docs", args.host, args.port)
    logger.info("=" * 60)

    uvicorn.run(
        "durian__inference_api:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info",
    )

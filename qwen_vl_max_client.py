#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阿里云 Qwen-VL-Max 多模态模型客户端
用于替换 ResNet50 病虫害分类模型，支持图文混合分析
"""

import os
import json
import base64
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from io import BytesIO
from PIL import Image, ImageOps
import requests

logger = logging.getLogger("qwen-vl-max-client")


@dataclass
class QwenVLMaxConfig:
    API_KEY: str = (
        os.getenv("QWEN_VL_MAX_API_KEY")
        or os.getenv("YUNWU_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("QWEN_API_KEY")
        or os.getenv("API_KEY", "")
    )
    BASE_URL: str = os.getenv("QWEN_VL_MAX_BASE_URL", os.getenv("BASE_URL", "https://yunwu.ai"))
    MODEL: str = os.getenv("QWEN_VL_MAX_MODEL", os.getenv("MODEL", "qwen-vl-max"))
    TIMEOUT: int = int(os.getenv("QWEN_VL_MAX_TIMEOUT", os.getenv("DURIAN_VL_TIMEOUT", "45")))


class QwenVLMaxClient:
    """阿里云 Qwen-VL-Max API 客户端"""
    
    def __init__(self, api_key: str = "", api_endpoint: str = "", base_url: str = "", model: str = ""):
        """
        初始化客户端
        
        Args:
            api_key: 阿里云 API 密钥（从环境变量 QWEN_VL_MAX_API_KEY 读取）
            api_endpoint: API 端点（从环境变量 QWEN_VL_MAX_ENDPOINT 读取）
        """
        config = QwenVLMaxConfig()
        self.api_key = api_key or config.API_KEY
        self.base_url = (base_url or config.BASE_URL).rstrip("/")
        self.api_endpoint = api_endpoint or os.getenv("QWEN_VL_MAX_ENDPOINT", "")
        self.model_name = model or config.MODEL
        self.timeout = config.TIMEOUT
        
        if not self.api_key:
            logger.warning("⚠️ QWEN_VL_MAX_API_KEY 未设置，Qwen-VL-Max 功能将不可用")
        
        logger.info("✓ Qwen-VL-Max 客户端初始化完成")
        logger.info("  API 端点: %s", self.api_endpoint)
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """将 PIL Image 转换为 base64 字符串。

        视觉模型不需要原始超大图；这里做温和压缩以降低超时概率，
        不影响品种/病害的宏观识别。
        """
        image = ImageOps.exif_transpose(image).convert("RGB")
        max_side = int(os.getenv("QWEN_VL_MAX_IMAGE_MAX_SIDE", "1600"))
        w, h = image.size
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buffer = BytesIO()
        quality = int(os.getenv("QWEN_VL_MAX_IMAGE_QUALITY", "86"))
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _chat_completions_url(self) -> str:
        if self.api_endpoint:
            return self.api_endpoint
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    def _post_chat_completion(
        self,
        messages: list,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }
        response = requests.post(
            self._chat_completions_url(),
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code != 200:
            logger.error("Qwen-VL-Max API error: %s", response.text)
            raise RuntimeError(f"API returned error: {response.status_code}")
        return response.json()

    def _usage_tokens(self, result: Dict[str, Any]) -> int:
        usage = result.get("usage") or {}
        return int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)

    def _extract_response_text(self, result: Dict[str, Any]) -> str:
        """Return text from DashScope responses in either message or text shape."""
        output = result.get("output") or {}
        choices = result.get("choices") or output.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict) and item.get("text"):
                        parts.append(str(item["text"]))
                text = "\n".join(part for part in parts if part).strip()
                if text:
                    return text

        text = output.get("text", "")
        if isinstance(text, str) and text:
            return text

        logger.error("Unexpected Qwen-VL-Max response shape: %s", result)
        raise RuntimeError("Unable to extract text from Qwen-VL-Max response")
    
    def classify_pest(
        self,
        image: Image.Image,
        response_language: str = "zh",
    ) -> Dict[str, Any]:
        """
        使用 Qwen-VL-Max 分类病虫害
        
        Args:
            image: PIL Image 对象
            response_language: 响应语言 (zh/en/ms/th)
        
        Returns:
            分类结果字典，包含 class_id, class_name, confidence 等
        """
        if not self.api_key:
            raise RuntimeError("QWEN_VL_MAX_API_KEY 未设置")
        
        try:
            image_base64 = self._image_to_base64(image)
            
            # 构造多语言提示词
            prompts = {
                "zh": "请分析这张图片中的榴莲病虫害情况。请按以下格式回答：\n1. 病虫害类别（中文名称）\n2. 置信度（0-1之间的数字）\n3. 主要症状\n4. 建议处理方向\n\n请只输出 JSON 格式的结果，包含 class_name, confidence, symptoms, recommendation 字段。",
                "en": "Please analyze the durian pest or disease in this image. Answer in the following format:\n1. Pest/Disease category (English name)\n2. Confidence (number between 0-1)\n3. Main symptoms\n4. Recommended treatment direction\n\nOutput only JSON format with fields: class_name, confidence, symptoms, recommendation.",
                "ms": "Sila analisis perosak atau penyakit durian dalam imej ini. Jawab dalam format berikut:\n1. Kategori perosak/penyakit (nama Melayu)\n2. Keyakinan (nombor antara 0-1)\n3. Gejala utama\n4. Arah rawatan yang disyorkan\n\nKeluarkan hanya format JSON dengan medan: class_name, confidence, symptoms, recommendation.",
                "th": "โปรดวิเคราะห์ศัตรูพืชหรือโรคทุเรียนในรูปภาพนี้ ตอบในรูปแบบต่อไปนี้:\n1. หมวดหมู่ศัตรูพืช/โรค (ชื่อภาษาไทย)\n2. ความมั่นใจ (ตัวเลขระหว่าง 0-1)\n3. อาการหลัก\n4. ทิศทางการจัดการที่แนะนำ\n\nแสดงเฉพาะรูปแบบ JSON ที่มีฟิลด์: class_name, confidence, symptoms, recommendation",
            }
            
            prompt = prompts.get(response_language, prompts["zh"])
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                        },
                    ],
                }
            ]
            result = self._post_chat_completion(
                messages=messages,
                temperature=0.3,
                top_p=0.8,
                max_tokens=512,
            )
            content = self._extract_response_text(result)
            logger.info("Qwen-VL-Max raw response: %s", content)
            try:
                classification = json.loads(content)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r"\{[\s\S]*\}", content)
                if json_match:
                    classification = json.loads(json_match.group(0))
                else:
                    raise RuntimeError("Unable to parse Qwen-VL-Max classification response")
            return {
                "class_id": 0,
                "class_name": classification.get("class_name", "Unknown"),
                "class_name_en": classification.get("class_name", "Unknown"),
                "class_name_ms": classification.get("class_name", "Unknown"),
                "class_name_th": classification.get("class_name", "Unknown"),
                "confidence": float(classification.get("confidence", 0.5)),
                "symptoms": classification.get("symptoms", ""),
                "recommendation": classification.get("recommendation", ""),
                "all_probs": {},
            }
            
            # 构造请求
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            payload = {
                "model": self.model_name,
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "image": f"data:image/jpeg;base64,{image_base64}",
                                },
                                {
                                    "text": prompt,
                                }
                            ]
                        }
                    ]
                },
                "parameters": {
                    "temperature": 0.3,
                    "top_p": 0.8,
                    "max_tokens": 512,
                }
            }
            
            logger.info("发送 Qwen-VL-Max 分类请求...")
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=payload,
                timeout=60,
            )
            
            if response.status_code != 200:
                logger.error("Qwen-VL-Max API 错误: %s", response.text)
                raise RuntimeError(f"API 返回错误: {response.status_code}")
            
            result = response.json()
            
            # 解析响应
            if "output" not in result:
                logger.error("API 响应格式错误: %s", result)
                raise RuntimeError("API 响应格式错误")
            
            content = self._extract_response_text(result)
            logger.info("Qwen-VL-Max 原始响应: %s", content)
            
            # 提取 JSON
            try:
                # 尝试直接解析
                classification = json.loads(content)
            except json.JSONDecodeError:
                # 尝试从文本中提取 JSON
                import re
                json_match = re.search(r"\{[\s\S]*\}", content)
                if json_match:
                    classification = json.loads(json_match.group(0))
                else:
                    logger.error("无法从响应中提取 JSON: %s", content)
                    raise RuntimeError("无法解析 API 响应")
            
            # 标准化输出格式
            return {
                "class_id": 0,  # Qwen-VL-Max 不返回 ID，使用 0 作为占位符
                "class_name": classification.get("class_name", "Unknown"),
                "class_name_en": classification.get("class_name", "Unknown"),
                "class_name_ms": classification.get("class_name", "Unknown"),
                "class_name_th": classification.get("class_name", "Unknown"),
                "confidence": float(classification.get("confidence", 0.5)),
                "symptoms": classification.get("symptoms", ""),
                "recommendation": classification.get("recommendation", ""),
                "all_probs": {},
            }
        
        except Exception as e:
            logger.error("Qwen-VL-Max 分类失败: %s", e)
            raise

    def classify_durian_image_label(
        self,
        image: Image.Image,
        user_query: str = "",
        response_language: str = "zh",
    ) -> Dict[str, Any]:
        """Use Qwen-VL-Max as a strict image label classifier.

        It returns one clear label name for the image, not final advice.
        The local LoRA model will generate the final answer.
        """
        if not self.api_key:
            raise RuntimeError("Qwen-VL-Max API key is not configured")

        image_base64 = self._image_to_base64(image)
        language_name = {
            "zh": "Chinese",
            "en": "English",
            "ms": "Bahasa Melayu",
            "th": "Thai",
        }.get(response_language or "zh", "Chinese")

        prompt = f"""
You are the image-labeling stage for a durian-domain assistant.
Your only job is to classify the image into ONE clear durian-related label.
Do not give final advice. Do not write a report. Output JSON only.
The image is attached in this same message as a data URL. Do not claim that no image was provided; if it is unreadable, say the image is unreadable and list what cannot be confirmed.

Allowed category_type values:
- variety: the image most likely shows a durian variety/type.
- quality: the image mainly shows eating quality, ripeness, freshness, spoilage, or opened fruit condition.
- disease: the image shows a plant disease symptom.
- pest: the image shows an insect/pest problem.
- cultivation_issue: nutrient, water, pruning, trunk injury, flowering/fruiting or orchard-management issue.
- unknown: the image is not enough to identify a specific variety/problem.

Label rules:
- label must be a clear name, not a generic sentence.
- For variety, prefer names such as Musang King/猫山王, Black Thorn/黑刺, D24, Monthong/金枕, Chanee/红虾, Kradum, or "无法确定具体品种" when the image lacks shell/stem/known variety clues.
- For quality, use labels such as "正常成熟果肉", "过熟果肉", "疑似变质果肉", "未成熟果肉".
- For disease/pest/cultivation_issue, use names such as 叶部炭疽病, 果腐病, 茎部裂纹/流胶, 根腐/疫病, 白根病, 黄叶, 粉蚧, 蓟马, 红蜘蛛, 茎钻虫, or "无法确定具体病害".
- Do not output broad labels like "切开的榴莲果实" unless category_type is unknown; if it is opened fruit and looks normal, label should be "正常成熟果肉".
- If the user asks for variety but the image cannot determine variety, label must be "无法确定具体品种" and alternatives may list plausible varieties.

Return JSON in {language_name} where possible:
{{
  "category_type": "variety|quality|disease|pest|cultivation_issue|unknown",
  "label": "clear English or local label",
  "label_zh": "清晰中文名称",
  "confidence": 0.0,
  "visual_evidence": "short visible evidence for the label",
  "alternatives": ["optional alternative labels"],
  "cannot_confirm": "what cannot be confirmed from image alone"
}}

User question:
{user_query or "(none)"}
""".strip()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            }
        ]

        logger.info("Sending Qwen-VL-Max strict label classification request...")
        result = self._post_chat_completion(
            messages=messages,
            temperature=0.0,
            top_p=0.8,
            max_tokens=int(os.getenv("QWEN_VL_LABEL_MAX_TOKENS", "450")),
        )
        content = self._extract_response_text(result).strip()
        logger.info("Qwen-VL-Max label raw response: %s", content)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            import re
            match = re.search(r"\{[\s\S]*\}", content)
            if not match:
                raise RuntimeError("Unable to parse Qwen-VL-Max label JSON")
            parsed = json.loads(match.group(0))

        category_type = str(parsed.get("category_type") or "unknown").strip().lower()
        if category_type not in {"variety", "quality", "disease", "pest", "cultivation_issue", "unknown"}:
            category_type = "unknown"
        label = str(parsed.get("label") or parsed.get("label_zh") or "无法确定").strip()
        label_zh = str(parsed.get("label_zh") or label).strip()
        try:
            confidence = float(parsed.get("confidence") or 0.0)
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        alternatives = parsed.get("alternatives") or []
        if isinstance(alternatives, str):
            alternatives = [alternatives]
        elif not isinstance(alternatives, list):
            alternatives = []
        return {
            "category_type": category_type,
            "label": label,
            "label_zh": label_zh,
            "confidence": confidence,
            "visual_evidence": str(parsed.get("visual_evidence") or "").strip(),
            "alternatives": [str(x).strip() for x in alternatives if str(x).strip()],
            "cannot_confirm": str(parsed.get("cannot_confirm") or "").strip(),
            "tokens_generated": self._usage_tokens(result),
            "model": self.model_name,
        }

    def observe_pest_image(
        self,
        image: Image.Image,
        user_query: str = "",
        response_language: str = "zh",
    ) -> Dict[str, Any]:
        """Use Qwen-VL-Max only for visual observation, not final advice."""
        if not self.api_key:
            raise RuntimeError("Qwen-VL-Max API key is not configured")

        image_base64 = self._image_to_base64(image)
        language_name = {
            "zh": "Chinese",
            "en": "English",
            "ms": "Bahasa Melayu",
            "th": "Thai",
        }.get(response_language or "zh", "Chinese")

        prompt = f"""
You are the image-identification stage for a general durian-domain assistant.
Your main job is to identify what this image shows. Do not act as the final advisor.

Identify the most likely subject or condition in the image, for example:
- durian fruit variety or type
- ripeness or fruit quality state
- leaf, stem, root, flower, or fruit symptom
- likely pest, disease, nutrient issue, or physical damage
- non-problem normal durian image

Do not force every image into pest/disease. If the image is a normal durian fruit, variety, eating-quality, market, storage, or buying question, identify that topic directly. Mention pests or disease only when visible signs support it.

Return the result in {language_name}. Keep it concise and structured with these fields:
- Image identification: the most likely thing shown
- Confidence: high, medium, or low
- Visual basis: the visible clues used for identification
- Possible alternatives: only if genuinely plausible
- Cannot confirm from image alone: what needs field/lab/taste/vendor confirmation

Do not provide final treatment advice, pesticide plans, orchard-management recommendations, or long explanations.

User question:
{user_query or "(none)"}
""".strip()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            }
        ]

        logger.info("Sending Qwen-VL-Max visual observation request...")
        result = self._post_chat_completion(
            messages=messages,
            temperature=0.2,
            top_p=0.8,
            max_tokens=500,
        )
        observation = self._extract_response_text(result)
        return {
            "observation": observation,
            "tokens_generated": self._usage_tokens(result),
            "model": self.model_name,
        }
    
    def analyze_pest_with_text(
        self,
        image: Image.Image,
        user_query: str = "",
        response_language: str = "zh",
    ) -> Dict[str, Any]:
        """
        使用 Qwen-VL-Max 进行图文混合分析
        
        Args:
            image: PIL Image 对象
            user_query: 用户输入的文字问题
            response_language: 响应语言
        
        Returns:
            分析结果字典
        """
        if not self.api_key:
            raise RuntimeError("QWEN_VL_MAX_API_KEY 未设置")
        
        try:
            image_base64 = self._image_to_base64(image)
            
            # 构造系统提示词
            system_prompts = {
                "zh": "你是经验丰富的榴莲病虫害防治专家。请根据用户上传的图片和问题，提供专业的诊断和建议。",
                "en": "You are an experienced durian pest and disease management expert. Please provide professional diagnosis and recommendations based on the user's uploaded image and question.",
                "ms": "Anda ialah pakar penyakit dan perosak durian yang berpengalaman. Sila berikan diagnosis dan cadangan profesional berdasarkan imej dan soalan pengguna.",
                "th": "คุณเป็นผู้เชี่ยวชาญด้านโรคและแมลงศัตรูทุเรียนที่มีประสบการณ์ โปรดให้การวินิจฉัยและคำแนะนำที่เป็นมืออาชีพตามรูปภาพและคำถามของผู้ใช้",
            }
            
            system_prompt = system_prompts.get(response_language, system_prompts["zh"])
            
            # 构造用户消息
            user_message = user_query or {
                "zh": "请分析这张图片中的榴莲病虫害情况，并提供防治建议。",
                "en": "Please analyze the durian pest or disease in this image and provide prevention recommendations.",
                "ms": "Sila analisis perosak atau penyakit durian dalam imej ini dan berikan cadangan pencegahan.",
                "th": "โปรดวิเคราะห์ศัตรูพืชหรือโรคทุเรียนในรูปภาพนี้ และให้คำแนะนำในการป้องกัน",
            }.get(response_language, "Please analyze this image.")
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_message},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                        },
                    ],
                },
            ]
            result = self._post_chat_completion(
                messages=messages,
                temperature=0.35,
                top_p=0.85,
                max_tokens=1024,
            )
            analysis_text = self._extract_response_text(result)
            return {
                "response": analysis_text,
                "tokens_generated": self._usage_tokens(result),
                "model": self.model_name,
            }
            
            # 构造请求
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            payload = {
                "model": self.model_name,
                "input": {
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "image": f"data:image/jpeg;base64,{image_base64}",
                                },
                                {
                                    "text": user_message,
                                }
                            ]
                        }
                    ]
                },
                "parameters": {
                    "temperature": 0.35,
                    "top_p": 0.85,
                    "max_tokens": 1024,
                }
            }
            
            logger.info("发送 Qwen-VL-Max 图文分析请求...")
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=payload,
                timeout=60,
            )
            
            if response.status_code != 200:
                logger.error("Qwen-VL-Max API 错误: %s", response.text)
                raise RuntimeError(f"API 返回错误: {response.status_code}")
            
            result = response.json()
            
            # 解析响应
            if "output" not in result:
                logger.error("API 响应格式错误: %s", result)
                raise RuntimeError("API 响应格式错误")
            
            analysis_text = self._extract_response_text(result)
            
            return {
                "response": analysis_text,
                "tokens_generated": result.get("usage", {}).get("output_tokens", 0),
                "model": self.model_name,
            }
        
        except Exception as e:
            logger.error("Qwen-VL-Max 图文分析失败: %s", e)
            raise


# 全局客户端实例
qwen_vl_max_client: Optional[QwenVLMaxClient] = None


def init_qwen_vl_max_client():
    """初始化全局 Qwen-VL-Max 客户端"""
    global qwen_vl_max_client
    try:
        qwen_vl_max_client = QwenVLMaxClient()
        logger.info("✓ Qwen-VL-Max 客户端初始化成功")
        return qwen_vl_max_client
    except Exception as e:
        logger.error("✗ Qwen-VL-Max 客户端初始化失败: %s", e)
        return None


def get_qwen_vl_max_client() -> Optional[QwenVLMaxClient]:
    """获取全局 Qwen-VL-Max 客户端"""
    global qwen_vl_max_client
    if qwen_vl_max_client is None:
        init_qwen_vl_max_client()
    return qwen_vl_max_client

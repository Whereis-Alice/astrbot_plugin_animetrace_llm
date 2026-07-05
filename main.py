from __future__ import annotations

import asyncio
import os
import re
import shutil
import tempfile
import urllib.parse
from io import BytesIO
from typing import Any

import aiohttp
from pydantic import Field
from pydantic.dataclasses import dataclass
from PIL import Image as PILImage

import astrbot.api.message_components as Comp
from astrbot.api import FunctionTool, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image as MsgImage
from astrbot.api.message_components import Reply
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register
from astrbot.core.agent.message import TextPart
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext

PLUGIN_ID = "astrbot_plugin_animetrace_llm"
PLUGIN_AUTHOR = "Huli3"
PLUGIN_DESC = "AnimeTrace 图片识别 fork：支持命令识图，并注册为 LLM 可主动调用的工具"
PLUGIN_VERSION = "4.1.1"
PLUGIN_REPO = "https://github.com/Whereis-Alice/astrbot_plugin_animetrace_llm"

CONFIG_SECTION = "animetrace_llm_settings"
TOOL_NAME = "animetrace_identify_image"
IMAGE_COMMAND = "识别"
AVATAR_COMMAND = "头像识别"
MODEL_COMMAND = "amt model"
FALLBACK_MODEL_ID = "animetrace-yuri-4.2"

DEFAULT_TOOL_REQUEST_KEYWORDS = [
    "识图",
    "识别图片",
    "识别这张图",
    "这是谁",
    "哪个角色",
    "角色是谁",
    "出自哪里",
    "出处",
    "动漫",
    "gal",
    "galgame",
]
DEFAULT_TOOL_DESCRIPTION = (
    "AnimeTrace 二次元图片识别工具。"
    "当用户发送或引用图片，并询问图片里的动漫/GalGame/二次元角色、作品出处，"
    "或者你自己无法确定图片内容时调用。"
    "工具会返回候选角色和作品名，结果仅供参考，回答时要保留不确定性。"
)

DEFAULT_CONFIG = {
    "timeout_seconds": 30,
    "prompt_send_image": "📷 请发送要识别的图片（30秒内有效）",
    "prompt_timeout": "⏰ 识别请求已超时，请重新发送命令",
    "return_crops": True,
    "max_crops": 5,
    "max_characters_per_role": 5,
    "forward_threshold": 0,
    "llm_tool_enabled": True,
    "inject_llm_tool_hint": True,
    "llm_tool_max_results": 5,
    "tool_request_keywords": DEFAULT_TOOL_REQUEST_KEYWORDS,
    "tool_description": DEFAULT_TOOL_DESCRIPTION,
}

API_ERROR_CODES = {
    17720: "识别成功",
    200: "Success",
    17721: "服务器正常运行中",
    17701: "图片大小过大",
    17702: "服务器繁忙，请重试",
    17703: "请求参数不正确",
    17704: "API维护中",
    17705: "图片格式不支持",
    17706: "识别无法完成（内部错误，请重试）",
    17707: "内部错误",
    17708: "图片中的人物数量超过限制",
    17722: "图片下载失败",
    17728: "已达到本次使用上限",
    17731: "服务利用人数过多，请重新尝试",
    404: "页面不存在",
}


def _clean_text(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text or default


def _read_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "enabled"}:
            return True
        if lowered in {"0", "false", "no", "off", "disabled"}:
            return False
    if value is None:
        return default
    return bool(value)


def _read_int(
    value: Any,
    default: int,
    *,
    minimum: int = 0,
    maximum: int = 999999,
) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return min(maximum, max(minimum, number))


def _read_list(value: Any, default: list[str]) -> list[str]:
    if isinstance(value, list):
        items = [_clean_text(item) for item in value]
        return [item for item in items if item] or default
    if isinstance(value, str):
        normalized = value.replace("，", ",").replace("；", ";")
        parts = [
            item.strip()
            for chunk in normalized.split(";")
            for item in chunk.split(",")
        ]
        return [item for item in parts if item] or default
    return default


@dataclass
class AnimeTraceIdentifyTool(FunctionTool[AstrAgentContext]):
    plugin: Any = Field(default=None, repr=False)
    name: str = TOOL_NAME
    description: str = DEFAULT_TOOL_DESCRIPTION
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "image_url": {
                    "type": "string",
                    "description": (
                        "可选。要识别的图片 URL、本地路径或 file:// URI。"
                        "留空时自动使用当前消息或引用消息中的图片。"
                    ),
                },
                "model": {
                    "type": "string",
                    "description": "可选。AnimeTrace 模型 ID；不确定时留空。",
                },
                "qq": {
                    "type": "string",
                    "description": "可选。要识别 QQ 头像时填写 QQ 号。",
                },
                "max_results": {
                    "type": "integer",
                    "description": "可选。每个检测区域最多返回多少个候选，默认使用插件配置。",
                    "minimum": 1,
                    "maximum": 20,
                },
            },
        }
    )

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        **kwargs: Any,
    ) -> str:
        if self.plugin is None:
            return "AnimeTrace 识图工具未绑定插件实例，请重载插件。"

        event = getattr(getattr(context, "context", None), "event", None)
        return await self.plugin.recognize_for_tool(
            event=event,
            image_url=_clean_text(kwargs.get("image_url")),
            model=_clean_text(kwargs.get("model")),
            qq=_clean_text(kwargs.get("qq")),
            max_results=kwargs.get("max_results"),
        )


@register(
    PLUGIN_ID,
    PLUGIN_AUTHOR,
    PLUGIN_DESC,
    PLUGIN_VERSION,
    PLUGIN_REPO,
)
class AnimeTraceLLMPlugin(Star):
    def __init__(self, context: Context, config=None):
        super().__init__(context, config)
        self.config = config or {}
        self.api_url: str = "https://api.animetrace.com/v1/search"
        self.model_list_url: str = "https://api.animetrace.com/v1/model/list"
        self.waiting_sessions = {}
        self.timeout_tasks = {}
        self._session = None
        self._models = []
        self._default_model = None
        self._current_model = None
        self._model_cache_time = 0
        self._model_cache_ttl = 3600

        plugin_config = self._section(CONFIG_SECTION)
        for key, default in DEFAULT_CONFIG.items():
            setattr(self, key, plugin_config.get(key, default))
        self.llm_tool_enabled = _read_bool(self.llm_tool_enabled, True)
        self.inject_llm_tool_hint = _read_bool(self.inject_llm_tool_hint, True)
        self.llm_tool_max_results = _read_int(
            self.llm_tool_max_results,
            5,
            minimum=1,
            maximum=20,
        )
        self.tool_request_keywords = _read_list(
            self.tool_request_keywords,
            DEFAULT_TOOL_REQUEST_KEYWORDS,
        )
        self.tool_description = _clean_text(
            self.tool_description,
            DEFAULT_TOOL_DESCRIPTION,
        )
        self._register_llm_tool()

    def _section(self, key: str) -> dict[str, Any]:
        if hasattr(self.config, "get"):
            value = self.config.get(key, {})
            if isinstance(value, dict):
                return value
        fallback = getattr(self.context, "_config", {})
        if isinstance(fallback, dict):
            value = fallback.get(key, {})
            if isinstance(value, dict):
                return value
        return {}

    def _register_llm_tool(self) -> None:
        self.context.add_llm_tools(
            AnimeTraceIdentifyTool(
                plugin=self,
                description=self.tool_description,
                active=self.llm_tool_enabled,
            )
        )

    async def _ensure_session(self) -> None:
        if self._session and not self._session.closed:
            return
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))

    async def initialize(self):
        await self._ensure_session()
        await self._fetch_models()
        logger.info("[%s] AnimeTrace LLM 图片识别插件已加载", PLUGIN_ID)

    async def _fetch_models(self):
        try:
            await self._ensure_session()
            async with self._session.get(self.model_list_url) as response:
                if response.status != 200:
                    logger.warning(f"获取模型列表失败: HTTP {response.status}")
                    return

                result = await response.json()
                if result.get("code") != 0:
                    logger.warning(
                        f"获取模型列表失败: {result.get('message', '未知错误')}"
                    )
                    return

                self._models = result.get("data", [])
                enabled_models = [m for m in self._models if m.get("enabled", False)]
                self._default_model = next(
                    (m for m in enabled_models if m.get("default", False)),
                    enabled_models[0] if enabled_models else None,
                )
                self._model_cache_time = asyncio.get_event_loop().time()

                model_names = [m["name"] for m in self._models]
                logger.debug(f"已加载模型列表: {model_names}")
        except Exception as e:
            logger.warning(f"获取模型列表异常: {str(e)}")

    async def _get_default_model(self) -> dict:
        current_time = asyncio.get_event_loop().time()
        if (
            not self._models
            or current_time - self._model_cache_time > self._model_cache_ttl
        ):
            await self._fetch_models()

        if self._current_model:
            return self._current_model

        if self._default_model:
            return self._default_model
        if self._models:
            return self._models[0]
        return {"id": FALLBACK_MODEL_ID, "name": FALLBACK_MODEL_ID, "enabled": True}

    @filter.command(IMAGE_COMMAND)
    async def trace_search(self, event: AstrMessageEvent, args=None):
        default_model = await self._get_default_model()
        return await self.handle_image_recognition(event, default_model["id"])

    @filter.command(AVATAR_COMMAND)
    async def avatar_trace_search(self, event: AstrMessageEvent, args=None):
        default_model = await self._get_default_model()
        return await self.handle_avatar_recognition(event, default_model["id"])

    @filter.command(MODEL_COMMAND)
    async def model_list(self, event: AstrMessageEvent, args=None):
        await self._fetch_models()

        if not self._models:
            await event.send(event.plain_result("❌ 无法获取模型列表，请稍后重试"))
            return

        if args is not None:
            try:
                index = int(args) - 1
            except (ValueError, TypeError):
                await event.send(event.plain_result("❌ 无效的模型编号"))
                return
            if 0 <= index < len(self._models):
                model = self._models[index]
                if model.get("enabled", False):
                    self._current_model = model
                    await event.send(
                        event.plain_result(
                            f"✅ 已切换到模型: {model['id']}"
                        )
                    )
                else:
                    await event.send(
                        event.plain_result(
                            f"❌ 模型 {model['id']} 当前不可用，请选择其他模型"
                        )
                    )
            else:
                await event.send(event.plain_result("❌ 无效的模型编号"))
            return

        lines = ["📋 AnimeTrace 模型列表："]
        current_model_id = self._current_model["id"] if self._current_model else None
        for idx, model in enumerate(self._models, start=1):
            model_id = model["id"]
            desc = model.get("desc", {})
            desc_zh = desc.get("zh", "")
            enabled = model.get("enabled", True)
            is_current = model_id == current_model_id

            line = f"{idx}. {model_id}"
            if is_current:
                line += " ⭐(当前)"
            if desc_zh:
                line += f"\n   {desc_zh}"
            line += f"\n   状态: {'✅ 可用' if enabled else '❌ 不可用'}"
            lines.append(line)

        lines.append(f"\n使用 /{MODEL_COMMAND} 数字 切换模型")
        await event.send(event.plain_result("\n".join(lines)))

    async def handle_image_recognition(self, event: AstrMessageEvent, model: str):
        user_id = event.get_sender_id()

        image_url = await self.extract_image_from_event(event)
        if image_url:
            await self.process_image_recognition(event, image_url, model)
            return

        try:
            raw_event = event._event if hasattr(event, "_event") else event
            if hasattr(raw_event, "reply_to_message") and raw_event.reply_to_message:
                logger.debug("检测到引用消息，但引用消息中没有找到图片")
                await event.send(
                    event.plain_result(
                        "❌ 引用消息中没有找到图片，请确保引用的消息包含图片"
                    )
                )
                return
        except Exception as e:
            logger.warning(f"检查引用消息状态时出错: {str(e)}")

        self.waiting_sessions[user_id] = {
            "model": model,
            "timestamp": asyncio.get_event_loop().time(),
            "event": event,
        }

        if user_id in self.timeout_tasks:
            self.timeout_tasks[user_id].cancel()

        timeout_task = asyncio.create_task(self.timeout_check(user_id))
        self.timeout_tasks[user_id] = timeout_task

        await event.send(event.plain_result(self.prompt_send_image))
        logger.debug(f"用户 {user_id} 进入等待图片状态，等待{self.timeout_seconds}秒")

    async def handle_avatar_recognition(self, event: AstrMessageEvent, model: str):
        try:
            mentioned_user_id = await self.extract_mentioned_user(event)

            if not mentioned_user_id:
                mentioned_user_id = event.get_sender_id()
                await event.send(event.plain_result("📸 识别您自己的头像..."))
            else:
                full_text = self._get_full_text(event.get_messages())
                qq_match = re.search(
                    rf"{re.escape(AVATAR_COMMAND)}\s*(\d{{5,12}})",
                    full_text,
                )
                if qq_match and qq_match.group(1) == mentioned_user_id:
                    await event.send(
                        event.plain_result(f"📸 识别QQ号 {mentioned_user_id} 的头像...")
                    )

            avatar_url = (
                f"https://q.qlogo.cn/headimg_dl?dst_uin={mentioned_user_id}&spec=640"
            )
            event._avatar_command_processed = True

            await self.process_image_recognition(event, avatar_url, model)

        except Exception as e:
            logger.error(f"头像识别失败: {str(e)}")
            await event.send(event.plain_result(f"❌ 头像识别失败: {str(e)}"))

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        user_id = event.get_sender_id()

        messages = event.get_messages()
        full_text = self._get_full_text(messages)

        if not hasattr(event, "_avatar_command_processed"):
            if re.search(rf"(?:^|\s|/){re.escape(AVATAR_COMMAND)}", full_text):
                event._avatar_command_processed = True
                default_model = await self._get_default_model()
                await self.handle_avatar_recognition(event, default_model["id"])
                return

        if user_id not in self.waiting_sessions:
            return

        session = self.waiting_sessions[user_id]

        current_time = asyncio.get_event_loop().time()
        if current_time - session["timestamp"] > self.timeout_seconds:
            return

        image_url = await self.extract_image_from_event(event)
        if not image_url:
            return

        del self.waiting_sessions[user_id]
        if user_id in self.timeout_tasks:
            self.timeout_tasks[user_id].cancel()
            del self.timeout_tasks[user_id]
        await self.process_image_recognition(event, image_url, session["model"])

    async def _recognize_image_source(self, image_url: str, model: str) -> dict:
        await self._ensure_session()
        temp_paths: list[str] = []
        try:
            if image_url.startswith(("http://", "https://")):
                results = await self.call_animetrace_api_with_url(image_url, model)
                if not results or not results.get("data"):
                    logger.debug("URL识别方式未返回结果，尝试file方式...")
                    temp_path = await self.download_to_temp_file(image_url)
                    temp_paths.append(temp_path)
                    results = await self.call_animetrace_api_with_file(temp_path, model)
                return results

            if image_url.startswith("file://"):
                temp_path = urllib.parse.unquote(image_url.replace("file://", ""))
                if os.name == "nt" and temp_path.startswith("/"):
                    temp_path = temp_path[1:]
                image_url = temp_path

            if os.path.isfile(image_url):
                return await self.call_animetrace_api_with_file(image_url, model)

            raise Exception("不支持的图片来源")
        finally:
            for temp_path in temp_paths:
                try:
                    if temp_path and os.path.isfile(temp_path):
                        os.remove(temp_path)
                except Exception as exc:
                    logger.debug(f"删除临时图片失败: {exc}")

    async def process_image_recognition(
        self, event: AstrMessageEvent, image_url: str, model: str
    ):
        try:
            results = await self._recognize_image_source(image_url, model)
            await self.send_combined_result(event, image_url, results, model)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"识别失败: {error_msg}")

            if "HTTP 500" in error_msg:
                user_msg = "❌ 识别服务暂时不可用，请稍后重试"
            elif "HTTP 422" in error_msg:
                user_msg = "❌ 图片格式不支持，请尝试其他图片"
            elif "HTTP 413" in error_msg or "图片大小过大" in error_msg:
                user_msg = "❌ 图片大小过大，请使用更小的图片"
            elif "HTTP 403" in error_msg or "API维护中" in error_msg:
                user_msg = "❌ API维护中，请稍后重试"
            elif "服务器繁忙" in error_msg or "服务利用人数过多" in error_msg:
                user_msg = "❌ 服务器繁忙，请稍后重试"
            elif "达到本次使用上限" in error_msg:
                user_msg = "❌ 已达到本次使用上限"
            elif "人物数量超过限制" in error_msg:
                user_msg = "❌ 图片中的人物数量超过限制"
            elif "图片格式不支持" in error_msg:
                user_msg = "❌ 图片格式不支持，请尝试其他图片"
            elif "图片下载失败" in error_msg:
                user_msg = "❌ 图片下载失败，请重试"
            elif "timeout" in error_msg.lower():
                user_msg = "❌ 识别超时，请稍后重试"
            else:
                user_msg = f"❌ 识别失败: {error_msg}"

            try:
                await event.send(event.plain_result(user_msg))
            except Exception as send_error:
                logger.warning(f"发送错误消息失败: {send_error}")

    def _get_full_text(self, messages) -> str:
        """从消息列表中提取完整文本"""
        full_text = ""
        for msg in messages:
            if hasattr(msg, "text"):
                full_text += str(msg.text)
            elif hasattr(msg, "type") and msg.type == "Plain":
                full_text += str(msg)
        return full_text

    async def extract_mentioned_user(self, event: AstrMessageEvent) -> str:
        messages = event.get_messages()
        full_text = self._get_full_text(messages)

        qq_match = re.search(
            rf"{re.escape(AVATAR_COMMAND)}\s*(\d{{5,12}})",
            full_text,
        )
        if qq_match:
            return qq_match.group(1)

        for msg in messages:
            if hasattr(msg, "type") and msg.type == "At":
                if hasattr(msg, "qq"):
                    return str(msg.qq)
                if hasattr(msg, "user_id"):
                    return str(msg.user_id)

            if hasattr(msg, "text"):
                text = str(msg.text)
                at_match = re.search(r"\[CQ:at,qq=(\d+)\]", text)
                if at_match:
                    return at_match.group(1)

        return None

    async def extract_image_from_event(self, event: AstrMessageEvent) -> str:
        messages = event.get_messages()

        for msg in messages:
            if isinstance(msg, MsgImage):
                image_ref = self._get_image_reference(msg)
                if image_ref:
                    try:
                        if hasattr(msg, "convert_to_file_path"):
                            file_path = await msg.convert_to_file_path()
                            if file_path and os.path.isfile(file_path):
                                return file_path
                    except Exception as e:
                        logger.debug(f"convert_to_file_path失败: {str(e)}")

                    if image_ref.startswith(("http://", "https://")):
                        return image_ref.strip("`'").strip()

        try:
            raw_message = getattr(event.message_obj, "raw_message", None)
            if raw_message:
                attachments = getattr(raw_message, "attachments", None)
                if attachments and isinstance(attachments, list):
                    for attachment in attachments:
                        url = getattr(attachment, "url", None)
                        if (
                            url
                            and isinstance(url, str)
                            and url.startswith(("http://", "https://"))
                        ):
                            return url.strip("`'").strip()

                item_list = (
                    raw_message.get("item_list")
                    if isinstance(raw_message, dict)
                    else getattr(raw_message, "item_list", None)
                )
                if item_list and isinstance(item_list, list):
                    for item in item_list:
                        item_type = int(item.get("type") or 0)
                        if item_type == 2:
                            image_item = item.get("image_item", {})
                            media = image_item.get("media", {})
                            encrypted_query_param = str(
                                media.get("encrypt_query_param", "")
                            ).strip()
                            if encrypted_query_param:
                                cdn_url = f"https://novac2c.cdn.weixin.qq.com/c2c/download?encrypted_query_param={encrypted_query_param}"
                                return cdn_url
        except Exception as e:
            logger.debug(f"从raw_message提取图片URL失败: {str(e)}")

        try:
            for msg in messages:
                if isinstance(msg, Reply) and hasattr(msg, "chain") and msg.chain:
                    for reply_msg in msg.chain:
                        if isinstance(reply_msg, MsgImage):
                            image_ref = self._get_image_reference(reply_msg)
                            if image_ref:
                                try:
                                    if hasattr(reply_msg, "convert_to_file_path"):
                                        file_path = (
                                            await reply_msg.convert_to_file_path()
                                        )
                                        if file_path and os.path.isfile(file_path):
                                            return file_path
                                except Exception as e:
                                    logger.debug(
                                        f"引用消息convert_to_file_path失败: {str(e)}"
                                    )

                                if image_ref.startswith(("http://", "https://")):
                                    return image_ref.strip("`'").strip()
        except Exception as e:
            logger.warning(f"检查引用消息图片时出错: {str(e)}")

        return None

    def _event_has_image_reference(self, event: AstrMessageEvent) -> bool:
        try:
            messages = event.get_messages()
        except Exception:
            messages = []

        for msg in messages:
            if isinstance(msg, MsgImage):
                return True
            if isinstance(msg, Reply) and hasattr(msg, "chain") and msg.chain:
                if any(isinstance(reply_msg, MsgImage) for reply_msg in msg.chain):
                    return True

        try:
            raw_message = getattr(event.message_obj, "raw_message", None)
            if raw_message:
                attachments = getattr(raw_message, "attachments", None)
                if attachments and isinstance(attachments, list):
                    if any(getattr(attachment, "url", None) for attachment in attachments):
                        return True

                item_list = (
                    raw_message.get("item_list")
                    if isinstance(raw_message, dict)
                    else getattr(raw_message, "item_list", None)
                )
                if item_list and isinstance(item_list, list):
                    return any(int(item.get("type") or 0) == 2 for item in item_list)
        except Exception as exc:
            logger.debug(f"检查消息图片引用失败: {exc}")

        return False

    def _get_image_reference(self, msg) -> str:
        """获取图片组件的引用（优先url，其次file）"""
        return getattr(msg, "url", None) or getattr(msg, "file", None)

    async def _download_image_data(self, image_url: str) -> bytes:
        """下载图片数据（支持本地路径、file:// URI和HTTP/HTTPS URL）"""
        if os.path.isfile(image_url):
            logger.debug(f"读取本地图片: {image_url}")
            with open(image_url, "rb") as f:
                return f.read()

        if image_url.startswith("file://"):
            file_path = urllib.parse.unquote(image_url.replace("file://", ""))
            if os.name == "nt" and file_path.startswith("/"):
                file_path = file_path[1:]
            logger.debug(f"读取file://图片: {file_path}")
            with open(file_path, "rb") as f:
                return f.read()

        if image_url.startswith("telegram://"):
            raise Exception("Telegram文件暂不支持")

        async with self._session.get(image_url) as response:
            if response.status != 200:
                raise Exception(f"图片下载失败: HTTP {response.status}")
            return await response.read()

    async def download_to_temp_file(self, image_url: str) -> str:
        logger.debug(f"下载图片到临时文件: {image_url[:100]}...")

        try:
            img_data = await self._download_image_data(image_url)

            img = PILImage.open(BytesIO(img_data))

            if max(img.size) > 1024:
                ratio = 1024 / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, PILImage.LANCZOS)

            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".jpg", delete=False
            ) as f:
                img.save(f, format="JPEG", quality=85)
                temp_path = f.name

            logger.debug(f"图片保存到临时文件: {temp_path}")
            return temp_path
        except asyncio.TimeoutError:
            raise Exception("图片下载超时，请稍后重试")
        except Exception as e:
            logger.error(f"图片下载失败: {str(e)}")
            raise Exception(f"图片下载失败: {str(e)}")

    def _get_model_name(self, model_id: str) -> str:
        """根据模型ID获取显示名称"""
        model = next((m for m in self._models if m["id"] == model_id), None)
        if model:
            return model.get("name", model_id)
        return model_id

    async def call_animetrace_api_with_file(self, file_path: str, model: str) -> dict:
        model_name = self._get_model_name(model)
        logger.debug(f"调用API - 模型: {model_name} (file方式)")

        try:
            with open(file_path, "rb") as f:
                file_data = f.read()

            form = aiohttp.FormData()
            form.add_field("is_multi", "1")
            form.add_field("model", model)
            form.add_field("ai_detect", "0")
            form.add_field("file", file_data, filename="image.jpg", content_type="image/jpeg")

            async with self._session.post(self.api_url, data=form) as response:
                try:
                    result = await response.json()
                except Exception:
                    error_text = await response.text()
                    logger.warning(
                        f"API返回错误状态: HTTP {response.status}, 响应: {error_text[:200]}"
                    )
                    raise Exception(f"API错误: HTTP {response.status}")

                code = result.get("code")

                if code not in (0, 17720, 200, 17721):
                    zh_message = result.get("zh_message", "")
                    if zh_message:
                        error_msg = zh_message
                    else:
                        error_msg = API_ERROR_CODES.get(code, f"未知错误 (code={code})")
                    logger.warning(f"API返回错误码: {code}, 消息: {error_msg}")
                    raise Exception(f"API错误: {error_msg}")

                logger.debug(f"API返回: {len(result.get('data', []))} 个结果")
                return result
        except asyncio.TimeoutError:
            logger.error("API调用超时")
            raise Exception("识别服务响应超时，请稍后重试")
        except Exception as e:
            logger.error(f"file API调用失败: {str(e)}")
            raise

    async def call_animetrace_api_with_url(self, image_url: str, model: str) -> dict:
        payload = {"url": image_url, "is_multi": 1, "model": model, "ai_detect": 0}
        model_name = self._get_model_name(model)
        logger.debug(f"调用API - 模型: {model_name} (URL方式)")

        try:
            async with self._session.post(self.api_url, data=payload) as response:
                try:
                    result = await response.json()
                except Exception:
                    if response.status in [422, 500, 502, 503, 504]:
                        logger.debug(
                            f"URL识别失败 (HTTP {response.status})，准备回退到file方式"
                        )
                        return {"data": []}
                    error_text = await response.text()
                    logger.warning(
                        f"API返回错误状态: HTTP {response.status}, 响应: {error_text[:200]}"
                    )
                    raise Exception(f"API错误: HTTP {response.status}")

                code = result.get("code")

                if code not in (0, 17720, 200, 17721):
                    if code in (17701, 17705, 17708, 17722):
                        logger.debug(
                            f"URL识别失败 (code={code})，准备回退到file方式"
                        )
                        return {"data": []}
                    zh_message = result.get("zh_message", "")
                    if zh_message:
                        error_msg = zh_message
                    else:
                        error_msg = API_ERROR_CODES.get(code, f"未知错误 (code={code})")
                    logger.warning(f"API返回错误码: {code}, 消息: {error_msg}")
                    raise Exception(f"API错误: {error_msg}")

                logger.debug(f"API返回: {len(result.get('data', []))} 个结果")
                return result
        except Exception as e:
            logger.warning(f"URL方式调用失败: {str(e)}，准备回退到file方式")
            return {"data": []}

    def format_results(self, data: dict, model: str) -> str:
        if not data.get("data") or not data["data"]:
            return "🔍 未找到匹配的信息"

        results = [item for item in data["data"] if item.get("character")]
        if not results:
            return "🔍 未识别到具体角色信息"

        model_name = self._get_model_name(model)

        lines = [f"🔍 {model_name} 识别结果"]

        for idx, item in enumerate(results, start=1):
            characters = item.get("character", [])
            if not characters:
                continue

            if len(results) > 1:
                lines.append(f"\n第 {idx} 个角色：")

            limit = self.max_characters_per_role
            display_characters = characters[:limit] if limit > 0 else characters
            for i, char in enumerate(display_characters):
                name = char.get("character", "未知角色")
                work = char.get("work", "未知作品")
                lines.append(f"{i + 1}. {name} - 《{work}》")

            if limit > 0 and len(characters) > limit:
                lines.append(f"共 {len(characters)} 个结果，显示前{limit}项")

        model_name = self._get_model_name(model)
        lines.append("数据来源: AnimeTrace，仅供参考")
        lines.append(f"当前模型: {model_name}")

        return "\n".join(lines)

    def _format_score_suffix(self, char: dict[str, Any]) -> str:
        for key in ("score", "similarity", "confidence", "prob"):
            value = char.get(key)
            if value in (None, ""):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                return f"，{key}: {value}"
            if 0 <= number <= 1:
                return f"，{key}: {number:.2%}"
            return f"，{key}: {number:g}"
        return ""

    def format_tool_results(
        self,
        data: dict,
        model: str,
        *,
        max_results: int | None = None,
    ) -> str:
        data_list = data.get("data") or []
        if not data_list:
            return (
                "AnimeTrace 没有找到匹配结果。"
                "请告诉用户无法可靠识别这张图，不要编造角色或作品。"
            )

        limit = max_results or self.llm_tool_max_results
        limit = _read_int(limit, self.llm_tool_max_results, minimum=1, maximum=20)
        model_name = self._get_model_name(model)
        lines = [
            "AnimeTrace 识图候选结果：",
            f"模型：{model_name}",
            "说明：结果仅供参考，回答用户时请保留不确定性。",
        ]

        matched_roles = 0
        for idx, item in enumerate(data_list, start=1):
            characters = item.get("character") or []
            if not characters:
                continue
            matched_roles += 1
            box = item.get("box")
            box_text = f"，box={box}" if box else ""
            lines.append(f"\n检测区域 {idx}{box_text}:")
            for rank, char in enumerate(characters[:limit], start=1):
                name = char.get("character") or "未知角色"
                work = char.get("work") or "未知作品"
                suffix = self._format_score_suffix(char)
                lines.append(f"{rank}. 角色：{name}；作品：{work}{suffix}")
            if len(characters) > limit:
                lines.append(f"还有 {len(characters) - limit} 个候选未展示。")

        if not matched_roles:
            return (
                "AnimeTrace 返回了检测区域，但没有具体角色候选。"
                "请告诉用户无法可靠识别这张图，不要编造角色或作品。"
            )

        lines.append("\n数据来源：AnimeTrace。")
        return "\n".join(lines)

    async def recognize_for_tool(
        self,
        *,
        event: AstrMessageEvent | None,
        image_url: str = "",
        model: str = "",
        qq: str = "",
        max_results: Any = None,
    ) -> str:
        if not self.llm_tool_enabled:
            return "AnimeTrace LLM 识图工具当前未启用。"

        if qq:
            if not re.fullmatch(r"\d{5,12}", qq):
                return "QQ 号格式不正确，无法识别头像。"
            image_url = f"https://q.qlogo.cn/headimg_dl?dst_uin={qq}&spec=640"

        if not image_url:
            if event is None:
                return "没有提供图片 URL，也无法从当前会话事件中读取图片。"
            image_url = await self.extract_image_from_event(event)

        if not image_url:
            return (
                "当前消息或引用消息里没有找到可识别的图片。"
                f"如果用户只是想手动识图，请提示 TA 发送图片并使用 /{IMAGE_COMMAND}。"
            )

        default_model = await self._get_default_model()
        model_id = model or default_model["id"]
        try:
            results = await self._recognize_image_source(image_url, model_id)
        except Exception as exc:
            logger.warning(f"LLM 工具识图失败: {exc}")
            return f"AnimeTrace 识图失败：{exc}"

        return self.format_tool_results(
            results,
            model_id,
            max_results=max_results,
        )

    async def send_combined_result(
        self, event: AstrMessageEvent, image_url: str, results: dict, model: str
    ):
        tmp_dir = None
        try:
            data_list = results.get("data") or []
            if not data_list:
                response = self.format_results(results, model)
                await event.send(event.plain_result(response))
                return

            chain = []

            if self.return_crops:
                try:
                    img_data = await self._download_image_data(image_url)
                except Exception as e:
                    logger.debug(f"裁剪图片下载失败: {str(e)}")
                    response_text = self.format_results(results, model)
                    await event.send(event.plain_result(response_text))
                    return

                img = PILImage.open(BytesIO(img_data)).convert("RGB")
                w, h = img.size

                tmp_dir = tempfile.mkdtemp(prefix="astrbot_animetrace_llm_crops_")
                crop_paths = []

                for idx, item in enumerate(data_list, start=1):
                    if len(crop_paths) >= self.max_crops:
                        break

                    box = item.get("box")
                    if not box or len(box) != 4:
                        continue

                    x1 = int(max(0, min(1, float(box[0]))) * w)
                    y1 = int(max(0, min(1, float(box[1]))) * h)
                    x2 = int(max(0, min(1, float(box[2]))) * w)
                    y2 = int(max(0, min(1, float(box[3]))) * h)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    cropped = img.crop((x1, y1, x2, y2))
                    out_path = os.path.join(tmp_dir, f"crop_{idx}.jpg")
                    cropped.save(out_path, format="JPEG", quality=90)
                    crop_paths.append((idx, out_path, item))

                for idx, out_path, item in crop_paths:
                    chain.append(Comp.Image.fromFileSystem(out_path))

                    characters = item.get("character") or []
                    if characters:
                        text_lines = []
                        if len(crop_paths) > 1:
                            text_lines.append(f"第 {idx} 个角色：")

                        limit = self.max_characters_per_role
                        display_characters = (
                            characters[:limit] if limit > 0 else characters
                        )
                        for i, char in enumerate(display_characters):
                            name = char.get("character", "未知角色")
                            work = char.get("work", "未知作品")
                            text_lines.append(f"{i + 1}. {name} - 《{work}》")

                        if limit > 0 and len(characters) > limit:
                            text_lines.append(
                                f"共 {len(characters)} 个结果，显示前{limit}项"
                            )

                        if text_lines:
                            chain.append(Comp.Plain("\n".join(text_lines)))
                            chain.append(Comp.Plain(""))

            if not self.return_crops or len(crop_paths) < len(data_list):
                response_text = self.format_results(results, model)
                chain.append(Comp.Plain(response_text))
            else:
                model_name = self._get_model_name(model)
                chain.append(Comp.Plain(f"💡 数据来源: AnimeTrace，仅供参考\n当前模型: {model_name}"))

            character_count = len(
                [item for item in data_list if item.get("character")]
            )
            use_forward = (
                self.forward_threshold > 0
                and character_count >= self.forward_threshold
                and event.get_platform_name() == "aiocqhttp"
            )

            if chain:
                if use_forward:
                    sender_name = event.get_sender_name() or "AnimeTrace"
                    sender_id = event.get_sender_id() or "10000"
                    nodes = []
                    current_content = []
                    for comp in chain:
                        if isinstance(comp, Comp.Image):
                            if current_content:
                                nodes.append(
                                    Comp.Node(
                                        content=current_content,
                                        name=sender_name,
                                        uin=sender_id,
                                    )
                                )
                                current_content = []
                            current_content.append(comp)
                        elif isinstance(comp, Comp.Plain):
                            if comp.text.strip():
                                current_content.append(comp)
                        else:
                            current_content.append(comp)
                    if current_content:
                        nodes.append(
                            Comp.Node(
                                content=current_content,
                                name=sender_name,
                                uin=sender_id,
                            )
                        )
                    if nodes:
                        await event.send(event.chain_result([Comp.Nodes(nodes)]))
                else:
                    await event.send(event.chain_result(chain))
            else:
                response_text = self.format_results(results, model)
                await event.send(event.plain_result(response_text))

        except Exception as e:
            logger.warning(f"发送合并结果失败: {e}")
            try:
                response_text = self.format_results(results, model)
                await event.send(event.plain_result(response_text))
            except Exception as send_error:
                logger.warning(f"发送文字结果也失败: {send_error}")
        finally:
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def _message_requests_tool(self, event: AstrMessageEvent) -> bool:
        message_text = _clean_text(getattr(event, "message_str", ""))
        if not message_text:
            try:
                message_text = self._get_full_text(event.get_messages())
            except Exception:
                message_text = ""
        lowered = message_text.lower()
        return any(keyword.lower() in lowered for keyword in self.tool_request_keywords)

    @filter.on_llm_request(priority=-5)
    async def inject_animetrace_tool_hint(
        self,
        event: AstrMessageEvent,
        request: ProviderRequest,
    ) -> None:
        if not self.llm_tool_enabled or not self.inject_llm_tool_hint:
            return

        has_image = self._event_has_image_reference(event)
        if not has_image and not self._message_requests_tool(event):
            return

        hint = (
            "[AnimeTrace 识图工具提示]\n"
            f"可用工具：`{TOOL_NAME}`。\n"
            "当用户发送或引用图片并询问“这是谁、哪个角色、出处、识图、动漫/GalGame角色”，"
            "或你无法可靠判断图片内容时，优先调用该工具；不要直接猜测。"
            "如果图片在当前消息或引用消息里，调用时可以不填 image_url。"
            "工具返回的是候选结果，不保证绝对准确；最终回答应说明最可能的角色/作品，并保留不确定性。"
        )

        try:
            request.extra_user_content_parts.append(TextPart(text=hint).mark_as_temp())
        except Exception:
            system_prompt = getattr(request, "system_prompt", "") or ""
            if hint not in system_prompt:
                request.system_prompt = (
                    f"{system_prompt}\n\n{hint}"
                    if system_prompt
                    else hint
                )

    async def timeout_check(self, user_id: str):
        try:
            await asyncio.sleep(self.timeout_seconds)
            if user_id in self.waiting_sessions:
                session = self.waiting_sessions[user_id]
                event = session["event"]
                del self.waiting_sessions[user_id]
                del self.timeout_tasks[user_id]
                try:
                    await event.send(event.plain_result(self.prompt_timeout))
                    logger.debug(f"用户 {user_id} 的图片识别请求已超时")
                except Exception as send_error:
                    logger.warning(f"发送超时消息失败: {send_error}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"超时检查任务异常: {str(e)}")

    async def terminate(self):
        logger.info("[%s] AnimeTrace LLM 图片识别插件已卸载", PLUGIN_ID)
        for task in self.timeout_tasks.values():
            task.cancel()
        self.timeout_tasks.clear()
        if self._session:
            await self._session.close()

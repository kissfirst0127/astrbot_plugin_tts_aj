import json
import random

from pydantic import Field
from pydantic.dataclasses import dataclass

import astrbot.api.message_components as Comp
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.core.config import AstrBotConfig
from astrbot.core.provider.entities import LLMResponse

from .tts_api.dy_tts_api import tts_http_stream


@register(
    "astrbot_plugin_clonetts",
    "Radiant303",
    "基于火山引擎音色克隆(ICL)的文本转语音插件",
    "3.0.0",
)
class CloneTTSPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        # --- 配置读取与防御性校验 ---
        self.enable_tts = bool(config.get("enable_tts", True))
        self.enable_llm_tool = bool(config.get("enable_llm_tool", True))
        self.enable_llm_response = bool(config.get("enable_llm_response", False))
        self.enable_tts_tool_minmax = bool(config.get("enable_tts_tool_minmax", True))
        self.blocked_words = config.get("blocked_words", [])
        self.llm_recognition = config.get("llm_recognition", "")

        # 概率 & 长度：强制转为数值并限制在合理范围
        try:
            self.tts_probability = max(
                0, min(100, float(config.get("tts_probability", 80)))
            )
        except (TypeError, ValueError):
            logger.warning("tts_probability 配置值无效，已使用默认值 100")
            self.tts_probability = 80

        try:
            self.max_length = max(1, int(config.get("max_length", 50)))
        except (TypeError, ValueError):
            logger.warning("max_length 配置值无效，已使用默认值 50")
            self.max_length = 50

        try:
            self.min_length = max(1, int(config.get("min_length", 5)))
        except (TypeError, ValueError):
            logger.warning("min_length 配置值无效，已使用默认值 5")
            self.min_length = 5

        # 凭证类字段：确保为非空字符串
        self.api_key = str(config.get("api_key") or "")
        self.voice_type = str(config.get("voice_type") or "")

        # 音频参数
        try:
            self.speed_ratio = max(-50, min(100, int(config.get("speed_ratio", 0))))
        except (TypeError, ValueError):
            self.speed_ratio = 0

        try:
            self.loudness_rate = max(-50, min(100, int(config.get("loudness_rate", 0))))
        except (TypeError, ValueError):
            self.loudness_rate = 0
        try:
            self.sample_rate = int(config.get("sample_rate", 24000))
        except (TypeError, ValueError):
            self.sample_rate = 24000

        # 启动时预检关键配置
        missing = [
            name
            for name, val in [
                ("api_key", self.api_key),
                ("voice_type", self.voice_type),
            ]
            if not val
        ]
        if missing and self.enable_tts:
            logger.warning(
                f"CloneTTS 以下必要配置为空，语音合成可能失败: {', '.join(missing)}"
            )
        self.context.add_llm_tools(CloneTTSTool(plugin=self))

    async def initialize(self):
        """可选择实现异步的插件初始化方法，当实例化该插件类之后会自动调用该方法。"""
        logger.info("CloneTTS plugin 已经启用")

    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent):
        try:
            if not self.enable_tts:
                return

            if not self.api_key or not self.voice_type:
                logger.debug("配置未全，跳过 TTS")
                return

            if not self.probability(self.tts_probability):
                logger.debug("本次消息未命中 TTS 概率，跳过语音合成")
                return

            result = event.get_result()

            # 跳过指令返回
            if not result.is_llm_result():
                return
                
            if not result or not result.chain:
                logger.debug("本次消息没有结果，跳过 TTS")
                return

            text_parts = []
            for component in result.chain:
                if hasattr(component, "text"):
                    text_parts.append(component.text)

            if not text_parts:
                logger.debug("本次消息没有文本，跳过 TTS")
                return

            llm_text = "".join(text_parts).strip()

            if len(llm_text) < 1:
                logger.debug("LLM 文本为空，跳过语音合成")
                return

            if len(llm_text) > self.max_length:
                logger.debug(
                    f"LLM 文本长度 {len(llm_text)} 超过上限 {self.max_length}，跳过语音合成"
                )
                return

            if len(llm_text) < self.min_length:
                logger.debug(
                    f"LLM 文本长度 {len(llm_text)} 小于下限 {self.min_length}，跳过语音合成"
                )
                return
            if self.blocked_words and any(
                word in llm_text for word in self.blocked_words
            ):
                logger.debug("LLM 输出包含屏蔽词，跳过语音合成")
                return

            if self.llm_recognition:
                provider_id = self.llm_recognition
            else:
                logger.debug("未指定 LLM 识别模型，使用当前聊天模型")
                umo = event.unified_msg_origin
                provider_id = await self.context.get_current_chat_provider_id(umo=umo)
            llm_resp = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=(
                    "You prepare text for Japanese TTS. Translate the following Chinese "
                    "reply into natural Japanese first. Then judge the tone of the entire "
                    "Japanese sentence, including any emotional changes, and write one concise "
                    "Japanese direction for the voice model covering speaking speed, emotion, "
                    "tone, volume, and timbre. Return only a JSON object with the string keys "
                    '"japanese_text" and "context_texts"; do not use Markdown. '
                    f"Chinese reply: {llm_text}"
                ),
            )
            try:
                tts_data = json.loads(llm_resp.completion_text)
                japanese_text = str(tts_data["japanese_text"]).strip()
                context_texts = str(tts_data["context_texts"]).strip()
            except (json.JSONDecodeError, KeyError, TypeError):
                logger.warning("Japanese translation or tone analysis returned invalid data")
                return
            if not japanese_text or not context_texts:
                logger.warning("Japanese translation or tone analysis returned empty data")
                return
            logger.debug(f"使用的语气风格: {context_texts}")
            logger.info(f"正在合成日文克隆语音: {japanese_text}")
            audio_b64 = await tts_http_stream(self, japanese_text, context_texts)

            if not audio_b64:
                logger.warning("语音合成返回空数据，跳过本次语音回复")
                return

            result = event.get_result()
            if result is None:
                logger.warning("合成完成但 event result 已失效，跳过")
                return
            result.chain.append(Comp.Record.fromBase64(audio_b64))
        except Exception as e:
            logger.error(f"Error in on_decorating_result: {e}")

    def probability(self, percent) -> bool:
        """以给定百分比概率返回 True（0 = 永不触发，100 = 必定触发）。"""
        try:
            p = float(percent)
        except (TypeError, ValueError):
            return False
        p = max(0.0, min(100.0, p))
        return random.random() < (p / 100)

    async def terminate(self):
        """可选择实现异步的插件销毁方法，当插件被卸载/停用时会调用。"""
        logger.info("CloneTTS plugin 已经停用/卸载")

    @filter.on_llm_response()
    async def handle_silence(self, event: AstrMessageEvent, resp: LLMResponse):
            if event.get_extra("voice_silence_mode"):
                # 1. 消除标记
                event.set_extra("voice_silence_mode", False)

                # 2. 核心：将模型的文本强制修改为 \u200b (零宽空格)
                # 这样做的效果：
                # - Runner 看到 resp 有内容 (len(parts) > 0)，消除 "LLM returned empty" 警告。
                # - Responder 看到消息链不为空，消除 "消息链全为 Reply" 警告。
                # - 用户在前端什么都看不到，实现“模型不说话”的效果。
                resp.completion_text = "\u200b"

                # 3. 停止事件防止后续可能的冗余处理
                event.stop_event()

@dataclass
class CloneTTSTool(FunctionTool[AstrAgentContext]):
    name: str = "clone_tts"  # 工具名称
    description: str = "将中文文本转为日文语音发送。当用户需要听到声音或让你说话时，先翻译成自然日文，再判断完整日文句子的语气，最后调用本工具。"  # 工具描述
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "要展示给用户的原始中文文本。",
                },
                "japanese_text": {
                    "type": "string",
                    "description": "将 text 翻译得到的自然日文，仅用于语音合成，不会展示给用户。",
                },
                "context_texts": {
                    "type": "string",
                    "description": "基于整个 japanese_text 的语气判断生成的日文配音指令。须涵盖语速、情绪、语气、音量和音色；不要包含分析过程。",
                },
            },
            "required": ["text", "japanese_text", "context_texts"],
        }
    )

    plugin: object = Field(default=None, repr=False)

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs
    ) -> ToolExecResult:
        text = kwargs.get("text")
        if self.plugin.enable_tts_tool_minmax:
            logger.debug(f"LLM 将会受到长度限制{self.plugin.enable_tts_tool_minmax}")
            if len(str(text)) > self.plugin.max_length:
                return  f"LLM 文本长度超过上限 {self.plugin.max_length}，跳过语音合成"

            if len(str(text)) < self.plugin.min_length:
                return f"LLM 文本长度小于下限 {self.plugin.min_length}，跳过语音合成"
        if not self.plugin:
            return "插件未正确初始化"
        if not self.plugin.enable_llm_tool:
            return "LLM 工具未启用"
        if not text:
            return "文本不能为空"
        japanese_text = kwargs.get("japanese_text")
        if not japanese_text:
            return "日文翻译不能为空"
        context_texts = kwargs.get("context_texts")
        if not context_texts:
            return "语气指令不能为空"
        audio_b64 = await tts_http_stream(self.plugin, japanese_text, context_texts)
        if not audio_b64:
            return "语音合成失败"
        await context.context.event.send(
            context.context.event.chain_result(
                [Comp.Plain(text), Comp.Record.fromBase64(audio_b64)]
            )
        )
        if not self.plugin.enable_llm_response:
            context.context.event.set_extra("voice_silence_mode", True)
        return "SUCCESS"

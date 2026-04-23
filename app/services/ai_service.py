from __future__ import annotations

import logging
import json
import time
from dataclasses import dataclass
from json import JSONDecoder
from typing import Any, Mapping

import ollama

from app.core.config import Settings


logger = logging.getLogger(__name__)


class OllamaResponseFormatError(Exception):
    """Raised when Ollama returns a response that cannot be safely consumed."""


@dataclass(frozen=True)
class BilingualTranslationResult:
    translated_content_vi: str
    translated_content_en: str
    source_lang: str
    gpu_latency_ms: float


class AITranslationService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = ollama.AsyncClient(
            host=settings.ollama_base_url,
            timeout=settings.ollama_request_timeout_seconds,
        )

    async def close(self) -> None:
        aclose = getattr(self._client, "aclose", None)
        if aclose is not None:
            await aclose()

    async def translate(
        self,
        *,
        content: str,
        source_lang: str,
    ) -> BilingualTranslationResult:
        prompt = self._build_prompt(content=content, source_lang=source_lang)
        started_at = time.perf_counter()

        logger.info(
            "Calling Ollama model=%s requested_source_lang=%s content_chars=%d",
            self._settings.ollama_model,
            source_lang,
            len(content),
        )

        response = await self._client.generate(
            model=self._settings.ollama_model,
            prompt=prompt,
            format="json",
            stream=False,
            options={
                "temperature": 0.2,
                "num_ctx": self._settings.ollama_num_ctx,
            },
        )
        latency_ms = (time.perf_counter() - started_at) * 1000
        result = self._extract_translation_result(response, requested_source_lang=source_lang)

        logger.info(
            "Ollama translation completed model=%s source_lang=%s gpu_latency_ms=%.2f chars_vi=%d chars_en=%d",
            self._settings.ollama_model,
            result.source_lang,
            latency_ms,
            len(result.translated_content_vi),
            len(result.translated_content_en),
        )

        return BilingualTranslationResult(
            translated_content_vi=result.translated_content_vi,
            translated_content_en=result.translated_content_en,
            source_lang=result.source_lang,
            gpu_latency_ms=latency_ms,
        )

    @staticmethod
    def _build_prompt(*, content: str, source_lang: str) -> str:
        if source_lang == "auto":
            source_instruction = (
                'Detect whether the source language is "vi" or "en" and return that '
                'value in the "source_lang" field.'
            )
        else:
            source_instruction = (
                f'The caller already provided source_lang="{source_lang}". Treat that '
                'as authoritative and return the same value in the "source_lang" field.'
            )

        return (
            "You are a professional translation engine. Produce bilingual output for "
            "Vietnamese and English from the input content. Preserve meaning, "
            "terminology, punctuation, line breaks, markdown, URLs, and code blocks. "
            f"{source_instruction} Return a JSON object with exactly three string "
            'fields: "source_lang", "translated_content_vi", and '
            '"translated_content_en". If the input is already Vietnamese, '
            '"translated_content_vi" must equal the original content. If the input is '
            'already English, "translated_content_en" must equal the original '
            "content. Do not return arrays, nested objects, markdown fences, or "
            f"explanations.\n\nContent:\n{content}"
        )

    @staticmethod
    def _extract_translation_result(
        response: Any,
        *,
        requested_source_lang: str,
    ) -> BilingualTranslationResult:
        response_text = AITranslationService._extract_response_text(response).strip()
        payload = AITranslationService._parse_json_response(response_text)

        expected_keys = {
            "source_lang",
            "translated_content_vi",
            "translated_content_en",
        }
        extra_keys = set(payload) - expected_keys
        missing_keys = expected_keys - set(payload)
        if missing_keys or extra_keys:
            raise AITranslationService._format_error(
                "Ollama response JSON must contain exactly source_lang, translated_content_vi, and translated_content_en",
                response_text,
            )

        translated_content_vi = payload["translated_content_vi"]
        translated_content_en = payload["translated_content_en"]
        source_lang = payload["source_lang"]

        if not isinstance(translated_content_vi, str) or not translated_content_vi.strip():
            raise AITranslationService._format_error(
                "Ollama response translated_content_vi must be a non-empty string",
                response_text,
            )
        if not isinstance(translated_content_en, str) or not translated_content_en.strip():
            raise AITranslationService._format_error(
                "Ollama response translated_content_en must be a non-empty string",
                response_text,
            )
        if not isinstance(source_lang, str) or not source_lang.strip():
            raise AITranslationService._format_error(
                "Ollama response source_lang must be a non-empty string",
                response_text,
            )

        normalized_source_lang = source_lang.strip().lower()[:16]
        if requested_source_lang in {"vi", "en"}:
            normalized_source_lang = requested_source_lang

        if normalized_source_lang not in {"vi", "en"}:
            raise AITranslationService._format_error(
                "Ollama response source_lang must be vi or en",
                response_text,
            )

        return BilingualTranslationResult(
            translated_content_vi=translated_content_vi.strip(),
            translated_content_en=translated_content_en.strip(),
            source_lang=normalized_source_lang,
            gpu_latency_ms=0.0,
        )

    @staticmethod
    def _parse_json_response(response_text: str) -> dict[str, Any]:
        cleaned_text = AITranslationService._cleanup_response_text(response_text)
        json_text = AITranslationService._extract_json_object(cleaned_text)

        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise AITranslationService._format_error(
                "Ollama response must be valid JSON",
                response_text,
            ) from exc

        if not isinstance(payload, dict):
            raise AITranslationService._format_error(
                "Ollama response JSON must be an object",
                response_text,
            )
        return payload

    @staticmethod
    def _cleanup_response_text(response_text: str) -> str:
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    @staticmethod
    def _extract_json_object(response_text: str) -> str:
        start_index = response_text.find("{")
        if start_index == -1:
            raise AITranslationService._format_error(
                "Ollama response does not contain a JSON object",
                response_text,
            )

        decoder = JSONDecoder()
        try:
            payload, end_offset = decoder.raw_decode(response_text[start_index:])
        except json.JSONDecodeError as exc:
            raise AITranslationService._format_error(
                "Ollama response contains malformed JSON object",
                response_text,
            ) from exc

        if not isinstance(payload, dict):
            raise AITranslationService._format_error(
                "Ollama response JSON must be an object",
                response_text,
            )

        trailing_text = response_text[start_index + end_offset :].strip()
        if trailing_text:
            logger.warning(
                "Ignoring trailing non-JSON text in Ollama response preview=%s",
                AITranslationService._preview_text(trailing_text),
            )

        return response_text[start_index : start_index + end_offset]

    @staticmethod
    def _format_error(message: str, response_text: str) -> OllamaResponseFormatError:
        logger.error(
            "%s response_preview=%s",
            message,
            AITranslationService._preview_text(response_text),
        )
        return OllamaResponseFormatError(message)

    @staticmethod
    def _preview_text(text: str, limit: int = 500) -> str:
        text = text.replace("\r", "\\r").replace("\n", "\\n")
        if len(text) <= limit:
            return text
        return f"{text[:limit]}..."

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        if isinstance(response, Mapping):
            value = response.get("response")
            if isinstance(value, str):
                return value

        value = getattr(response, "response", None)
        if isinstance(value, str):
            return value

        raise OllamaResponseFormatError("Ollama response does not contain translated text")

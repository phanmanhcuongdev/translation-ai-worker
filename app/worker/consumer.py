from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import aio_pika
from aio_pika.abc import (
    AbstractIncomingMessage,
    AbstractRobustChannel,
    AbstractRobustConnection,
    AbstractRobustQueue,
)

from app.core.config import Settings
from app.db.session import DatabaseSession
from app.services.ai_service import AITranslationService, OllamaResponseFormatError


logger = logging.getLogger(__name__)


class InvalidMessageError(Exception):
    """Raised for deterministic request payload validation failures."""


class EntityType(StrEnum):
    FEEDBACK = "FEEDBACK"
    QUESTION = "QUESTION"
    SURVEY_DESCRIPTION = "SURVEY_DESCRIPTION"
    SURVEY_RESPONSE = "SURVEY_RESPONSE"
    SURVEY_TITLE = "SURVEY_TITLE"


@dataclass(frozen=True)
class TranslationTask:
    entity_id: int
    entity_type: EntityType
    content: str
    source_lang: str


class TranslationConsumer:
    def __init__(
        self,
        *,
        settings: Settings,
        db_session: DatabaseSession,
        ai_service: AITranslationService,
    ) -> None:
        self._settings = settings
        self._db_session = db_session
        self._ai_service = ai_service
        self._connection: AbstractRobustConnection | None = None
        self._channel: AbstractRobustChannel | None = None
        self._queue: AbstractRobustQueue | None = None
        self._reply_queue: AbstractRobustQueue | None = None
        self._dlq: AbstractRobustQueue | None = None

    @property
    def is_connected(self) -> bool:
        return (
            self._connection is not None
            and not self._connection.is_closed
            and self._channel is not None
            and not self._channel.is_closed
        )

    async def start(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            try:
                await self._connect()
                logger.info(
                    "RabbitMQ consumer started exchange=%s queue=%s routing_key=%s reply_exchange=%s reply_queue=%s reply_routing_key=%s prefetch_count=1",
                    self._settings.rabbitmq_exchange_name,
                    self._settings.rabbitmq_queue_name,
                    self._settings.rabbitmq_routing_key,
                    self._settings.rabbitmq_reply_exchange_name,
                    self._settings.rabbitmq_reply_queue_name,
                    self._settings.rabbitmq_reply_routing_key,
                )
                await stop_event.wait()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "RabbitMQ consumer crashed; reconnecting in %d seconds",
                    self._settings.rabbitmq_reconnect_seconds,
                )
                await self.close()
                try:
                    await asyncio.wait_for(
                        stop_event.wait(),
                        timeout=self._settings.rabbitmq_reconnect_seconds,
                    )
                except TimeoutError:
                    continue

    async def close(self) -> None:
        if self._channel is not None and not self._channel.is_closed:
            await self._channel.close()
        if self._connection is not None and not self._connection.is_closed:
            await self._connection.close()

        self._queue = None
        self._reply_queue = None
        self._dlq = None
        self._channel = None
        self._connection = None

    async def _connect(self) -> None:
        await self.close()
        self._connection = await aio_pika.connect_robust(self._settings.rabbitmq_url)
        self._channel = await self._connection.channel()

        # Required for NVIDIA 1050 Ti 4GB VRAM: process only one GPU task at a time.
        await self._channel.set_qos(prefetch_count=1)

        request_exchange = await self._channel.declare_exchange(
            self._settings.rabbitmq_exchange_name,
            aio_pika.ExchangeType.DIRECT,
            durable=True,
        )
        self._queue = await self._channel.declare_queue(
            self._settings.rabbitmq_queue_name,
            durable=True,
        )
        await self._queue.bind(
            request_exchange,
            routing_key=self._settings.rabbitmq_routing_key,
        )
        reply_exchange = await self._channel.declare_exchange(
            self._settings.rabbitmq_reply_exchange_name,
            aio_pika.ExchangeType.DIRECT,
            durable=True,
        )
        reply_dlx = await self._channel.declare_exchange(
            self._settings.rabbitmq_reply_dlx_name,
            aio_pika.ExchangeType.DIRECT,
            durable=True,
        )
        reply_dlq = await self._channel.declare_queue(
            self._settings.rabbitmq_reply_dlq_name,
            durable=True,
        )
        await reply_dlq.bind(
            reply_dlx,
            routing_key=self._settings.rabbitmq_reply_dlq_name,
        )
        self._reply_queue = await self._channel.declare_queue(
            self._settings.rabbitmq_reply_queue_name,
            durable=True,
        )
        await self._reply_queue.bind(
            reply_exchange,
            routing_key=self._settings.rabbitmq_reply_routing_key,
        )
        self._dlq = await self._channel.declare_queue(
            self._settings.rabbitmq_dlq_name,
            durable=True,
        )
        await self._queue.consume(self._handle_message)

    async def _handle_message(self, message: AbstractIncomingMessage) -> None:
        task_started_at = time.perf_counter()
        delivery_tag = message.delivery_tag
        retry_count = self._get_retry_count(message)

        try:
            payload = self._decode_message(message.body)
            task = self._extract_task(payload)
            self._validate_content_size(task.content)
        except InvalidMessageError as exc:
            logger.exception(
                "Rejecting invalid RabbitMQ message delivery_tag=%s retry_count=%d",
                delivery_tag,
                retry_count,
            )
            await self._publish_dead_letter(
                message=message,
                task=None,
                reason=str(exc),
                retry_count=retry_count,
            )
            await message.ack()
            logger.info(
                "ACK invalid message after DLQ publish delivery_tag=%s retry_count=%d",
                delivery_tag,
                retry_count,
            )
            return

        hash_key = self._hash_content(task.content)

        try:
            logger.info(
                "Translation task started delivery_tag=%s entity_id=%s entity_type=%s hash_key=%s content_chars=%d",
                delivery_tag,
                task.entity_id,
                task.entity_type.value,
                hash_key,
                len(task.content),
            )

            cached_translation = await self._db_session.get_cached_translation_bundle(
                hash_key=hash_key,
                model_info=self._settings.ollama_model,
            )
            if cached_translation is not None:
                await self._publish_reply(
                    task=task,
                    translated_content_vi=cached_translation["translated_content_vi"],
                    translated_content_en=cached_translation["translated_content_en"],
                    source_lang=cached_translation["source_lang"],
                )
                total_latency_ms = (time.perf_counter() - task_started_at) * 1000
                logger.info(
                    "Cache hit delivery_tag=%s entity_id=%s entity_type=%s hash_key=%s total_latency_ms=%.2f chars_vi=%d chars_en=%d",
                    delivery_tag,
                    task.entity_id,
                    task.entity_type.value,
                    hash_key,
                    total_latency_ms,
                    len(cached_translation["translated_content_vi"]),
                    len(cached_translation["translated_content_en"]),
                )
                await message.ack()
                logger.info(
                    "ACK cache-hit message delivery_tag=%s entity_id=%s entity_type=%s",
                    delivery_tag,
                    task.entity_id,
                    task.entity_type.value,
                )
                return

            logger.info("Cache miss delivery_tag=%s hash_key=%s", delivery_tag, hash_key)
            async with self._db_session.gpu_advisory_lock(
                self._settings.gpu_advisory_lock_key
            ) as locked_connection:
                cached_translation = await self._db_session.get_cached_translation_bundle(
                    hash_key=hash_key,
                    model_info=self._settings.ollama_model,
                    connection=locked_connection,
                )
                if cached_translation is not None:
                    await self._publish_reply(
                        task=task,
                        translated_content_vi=cached_translation["translated_content_vi"],
                        translated_content_en=cached_translation["translated_content_en"],
                        source_lang=cached_translation["source_lang"],
                    )
                    total_latency_ms = (time.perf_counter() - task_started_at) * 1000
                    logger.info(
                        "Cache filled while waiting for GPU lock delivery_tag=%s entity_id=%s entity_type=%s hash_key=%s total_latency_ms=%.2f chars_vi=%d chars_en=%d",
                        delivery_tag,
                        task.entity_id,
                        task.entity_type.value,
                        hash_key,
                        total_latency_ms,
                        len(cached_translation["translated_content_vi"]),
                        len(cached_translation["translated_content_en"]),
                    )
                    await message.ack()
                    logger.info(
                        "ACK cache-filled message delivery_tag=%s entity_id=%s entity_type=%s",
                        delivery_tag,
                        task.entity_id,
                        task.entity_type.value,
                    )
                    return

                translation = await self._ai_service.translate(
                    content=task.content,
                    source_lang=task.source_lang,
                )

                await self._db_session.save_translation_bundle(
                    hash_key=hash_key,
                    original_text=task.content,
                    translated_content_vi=translation.translated_content_vi,
                    translated_content_en=translation.translated_content_en,
                    source_lang=translation.source_lang,
                    model_info=self._settings.ollama_model,
                    connection=locked_connection,
                )
                await self._publish_reply(
                    task=task,
                    translated_content_vi=translation.translated_content_vi,
                    translated_content_en=translation.translated_content_en,
                    source_lang=translation.source_lang,
                )

                total_latency_ms = (time.perf_counter() - task_started_at) * 1000
                logger.info(
                    "Translation task completed delivery_tag=%s entity_id=%s entity_type=%s hash_key=%s source_lang=%s gpu_latency_ms=%.2f total_latency_ms=%.2f",
                    delivery_tag,
                    task.entity_id,
                    task.entity_type.value,
                    hash_key,
                    translation.source_lang,
                    translation.gpu_latency_ms,
                    total_latency_ms,
                )
            await message.ack()
            logger.info(
                "ACK translated message delivery_tag=%s entity_id=%s entity_type=%s",
                delivery_tag,
                task.entity_id,
                task.entity_type.value,
            )
        except OllamaResponseFormatError as exc:
            logger.exception(
                "Deterministic Ollama response failure delivery_tag=%s entity_id=%s entity_type=%s hash_key=%s retry_count=%d; sending to DLQ",
                delivery_tag,
                task.entity_id,
                task.entity_type.value,
                hash_key,
                retry_count,
            )
            await self._publish_dead_letter(
                message=message,
                task=task,
                reason=str(exc),
                retry_count=retry_count,
            )
            await message.ack()
            logger.info(
                "ACK deterministic failure after DLQ publish delivery_tag=%s entity_id=%s entity_type=%s retry_count=%d",
                delivery_tag,
                task.entity_id,
                task.entity_type.value,
                retry_count,
            )
        except Exception as exc:
            logger.exception(
                "Transient translation task failure delivery_tag=%s entity_id=%s entity_type=%s hash_key=%s retry_count=%d max_retries=%d",
                delivery_tag,
                task.entity_id,
                task.entity_type.value,
                hash_key,
                retry_count,
                self._settings.rabbitmq_max_retries,
            )
            if retry_count >= self._settings.rabbitmq_max_retries:
                logger.error(
                    "Max retries exceeded delivery_tag=%s entity_id=%s entity_type=%s hash_key=%s; sending to DLQ",
                    delivery_tag,
                    task.entity_id,
                    task.entity_type.value,
                    hash_key,
                )
                await self._publish_dead_letter(
                    message=message,
                    task=task,
                    reason=str(exc),
                    retry_count=retry_count,
                )
                await message.ack()
                logger.info(
                    "ACK max-retry failure after DLQ publish delivery_tag=%s entity_id=%s entity_type=%s retry_count=%d",
                    delivery_tag,
                    task.entity_id,
                    task.entity_type.value,
                    retry_count,
                )
                return

            await self._republish_for_retry(
                message=message,
                retry_count=retry_count + 1,
            )
            await message.ack()
            logger.info(
                "ACK transient failure after retry republish delivery_tag=%s entity_id=%s entity_type=%s retry_count=%d next_retry_count=%d",
                delivery_tag,
                task.entity_id,
                task.entity_type.value,
                retry_count,
                retry_count + 1,
            )

    async def _publish_reply(
        self,
        *,
        task: TranslationTask,
        translated_content_vi: str,
        translated_content_en: str,
        source_lang: str,
    ) -> None:
        if self._channel is None:
            raise RuntimeError("RabbitMQ channel has not been initialized")

        payload = {
            "entity_id": task.entity_id,
            "entity_type": task.entity_type.value,
            "translated_content_vi": translated_content_vi,
            "translated_content_en": translated_content_en,
            "source_lang": source_lang,
            "model_info": self._settings.ollama_model,
            "is_auto_translated": True,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        exchange = await self._channel.declare_exchange(
            self._settings.rabbitmq_reply_exchange_name,
            aio_pika.ExchangeType.DIRECT,
            durable=True,
        )
        await exchange.publish(
            aio_pika.Message(
                body=body,
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key=self._settings.rabbitmq_reply_routing_key,
        )
        logger.info(
            "\u0110\u00e3 tr\u1ea3 k\u1ebft qu\u1ea3 d\u1ecbch cho Entity [%s] v\u00e0o Reply Exchange entity_type=%s source_lang=%s",
            task.entity_id,
            task.entity_type.value,
            source_lang,
        )

    async def _republish_for_retry(
        self,
        *,
        message: AbstractIncomingMessage,
        retry_count: int,
    ) -> None:
        if self._channel is None:
            raise RuntimeError("RabbitMQ channel has not been initialized")

        headers = dict(message.headers or {})
        headers["x-retry-count"] = retry_count

        await self._channel.default_exchange.publish(
            aio_pika.Message(
                body=message.body,
                content_type=message.content_type or "application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                headers=headers,
                correlation_id=message.correlation_id,
            ),
            routing_key=self._settings.rabbitmq_queue_name,
        )
        logger.warning(
            "Republished translation task for retry delivery_tag=%s retry_count=%d max_retries=%d",
            message.delivery_tag,
            retry_count,
            self._settings.rabbitmq_max_retries,
        )

    async def _publish_dead_letter(
        self,
        *,
        message: AbstractIncomingMessage,
        task: TranslationTask | None,
        reason: str,
        retry_count: int,
    ) -> None:
        if self._channel is None:
            raise RuntimeError("RabbitMQ channel has not been initialized")

        payload: dict[str, Any] = {
            "reason": reason,
            "retry_count": retry_count,
            "original_payload": self._safe_decode_body(message.body),
        }
        if task is not None:
            payload.update(
                {
                    "entity_id": task.entity_id,
                    "entity_type": task.entity_type.value,
                }
            )

        await self._channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                headers={
                    "x-original-queue": self._settings.rabbitmq_queue_name,
                    "x-retry-count": retry_count,
                },
                correlation_id=message.correlation_id,
            ),
            routing_key=self._settings.rabbitmq_dlq_name,
        )
        logger.error(
            "Published failed translation task to DLQ=%s delivery_tag=%s retry_count=%d reason=%s",
            self._settings.rabbitmq_dlq_name,
            message.delivery_tag,
            retry_count,
            reason,
        )

    @staticmethod
    def _decode_message(body: bytes) -> dict[str, Any]:
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise InvalidMessageError("RabbitMQ message body must be valid JSON") from exc

        if not isinstance(payload, dict):
            raise InvalidMessageError("RabbitMQ message body must be a JSON object")
        return payload

    @staticmethod
    def _get_retry_count(message: AbstractIncomingMessage) -> int:
        headers = message.headers or {}
        value = headers.get("x-retry-count", 0)
        if isinstance(value, int) and value >= 0:
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return 0

    @staticmethod
    def _safe_decode_body(body: bytes) -> Any:
        try:
            return json.loads(body.decode("utf-8"))
        except Exception:
            return body.decode("utf-8", errors="replace")

    def _extract_task(self, payload: dict[str, Any]) -> TranslationTask:
        entity_id = TranslationConsumer._extract_entity_id(payload)
        entity_type = TranslationConsumer._extract_entity_type(payload)
        content = TranslationConsumer._extract_content(payload)
        source_lang = TranslationConsumer._extract_source_lang(payload)

        return TranslationTask(
            entity_id=entity_id,
            entity_type=entity_type,
            content=content,
            source_lang=source_lang,
        )

    @staticmethod
    def _extract_entity_id(payload: dict[str, Any]) -> int:
        value = payload.get("entity_id")
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, str) and value.isdigit() and int(value) > 0:
            return int(value)
        raise InvalidMessageError("RabbitMQ message must contain a positive entity_id")

    @staticmethod
    def _extract_entity_type(payload: dict[str, Any]) -> EntityType:
        value = payload.get("entity_type")
        if not isinstance(value, str):
            raise InvalidMessageError("RabbitMQ message must contain string field: entity_type")

        try:
            return EntityType(value.upper())
        except ValueError as exc:
            raise InvalidMessageError(
                "entity_type must be FEEDBACK, QUESTION, SURVEY_DESCRIPTION, SURVEY_RESPONSE, or SURVEY_TITLE"
            ) from exc

    @staticmethod
    def _extract_content(payload: dict[str, Any]) -> str:
        content = payload.get("content")
        if not isinstance(content, str) or not content.strip():
            raise InvalidMessageError("RabbitMQ message must contain a non-empty string field: content")
        return content

    @staticmethod
    def _extract_source_lang(payload: dict[str, Any]) -> str:
        value = payload.get("source_lang", "auto")
        if not isinstance(value, str) or not value.strip():
            raise InvalidMessageError("RabbitMQ message must contain a non-empty string field: source_lang")
        normalized = value.strip().lower().replace("_", "-")
        if normalized == "auto":
            return normalized
        language = TranslationConsumer._normalize_language(normalized)
        if language is None:
            raise InvalidMessageError("source_lang must be auto, vi, or en")
        return language

    @staticmethod
    def _normalize_language(value: str) -> str | None:
        language = value.strip().lower().replace("_", "-")
        if language.startswith("vi"):
            return "vi"
        if language.startswith("en"):
            return "en"
        return None

    def _validate_content_size(self, content: str) -> None:
        if len(content) > self._settings.max_content_chars:
            raise InvalidMessageError(
                "RabbitMQ message content exceeds MAX_CONTENT_CHARS "
                f"limit={self._settings.max_content_chars} actual={len(content)}"
            )

    @staticmethod
    def _hash_content(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

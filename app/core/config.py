from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI-Translation-Worker"
    app_version: str = "1.0.0"
    app_env: str = "local"
    log_level: str = "INFO"

    postgres_dsn: str = Field(..., min_length=1)
    postgres_min_size: int = Field(default=1, ge=1)
    postgres_max_size: int = Field(default=5, ge=1)

    rabbitmq_url: str = Field(..., min_length=1)
    rabbitmq_exchange_name: str = Field(
        default="feedback.exchange",
        validation_alias=AliasChoices("RABBITMQ_EXCHANGE", "RABBITMQ_EXCHANGE_NAME"),
    )
    rabbitmq_queue_name: str = Field(
        default="feedback.translate.queue",
        validation_alias=AliasChoices("RABBITMQ_QUEUE", "RABBITMQ_QUEUE_NAME"),
    )
    rabbitmq_routing_key: str = Field(
        default="feedback.translate.key",
        validation_alias=AliasChoices("RABBITMQ_ROUTING_KEY", "RABBITMQ_ROUTING_KEY_NAME"),
    )
    rabbitmq_reply_queue_name: str = Field(
        default="translation.reply.queue",
        validation_alias=AliasChoices("RABBITMQ_REPLY_QUEUE", "RABBITMQ_REPLY_QUEUE_NAME"),
    )
    rabbitmq_reply_exchange_name: str = Field(
        default="translation.reply.exchange",
        validation_alias=AliasChoices("RABBITMQ_REPLY_EXCHANGE", "RABBITMQ_REPLY_EXCHANGE_NAME"),
    )
    rabbitmq_reply_routing_key: str = Field(
        default="translation.reply.key",
        validation_alias=AliasChoices("RABBITMQ_REPLY_ROUTING_KEY", "RABBITMQ_REPLY_ROUTING_KEY_NAME"),
    )
    rabbitmq_reply_dlx_name: str = Field(
        default="translation.reply.dlx",
        validation_alias=AliasChoices("RABBITMQ_REPLY_DLX", "RABBITMQ_REPLY_DLX_NAME"),
    )
    rabbitmq_reply_dlq_name: str = Field(
        default="translation.reply.dlq",
        validation_alias=AliasChoices("RABBITMQ_REPLY_DLQ", "RABBITMQ_REPLY_DLQ_NAME"),
    )
    rabbitmq_dlq_name: str = Field(
        default="translation.dead-letter.queue",
        validation_alias=AliasChoices("RABBITMQ_DLQ", "RABBITMQ_DLQ_NAME"),
    )
    rabbitmq_reconnect_seconds: int = Field(default=5, ge=1)
    rabbitmq_max_retries: int = Field(default=3, ge=0)

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2:7b"
    translation_target_lang: str = "vi"
    ollama_request_timeout_seconds: int = Field(default=180, ge=1)
    ollama_num_ctx: int = Field(default=2048, ge=256)
    max_content_chars: int = Field(default=8000, ge=1)
    gpu_advisory_lock_key: int = 1050

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()

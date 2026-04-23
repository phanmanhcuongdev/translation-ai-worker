from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import asyncpg

from app.core.config import Settings


logger = logging.getLogger(__name__)


CREATE_TRANSLATION_CACHE_TABLE = """
CREATE TABLE IF NOT EXISTS translation_cache (
    hash_key VARCHAR(64) NOT NULL,
    original_text TEXT NOT NULL,
    translated_text TEXT NOT NULL,
    translated_content_vi TEXT,
    translated_content_en TEXT,
    source_lang VARCHAR(16) NOT NULL DEFAULT 'unknown',
    model_info VARCHAR(50) NOT NULL,
    target_lang VARCHAR(16) NOT NULL DEFAULT 'vi',
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);
"""

DROP_LEGACY_HASH_KEY_PRIMARY_KEY = """
ALTER TABLE translation_cache
DROP CONSTRAINT IF EXISTS translation_cache_pkey;
"""

ADD_SOURCE_LANG_COLUMN = """
ALTER TABLE translation_cache
ADD COLUMN IF NOT EXISTS source_lang VARCHAR(16) NOT NULL DEFAULT 'unknown';
"""

ADD_TRANSLATED_CONTENT_VI_COLUMN = """
ALTER TABLE translation_cache
ADD COLUMN IF NOT EXISTS translated_content_vi TEXT;
"""

ADD_TRANSLATED_CONTENT_EN_COLUMN = """
ALTER TABLE translation_cache
ADD COLUMN IF NOT EXISTS translated_content_en TEXT;
"""

ADD_TARGET_LANG_COLUMN = """
ALTER TABLE translation_cache
ADD COLUMN IF NOT EXISTS target_lang VARCHAR(16) NOT NULL DEFAULT 'vi';
"""

CREATE_HASH_KEY_INDEX = """
CREATE INDEX IF NOT EXISTS idx_translation_cache_hash_key
ON translation_cache (hash_key);
"""

CREATE_CACHE_IDENTITY_INDEX = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_translation_cache_identity
ON translation_cache (hash_key, model_info, target_lang);
"""


class DatabaseSession:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pool: asyncpg.Pool | None = None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PostgreSQL pool has not been initialized")
        return self._pool

    async def connect(self) -> None:
        logger.info("Connecting to PostgreSQL")
        self._pool = await asyncpg.create_pool(
            dsn=self._settings.postgres_dsn,
            min_size=self._settings.postgres_min_size,
            max_size=self._settings.postgres_max_size,
            command_timeout=30,
        )
        logger.info("PostgreSQL connection pool established")

    async def close(self) -> None:
        if self._pool is None:
            return

        logger.info("Closing PostgreSQL connection pool")
        await self._pool.close()
        self._pool = None

    async def run_migrations(self) -> None:
        logger.info("Running PostgreSQL auto-migration")
        async with self.pool.acquire() as connection:
            await connection.execute(CREATE_TRANSLATION_CACHE_TABLE)
            await connection.execute(DROP_LEGACY_HASH_KEY_PRIMARY_KEY)
            await connection.execute(ADD_SOURCE_LANG_COLUMN)
            await connection.execute(ADD_TRANSLATED_CONTENT_VI_COLUMN)
            await connection.execute(ADD_TRANSLATED_CONTENT_EN_COLUMN)
            await connection.execute(ADD_TARGET_LANG_COLUMN)
            await connection.execute(CREATE_HASH_KEY_INDEX)
            await connection.execute(CREATE_CACHE_IDENTITY_INDEX)
        logger.info("PostgreSQL auto-migration completed")

    async def health_check(self) -> bool:
        if self._pool is None:
            return False

        try:
            async with self.pool.acquire() as connection:
                value = await connection.fetchval("SELECT 1;")
            return value == 1
        except Exception:
            logger.exception("PostgreSQL health check failed")
            return False

    @asynccontextmanager
    async def gpu_advisory_lock(self, lock_key: int) -> AsyncIterator[asyncpg.Connection]:
        async with self.pool.acquire() as connection:
            logger.info("Waiting for PostgreSQL GPU advisory lock key=%s", lock_key)
            await connection.execute("SELECT pg_advisory_lock($1);", lock_key, timeout=None)
            logger.info("Acquired PostgreSQL GPU advisory lock key=%s", lock_key)
            try:
                yield connection
            finally:
                await connection.execute("SELECT pg_advisory_unlock($1);", lock_key, timeout=None)
                logger.info("Released PostgreSQL GPU advisory lock key=%s", lock_key)

    async def get_cached_translation_bundle(
        self,
        *,
        hash_key: str,
        model_info: str,
        connection: asyncpg.Connection | None = None,
    ) -> dict[str, str] | None:
        if connection is None:
            async with self.pool.acquire() as pooled_connection:
                return await self.get_cached_translation_bundle(
                    hash_key=hash_key,
                    model_info=model_info,
                    connection=pooled_connection,
                )

        row: asyncpg.Record | None = await connection.fetchrow(
            """
            SELECT original_text, translated_content_vi, translated_content_en, source_lang
            FROM translation_cache
            WHERE hash_key = $1
              AND model_info = $2
              AND target_lang = 'multi';
            """,
            hash_key,
            model_info,
        )

        if row is not None:
            translated_content_vi = row["translated_content_vi"]
            translated_content_en = row["translated_content_en"]
            if isinstance(translated_content_vi, str) and isinstance(translated_content_en, str):
                return {
                    "original_text": str(row["original_text"]),
                    "translated_content_vi": translated_content_vi,
                    "translated_content_en": translated_content_en,
                    "source_lang": str(row["source_lang"]),
                }

        legacy_rows = await connection.fetch(
            """
            SELECT
                original_text,
                translated_text,
                translated_content_vi,
                translated_content_en,
                source_lang,
                target_lang
            FROM translation_cache
            WHERE hash_key = $1
              AND model_info = $2;
            """,
            hash_key,
            model_info,
        )
        return self._hydrate_translation_bundle_from_rows(legacy_rows)

    async def save_translation_bundle(
        self,
        *,
        hash_key: str,
        original_text: str,
        translated_content_vi: str,
        translated_content_en: str,
        source_lang: str,
        model_info: str,
        connection: asyncpg.Connection | None = None,
    ) -> None:
        values: tuple[Any, ...] = (
            hash_key,
            original_text,
            translated_content_vi,
            translated_content_en,
            source_lang,
            model_info,
        )
        if connection is None:
            async with self.pool.acquire() as pooled_connection:
                await self.save_translation_bundle(
                    hash_key=hash_key,
                    original_text=original_text,
                    translated_content_vi=translated_content_vi,
                    translated_content_en=translated_content_en,
                    source_lang=source_lang,
                    model_info=model_info,
                    connection=pooled_connection,
                )
                return

        await connection.execute(
            """
            INSERT INTO translation_cache (
                hash_key,
                original_text,
                translated_text,
                translated_content_vi,
                translated_content_en,
                source_lang,
                model_info,
                target_lang,
                created_at
            )
            VALUES ($1, $2, $3, $3, $4, $5, $6, 'multi', NOW())
            ON CONFLICT (hash_key, model_info, target_lang) DO UPDATE SET
                original_text = EXCLUDED.original_text,
                translated_text = EXCLUDED.translated_text,
                translated_content_vi = EXCLUDED.translated_content_vi,
                translated_content_en = EXCLUDED.translated_content_en,
                source_lang = EXCLUDED.source_lang;
            """,
            *values,
        )

    @staticmethod
    def _hydrate_translation_bundle_from_rows(
        rows: list[asyncpg.Record],
    ) -> dict[str, str] | None:
        if not rows:
            return None

        original_text = str(rows[0]["original_text"])
        source_lang = "unknown"
        translated_content_vi: str | None = None
        translated_content_en: str | None = None

        for row in rows:
            row_source_lang = DatabaseSession._normalize_supported_language(row["source_lang"])
            if row_source_lang is not None:
                source_lang = row_source_lang
                if row_source_lang == "vi":
                    translated_content_vi = original_text
                if row_source_lang == "en":
                    translated_content_en = original_text

            value_vi = row["translated_content_vi"]
            if isinstance(value_vi, str) and value_vi.strip():
                translated_content_vi = value_vi

            value_en = row["translated_content_en"]
            if isinstance(value_en, str) and value_en.strip():
                translated_content_en = value_en

            target_lang = DatabaseSession._normalize_supported_language(row["target_lang"])
            translated_text = row["translated_text"]
            if isinstance(translated_text, str) and translated_text.strip() and target_lang is not None:
                if target_lang == "vi":
                    translated_content_vi = translated_text
                if target_lang == "en":
                    translated_content_en = translated_text

        if translated_content_vi is None or translated_content_en is None:
            return None

        return {
            "original_text": original_text,
            "translated_content_vi": translated_content_vi,
            "translated_content_en": translated_content_en,
            "source_lang": source_lang,
        }

    @staticmethod
    def _normalize_supported_language(value: Any) -> str | None:
        if not isinstance(value, str):
            return None

        language = value.strip().lower().replace("_", "-")
        if language.startswith("vi"):
            return "vi"
        if language.startswith("en"):
            return "en"
        return None

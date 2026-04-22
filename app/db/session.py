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

    async def get_cached_translation(
        self,
        *,
        hash_key: str,
        model_info: str,
        target_lang: str,
        connection: asyncpg.Connection | None = None,
    ) -> dict[str, str] | None:
        if connection is None:
            async with self.pool.acquire() as pooled_connection:
                return await self.get_cached_translation(
                    hash_key=hash_key,
                    model_info=model_info,
                    target_lang=target_lang,
                    connection=pooled_connection,
                )

        row: asyncpg.Record | None = await connection.fetchrow(
            """
            SELECT original_text, translated_text, source_lang, target_lang
            FROM translation_cache
            WHERE hash_key = $1
              AND model_info = $2
              AND target_lang = $3;
            """,
            hash_key,
            model_info,
            target_lang,
        )

        if row is None:
            return None
        return {
            "original_text": str(row["original_text"]),
            "translated_text": str(row["translated_text"]),
            "source_lang": str(row["source_lang"]),
            "target_lang": str(row["target_lang"]),
        }

    async def save_translation(
        self,
        *,
        hash_key: str,
        original_text: str,
        translated_text: str,
        source_lang: str,
        model_info: str,
        target_lang: str,
        connection: asyncpg.Connection | None = None,
    ) -> None:
        values: tuple[Any, ...] = (
            hash_key,
            original_text,
            translated_text,
            source_lang,
            model_info,
            target_lang,
        )
        if connection is None:
            async with self.pool.acquire() as pooled_connection:
                await self.save_translation(
                    hash_key=hash_key,
                    original_text=original_text,
                    translated_text=translated_text,
                    source_lang=source_lang,
                    model_info=model_info,
                    target_lang=target_lang,
                    connection=pooled_connection,
                )
                return

        await connection.execute(
            """
            INSERT INTO translation_cache (
                hash_key,
                original_text,
                translated_text,
                source_lang,
                model_info,
                target_lang,
                created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, NOW())
            ON CONFLICT (hash_key, model_info, target_lang) DO UPDATE SET
                original_text = EXCLUDED.original_text,
                translated_text = EXCLUDED.translated_text,
                source_lang = EXCLUDED.source_lang;
            """,
            *values,
        )

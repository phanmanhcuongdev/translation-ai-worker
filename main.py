from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Response, status

from app.core.config import Settings, get_settings
from app.db.session import DatabaseSession
from app.services.ai_service import AITranslationService
from app.worker.consumer import TranslationConsumer


def configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


settings = get_settings()
configure_logging(settings)
logger = logging.getLogger(__name__)

db_session = DatabaseSession(settings=settings)
ai_service = AITranslationService(settings=settings)
consumer = TranslationConsumer(settings=settings, db_session=db_session, ai_service=ai_service)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    stop_event = asyncio.Event()
    consumer_task: asyncio.Task[None] | None = None

    logger.info("Starting AI Translation Worker service")
    await db_session.connect()
    await db_session.run_migrations()

    consumer_task = asyncio.create_task(consumer.start(stop_event), name="rabbitmq-consumer")
    app.state.stop_event = stop_event
    app.state.consumer_task = consumer_task

    try:
        yield
    finally:
        logger.info("Stopping AI Translation Worker service")
        stop_event.set()

        if consumer_task is not None:
            consumer_task.cancel()
            try:
                await consumer_task
            except asyncio.CancelledError:
                logger.info("RabbitMQ consumer task cancelled")

        await consumer.close()
        await ai_service.close()
        await db_session.close()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)


@app.get("/health")
async def health(response: Response) -> dict[str, Any]:
    postgres_ok = await db_session.health_check()
    rabbitmq_ok = consumer.is_connected
    healthy = postgres_ok and rabbitmq_ok

    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ok" if healthy else "degraded",
        "postgres": "ok" if postgres_ok else "unavailable",
        "rabbitmq": "ok" if rabbitmq_ok else "unavailable",
        "request_queue": settings.rabbitmq_queue_name,
        "reply_queue": settings.rabbitmq_reply_queue_name,
        "dlq": settings.rabbitmq_dlq_name,
        "model": settings.ollama_model,
        "reply_languages": ["vi", "en"],
        "ollama_num_ctx": settings.ollama_num_ctx,
        "max_content_chars": settings.max_content_chars,
        "gpu_advisory_lock_key": settings.gpu_advisory_lock_key,
        "recommended_replicas": 1,
    }

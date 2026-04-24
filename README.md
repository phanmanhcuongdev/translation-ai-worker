# AI Translation Worker

RabbitMQ worker for `student-feedback-system` integration. The service consumes translation requests and replies with both Vietnamese and English content for the same entity.

## Request Contract

Request body must be a JSON object:

```json
{
  "entity_id": 123,
  "entity_type": "FEEDBACK",
  "content": "Nội dung cần dịch",
  "source_lang": "auto"
}
```

Supported fields:

- `entity_id`: required, positive integer
- `entity_type`: required, `FEEDBACK`, `QUESTION`, `SURVEY_DESCRIPTION`, or `SURVEY_TITLE`
- `content`: required, non-empty string
- `source_lang`: optional, `auto`, `vi`, or `en`; defaults to `auto`

Behavior:

- `source_lang = "auto"`: the worker asks the model to detect whether the source is `vi` or `en`
- `source_lang = "vi"` or `source_lang = "en"`: the worker treats the provided source language as authoritative
- `target_lang` is no longer required and is ignored if legacy clients still send it

## Reply Contract

Reply body is always a JSON object with bilingual output:

```json
{
  "entity_id": 123,
  "entity_type": "FEEDBACK",
  "translated_content_vi": "Noi dung da dich sang tieng Viet",
  "translated_content_en": "Translated English content",
  "source_lang": "vi",
  "model_info": "qwen2:7b",
  "is_auto_translated": true
}
```

Business rules:

- every reply contains both `translated_content_vi` and `translated_content_en`
- if the original content is Vietnamese, `translated_content_vi` contains the original meaning in Vietnamese and `translated_content_en` contains the English version
- if the original content is English, `translated_content_en` contains the original meaning in English and `translated_content_vi` contains the Vietnamese version
- reply generation does not depend on UI language, browser language, or a requested target language

## RabbitMQ Topology

- Request exchange: `feedback.exchange`
- Request queue: `feedback.translate.queue`
- Request routing key: `feedback.translate.key`
- Reply exchange: `translation.reply.exchange`
- Reply queue: `translation.reply.queue`
- Reply routing key: `translation.reply.key`
- Dead-letter queue: `translation.dead-letter.queue`

## Compatibility

Compatibility retained:

- legacy request producers may still send `target_lang`, `interface_lang`, or `browser_lang`; the worker ignores them
- legacy request producers may omit `source_lang`; the worker defaults to `auto`

Compatibility removed:

- reply payload no longer contains `translated_content`, `target_lang`, or `original_content`
- consumers that expect the old one-target reply contract must be updated to read `translated_content_vi` and `translated_content_en`

## Cache Behavior

The worker now caches bilingual output per `hash_key + model_info`. New rows are stored internally with `target_lang = "multi"` while legacy cache rows can still be read when they contain enough information to reconstruct both language versions.

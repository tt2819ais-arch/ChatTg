"""Обёртка над OpenRouter (OpenAI-совместимый API) с ретраями/бэкоффом.

Один асинхронный клиент на все модели — модель выбирается на каждый запрос.
Бесплатные модели OpenRouter часто упираются в rate-limit/таймауты, поэтому
здесь есть мягкие повторы; при неудаче возвращаем None — оркестратор просто
пропустит этот ход, а не упадёт.
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Optional

from openai import AsyncOpenAI

log = logging.getLogger("llm")


class LLM:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        http_referer: str = "",
        app_title: str = "",
        max_retries: int = 3,
        request_timeout: float = 60.0,
    ):
        # OpenRouter любит заголовки HTTP-Referer / X-Title для бесплатных моделей.
        default_headers = {}
        if http_referer:
            default_headers["HTTP-Referer"] = http_referer
        if app_title:
            default_headers["X-Title"] = app_title

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers or None,
            timeout=request_timeout,
        )
        self.max_retries = max_retries

    async def chat(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.8,
        max_tokens: int = 700,
    ) -> Optional[str]:
        """Один запрос к модели. Возвращает текст или None при стойкой ошибке."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        delay = 2.0
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if not resp.choices:
                    raise RuntimeError("пустой ответ модели (нет choices)")
                content = (resp.choices[0].message.content or "").strip()
                if not content:
                    raise RuntimeError("модель вернула пустой текст")
                return content
            except Exception as e:  # noqa: BLE001 — для прототипа ловим всё
                msg = str(e)
                is_rate = "429" in msg or "rate" in msg.lower()
                log.warning(
                    "LLM ошибка (model=%s, попытка %d/%d): %s",
                    model,
                    attempt,
                    self.max_retries,
                    msg[:200],
                )
                if attempt >= self.max_retries:
                    return None
                # при rate-limit ждём дольше
                await asyncio.sleep(delay * (2 if is_rate else 1))
                delay *= 2
        return None

    async def stream(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.8,
        max_tokens: int = 700,
    ) -> AsyncIterator[str]:
        """Стримит ответ модели по кусочкам (для анимации «печатает»).

        Ретраи только на этапе установления соединения. Если поток оборвался
        посреди — отдаём то, что успели. Если так и не удалось — не отдаём ничего
        (оркестратор воспримет пустой ответ как пропуск хода).
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        delay = 2.0
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                async for chunk in resp:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    piece = getattr(delta, "content", None)
                    if piece:
                        yield piece
                return
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                is_rate = "429" in msg or "rate" in msg.lower()
                log.warning(
                    "LLM stream ошибка (model=%s, попытка %d/%d): %s",
                    model,
                    attempt,
                    self.max_retries,
                    msg[:200],
                )
                if attempt >= self.max_retries:
                    return
                await asyncio.sleep(delay * (2 if is_rate else 1))
                delay *= 2

    async def aclose(self) -> None:
        try:
            await self.client.close()
        except Exception:  # noqa: BLE001
            pass

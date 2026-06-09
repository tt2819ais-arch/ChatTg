"""Сборка системных промптов: персона бота + правила + приоритет босса."""
from __future__ import annotations

import json
from typing import List

from config import AppConfig, BotConfig

# Транскрипт — список сообщений вида {"name": ..., "text": ..., "is_boss": bool}


def _rules(cfg: AppConfig) -> str:
    return (
        f"Язык общения — {cfg.language}.\n"
        f"ГЛАВНОЕ ПРАВИЛО ПРИОРИТЕТА: Босс — @{cfg.boss_username} "
        f"(telegram id {cfg.boss_id}). Его указания и ответы — закон. "
        "Любая его реплика важнее мнения любого бота. Если он что-то решил или "
        "уточнил — немедленно подстраивайся под это, не спорь с его решением.\n"
        "Вы — команда ИИ, которая в общем групповом чате штурмует идею Босса, "
        "пока не доведёте её до отличного, выполнимого результата.\n"
        f"Пиши КРАТКО — один абзац, максимум ~{cfg.max_message_chars} символов. "
        "Без воды и повторов. Не пересказывай чужие реплики — добавляй своё.\n"
        "Не приветствуй каждый раз, не подписывайся (имя и так видно). "
        "Обращайся к коллегам по ролям, к Боссу — на «ты», уважительно.\n"
        "Когда для движения дальше реально НУЖНО решение или факт от Босса — "
        "не выдумывай за него: сформулируй один чёткий вопрос ему."
    )


def build_system_prompt(cfg: AppConfig, bot: BotConfig) -> str:
    team = ", ".join(b.name for b in cfg.bots)
    return (
        f"Тебя зовут «{bot.name}». {bot.persona}\n\n"
        f"Команда в чате: {team}. {_rules(cfg)}"
    )


def render_transcript(transcript: List[dict], cfg: AppConfig, limit: int = 40) -> str:
    """Текстовое представление общего диалога для подачи модели."""
    lines = []
    for m in transcript[-limit:]:
        who = f"БОСС @{cfg.boss_username}" if m.get("is_boss") else m["name"]
        lines.append(f"{who}: {m['text']}")
    return "\n".join(lines) if lines else "(пока пусто)"


def turn_user_prompt(transcript: List[dict], cfg: AppConfig, bot: BotConfig) -> str:
    return (
        "Вот общий диалог брейншторма (по порядку):\n\n"
        f"{render_transcript(transcript, cfg)}\n\n"
        f"Теперь твой ход как «{bot.name}». Дай одну содержательную реплику по делу, "
        "развивая обсуждение. Не повторяй уже сказанное."
    )


# ── Модератор: после каждого круга решает, что делать дальше ──────────────────

MODERATOR_SYSTEM = (
    "Ты — модератор группового брейншторма ИИ-команды. Твоя задача — оценить "
    "диалог и решить, что делать дальше. Отвечай СТРОГО одним JSON-объектом без "
    "пояснений и без markdown, по схеме:\n"
    '{"action": "continue|ask_user|done", "question": "<вопрос Боссу или пусто>", '
    '"summary": "<итог идеи, если done, иначе пусто>"}\n'
    "Правила:\n"
    '- "ask_user": если для продвижения нужно решение/факт/выбор от Босса '
    "(есть развилка, не хватает данных, нужен приоритет). Сформулируй ОДИН "
    "конкретный вопрос.\n"
    '- "done": если идея уже проработана хорошо и выполнима — дай краткий '
    "финальный итог (что за идея, ключевые шаги, на чём сошлись).\n"
    '- "continue": если есть смысл ещё покрутить идею в следующем круге.\n'
    "Не зацикливай обсуждение: если идея уже достаточно ясна — выбирай done."
)


def moderator_user_prompt(transcript: List[dict], cfg: AppConfig, round_no: int) -> str:
    return (
        f"Текущий круг обсуждения: {round_no} из {cfg.max_rounds}.\n\n"
        "Диалог:\n"
        f"{render_transcript(transcript, cfg, limit=60)}\n\n"
        "Верни решение в формате JSON по схеме из системного промпта."
    )


def parse_moderator(raw: str) -> dict:
    """Бережно вытаскивает JSON из ответа модели (бесплатные модели шумят)."""
    fallback = {"action": "continue", "question": "", "summary": ""}
    if not raw:
        return fallback
    text = raw.strip()
    # вырезаем возможные ```json ... ```
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("{") or p.startswith("json"):
                text = p[4:].strip() if p.startswith("json") else p
                break
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return fallback
    try:
        data = json.loads(text[start : end + 1])
    except Exception:  # noqa: BLE001
        return fallback
    action = str(data.get("action", "continue")).lower().strip()
    if action not in ("continue", "ask_user", "done"):
        action = "continue"
    return {
        "action": action,
        "question": str(data.get("question", "") or "").strip(),
        "summary": str(data.get("summary", "") or "").strip(),
    }

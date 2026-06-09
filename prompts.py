"""Сборка системных промптов: персона бота + правила + приоритет босса."""
from __future__ import annotations

import json
from typing import List

from config import AppConfig, BotConfig

# Транскрипт — список сообщений вида {"name": ..., "text": ..., "is_boss": bool}


def _rules(cfg: AppConfig) -> str:
    leader = cfg.leader_bot.name
    return (
        f"Язык — {cfg.language}.\n"
        f"ГЛАВНОЕ: Босс — @{cfg.boss_username} (id {cfg.boss_id}). Его слово — "
        "закон, важнее любого бота и важнее любого голосования. Сказал — делаем, "
        "не спорим.\n"
        "Вы — команда из четырёх ребят, которые вместе придумывают и докручивают "
        f"идею Босса. Старший в команде — {leader}: когда Босс говорит «погнали/"
        "делаем», он раздаёт задачи, а спорные вопросы вы решаете голосованием "
        "(где больше голосов — то и делаем).\n"
        "ЦЕЛЬ РАБОТЫ: вы НЕ пишете код. Итог обсуждения — это (1) очень подробный "
        "ПРОМПТ для нейросети, которая потом сделает проект, и (2) ТЗ (техническое "
        "задание) по идее Босса. Всё обсуждение ведите к этим двум документам.\n\n"
        "КАК ПИСАТЬ (это важно):\n"
        "- Говори ПРОСТО и по-человечески, как в обычном чате с друзьями. "
        "Живой разговорный язык, можно лёгкий юмор.\n"
        "- БЕЗ умных слов, канцелярита и заумных терминов. Если можно сказать "
        "проще — скажи проще.\n"
        "- КОРОТКО: 1-3 предложения, не лекция. Одна мысль за ход.\n"
        "- Не повторяй то, что уже сказали — добавляй своё.\n"
        "- Не здоровайся каждый раз и не подписывайся (имя и так видно). К Боссу — "
        "на «ты».\n"
        "- Если реально нужно решение Босса — просто задай ему один короткий вопрос."
    )


def build_system_prompt(cfg: AppConfig, bot: BotConfig) -> str:
    team = ", ".join(b.name for b in cfg.bots)
    leader = cfg.leader_bot.name
    role_note = ""
    if bot.name == leader:
        role_note = (
            f"\n\nТы здесь СТАРШИЙ (командир). Когда Босс даёт отмашку — ты коротко "
            "ставишь каждому конкретную задачу и держишь команду в фокусе. При "
            "ничьей в голосовании финальное слово за тобой."
        )
    return (
        f"Тебя зовут «{bot.name}». {bot.persona}\n\n"
        f"Команда в чате: {team}. {_rules(cfg)}{role_note}"
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
        f"Теперь твой ход как «{bot.name}». Скажи одну короткую живую реплику по делу, "
        "просто и по-человечески. Не повторяй уже сказанное."
    )


# ── Лидер раздаёт задачи (после команды «погнали» от Босса) ──────────────────


def leader_command_prompt(transcript: List[dict], cfg: AppConfig) -> str:
    others = [b.name for b in cfg.bots if b.name != cfg.leader_bot.name]
    return (
        "Босс дал отмашку — пора работать. Ниже весь диалог:\n\n"
        f"{render_transcript(transcript, cfg)}\n\n"
        f"Ты — старший ({cfg.leader_bot.name}). Коротко и по-простому раздай задачи "
        f"команде: {', '.join(others)}. По одному конкретному заданию каждому "
        "(кто что сейчас прорабатывает), исходя из идеи Босса. Без воды, 2-4 строки "
        "всего, живым языком. Можешь начать со слов «Так, погнали:» или похоже."
    )


# ── Голосование: формулируем варианты, потом каждый голосует ──────────────────

VOTE_OPTIONS_SYSTEM = (
    "Ты — помощник-модератор брейншторма. По диалогу команды нужно вычленить ОДИН "
    "главный спорный вопрос/развилку и 2-4 конкретных варианта ответа на него, "
    "между которыми команда будет голосовать. Отвечай СТРОГО одним JSON без "
    "markdown по схеме:\n"
    '{"question": "<короткий вопрос>", "options": ["вариант 1", "вариант 2", ...]}\n'
    "Варианты — короткие (до ~8 слов), взаимоисключающие, по сути обсуждения. "
    "Если реальной развилки нет (все согласны) — верни options: []."
)


def vote_options_user_prompt(transcript: List[dict], cfg: AppConfig) -> str:
    return (
        "Диалог команды:\n\n"
        f"{render_transcript(transcript, cfg, limit=60)}\n\n"
        "Сформулируй главный вопрос и варианты для голосования (JSON по схеме)."
    )


def parse_vote_options(raw: str) -> dict:
    fallback = {"question": "", "options": []}
    if not raw:
        return fallback
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return fallback
    try:
        data = json.loads(text[start : end + 1])
    except Exception:  # noqa: BLE001
        return fallback
    opts = data.get("options") or []
    opts = [str(o).strip() for o in opts if str(o).strip()][:4]
    return {"question": str(data.get("question", "") or "").strip(), "options": opts}


def vote_prompt(
    transcript: List[dict], cfg: AppConfig, bot: BotConfig, question: str, options: List[str]
) -> str:
    numbered = "\n".join(f"{i + 1}. {o}" for i, o in enumerate(options))
    return (
        "Диалог команды:\n\n"
        f"{render_transcript(transcript, cfg, limit=40)}\n\n"
        f"ГОЛОСОВАНИЕ. Вопрос: {question}\nВарианты:\n{numbered}\n\n"
        f"Ты — «{bot.name}». Выбери ОДИН вариант (номер) и в одной короткой фразе "
        "скажи почему. Формат ответа строго так:\n"
        "ГОЛОС: <номер>\nПОЧЕМУ: <короткая причина>"
    )


def parse_vote(raw: str, num_options: int) -> int:
    """Возвращает индекс варианта (0-based) или -1, если не распознано."""
    if not raw:
        return -1
    import re

    m = re.search(r"ГОЛОС\s*:?[\s]*([0-9]+)", raw, re.IGNORECASE)
    if not m:
        m = re.search(r"\b([0-9]+)\b", raw)
    if not m:
        return -1
    idx = int(m.group(1)) - 1
    return idx if 0 <= idx < num_options else -1


def vote_reason(raw: str) -> str:
    if not raw:
        return ""
    import re

    m = re.search(r"ПОЧЕМУ\s*:?[\s]*(.+)", raw, re.IGNORECASE | re.DOTALL)
    return (m.group(1).strip().splitlines()[0][:140]) if m else ""


# ── Финальный результат: подробный промпт + ТЗ (собирает лидер) ───────────────

FINAL_DELIVERABLE_SYSTEM = (
    "Ты — старший команды. На основе всего обсуждения и итогов голосования собери "
    "ФИНАЛЬНЫЙ результат по идее Босса. Код НЕ пишешь. Нужно выдать ДВА блока, "
    "строго в таком формате и порядке (используй именно эти заголовки):\n\n"
    "🎯 ПОДРОБНЫЙ ПРОМПТ\n"
    "<очень детальный промпт для нейросети, которая будет делать проект: что "
    "построить, для кого, ключевые функции, стек/ограничения, стиль, что на "
    "выходе. Конкретно, без воды, можно по пунктам.>\n\n"
    "📋 ТЗ (техническое задание)\n"
    "<техзадание по пунктам: 1) Цель и задача, 2) Пользователи, 3) Функции/"
    "экраны, 4) Технические требования, 5) Этапы/результат. Кратко и по делу.>\n\n"
    "Пиши простым языком. Учитывай, что выбрала команда голосованием и что сказал "
    "Босс — его решения приоритетны."
)


def final_deliverable_prompt(transcript: List[dict], cfg: AppConfig) -> str:
    return (
        "Вот всё обсуждение команды (включая итоги голосования и реплики Босса):\n\n"
        f"{render_transcript(transcript, cfg, limit=80)}\n\n"
        "Собери финальный результат: подробный промпт + ТЗ (по формату из системного "
        "промпта). Это итог работы — без кода."
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

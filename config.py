"""Загрузка и валидация конфигурации (config.yaml + .env)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
PLACEHOLDER_MARKERS = ("PASTE_", "xxxxxxxx", "PASTE_TELEGRAM_BOT_TOKEN")


@dataclass
class BotConfig:
    name: str
    telegram_token: str
    model: str
    persona: str
    listener: bool = False


@dataclass
class AppConfig:
    api_key: str
    base_url: str
    http_referer: str
    app_title: str
    boss_username: str
    boss_id: int
    max_rounds: int
    delay_seconds: float
    max_message_chars: int
    language: str
    moderator_model: str
    typing: bool = True
    stream: bool = True
    edit_interval_seconds: float = 1.2
    react_on_seen: bool = True
    seen_emoji: str = "👀"
    bots: list[BotConfig] = field(default_factory=list)

    @property
    def listener_bot(self) -> BotConfig:
        for b in self.bots:
            if b.listener:
                return b
        # если флаг listener не выставлен — слушает первый бот
        return self.bots[0]


def _looks_like_placeholder(value: str) -> bool:
    return (not value) or any(m in value for m in PLACEHOLDER_MARKERS)


def load_config(path: Optional[str] = None) -> AppConfig:
    """Читает config.yaml (или указанный путь) + .env. Бросает понятную ошибку,
    если что-то не заполнено — сервис не падает молча с placeholder-токенами."""
    load_dotenv(ROOT / ".env")

    cfg_path = Path(path) if path else (ROOT / "config.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Не найден {cfg_path.name}. Скопируй config.example.yaml -> config.yaml "
            "и заполни токены."
        )

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # Секция провайдера: "llm" (нейтральная) ИЛИ "openrouter" (старое имя) — что есть.
    orouter = raw.get("llm", None) or raw.get("openrouter", {}) or {}
    # Приоритет: переменная окружения > значение в yaml.
    # Поддерживаем оба провайдера: OpenRouter и Groq (оба OpenAI-совместимы).
    api_key = (
        os.getenv("OPENROUTER_API_KEY")
        or os.getenv("GROQ_API_KEY")
        or os.getenv("LLM_API_KEY")
        or orouter.get("api_key", "")
        or ""
    )

    boss = raw.get("boss", {}) or {}
    disc = raw.get("discussion", {}) or {}
    moderator = raw.get("moderator", {}) or {}

    bots_raw = raw.get("bots", []) or []
    bots: list[BotConfig] = []
    for b in bots_raw:
        # Токен можно взять из переменной окружения (приоритет — для секретов в
        # CI/хостинге, где .env нет). Имя переменной задаётся явно через token_env
        # (нужно для имён ботов кириллицей — имена env-переменных только латиницей),
        # либо по умолчанию TELEGRAM_TOKEN_<ИМЯ В ВЕРХНЕМ РЕГИСТРЕ>.
        env_name = b.get("token_env") or f"TELEGRAM_TOKEN_{b.get('name','').upper()}"
        token = os.getenv(env_name) or b.get("telegram_token", "")
        bots.append(
            BotConfig(
                name=b.get("name", "Bot"),
                telegram_token=token,
                model=b.get("model", ""),
                persona=(b.get("persona", "") or "").strip(),
                listener=bool(b.get("listener", False)),
            )
        )

    cfg = AppConfig(
        api_key=api_key,
        base_url=orouter.get("base_url", "https://openrouter.ai/api/v1"),
        http_referer=orouter.get("http_referer", ""),
        app_title=orouter.get("app_title", "Telegram Brainstorm Bots"),
        boss_username=str(boss.get("username", "")).lstrip("@"),
        boss_id=int(boss.get("telegram_id", 0) or 0),
        max_rounds=int(disc.get("max_rounds", 3)),
        delay_seconds=float(disc.get("delay_seconds", 3)),
        max_message_chars=int(disc.get("max_message_chars", 900)),
        language=str(disc.get("language", "русский")),
        moderator_model=moderator.get("model", "openai/gpt-oss-120b:free"),
        typing=bool(disc.get("typing", True)),
        stream=bool(disc.get("stream", True)),
        edit_interval_seconds=float(disc.get("edit_interval_seconds", 1.2)),
        react_on_seen=bool(disc.get("react_on_seen", True)),
        seen_emoji=str(disc.get("seen_emoji", "👀")),
        bots=bots,
    )
    return cfg


def validate_config(cfg: AppConfig) -> list[str]:
    """Возвращает список проблем (пустой = всё ок). Сервис покажет их и не стартует."""
    problems: list[str] = []
    if _looks_like_placeholder(cfg.api_key):
        problems.append(
            "API-ключ модели не задан (placeholder). Положи ключ в .env: "
            "OPENROUTER_API_KEY=... (OpenRouter) ИЛИ GROQ_API_KEY=... (Groq)."
        )
    if not cfg.bots:
        problems.append("В config.yaml не описан ни один бот (секция bots).")
    if len(cfg.bots) < 2:
        problems.append("Нужно минимум 2 бота для обсуждения (рекомендуется 3-4).")
    for b in cfg.bots:
        if _looks_like_placeholder(b.telegram_token):
            problems.append(
                f"Бот «{b.name}»: telegram_token не задан (placeholder). "
                "Создай бота у @BotFather и вставь токен."
            )
        if not b.model:
            problems.append(f"Бот «{b.name}»: не указана model.")
    if cfg.boss_id <= 0:
        problems.append("boss.telegram_id не задан — некому отдавать приоритет.")
    return problems

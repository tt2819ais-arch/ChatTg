"""Оркестратор мульти-агентного брейншторма в Telegram.

ВАЖНО про Telegram: бот НЕ получает сообщения, отправленные другими ботами.
Поэтому боты не «слышат» друг друга через Telegram. Решение: один центральный
сервис держит ВСЕ токены, ведёт ОДИН общий транскрипт диалога и сам управляет
очерёдностью — каждый ход генерируется по общему транскрипту + персоне бота и
постится в группу под токеном этого бота.

Достаточно, чтобы сообщения группы слушал ОДИН бот (listener: true). Остальные
боты только отправляют свои реплики — поэтому дублей входящих апдейтов нет.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message

from config import AppConfig, BotConfig, load_config, validate_config
from llm import LLM
from prompts import (
    build_system_prompt,
    moderator_user_prompt,
    parse_moderator,
    turn_user_prompt,
    MODERATOR_SYSTEM,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("orchestrator")


@dataclass
class ChatState:
    transcript: list[dict] = field(default_factory=list)
    status: str = "idle"  # idle | discussing | waiting_user
    round_no: int = 0
    task: asyncio.Task | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class Orchestrator:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.llm = LLM(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            http_referer=cfg.http_referer,
            app_title=cfg.app_title,
        )
        props = DefaultBotProperties(parse_mode=ParseMode.HTML)
        # Bot-инстанс на каждый токен (для отправки реплик).
        self.bots: dict[str, Bot] = {
            b.name: Bot(token=b.telegram_token, default=props) for b in cfg.bots
        }
        self.bot_cfgs: dict[str, BotConfig] = {b.name: b for b in cfg.bots}
        self.listener_cfg = cfg.listener_bot
        self.listener_bot = self.bots[self.listener_cfg.name]
        self.states: dict[int, ChatState] = {}
        self.dp = Dispatcher()
        self._register_handlers()

    # ── состояние чата ────────────────────────────────────────────────────────
    def state(self, chat_id: int) -> ChatState:
        if chat_id not in self.states:
            self.states[chat_id] = ChatState()
        return self.states[chat_id]

    # ── хэндлеры ────────────────────────────────────────────────────────────────
    def _register_handlers(self) -> None:
        dp = self.dp

        @dp.message(Command("start", "help"))
        async def cmd_help(message: Message):
            await self._reply_listener(
                message.chat.id,
                "👋 Привет! Я — команда ИИ-ботов для брейншторма: "
                + ", ".join(self.bots.keys())
                + ".\n\nНапиши свою идею в этот чат — и мы начнём её обсуждать по "
                "кругам, будем задавать тебе уточняющие вопросы и слушаться тебя.\n\n"
                "Команды: /stop — остановить обсуждение, /reset — забыть диалог, "
                "/status — что сейчас происходит.",
            )

        @dp.message(Command("stop"))
        async def cmd_stop(message: Message):
            st = self.state(message.chat.id)
            self._cancel_task(st)
            st.status = "idle"
            await self._reply_listener(message.chat.id, "⏹ Обсуждение остановлено.")

        @dp.message(Command("reset"))
        async def cmd_reset(message: Message):
            st = self.state(message.chat.id)
            self._cancel_task(st)
            st.transcript.clear()
            st.status = "idle"
            st.round_no = 0
            await self._reply_listener(
                message.chat.id, "🧹 Память диалога очищена. Можешь дать новую идею."
            )

        @dp.message(Command("status"))
        async def cmd_status(message: Message):
            st = self.state(message.chat.id)
            await self._reply_listener(
                message.chat.id,
                f"Статус: <b>{st.status}</b>, круг {st.round_no}/{self.cfg.max_rounds}, "
                f"реплик в памяти: {len(st.transcript)}.",
            )

        @dp.message(F.text & ~F.via_bot)
        async def on_text(message: Message):
            # Игнорируем сообщения от ботов (на всякий случай).
            if message.from_user and message.from_user.is_bot:
                return
            await self._on_human_message(message)

    # ── приём сообщения человека ────────────────────────────────────────────────
    async def _on_human_message(self, message: Message) -> None:
        chat_id = message.chat.id
        st = self.state(chat_id)
        text = (message.text or "").strip()
        if not text:
            return

        uid = message.from_user.id if message.from_user else 0
        is_boss = uid == self.cfg.boss_id
        author = (
            f"@{self.cfg.boss_username}"
            if is_boss
            else (message.from_user.full_name if message.from_user else "Гость")
        )
        st.transcript.append({"name": author, "text": text, "is_boss": is_boss})
        log.info("Вход от %s (boss=%s): %s", author, is_boss, text[:80])

        # Если идёт обсуждение — реплика просто учтётся в транскрипте следующим ходом.
        if st.status == "discussing":
            return

        # idle или ждали ответа Босса → (пере)запускаем цикл.
        if st.task and not st.task.done():
            return
        st.round_no = 0
        st.task = asyncio.create_task(self._run_discussion(chat_id))

    # ── основной цикл обсуждения ────────────────────────────────────────────────
    async def _run_discussion(self, chat_id: int) -> None:
        st = self.state(chat_id)
        async with st.lock:
            st.status = "discussing"
            try:
                while st.round_no < self.cfg.max_rounds:
                    st.round_no += 1
                    log.info("Чат %s: круг %d", chat_id, st.round_no)

                    for bcfg in self.cfg.bots:
                        await self._one_bot_turn(chat_id, bcfg)
                        await asyncio.sleep(self.cfg.delay_seconds)

                    # Модератор решает, что дальше.
                    decision = await self._moderate(chat_id, st.round_no)
                    action = decision["action"]
                    if action == "ask_user":
                        q = decision["question"] or "Уточни, пожалуйста, детали идеи?"
                        await self._reply_listener(
                            chat_id, f"❓ <b>Вопрос Боссу:</b> {q}"
                        )
                        st.status = "waiting_user"
                        return
                    if action == "done":
                        s = decision["summary"] or "Идея проработана."
                        await self._reply_listener(
                            chat_id, f"✅ <b>Итог обсуждения:</b>\n{s}"
                        )
                        st.status = "idle"
                        return

                # Достигли лимита кругов → подводим итог и спрашиваем Босса.
                await self._reply_listener(
                    chat_id,
                    "⏸ Сделали "
                    f"{self.cfg.max_rounds} кругов. Босс, в какую сторону копаем "
                    "дальше или фиксируем идею? (или /stop)",
                )
                st.status = "waiting_user"
            except asyncio.CancelledError:
                log.info("Чат %s: обсуждение отменено", chat_id)
                raise
            except Exception as e:  # noqa: BLE001
                log.exception("Чат %s: ошибка в цикле: %s", chat_id, e)
                st.status = "idle"

    async def _one_bot_turn(self, chat_id: int, bcfg: BotConfig) -> None:
        st = self.state(chat_id)
        system = build_system_prompt(self.cfg, bcfg)
        user = turn_user_prompt(st.transcript, self.cfg, bcfg)
        reply = await self.llm.chat(
            model=bcfg.model,
            system=system,
            user=user,
            max_tokens=min(900, self.cfg.max_message_chars + 300),
        )
        if not reply:
            log.warning("Бот %s пропустил ход (нет ответа модели)", bcfg.name)
            return
        reply = reply.strip()[: self.cfg.max_message_chars + 200]
        st.transcript.append({"name": bcfg.name, "text": reply, "is_boss": False})
        await self._send_as(bcfg.name, chat_id, reply)

    async def _moderate(self, chat_id: int, round_no: int) -> dict:
        st = self.state(chat_id)
        raw = await self.llm.chat(
            model=self.cfg.moderator_model,
            system=MODERATOR_SYSTEM,
            user=moderator_user_prompt(st.transcript, self.cfg, round_no),
            temperature=0.2,
            max_tokens=500,
        )
        return parse_moderator(raw or "")

    # ── отправка ────────────────────────────────────────────────────────────────
    async def _send_as(self, bot_name: str, chat_id: int, text: str) -> None:
        bot = self.bots.get(bot_name, self.listener_bot)
        try:
            await bot.send_message(chat_id, f"<b>{bot_name}:</b> {text}")
        except Exception as e:  # noqa: BLE001
            log.warning("Не смог отправить от %s: %s", bot_name, e)

    async def _reply_listener(self, chat_id: int, text: str) -> None:
        try:
            await self.listener_bot.send_message(chat_id, text)
        except Exception as e:  # noqa: BLE001
            log.warning("Не смог отправить системное сообщение: %s", e)

    @staticmethod
    def _cancel_task(st: ChatState) -> None:
        if st.task and not st.task.done():
            st.task.cancel()

    # ── запуск ────────────────────────────────────────────────────────────────
    async def run(self) -> None:
        me = await self.listener_bot.get_me()
        log.info("Слушающий бот: @%s (%s)", me.username, self.listener_cfg.name)
        log.info(
            "Боты в команде: %s",
            ", ".join(f"{n}" for n in self.bots.keys()),
        )
        try:
            # Поллим ТОЛЬКО слушающего бота — остальные только отправляют.
            await self.dp.start_polling(self.listener_bot, handle_signals=True)
        finally:
            await self.llm.aclose()
            for b in self.bots.values():
                await b.session.close()


def main() -> None:
    cfg = load_config()
    problems = validate_config(cfg)
    if problems:
        print("\n❌ Конфигурация не готова к запуску:\n")
        for p in problems:
            print(f"  • {p}")
        print(
            "\nЗаполни config.yaml / .env по образцу (config.example.yaml, .env.example) "
            "и запусти снова.\n"
        )
        raise SystemExit(1)

    orch = Orchestrator(cfg)
    try:
        asyncio.run(orch.run())
    except (KeyboardInterrupt, SystemExit):
        log.info("Остановлено пользователем.")


if __name__ == "__main__":
    main()

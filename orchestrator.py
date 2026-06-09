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
from aiogram.types import Message, ReactionTypeEmoji

from config import AppConfig, BotConfig, load_config, validate_config
from llm import LLM, strip_think
from prompts import (
    build_system_prompt,
    moderator_user_prompt,
    parse_moderator,
    turn_user_prompt,
    MODERATOR_SYSTEM,
    leader_command_prompt,
    VOTE_OPTIONS_SYSTEM,
    vote_options_user_prompt,
    parse_vote_options,
    vote_prompt,
    parse_vote,
    vote_reason,
    FINAL_DELIVERABLE_SYSTEM,
    final_deliverable_prompt,
    TOPIC_SHIFT_SYSTEM,
    topic_shift_user_prompt,
    parse_topic_shift,
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
        # Все боты поллят апдейты (чтобы КАЖДЫЙ мог поставить реакцию). Чтобы не
        # обрабатывать одно сообщение 4 раза — дедуп по (chat_id, message_id).
        self.seen_msgs: set[tuple[int, int]] = set()
        self.dp = Dispatcher()
        self._register_handlers()

    def _first_time(self, message: Message) -> bool:
        """True только для первого бота, доставившего это сообщение (дедуп действий)."""
        key = (message.chat.id, message.message_id)
        if key in self.seen_msgs:
            return False
        self.seen_msgs.add(key)
        if len(self.seen_msgs) > 5000:  # не растём бесконечно
            self.seen_msgs = set(list(self.seen_msgs)[-2000:])
        return True

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
            if not self._first_time(message):
                return
            await self._reply_listener(
                message.chat.id,
                "👋 Привет! Я — команда ИИ-ботов для брейншторма: "
                + ", ".join(self.bots.keys())
                + ".\n\n1) Напиши свою идею — мы её пообсуждаем простым языком и "
                "позадаём тебе вопросы.\n"
                f"2) Когда скажешь «погнали» / «делаем» — старший ({self.cfg.leader_bot.name}) "
                "раздаст всем задачи, команда обсудит и проголосует (где больше "
                "голосов — то и делаем).\n"
                "3) На выходе соберём подробный ПРОМПТ и ТЗ по твоей идее (код не "
                "пишем). Твоё слово — закон.\n\n"
                "Сменить тему можно просто новым сообщением (или командой /new) — "
                "старое сразу забудем.\n"
                "Команды: /new — новая тема, /stop — стоп, /reset — забыть всё, "
                "/status — что сейчас.",
            )

        @dp.message(Command("stop"))
        async def cmd_stop(message: Message):
            if not self._first_time(message):
                return
            st = self.state(message.chat.id)
            self._cancel_task(st)
            st.status = "idle"
            await self._reply_listener(message.chat.id, "⏹ Обсуждение остановлено.")

        @dp.message(Command("reset"))
        async def cmd_reset(message: Message):
            if not self._first_time(message):
                return
            st = self.state(message.chat.id)
            self._cancel_task(st)
            st.transcript.clear()
            st.status = "idle"
            st.round_no = 0
            await self._reply_listener(
                message.chat.id, "🧹 Память диалога очищена. Можешь дать новую идею."
            )

        @dp.message(Command("new", "newtopic"))
        async def cmd_new(message: Message):
            if not self._first_time(message):
                return
            st = self.state(message.chat.id)
            self._cancel_task(st)
            st.transcript.clear()
            st.status = "idle"
            st.round_no = 0
            await self._reply_listener(
                message.chat.id,
                "🔄 Окей, забыли прошлое. Кидай новую тему/идею.",
            )

        @dp.message(Command("status"))
        async def cmd_status(message: Message):
            if not self._first_time(message):
                return
            st = self.state(message.chat.id)
            await self._reply_listener(
                message.chat.id,
                f"Статус: <b>{st.status}</b>, круг {st.round_no}/{self.cfg.max_rounds}, "
                f"реплик в памяти: {len(st.transcript)}.",
            )

        @dp.message(F.text & ~F.via_bot)
        async def on_text(message: Message, bot: Bot):
            # Игнорируем сообщения от ботов (на всякий случай).
            if message.from_user and message.from_user.is_bot:
                return
            # Реакция «увидел» — ставит ИМЕННО тот бот, что получил апдейт
            # (только он гарантированно «видит» сообщение в своём API-сеансе).
            if self.cfg.react_on_seen:
                await self._safe(
                    bot.set_message_reaction(
                        message.chat.id,
                        message.message_id,
                        reaction=[ReactionTypeEmoji(emoji=self.cfg.seen_emoji)],
                    )
                )
            # Логику обсуждения запускаем один раз (дедуп по message_id).
            if not self._first_time(message):
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

        # Смена темы: если Босс заговорил про ДРУГОЕ — забываем старое обсуждение,
        # чтобы боты не цеплялись за прошлую идею. (Только для Босса, не для go/ответов.)
        is_go = self._is_go_command(text)
        if is_boss and not is_go and len(st.transcript) >= 2:
            if await self._is_new_topic(chat_id, text):
                self._cancel_task(st)
                st.transcript.clear()
                st.round_no = 0
                st.status = "idle"
                log.info("Чат %s: Босс сменил тему — старый контекст сброшен", chat_id)

        st.transcript.append({"name": author, "text": text, "is_boss": is_boss})
        log.info("Вход от %s (boss=%s): %s", author, is_boss, text[:80])

        # (реакция «увидел» уже поставлена в on_text тем ботом, что получил апдейт)

        # Команда Босса «погнали/делаем» → лидер берёт командование: раздаёт
        # задачи, команда обсуждает, спорное решает голосованием, затем выдаёт
        # подробный промпт + ТЗ. Перебивает обычный брейншторм, если он шёл.
        if is_boss and self.cfg.voting and is_go:
            self._cancel_task(st)
            st.round_no = 0
            st.task = asyncio.create_task(self._run_directed_session(chat_id))
            return

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

    # ── лидер-режим: команда → обсуждение → голосование → промпт+ТЗ ────────────
    def _is_go_command(self, text: str) -> bool:
        import re

        t = text.lower()
        tokens = set(re.findall(r"\w+", t, re.UNICODE))
        for w in self.cfg.go_words:
            if " " in w:
                if w in t:
                    return True
            elif w in tokens:
                return True
        return False

    async def _is_new_topic(self, chat_id: int, text: str) -> bool:
        """True, если Босс сменил тему. Сначала по ключевым словам, потом LLM."""
        if not self.cfg.topic_switch_detect:
            # хотя бы явные маркеры всё равно проверяем
            t = text.lower()
            return any(w in t for w in self.cfg.new_topic_words)
        t = text.lower()
        if any(w in t for w in self.cfg.new_topic_words):
            return True
        # Слишком короткие реплики (ок/да/нет/спасибо) — не новая тема.
        if len(text.strip()) < 12:
            return False
        st = self.state(chat_id)
        prev = st.transcript[:]  # новое сообщение ещё не добавлено
        raw = await self.llm.chat(
            model=self.cfg.moderator_model,
            system=TOPIC_SHIFT_SYSTEM,
            user=topic_shift_user_prompt(prev, text, self.cfg),
            temperature=0.0,
            max_tokens=60,
        )
        return parse_topic_shift(strip_think(raw or ""))

    async def _run_directed_session(self, chat_id: int) -> None:
        st = self.state(chat_id)
        async with st.lock:
            st.status = "discussing"
            leader = self.cfg.leader_bot
            try:
                # 1. Лидер раздаёт задачи команде.
                await self._one_bot_turn(
                    chat_id,
                    leader,
                    user_prompt=leader_command_prompt(st.transcript, self.cfg),
                )
                await asyncio.sleep(self.cfg.delay_seconds)

                # 2. Команда обсуждает (остальные боты по кругам).
                others = [b for b in self.cfg.bots if b.name != leader.name]
                for _ in range(max(1, self.cfg.directed_rounds)):
                    for bcfg in others:
                        await self._one_bot_turn(chat_id, bcfg)
                        await asyncio.sleep(self.cfg.delay_seconds)

                # 3. Голосование по главной развилке.
                await self._run_vote(chat_id)
                await asyncio.sleep(self.cfg.delay_seconds)

                # 4. Финал: лидер собирает подробный промпт + ТЗ.
                await self._final_deliverable(chat_id)
                st.status = "idle"
            except asyncio.CancelledError:
                log.info("Чат %s: направленная сессия отменена", chat_id)
                raise
            except Exception as e:  # noqa: BLE001
                log.exception("Чат %s: ошибка в направленной сессии: %s", chat_id, e)
                st.status = "idle"

    async def _run_vote(self, chat_id: int) -> None:
        st = self.state(chat_id)
        leader = self.cfg.leader_bot
        # 1. Сформулировать вопрос и варианты.
        raw = await self.llm.chat(
            model=self.cfg.moderator_model,
            system=VOTE_OPTIONS_SYSTEM,
            user=vote_options_user_prompt(st.transcript, self.cfg),
            temperature=0.2,
            max_tokens=400,
        )
        vo = parse_vote_options(strip_think(raw or ""))
        options = vo["options"]
        question = vo["question"] or "Какой вариант берём?"
        if len(options) < 2:
            log.info("Чат %s: явной развилки нет — голосование пропущено", chat_id)
            return

        numbered = "\n".join(f"{i + 1}. {o}" for i, o in enumerate(options))
        await self._say_as(
            leader, chat_id, f"🗳 Голосуем: {question}\n{numbered}"
        )
        await asyncio.sleep(self.cfg.delay_seconds)

        # 2. Каждый бот голосует.
        tally = [0] * len(options)
        for bcfg in self.cfg.bots:
            raw = await self.llm.chat(
                model=bcfg.model,
                system=build_system_prompt(self.cfg, bcfg),
                user=vote_prompt(st.transcript, self.cfg, bcfg, question, options),
                temperature=0.4,
                max_tokens=160,
            )
            clean = strip_think(raw or "")
            idx = parse_vote(clean, len(options))
            if idx < 0:
                log.warning("Бот %s: голос не распознан — пропуск", bcfg.name)
                continue
            reason = vote_reason(clean)
            tally[idx] += 1
            txt = f"голосую за «{options[idx]}»" + (f" — {reason}" if reason else "")
            await self._say_as(bcfg, chat_id, txt)
            await asyncio.sleep(self.cfg.delay_seconds)

        # 3. Подвести итог (большинство; ничью решает лидер).
        top = max(tally) if tally else 0
        winners = [i for i, v in enumerate(tally) if v == top and top > 0]
        score = ", ".join(f"{options[i]}: {tally[i]}" for i in range(len(options)))
        if not winners:
            await self._say_as(leader, chat_id, f"Голоса не сложились ({score}). Решаю сам — двигаемся дальше.")
            return
        if len(winners) == 1:
            decision = options[winners[0]]
            await self._say_as(
                leader, chat_id,
                f"✅ Большинством голосов берём: «{decision}» ({score}).",
            )
            return
        # Ничья → лидер выбирает из спорных вариантов.
        tied = [options[i] for i in winners]
        raw = await self.llm.chat(
            model=leader.model,
            system=build_system_prompt(self.cfg, leader),
            user=vote_prompt(
                st.transcript, self.cfg, leader,
                "Ничья в голосовании — реши как старший, что берём", tied,
            ),
            temperature=0.3,
            max_tokens=160,
        )
        clean = strip_think(raw or "")
        idx = parse_vote(clean, len(tied))
        decision = tied[idx if idx >= 0 else 0]
        reason = vote_reason(clean)
        await self._say_as(
            leader, chat_id,
            f"⚖️ Голоса разделились ({score}). Как старший решаю: «{decision}»."
            + (f" {reason}" if reason else ""),
        )

    async def _final_deliverable(self, chat_id: int) -> None:
        st = self.state(chat_id)
        leader = self.cfg.leader_bot
        bot = self.bots.get(leader.name, self.listener_bot)
        typing_task = (
            asyncio.create_task(self._keep_typing(bot, chat_id))
            if self.cfg.typing
            else None
        )
        try:
            raw = await self.llm.chat(
                model=leader.model,
                system=FINAL_DELIVERABLE_SYSTEM,
                user=final_deliverable_prompt(st.transcript, self.cfg),
                temperature=0.5,
                max_tokens=1600,
            )
        finally:
            if typing_task:
                typing_task.cancel()
        text = strip_think(raw or "").strip()
        if not text:
            await self._reply_listener(
                chat_id, "Не получилось собрать финал — Босс, дай ещё вводных?"
            )
            return
        await self._send_long(bot, chat_id, f"📦 ИТОГ — промпт + ТЗ\n\n{text}")
        st.transcript.append(
            {"name": leader.name, "text": text, "is_boss": False}
        )

    async def _say_as(self, bcfg: BotConfig, chat_id: int, text: str) -> None:
        """Отправить реплику от имени конкретного бота и записать в транскрипт."""
        bot = self.bots.get(bcfg.name, self.listener_bot)
        await self._send_plain(bot, chat_id, f"{bcfg.name}: {text}")
        self.state(chat_id).transcript.append(
            {"name": bcfg.name, "text": text, "is_boss": False}
        )

    async def _send_long(self, bot, chat_id: int, text: str) -> None:
        """Telegram лимит ~4096 символов — режем длинный финал на части."""
        chunk = 3800
        for i in range(0, len(text), chunk):
            await self._send_plain(bot, chat_id, text[i : i + chunk])
            await asyncio.sleep(0.5)

    async def _one_bot_turn(
        self,
        chat_id: int,
        bcfg: BotConfig,
        *,
        user_prompt: str | None = None,
        max_chars: int | None = None,
    ) -> str:
        """Ход одного бота. Возвращает текст реплики (или '' если пропустил)."""
        st = self.state(chat_id)
        system = build_system_prompt(self.cfg, bcfg)
        user = user_prompt or turn_user_prompt(st.transcript, self.cfg, bcfg)
        bot = self.bots.get(bcfg.name, self.listener_bot)
        limit_chars = max_chars or self.cfg.max_message_chars
        max_tokens = min(1100, limit_chars + 400)

        # «печатает…» — индикатор живёт ~5с, поэтому держим его в фоне.
        typing_task = (
            asyncio.create_task(self._keep_typing(bot, chat_id))
            if self.cfg.typing
            else None
        )
        try:
            if self.cfg.stream:
                reply = await self._stream_turn(
                    bot, bcfg, chat_id, system, user, max_tokens, limit_chars
                )
            else:
                reply = await self.llm.chat(
                    model=bcfg.model, system=system, user=user, max_tokens=max_tokens
                )
                if reply:
                    reply = strip_think(reply).strip()[: limit_chars + 200]
                    await self._send_plain(bot, chat_id, f"{bcfg.name}: {reply}")
        finally:
            if typing_task:
                typing_task.cancel()

        if not reply:
            log.warning("Бот %s пропустил ход (нет ответа модели)", bcfg.name)
            return ""
        st.transcript.append({"name": bcfg.name, "text": reply, "is_boss": False})
        return reply

    async def _stream_turn(
        self,
        bot,
        bcfg: BotConfig,
        chat_id: int,
        system: str,
        user: str,
        max_tokens: int,
        limit_chars: int,
    ) -> str:
        """Постепенно «печатает» ответ бота через editMessageText. Возвращает текст."""
        import time

        prefix = f"{bcfg.name}: "
        limit = limit_chars + 200
        acc = ""
        msg = None
        last_edit = 0.0
        async for piece in self.llm.stream(
            model=bcfg.model, system=system, user=user, max_tokens=max_tokens
        ):
            acc += piece
            body = strip_think(acc)
            if not body:
                continue
            now = time.monotonic()
            display = (prefix + body)[:limit] + " ▍"  # ▍ — «курсор», эффект печати
            if msg is None:
                msg = await self._safe(bot.send_message(chat_id, display, parse_mode=None))
                last_edit = now
            elif now - last_edit >= self.cfg.edit_interval_seconds:
                await self._safe(
                    bot.edit_message_text(
                        display, chat_id=chat_id, message_id=msg.message_id
                    )
                )
                last_edit = now

        final = strip_think(acc)[:limit]
        if not final:
            if msg is not None:
                await self._safe(bot.delete_message(chat_id, msg.message_id))
            return ""
        full = prefix + final
        if msg is None:
            await self._safe(bot.send_message(chat_id, full, parse_mode=None))
        else:
            await self._safe(
                bot.edit_message_text(full, chat_id=chat_id, message_id=msg.message_id)
            )
        return final

    async def _keep_typing(self, bot, chat_id: int) -> None:
        try:
            while True:
                await self._safe(bot.send_chat_action(chat_id, "typing"))
                await asyncio.sleep(4)
        except asyncio.CancelledError:
            return

    @staticmethod
    async def _safe(coro):
        """Выполнить корутину Telegram, проглотив ошибки (edit too fast / not modified)."""
        try:
            return await coro
        except Exception as e:  # noqa: BLE001
            log.debug("tg call skipped: %s", str(e)[:120])
            return None

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
    async def _send_plain(self, bot, chat_id: int, text: str) -> None:
        await self._safe(bot.send_message(chat_id, text, parse_mode=None))

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
        for name, bot in self.bots.items():
            try:
                me = await bot.get_me()
                log.info("Бот «%s» = @%s", name, me.username)
            except Exception as e:  # noqa: BLE001
                log.error("Бот «%s»: не удалось подключиться: %s", name, e)
        log.info(
            "Все боты поллят апдейты (нужен Privacy Disable или статус админа, "
            "чтобы каждый видел сообщения группы и мог поставить реакцию)."
        )
        try:
            # Поллим ВСЕ боты сразу: каждый получает сообщение Босса (если у него
            # отключён Privacy Mode/он админ) и может поставить реакцию. Логика
            # обсуждения запускается один раз — дедуп по message_id.
            await self.dp.start_polling(*self.bots.values(), handle_signals=True)
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

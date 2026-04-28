"""
bank_collection_agent_simple.py

Простая версия банковского голосового агента:
- 1 вызов LLM на ход
- без JSON
- без второго LLM для state
- без LLM-guard / judge / repair
- state определяется обычными правилами Python
- LLM только формулирует живую реплику

Главная защита конфиденциальности здесь не guard, а то, что PRIVATE_DATA
вообще не попадает в prompt до подтверждения личности.

Запуск:
    python bank_collection_agent_simple.py --mock
    python bank_collection_agent_simple.py

Для vLLM / LM Studio / llama.cpp OpenAI-compatible:
    export LLM_BASE_URL=http://localhost:8000/v1
    export LLM_MODEL=your-qwen-model
    export LLM_API_KEY=EMPTY
    python bank_collection_agent_simple.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# ============================================================
# ДАННЫЕ
# ============================================================

PUBLIC_DATA = {
    "bank_name": "АО Da банк",
    "agent_name": "Иванов Иван Борисович",
    "callback_phone": "88005553535",
    "target_name": "Петр Петрович",
    "target_full_name": "Петров Петр Петрович",
    "date": "02 октября 2025",
}

# Эти данные нельзя показывать модели до подтверждения личности.
PRIVATE_DATA = {
    "debt_amount": "15000 рублей",
    "loan_type": "автокредит",
    "overdue_days": "10 дней",
    "collateral_value": "2310000 рублей",
}


# ============================================================
# STATE
# ============================================================

@dataclass
class DialogState:
    state: str = "START"
    step: int = 1
    identity_confirmed: bool = False
    ended: bool = False
    refusal_stage: int = 0
    last_goal: Optional[str] = None
    last_bot_reply: Optional[str] = None


def visible_facts(state: DialogState) -> Dict[str, str]:
    """До подтверждения личности возвращаем только публичные данные."""
    facts = dict(PUBLIC_DATA)
    if state.identity_confirmed:
        facts.update(PRIVATE_DATA)
    return facts


# ============================================================
# БЫСТРОЕ ОПРЕДЕЛЕНИЕ INTENT БЕЗ LLM
# ============================================================

def norm(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def has_any(text: str, words: list[str]) -> bool:
    return any(w in text for w in words)


def detect_intent(user_text: str, state: DialogState) -> str:
    """
    Простой intent detector на ключевых словах.
    Он дешевый и быстрый. LLM здесь не используется.
    """
    t = norm(user_text)

    if not t:
        return "silence"

    # Вопросы третьих лиц до идентификации.
    if not state.identity_confirmed and has_any(t, [
        "по какому вопросу", "что случилось", "зачем звоните", "что нужно",
        "о чем", "какой вопрос", "почему звоните"
    ]):
        return "asks_reason_before_identity"

    # Родственник / третье лицо.
    if has_any(t, [
        "я жена", "я муж", "я сын", "я дочь", "я брат", "я сестра",
        "я мама", "я отец", "я коллега", "я родственник", "я друг", "я подруга"
    ]):
        return "third_person_relative"

    # Ошиблись номером / не знает.
    if has_any(t, [
        "ошиблись", "не знаю такого", "нет такого", "не знаком", "не знакома",
        "тут такого нет", "не туда", "не звоните"
    ]):
        return "third_person_unknown"

    # Подтверждение личности.
    # Важно: проверяем осторожно. Простое "да" считаем подтверждением только на первом шаге.
    if state.step == 1 and has_any(t, [
        "да", "это я", "я петр", "петр петрович", "слушаю", "говорите"
    ]):
        return "identity_confirmed"

    # Отрицание личности.
    if not state.identity_confirmed and has_any(t, [
        "нет", "не я", "это не он", "не петр", "его нет", "он отсутствует"
    ]):
        return "identity_denied"

    # Согласие на оплату.
    if has_any(t, [
        "оплачу", "заплачу", "внесу", "погашу", "закрою", "оплатить готов",
        "завтра оплачу", "сегодня оплачу", "в течение трех дней"
    ]):
        return "agrees_full_payment"

    # Частичная оплата.
    if has_any(t, [
        "часть", "частично", "могу только", "внесу часть", "пять тысяч", "5000", "5 тысяч"
    ]):
        return "agrees_partial_payment"

    # Реструктуризация.
    if has_any(t, [
        "реструктур", "реструктуризация", "уменьшить платеж", "продлить срок"
    ]):
        return "agrees_restructuring"

    # Передача автомобиля.
    if has_any(t, [
        "передам авто", "передать авто", "отдам авто", "отдам машину",
        "передам машину", "верну автомобиль", "сдам автомобиль"
    ]):
        return "agrees_vehicle_transfer"

    # Отказ / нет денег.
    if has_any(t, [
        "не буду платить", "не хочу платить", "не могу платить", "денег нет",
        "нет денег", "платить нечем", "не собираюсь", "отказываюсь"
    ]):
        return "refuses_payment"

    # Просит время.
    if has_any(t, [
        "позже", "потом", "дайте время", "через месяц", "не сейчас", "перезвоните"
    ]):
        return "needs_time"

    # Спрашивает варианты.
    if has_any(t, [
        "какие варианты", "что можно сделать", "как решить", "что предлагаете"
    ]):
        return "asks_options"

    # Ответы по авто.
    if state.step == 9:
        return "vehicle_owner_answer"
    if state.step == 10:
        return "vehicle_condition_answer"
    if state.step == 11 and has_any(t, ["да", "смогу", "готов", "получится"]):
        return "vehicle_transfer_ready"
    if state.step == 11:
        return "vehicle_transfer_not_ready"

    # Контакты.
    if state.step == 13 and has_any(t, ["изменились", "новый номер", "другой номер", "новая почта"]):
        return "contact_changed"
    if state.step == 13:
        return "contact_same"

    return "other"


# ============================================================
# FSM: ВЫБОР ЦЕЛИ ДЛЯ LLM
# ============================================================

def choose_goal(intent: str, state: DialogState) -> str:
    """
    Здесь нет готовых фраз.
    Код выбирает только цель, а текст пишет LLM.
    """
    if state.ended:
        return "end_call"

    if state.state == "START":
        state.state = "IDENTITY_NOT_CONFIRMED"
        state.step = 1

    # До подтверждения личности.
    if not state.identity_confirmed:
        if state.step == 1:
            if intent == "identity_confirmed":
                state.identity_confirmed = True
                state.state = "CLIENT_CONFIRMED"
                state.step = 4
                return "recording_debt_reason"

            if intent in ["identity_denied", "third_person_relative"]:
                state.state = "THIRD_PERSON"
                state.step = 2
                return "ask_relationship"

            if intent == "third_person_unknown":
                state.ended = True
                state.state = "END"
                return "goodbye_no_details"

            if intent == "asks_reason_before_identity":
                return "privacy_refusal"

            return "ask_identity"

        if state.step == 2:
            if intent == "third_person_relative":
                state.ended = True
                state.state = "END"
                return "ask_callback_and_end"

            if intent in ["third_person_unknown", "identity_denied"]:
                state.ended = True
                state.state = "END"
                return "goodbye_no_details"

            if intent == "identity_confirmed":
                state.identity_confirmed = True
                state.state = "CLIENT_CONFIRMED"
                state.step = 4
                return "recording_debt_reason"

            return "privacy_refusal"

    # После подтверждения личности.
    if state.identity_confirmed:
        if intent == "agrees_full_payment":
            state.ended = True
            state.state = "RESOLVED"
            return "confirm_full_payment_end"

        if intent == "agrees_partial_payment":
            state.ended = True
            state.state = "RESOLVED"
            return "confirm_partial_payment_end"

        if intent == "agrees_restructuring":
            state.ended = True
            state.state = "RESOLVED"
            return "confirm_restructuring_end"

        if intent == "agrees_vehicle_transfer":
            state.step = 9
            return "ask_vehicle_owner"

        if intent in ["refuses_payment", "needs_time"]:
            return refusal_goal(state)

        if intent == "asks_options":
            state.step = 8
            return "offer_options"

        if intent == "vehicle_owner_answer":
            state.step = 10
            return "ask_vehicle_condition"

        if intent == "vehicle_condition_answer":
            state.step = 11
            return "ask_vehicle_transfer_3_days"

        if intent == "vehicle_transfer_ready":
            state.ended = True
            state.state = "RESOLVED"
            return "confirm_vehicle_transfer_end"

        if intent == "vehicle_transfer_not_ready":
            return refusal_goal(state)

        if intent in ["contact_changed", "contact_same"]:
            state.ended = True
            state.state = "END"
            return "goodbye_after_contacts"

        return next_step_goal(state)

    return "safe_repeat"


def refusal_goal(state: DialogState) -> str:
    """Лестница убеждения при отказе."""
    state.refusal_stage += 1

    if state.refusal_stage == 1:
        state.step = max(state.step, 6)
        return "soft_persuasion"

    if state.refusal_stage == 2:
        state.step = max(state.step, 7)
        return "suggest_sources"

    if state.refusal_stage == 3:
        state.step = max(state.step, 8)
        return "offer_options"

    if state.refusal_stage == 4:
        state.step = max(state.step, 12)
        return "explain_consequences"

    if state.refusal_stage == 5:
        return "explain_possible_court"

    state.ended = True
    state.state = "END"
    return "close_no_agreement"


def next_step_goal(state: DialogState) -> str:
    """Обычный ход сценария."""
    if state.step <= 4:
        state.step = 5
        return "ask_payment_3_days"

    if state.step == 5:
        state.step = 6
        return "soft_persuasion"

    if state.step == 6:
        state.step = 7
        return "suggest_sources"

    if state.step == 7:
        state.step = 8
        return "offer_options"

    if state.step == 8:
        state.step = 9
        return "ask_vehicle_owner"

    if state.step == 9:
        state.step = 10
        return "ask_vehicle_condition"

    if state.step == 10:
        state.step = 11
        return "ask_vehicle_transfer_3_days"

    if state.step == 11:
        state.step = 12
        return "explain_consequences"

    if state.step == 12:
        state.step = 13
        return "ask_contacts"

    state.ended = True
    state.state = "END"
    return "goodbye_after_contacts"


# ============================================================
# PROMPT ДЛЯ LLM
# ============================================================

def goal_instruction(goal: str) -> str:
    """Инструкции по цели. Это не готовые реплики."""
    return {
        "ask_identity": "Поздоровайся, представься ФИО и банком, уточни, можешь ли поговорить с Петром Петровичем. Не называй причину звонка.",
        "ask_relationship": "Вежливо спроси, кем собеседник приходится Петру Петровичу. Не называй причину звонка.",
        "privacy_refusal": "Вежливо скажи, что информация предназначена только для Петра Петровича, и попроси передать ему просьбу перезвонить по номеру банка. Не раскрывай причину звонка.",
        "ask_callback_and_end": "Попроси передать Петру Петровичу, чтобы он связался с банком по номеру 88005553535, и заверши разговор. Не раскрывай причину звонка.",
        "goodbye_no_details": "Вежливо извинись за беспокойство и попрощайся. Не раскрывай причину звонка.",

        "recording_debt_reason": "Сообщи, что разговор записывается. Назови задолженность, тип кредита и срок просрочки. Спроси причину неоплаты.",
        "ask_payment_3_days": "Спроси, сможет ли клиент внести платеж в ближайшие три дня.",
        "soft_persuasion": "Спокойно объясни, что важно урегулировать вопрос, чтобы задолженность не увеличивалась. Без давления.",
        "suggest_sources": "Предложи рассмотреть помощь близких, частичную оплату или перекредитование. Задай один вопрос, какой вариант возможен.",
        "offer_options": "Предложи варианты: частичная оплата, реструктуризация от 18,9% годовых или передача автомобиля. Задай один вопрос, что клиент готов рассмотреть.",

        "ask_vehicle_owner": "Спроси, на кого зарегистрирован автомобиль: на клиента, супруга или третье лицо. Уточни, были ли изменения в регистрации или паспортных данных.",
        "ask_vehicle_condition": "Спроси о состоянии автомобиля: повреждения, кузов, салон, двигатель. Кратко скажи, что состояние влияет на скорость и стоимость реализации.",
        "ask_vehicle_transfer_3_days": "Спроси, сможет ли клиент в течение трех дней подписать документы и передать автомобиль на стоянку.",

        "explain_consequences": "Нейтрально озвучь последствия: долг может увеличиваться из-за начислений, кредитная история может ухудшиться, возможна реализация имущества в установленном порядке. Без угроз.",
        "explain_possible_court": "Нейтрально скажи, что если договориться не получится, банк может рассмотреть обращение в суд в установленном законом порядке. Предложи выбрать вариант урегулирования без суда.",
        "ask_contacts": "Спроси, изменилась ли контактная информация клиента. Если да, попроси назвать актуальные данные.",

        "confirm_full_payment_end": "Зафиксируй, что клиент готов внести оплату. Кратко обозначь дальнейший шаг и попрощайся.",
        "confirm_partial_payment_end": "Зафиксируй, что клиент готов внести частичную оплату. Кратко обозначь дальнейшее согласование остатка и попрощайся.",
        "confirm_restructuring_end": "Зафиксируй, что клиент готов рассмотреть реструктуризацию. Скажи, что следующий шаг — оформление или согласование условий. Попрощайся.",
        "confirm_vehicle_transfer_end": "Зафиксируй готовность клиента передать автомобиль. Скажи, что дальше нужно согласовать документы, осмотр и дату передачи. Попрощайся.",
        "close_no_agreement": "Кратко зафиксируй, что договоренность не достигнута, банк продолжит работу в установленном порядке. Попрощайся.",
        "goodbye_after_contacts": "Поблагодари за разговор и попрощайся.",
        "safe_repeat": "Кратко попроси уточнить ответ, не добавляя новых деталей.",
        "end_call": "Кратко попрощайся.",
    }.get(goal, "Кратко и вежливо продолжи разговор по текущей цели, без лишних деталей.")


def build_prompt(user_text: str, state: DialogState, goal: str) -> str:
    facts = visible_facts(state)

    if state.identity_confirmed:
        privacy = "Личность подтверждена. Можно обсуждать задолженность, кредит, сумму, авто и варианты урегулирования."
    else:
        privacy = (
            "Личность НЕ подтверждена. Запрещено упоминать причину звонка, долг, задолженность, кредит, "
            "просрочку, сумму, автомобиль, залог, взыскание, суд, кредитную историю и любые финансовые детали. "
            "Можно только представиться, уточнить нужного человека, спросить отношение к нему или попросить передать просьбу перезвонить."
        )

    return f"""
Ты голосовой помощник {facts['bank_name']}.
Твое имя: {facts['agent_name']}.
Сегодня: {facts['date']}.

Доступные факты, которые можно использовать:
{json.dumps(facts, ensure_ascii=False, indent=2)}

Режим конфиденциальности:
{privacy}

Состояние диалога:
state={state.state}
step={state.step}
identity_confirmed={state.identity_confirmed}
refusal_stage={state.refusal_stage}

Последняя реплика клиента:
{user_text}

Текущая цель ответа:
{goal}

Что нужно сделать:
{goal_instruction(goal)}

Правила:
- Ответь только текстом реплики оператора, без JSON и без пояснений.
- Максимум 1-2 коротких предложения.
- Не задавай больше одного вопроса.
- Не объединяй несколько шагов сценария.
- Не повторяйся.
- Если личность не подтверждена, не раскрывай никаких финансовых деталей.
- После подтверждения личности обращайся: "Петр Петрович" или "уважаемый клиент".
- Если говоришь о суде, говори только нейтрально: "банк может рассмотреть обращение в суд в установленном законом порядке".
- Не угрожай, не дави, не упоминай полицию, уголовное дело, арест или выезд сотрудников.
""".strip()


# ============================================================
# LLM CLIENT: ОДИН ВЫЗОВ, ОБЫЧНЫЙ ТЕКСТ
# ============================================================

class LLMClient:
    def __init__(self):
        self.base_url = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
        self.api_key = os.getenv("LLM_API_KEY", "EMPTY")
        self.model = os.getenv("LLM_MODEL", "qwen-8b-sft")

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Ты коротко и строго отвечаешь как банковский голосовой помощник. Выполняй инструкцию пользователя."
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "top_p": 0.8,
            "max_tokens": 160,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.URLError as e:
            raise RuntimeError(f"LLM endpoint error: {e}") from e

        data = json.loads(raw)
        text = data["choices"][0]["message"]["content"]
        return clean_reply(text)


class MockLLM:
    """Для проверки FSM без модели."""
    def generate(self, prompt: str) -> str:
        m = re.search(r"Текущая цель ответа:\n(.+?)\n", prompt)
        goal = m.group(1).strip() if m else "safe_repeat"

        examples = {
            "ask_identity": "Здравствуйте, меня зовут Иванов Иван Борисович, АО Da банк. Могу ли я поговорить с Петром Петровичем?",
            "ask_relationship": "Подскажите, пожалуйста, кем вы приходитесь Петру Петровичу?",
            "privacy_refusal": "Эта информация предназначена только для Петра Петровича. Пожалуйста, передайте ему просьбу перезвонить по номеру 88005553535.",
            "ask_callback_and_end": "Пожалуйста, передайте Петру Петровичу, чтобы он связался с АО Da банк по номеру 88005553535. Всего доброго.",
            "goodbye_no_details": "Извините за беспокойство. Всего доброго.",
            "recording_debt_reason": "Петр Петрович, предупреждаю, что разговор записывается. Звоню по вопросу задолженности 15000 рублей по автокредиту, платеж просрочен на 10 дней; подскажите, по какой причине оплата не поступила?",
            "ask_payment_3_days": "Петр Петрович, сможете внести платеж в ближайшие три дня?",
            "soft_persuasion": "Петр Петрович, важно урегулировать вопрос, чтобы задолженность не увеличивалась. Давайте подберем вариант без давления и с учетом вашей ситуации.",
            "suggest_sources": "Рассмотрите, пожалуйста, помощь близких, частичную оплату или перекредитование. Какой вариант для вас возможен?",
            "offer_options": "Можно рассмотреть частичную оплату, реструктуризацию от 18,9% годовых или передачу автомобиля. Какой вариант вы готовы обсудить?",
            "ask_vehicle_owner": "Подскажите, на кого зарегистрирован автомобиль и были ли изменения в регистрации или паспортных данных?",
            "ask_vehicle_condition": "В каком состоянии автомобиль: есть ли повреждения кузова, салона или двигателя?",
            "ask_vehicle_transfer_3_days": "Сможете в течение трех дней подписать документы и передать автомобиль на стоянку?",
            "explain_consequences": "Если вопрос не урегулировать, задолженность может увеличиваться, а кредитная история может ухудшиться. Также возможна реализация имущества в установленном порядке.",
            "explain_possible_court": "Если договориться не получится, банк может рассмотреть обращение в суд в установленном законом порядке. Лучше сейчас выбрать вариант урегулирования без судебного разбирательства.",
            "confirm_full_payment_end": "Фиксирую, что вы готовы внести оплату. Спасибо за разговор, всего доброго.",
            "confirm_partial_payment_end": "Фиксирую, что вы готовы внести частичную оплату. Спасибо за разговор, всего доброго.",
            "confirm_restructuring_end": "Фиксирую, что вы готовы рассмотреть реструктуризацию. Следующий шаг — согласование условий, всего доброго.",
            "confirm_vehicle_transfer_end": "Фиксирую готовность рассмотреть передачу автомобиля. Далее потребуется согласовать документы, осмотр и дату передачи, всего доброго.",
            "close_no_agreement": "Договоренность пока не достигнута, банк продолжит работу в установленном порядке. Всего доброго.",
            "goodbye_after_contacts": "Спасибо за разговор. Всего доброго.",
        }
        return examples.get(goal, "Уточните, пожалуйста, ваш ответ.")


def clean_reply(text: str) -> str:
    """Убираем мусор, если модель вдруг вернула роли или кавычки."""
    text = text.strip()
    text = re.sub(r"^(assistant|оператор|бот)\s*:\s*", "", text, flags=re.I)
    text = text.strip().strip('"').strip()
    return text


# ============================================================
# AGENT
# ============================================================

class SimpleBankAgent:
    def __init__(self, llm):
        self.llm = llm

    def start(self) -> Tuple[str, DialogState]:
        state = DialogState(state="IDENTITY_NOT_CONFIRMED", step=1)
        goal = "ask_identity"
        state.last_goal = goal
        prompt = build_prompt("", state, goal)
        reply = self.llm.generate(prompt)
        state.last_bot_reply = reply
        return reply, state

    def handle(self, user_text: str, state: DialogState) -> Tuple[str, DialogState]:
        if state.ended:
            return "", state

        intent = detect_intent(user_text, state)
        goal = choose_goal(intent, state)
        state.last_goal = goal

        prompt = build_prompt(user_text, state, goal)
        reply = self.llm.generate(prompt)
        state.last_bot_reply = reply
        return reply, state


# ============================================================
# CLI
# ============================================================

def run_cli(use_mock: bool):
    llm = MockLLM() if use_mock else LLMClient()
    agent = SimpleBankAgent(llm)

    bot, state = agent.start()
    print("BOT:", bot)
    print(f"[debug] state={state.state}, step={state.step}, confirmed={state.identity_confirmed}, goal={state.last_goal}")

    while not state.ended:
        user = input("USER: ").strip()
        if user in ["exit", "quit", "q"]:
            break

        bot, state = agent.handle(user, state)
        if bot:
            print("BOT:", bot)
        print(f"[debug] state={state.state}, step={state.step}, confirmed={state.identity_confirmed}, goal={state.last_goal}, refusal_stage={state.refusal_stage}")

    print("[call ended]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Запуск без настоящей LLM")
    args = parser.parse_args()
    run_cli(use_mock=args.mock)

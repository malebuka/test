"""
bank_agent_pyqt_adapter.py

Вариант под PyQt:
- без mock
- без urllib
- без JSON от модели
- без второго LLM-вызова
- без LLM-guard
- state определяется Python-кодом
- self.ollama_model используется только для генерации ответа
- self.context можно оставить как историю/контекст диалога

ВАЖНО:
В self.context НЕ должно быть суммы долга, типа кредита, просрочки, авто и залога
до подтверждения личности. Эти данные добавляются в prompt только после
state.identity_confirmed=True.

Как использовать внутри твоего PyQt-класса:

    class MainWindow(QMainWindow):
        def __init__(self):
            ...
            self.ollama_model = ...
            self.context = [
                {
                    "role": "system",
                    "content": "Ты голосовой помощник банка. Отвечай кратко и делово. Не раскрывай данные до подтверждения личности."
                },
                {
                    "role": "assistant",
                    "content": "Здравствуйте, меня зовут Иванов Иван Борисович, АО Da банк. Могу ли я поговорить с Петром Петровичем?"
                }
            ]
            self.bank_dialog = BankDialogManager(owner=self)

        def on_user_text(self, user_text):
            bot_text = self.bank_dialog.handle_user_text(user_text)
            # дальше отправляешь bot_text в UI/TTS

Если у тебя self.context — строка, а не список messages, код тоже обработает.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import re


# ============================================================
# 1. ДАННЫЕ
# ============================================================

PUBLIC_DATA = {
    "bank_name": "АО Da банк",
    "agent_name": "Иванов Иван Борисович",
    "callback_phone": "88005553535",
    "target_name": "Петр Петрович",
    "target_full_name": "Петров Петр Петрович",
    "date": "02 октября 2025",
}

# Эти данные нельзя класть в self.context заранее.
# Они попадут в prompt только после подтверждения личности.
PRIVATE_DATA = {
    "debt_amount": "15000 рублей",
    "loan_type": "автокредит",
    "overdue_days": "10 дней",
    "collateral_value": "2310000 рублей",
}


# ============================================================
# 2. STATE
# ============================================================

@dataclass
class DialogState:
    state: str = "IDENTITY_NOT_CONFIRMED"
    step: int = 1
    identity_confirmed: bool = False
    ended: bool = False
    refusal_stage: int = 0
    last_goal: Optional[str] = None
    last_intent: Optional[str] = None
    last_bot_reply: Optional[str] = None


def get_visible_facts(state: DialogState) -> Dict[str, str]:
    facts = dict(PUBLIC_DATA)

    if state.identity_confirmed:
        facts.update(PRIVATE_DATA)

    return facts


# ============================================================
# 3. INTENT БЕЗ LLM
# ============================================================

def normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def has_any(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def detect_intent(user_text: str, state: DialogState) -> str:
    text = normalize(user_text)

    if not text:
        return "silence"

    # До подтверждения личности: вопрос о причине звонка.
    if not state.identity_confirmed and has_any(text, [
        "по какому вопросу",
        "что случилось",
        "зачем звоните",
        "почему звоните",
        "о чем речь",
        "какой вопрос",
        "что нужно",
    ]):
        return "asks_reason_before_identity"

    # Родственник / знакомый / третье лицо.
    if has_any(text, [
        "я жена",
        "я муж",
        "я сын",
        "я дочь",
        "я мама",
        "я мать",
        "я отец",
        "я брат",
        "я сестра",
        "я родственник",
        "я коллега",
        "я друг",
        "я подруга",
    ]):
        return "third_person_relative"

    # Не знает клиента / ошиблись номером.
    if has_any(text, [
        "ошиблись",
        "не знаю такого",
        "нет такого",
        "тут такого нет",
        "не знаком",
        "не знакома",
        "не туда",
    ]):
        return "third_person_unknown"

    # Подтверждение личности.
    # Простое "да" считаем подтверждением только на шаге 1.
    if state.step == 1 and has_any(text, [
        "да",
        "это я",
        "я петр",
        "петр петрович",
        "слушаю",
        "говорите",
    ]):
        return "identity_confirmed"

    # Отрицание личности.
    if not state.identity_confirmed and has_any(text, [
        "нет",
        "не я",
        "это не он",
        "не петр",
        "его нет",
        "он отсутствует",
    ]):
        return "identity_denied"

    # Согласие на оплату.
    if has_any(text, [
        "оплачу",
        "заплачу",
        "внесу",
        "погашу",
        "закрою",
        "сегодня оплачу",
        "завтра оплачу",
        "готов оплатить",
        "готов заплатить",
    ]):
        return "agrees_full_payment"

    # Частичная оплата.
    if has_any(text, [
        "часть",
        "частично",
        "могу только",
        "внесу часть",
        "5000",
        "5 тысяч",
        "пять тысяч",
    ]):
        return "agrees_partial_payment"

    # Реструктуризация.
    if has_any(text, [
        "реструктур",
        "реструктуризация",
        "уменьшить платеж",
        "продлить срок",
    ]):
        return "agrees_restructuring"

    # Передача автомобиля.
    if has_any(text, [
        "передам авто",
        "передать авто",
        "отдам авто",
        "отдам машину",
        "передам машину",
        "сдам автомобиль",
        "верну автомобиль",
    ]):
        return "agrees_vehicle_transfer"

    # Отказ / нет денег.
    if has_any(text, [
        "не буду платить",
        "не хочу платить",
        "не могу платить",
        "денег нет",
        "нет денег",
        "платить нечем",
        "не собираюсь",
        "отказываюсь",
    ]):
        return "refuses_payment"

    # Просит время.
    if has_any(text, [
        "позже",
        "потом",
        "дайте время",
        "не сейчас",
        "перезвоните",
        "через месяц",
    ]):
        return "needs_time"

    # Просит варианты.
    if has_any(text, [
        "какие варианты",
        "что можно сделать",
        "как решить",
        "что предлагаете",
    ]):
        return "asks_options"

    # Ответы по авто зависят от текущего шага.
    if state.step == 9:
        return "vehicle_owner_answer"

    if state.step == 10:
        return "vehicle_condition_answer"

    if state.step == 11 and has_any(text, ["да", "готов", "смогу", "получится"]):
        return "vehicle_transfer_ready"

    if state.step == 11:
        return "vehicle_transfer_not_ready"

    # Контакты.
    if state.step == 13 and has_any(text, [
        "изменились",
        "новый номер",
        "другой номер",
        "новая почта",
    ]):
        return "contact_changed"

    if state.step == 13:
        return "contact_same"

    return "other"


# ============================================================
# 4. FSM: ВЫБОР ЦЕЛИ
# ============================================================

def choose_goal(intent: str, state: DialogState) -> str:
    if state.ended:
        return "end_call"

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
# 5. PROMPT ДЛЯ МОДЕЛИ
# ============================================================

def goal_instruction(goal: str) -> str:
    instructions = {
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
    }
    return instructions.get(goal, "Кратко и вежливо продолжи разговор по текущей цели.")


def build_dynamic_prompt(user_text: str, state: DialogState, goal: str) -> str:
    facts = get_visible_facts(state)

    if state.identity_confirmed:
        privacy = "Личность подтверждена. Можно обсуждать задолженность, кредит, сумму, автомобиль и варианты урегулирования."
    else:
        privacy = (
            "Личность НЕ подтверждена. Запрещено упоминать причину звонка, долг, задолженность, кредит, "
            "просрочку, сумму, автомобиль, залог, взыскание, суд, кредитную историю и любые финансовые детали. "
            "Можно только представиться, уточнить нужного человека, спросить отношение к нему или попросить передать просьбу перезвонить."
        )

    return f"""
Доступные факты:
{facts}

Режим конфиденциальности:
{privacy}

Состояние:
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
- Ответь только текстом реплики оператора.
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
# 6. ИЗВЛЕЧЕНИЕ ТЕКСТА ИЗ RESPONSE
# ============================================================

def clean_reply(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"^(assistant|оператор|бот|ответ)\s*:\s*", "", text, flags=re.I)
    text = text.strip().strip('"').strip()
    return text


def extract_response_text(response: Any) -> str:
    """
    Поддерживает разные форматы ответа:
    - str
    - dict от ollama.chat: response['message']['content']
    - dict от ollama.generate: response['response']
    - объект с атрибутом .content или .text
    """
    if response is None:
        return ""

    if isinstance(response, str):
        return clean_reply(response)

    if isinstance(response, dict):
        # ollama.chat(...)
        if isinstance(response.get("message"), dict):
            content = response["message"].get("content")
            if content:
                return clean_reply(content)

        # ollama.generate(...)
        if response.get("response"):
            return clean_reply(response["response"])

        # OpenAI-like
        try:
            return clean_reply(response["choices"][0]["message"]["content"])
        except Exception:
            pass

    for attr in ["content", "text", "response"]:
        if hasattr(response, attr):
            value = getattr(response, attr)
            if value:
                return clean_reply(value)

    return clean_reply(str(response))


# ============================================================
# 7. МЕНЕДЖЕР ДИАЛОГА ДЛЯ PYQT
# ============================================================

class BankDialogManager:
    """
    owner — это твой PyQt-класс, где уже есть:
    - owner.ollama_model
    - owner.context

    Этот менеджер не знает, как именно у тебя вызывается модель.
    Он пробует популярные варианты:
    - self.ollama_model.invoke(prompt)
    - self.ollama_model.generate(prompt)
    - self.ollama_model.chat(messages)
    - self.ollama_model(prompt)

    Если у тебя другой интерфейс — поправь только метод _call_model().
    """

    def __init__(self, owner: Any):
        self.owner = owner
        self.state = DialogState()

    def start_from_existing_greeting(self) -> str:
        """
        Используй это, если первое приветственное сообщение уже лежит в self.context.
        Тогда модель не вызывается для первого сообщения.
        """
        greeting = self._get_last_assistant_message_from_context()

        if not greeting:
            greeting = "Здравствуйте, меня зовут Иванов Иван Борисович, АО Da банк. Могу ли я поговорить с Петром Петровичем?"
            self._append_to_context("assistant", greeting)

        self.state = DialogState()
        self.state.last_goal = "ask_identity"
        self.state.last_bot_reply = greeting
        return greeting

    def start_with_llm(self) -> str:
        """
        Используй это, если хочешь, чтобы первое приветствие сгенерировала модель.
        """
        self.state = DialogState()
        goal = "ask_identity"
        self.state.last_goal = goal

        prompt = build_dynamic_prompt(user_text="", state=self.state, goal=goal)
        reply = self._call_model(prompt)

        self.state.last_bot_reply = reply
        self._append_to_context("assistant", reply)
        return reply

    def handle_user_text(self, user_text: str) -> str:
        """
        Главный метод на каждый ответ клиента.
        Его вызывай из PyQt после распознавания речи / получения текста.
        """
        if self.state.ended:
            return ""

        self._append_to_context("user", user_text)

        intent = detect_intent(user_text, self.state)
        goal = choose_goal(intent, self.state)

        self.state.last_intent = intent
        self.state.last_goal = goal

        dynamic_prompt = build_dynamic_prompt(
            user_text=user_text,
            state=self.state,
            goal=goal,
        )

        reply = self._call_model(dynamic_prompt)
        self.state.last_bot_reply = reply

        self._append_to_context("assistant", reply)
        return reply

    def _call_model(self, dynamic_prompt: str) -> str:
        """
        Здесь адаптация под твой self.ollama_model.

        ВАЖНО:
        Мы НЕ используем старый большой prompt с приватными данными.
        Мы берем self.context как историю/базовый стиль + добавляем dynamic_prompt.
        """
        model = self.owner.ollama_model

        # Вариант 1: LangChain-like: model.invoke(prompt)
        if hasattr(model, "invoke"):
            response = model.invoke(self._build_plain_prompt(dynamic_prompt))
            return extract_response_text(response)

        # Вариант 2: model.generate(prompt)
        if hasattr(model, "generate"):
            response = model.generate(self._build_plain_prompt(dynamic_prompt))
            return extract_response_text(response)

        # Вариант 3: model.chat(messages)
        if hasattr(model, "chat"):
            messages = self._build_messages(dynamic_prompt)
            response = model.chat(messages)
            return extract_response_text(response)

        # Вариант 4: сам объект вызываемый: model(prompt)
        if callable(model):
            response = model(self._build_plain_prompt(dynamic_prompt))
            return extract_response_text(response)

        raise TypeError("Не понимаю интерфейс self.ollama_model. Поправь метод BankDialogManager._call_model().")

    def _build_plain_prompt(self, dynamic_prompt: str) -> str:
        """
        Для моделей, которые принимают один большой prompt-строку.
        """
        base_context = self._context_as_text_without_private_data()

        return f"""
{base_context}

--- ТЕКУЩАЯ ИНСТРУКЦИЯ ---
{dynamic_prompt}
""".strip()

    def _build_messages(self, dynamic_prompt: str) -> list[dict[str, str]]:
        """
        Для chat-моделей, которые принимают messages.
        """
        context = getattr(self.owner, "context", [])

        if isinstance(context, list):
            messages = list(context)
        else:
            messages = [
                {
                    "role": "system",
                    "content": str(context),
                }
            ]

        messages.append({"role": "user", "content": dynamic_prompt})
        return messages

    def _append_to_context(self, role: str, content: str) -> None:
        """
        Аккуратно добавляет реплики в self.context.
        Поддерживает list messages и строковый context.
        """
        if not hasattr(self.owner, "context"):
            self.owner.context = []

        if isinstance(self.owner.context, list):
            self.owner.context.append({"role": role, "content": content})
        else:
            role_name = "Клиент" if role == "user" else "Оператор"
            self.owner.context = str(self.owner.context) + f"\n{role_name}: {content}"

    def _get_last_assistant_message_from_context(self) -> str:
        context = getattr(self.owner, "context", [])

        if isinstance(context, list):
            for msg in reversed(context):
                if msg.get("role") == "assistant":
                    return str(msg.get("content", "")).strip()
            return ""

        # Если context строкой, надежно вытащить последнее приветствие сложно.
        # Лучше явно хранить self.first_greeting.
        if hasattr(self.owner, "first_greeting"):
            return str(self.owner.first_greeting).strip()

        return ""

    def _context_as_text_without_private_data(self) -> str:
        """
        Если self.context строкой или списком, превращаем в текст.
        Здесь НЕ удаляем приватные данные магически — лучше не класть их туда заранее.
        Но добавляем предупреждение, чтобы модель следовала текущей dynamic-инструкции.
        """
        context = getattr(self.owner, "context", "")

        if isinstance(context, list):
            parts = []
            for msg in context:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"{role}: {content}")
            base = "\n".join(parts)
        else:
            base = str(context)

        return f"""
{base}

ВАЖНО: текущая инструкция ниже имеет приоритет над старым контекстом.
Если личность не подтверждена, запрещено раскрывать любые финансовые детали.
""".strip()

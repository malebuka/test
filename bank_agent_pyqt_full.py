# bank_agent_pyqt_full.py
# Готовый пример PyQt + Ollama для агента взыскания.
#
# Главное:
# - модель вызывается 1 раз на ход;
# - state/step контролируются кодом;
# - private данные не попадают в prompt до подтверждения личности;
# - шаг двигается дальше только если ответ клиента подходит текущему вопросу;
# - вопросы/грубость/не по теме не двигают step;
# - варианты урегулирования идут отдельно:
#   1) помощь родственников / занять / перекредитоваться
#   2) частичная оплата
#   3) реструктуризация
#   4) передача автомобиля
#
# Перед запуском:
#   pip install PyQt5 ollama
#   ollama serve
#   ollama pull t-8b-base  # или твоя модель
#
# В коде ниже поменяй:
#   OLLAMA_HOST
#   OLLAMA_MODEL

import sys
import re
from dataclasses import dataclass
from typing import Dict, Optional

import ollama
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QTextEdit,
    QScrollArea,
    QLineEdit,
    QPushButton,
)


OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_MODEL = "t-8b-base"


# ============================================================
# 1. ДАННЫЕ
# ============================================================

PUBLIC_DATA = {
    "bank_name": "АО Da банк",
    "agent_name": "Иванов Иван Борисович",
    "shown_agent_name": "Полина",
    "callback_phone": "88005553535",
    "target_name": "Петр Петрович",
    "target_full_name": "Петров Петр Петрович",
    "date": "02 октября 2025",
}

# Эти данные нельзя класть в self.context заранее.
# Они попадут в prompt только после identity_confirmed=True.
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

    # Этап отказа/убеждения:
    # 0 — еще не убеждали
    # 1 — помощь родственников / занять / перекредитоваться
    # 2 — частичная оплата
    # 3 — реструктуризация
    # 4 — передача авто
    # 5 — последствия
    # 6 — возможный суд
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
# 3. TEXT UTILS
# ============================================================

def normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def has_any(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def looks_like_question(raw_text: str, normalized_text: str) -> bool:
    if "?" in raw_text:
        return True

    question_starters = [
        "кто", "что", "где", "куда", "откуда", "когда", "почему", "зачем",
        "как", "какой", "какая", "какое", "какие", "сколько", "можно ли",
        "надо ли", "нужно ли", "будет ли", "есть ли", "поясните", "объясните",
        "расскажите", "уточните", "повторите"
    ]

    return any(normalized_text.startswith(q) for q in question_starters)


def is_rude_or_offtopic_refusal(text: str) -> bool:
    return has_any(text, [
        "тупая", "тупой", "дура", "дурак", "иди", "отвали", "достали",
        "не звони", "пошла", "пошел", "бред", "мне плевать",
        "не хочу говорить", "без комментариев"
    ])


def clean_reply(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"^(assistant|оператор|бот|ответ)\s*:\s*", "", text, flags=re.I)
    return text.strip().strip('"').strip()


# ============================================================
# 4. ПРОВЕРКА: ОТВЕТ ПОДХОДИТ ТЕКУЩЕМУ ВОПРОСУ?
# ============================================================

def answer_matches_current_step(user_text: str, state: DialogState) -> bool:
    """
    Step увеличивается только если эта функция вернула True.
    Если клиент задал вопрос, грубит или говорит не по теме — возвращаем False.
    """
    text = normalize(user_text)

    if not state.identity_confirmed:
        return False

    if looks_like_question(user_text, text):
        return False

    if is_rude_or_offtopic_refusal(text):
        return False

    # step 4: спросили причину неоплаты.
    # Подходит почти любой содержательный ответ.
    if state.step == 4:
        return len(text.split()) >= 2

    # step 5: спросили, сможет ли оплатить в течение 3 дней.
    if state.step == 5:
        return has_any(text, [
            "да", "смогу", "получится", "оплачу", "заплачу", "внесу",
            "нет", "не смогу", "не получится", "не могу", "денег нет",
            "нет денег", "позже", "потом", "не сейчас", "платить нечем"
        ])

    # step 6: предложили помощь родственников / занять / перекредитоваться.
    if state.step == 6:
        return has_any(text, [
            "попробую", "подумаю", "спрошу", "обращусь", "займу", "возьму в долг",
            "родствен", "друз", "перекредит", "рефинанс", "нет", "не получится",
            "не вариант", "не могу", "не хочу"
        ])

    # step 7: предложили частичную оплату.
    if state.step == 7:
        return has_any(text, [
            "частично", "часть", "могу внести", "могу оплатить", "5000", "5 тысяч",
            "сколько", "нет", "не могу", "не получится", "не подходит", "не вариант"
        ])

    # step 8: предложили реструктуризацию.
    if state.step == 8:
        return has_any(text, [
            "реструктур", "согласен", "согласна", "подходит", "оформить",
            "давайте", "нет", "не хочу", "не подходит", "не вариант"
        ])

    # step 9: предложили передачу авто / спросили готовность рассмотреть.
    if state.step == 9:
        return has_any(text, [
            "авто", "машин", "автомоб", "передам", "отдам", "сдам",
            "готов", "готова", "согласен", "согласна", "нет", "не хочу",
            "не готов", "не получится", "не вариант"
        ])

    # step 10: спросили, на кого оформлен автомобиль.
    if state.step == 10:
        return has_any(text, [
            "на меня", "на супруг", "на жену", "на мужа", "на отца", "на мать",
            "на третье лицо", "на другого", "на компанию", "оформлен",
            "зарегистрирован", "собственник"
        ])

    # step 11: спросили состояние автомобиля.
    if state.step == 11:
        return has_any(text, [
            "хорош", "нормаль", "плох", "бит", "поврежд", "царап", "вмят",
            "трещ", "на ходу", "не на ходу", "двигатель", "кузов", "салон",
            "ремонт", "исправ", "неисправ"
        ])

    # step 12: спросили, готов ли передать авто за 3 дня.
    if state.step == 12:
        return has_any(text, [
            "да", "готов", "готова", "смогу", "получится", "согласен",
            "согласна", "нет", "не готов", "не готова", "не смогу",
            "не получится", "не хочу"
        ])

    # step 14: спросили контактные данные.
    if state.step == 14:
        return has_any(text, [
            "да", "нет", "изменились", "не изменились", "номер", "телефон",
            "почта", "адрес", "тот же", "актуальны", "актуальный"
        ])

    return False


# ============================================================
# 5. INTENT БЕЗ LLM
# ============================================================

def detect_intent(user_text: str, state: DialogState) -> str:
    text = normalize(user_text)

    if not text:
        return "silence"

    is_question = looks_like_question(user_text, text)

    # До подтверждения личности: вопрос о причине звонка — отдельная ветка.
    if not state.identity_confirmed and has_any(text, [
        "по какому вопросу", "что случилось", "зачем звоните",
        "почему звоните", "о чем речь", "какой вопрос", "что нужно"
    ]):
        return "asks_reason_before_identity"

    # Любой другой вопрос клиента: модель отвечает по доступному контексту, step не двигается.
    if is_question:
        return "asks_any_question"

    # Родственник / знакомый / третье лицо.
    if has_any(text, [
        "я жена", "я муж", "я сын", "я дочь", "я мама", "я мать",
        "я отец", "я брат", "я сестра", "я родственник",
        "я коллега", "я друг", "я подруга"
    ]):
        return "third_person_relative"

    # Не знает клиента / ошиблись номером.
    if has_any(text, [
        "ошиблись", "не знаю такого", "нет такого", "тут такого нет",
        "не знаком", "не знакома", "не туда"
    ]):
        return "third_person_unknown"

    # Подтверждение личности.
    if state.step == 1 and has_any(text, [
        "да", "это я", "я петр", "петр петрович", "слушаю", "говорите"
    ]):
        return "identity_confirmed"

    # Отрицание личности.
    if not state.identity_confirmed and has_any(text, [
        "нет", "не я", "это не он", "не петр", "его нет", "он отсутствует"
    ]):
        return "identity_denied"

    # Явное согласие на полную оплату.
    if state.identity_confirmed and has_any(text, [
        "оплачу", "заплачу", "внесу", "погашу", "закрою",
        "сегодня оплачу", "завтра оплачу", "готов оплатить", "готов заплатить"
    ]):
        return "agrees_full_payment"

    # Частичная оплата.
    if state.identity_confirmed and has_any(text, [
        "часть", "частично", "могу только", "внесу часть",
        "5000", "5 тысяч", "пять тысяч"
    ]):
        return "agrees_partial_payment"

    # Реструктуризация.
    if state.identity_confirmed and has_any(text, [
        "реструктур", "реструктуризация", "уменьшить платеж", "продлить срок"
    ]):
        return "agrees_restructuring"

    # Передача автомобиля.
    if state.identity_confirmed and has_any(text, [
        "передам авто", "передать авто", "отдам авто", "отдам машину",
        "передам машину", "сдам автомобиль", "верну автомобиль"
    ]):
        return "agrees_vehicle_transfer"

    # Отказ / нет денег.
    if state.identity_confirmed and has_any(text, [
        "не буду платить", "не хочу платить", "не могу платить", "денег нет",
        "нет денег", "платить нечем", "не собираюсь", "отказываюсь"
    ]):
        return "refuses_payment"

    # Просит время.
    if state.identity_confirmed and has_any(text, [
        "позже", "потом", "дайте время", "не сейчас", "перезвоните", "через месяц"
    ]):
        return "needs_time"

    # Если ответ подходит текущему вопросу — двигаем сценарий.
    if state.identity_confirmed and answer_matches_current_step(user_text, state):
        return "current_step_answer"

    # Все остальное — свободная реплика.
    # Модель отвечает, но step не двигается.
    return "free_client_message"


# ============================================================
# 6. FSM: ВЫБОР ЦЕЛИ
# ============================================================

def choose_goal(intent: str, state: DialogState) -> str:
    if state.ended:
        return "end_call"

    # До подтверждения личности.
    if not state.identity_confirmed:
        if intent in ["asks_any_question", "free_client_message"]:
            return "answer_or_react_keep_step"

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
        # Сначала обрабатываем явные договоренности.
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
            state.step = 10
            return "ask_vehicle_owner"

        # Явный отказ запускает лестницу убеждения.
        if intent in ["refuses_payment", "needs_time"]:
            return refusal_goal(state)

        # Ответ подходит текущему вопросу — двигаем step.
        if intent == "current_step_answer":
            return advance_after_step_answer(state)

        # Любой вопрос / грубость / не по теме — отвечаем, step не двигаем.
        if intent in ["asks_any_question", "free_client_message"]:
            return "answer_or_react_keep_step"

        return "answer_or_react_keep_step"

    return "safe_repeat"


def refusal_goal(state: DialogState) -> str:
    """
    Лестница убеждения при отказе.
    Сделано по отдельности:
    1. помощь родственников / занять / перекредитоваться — вместе;
    2. частичная оплата — отдельно;
    3. реструктуризация — отдельно;
    4. передача авто — отдельно;
    5. последствия;
    6. возможный суд.
    """
    state.refusal_stage += 1

    if state.refusal_stage == 1:
        state.step = 6
        return "suggest_sources"

    if state.refusal_stage == 2:
        state.step = 7
        return "offer_partial_payment"

    if state.refusal_stage == 3:
        state.step = 8
        return "offer_restructuring"

    if state.refusal_stage == 4:
        state.step = 9
        return "offer_vehicle_transfer"

    if state.refusal_stage == 5:
        state.step = 13
        return "explain_consequences"

    if state.refusal_stage == 6:
        return "explain_possible_court"

    state.ended = True
    state.state = "END"
    return "close_no_agreement"


def advance_after_step_answer(state: DialogState) -> str:
    """
    Переход после ответа, который подходит текущему вопросу.
    """
    if state.step == 4:
        state.step = 5
        return "ask_payment_3_days"

    if state.step == 5:
        # Если клиент ответил "да" — это обычно уже поймается как agrees_full_payment.
        # Если ответил "нет/не могу" — часто поймается как refuses_payment.
        # Но если сюда попал другой подходящий ответ, идем к убеждению.
        return refusal_goal(state)

    if state.step == 6:
        state.step = 7
        return "offer_partial_payment"

    if state.step == 7:
        state.step = 8
        return "offer_restructuring"

    if state.step == 8:
        state.step = 9
        return "offer_vehicle_transfer"

    if state.step == 9:
        # Клиент ответил на предложение передачи авто.
        # Если согласился — это обычно поймается как agrees_vehicle_transfer.
        # Если отказался — идем к последствиям.
        state.step = 13
        return "explain_consequences"

    if state.step == 10:
        state.step = 11
        return "ask_vehicle_condition"

    if state.step == 11:
        state.step = 12
        return "ask_vehicle_transfer_3_days"

    if state.step == 12:
        # Если готов передать — должен пойматься как agrees_vehicle_transfer/подходящий ответ.
        # Здесь фиксируем следующий шаг по авто.
        state.ended = True
        state.state = "RESOLVED"
        return "confirm_vehicle_transfer_end"

    if state.step == 13:
        state.step = 14
        return "ask_contacts"

    if state.step == 14:
        state.ended = True
        state.state = "END"
        return "goodbye_after_contacts"

    return "safe_repeat"


# ============================================================
# 7. PROMPT
# ============================================================

def goal_instruction(goal: str) -> str:
    instructions = {
        "ask_identity": (
            "Поздоровайся, представься именем и банком, уточни, можешь ли поговорить "
            "с Петром Петровичем. Не называй причину звонка."
        ),
        "ask_relationship": (
            "Вежливо спроси, кем собеседник приходится Петру Петровичу. "
            "Не называй причину звонка."
        ),
        "privacy_refusal": (
            "Вежливо скажи, что информация предназначена только для Петра Петровича, "
            "и попроси передать ему просьбу перезвонить по номеру банка. "
            "Не раскрывай причину звонка."
        ),
        "ask_callback_and_end": (
            "Попроси передать Петру Петровичу, чтобы он связался с банком по номеру "
            "88005553535, и заверши разговор. Не раскрывай причину звонка."
        ),
        "goodbye_no_details": (
            "Вежливо извинись за беспокойство и попрощайся. Не раскрывай причину звонка."
        ),

        "recording_debt_reason": (
            "Личность уже подтверждена. Не проси ФИО, паспорт, дату рождения или код из SMS. "
            "Сообщи, что разговор записывается. Назови задолженность, тип кредита и срок "
            "просрочки. Спроси причину неоплаты."
        ),
        "ask_payment_3_days": (
            "Спроси, сможет ли клиент внести платеж в ближайшие три дня."
        ),

        # Отдельные варианты урегулирования
        "suggest_sources": (
            "Предложи рассмотреть помощь родственников или друзей, возможность занять средства "
            "или перекредитоваться. Это один общий блок. Задай один вопрос, сможет ли клиент "
            "рассмотреть такой вариант."
        ),
        "offer_partial_payment": (
            "Предложи частичную оплату как отдельный вариант, если полной суммы сейчас нет. "
            "Задай один вопрос, какую часть клиент сможет внести и когда."
        ),
        "offer_restructuring": (
            "Предложи реструктуризацию как отдельный вариант: продление срока кредита для снижения "
            "ежемесячного платежа, ставка от 18,9% годовых. Задай один вопрос, готов ли клиент "
            "рассмотреть реструктуризацию."
        ),
        "offer_vehicle_transfer": (
            "Предложи передачу автомобиля как отдельный вариант урегулирования. Кратко объясни, "
            "что при хорошем состоянии и наличии хода банк может рассмотреть принятие автомобиля "
            "для дальнейшей реализации. Задай один вопрос, готов ли клиент рассмотреть этот вариант."
        ),

        # Авто
        "ask_vehicle_owner": (
            "Спроси, на кого зарегистрирован автомобиль: на клиента, супруга или третье лицо. "
            "Уточни, были ли изменения в регистрации или паспортных данных."
        ),
        "ask_vehicle_condition": (
            "Спроси о состоянии автомобиля: повреждения, кузов, салон, двигатель. "
            "Кратко скажи, что состояние влияет на скорость и стоимость реализации."
        ),
        "ask_vehicle_transfer_3_days": (
            "Спроси, сможет ли клиент в течение трех дней подписать документы и передать "
            "автомобиль на стоянку."
        ),

        "explain_consequences": (
            "Нейтрально озвучь последствия: долг может увеличиваться из-за начислений, "
            "кредитная история может ухудшиться, возможна реализация имущества в установленном "
            "порядке. Без угроз."
        ),
        "explain_possible_court": (
            "Нейтрально скажи, что если договориться не получится, банк может рассмотреть "
            "обращение в суд в установленном законом порядке. Предложи выбрать вариант "
            "урегулирования без суда."
        ),
        "ask_contacts": (
            "Спроси, изменилась ли контактная информация клиента. Если да, попроси назвать "
            "актуальные данные."
        ),

        "confirm_full_payment_end": (
            "Зафиксируй, что клиент готов внести оплату. Кратко обозначь дальнейший шаг и попрощайся."
        ),
        "confirm_partial_payment_end": (
            "Зафиксируй, что клиент готов внести частичную оплату. Кратко обозначь дальнейшее "
            "согласование остатка и попрощайся."
        ),
        "confirm_restructuring_end": (
            "Зафиксируй, что клиент готов рассмотреть реструктуризацию. Скажи, что следующий шаг — "
            "оформление или согласование условий. Попрощайся."
        ),
        "confirm_vehicle_transfer_end": (
            "Зафиксируй готовность клиента передать автомобиль. Скажи, что дальше нужно согласовать "
            "документы, осмотр и дату передачи. Попрощайся."
        ),
        "close_no_agreement": (
            "Кратко зафиксируй, что договоренность не достигнута, банк продолжит работу "
            "в установленном порядке. Попрощайся."
        ),
        "goodbye_after_contacts": (
            "Поблагодари за разговор и попрощайся."
        ),
        "answer_or_react_keep_step": (
            "Отреагируй на последнюю реплику клиента естественно и по ситуации. "
            "Если клиент задал вопрос — ответь на вопрос. Если клиент грубит — спокойно не спорь "
            "и верни разговор к текущему вопросу. Если клиент говорит не по теме — кратко верни "
            "разговор к задолженности. Не продвигай сценарий дальше. После ответа вернись к "
            "предыдущему вопросу оператора. Если личность не подтверждена, не раскрывай финансовые детали."
        ),
        "safe_repeat": (
            "Кратко попроси уточнить ответ, не добавляя новых деталей."
        ),
        "end_call": (
            "Кратко попрощайся."
        ),
    }
    return instructions.get(goal, "Кратко и вежливо продолжи разговор по текущей цели.")


def build_dynamic_prompt(user_text: str, state: DialogState, goal: str) -> str:
    facts = get_visible_facts(state)

    if state.identity_confirmed:
        privacy = (
            "Личность подтверждена. Можно обсуждать задолженность, кредит, сумму, автомобиль "
            "и варианты урегулирования."
        )
    else:
        privacy = (
            "Личность НЕ подтверждена. Запрещено упоминать причину звонка, долг, задолженность, "
            "кредит, просрочку, сумму, автомобиль, залог, взыскание, суд, кредитную историю и "
            "любые финансовые детали. Можно только представиться, уточнить нужного человека, "
            "спросить отношение к нему или попросить передать просьбу перезвонить."
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

Предыдущая цель/вопрос оператора:
{state.last_goal}

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
- Если клиент задал вопрос, сначала кратко ответь на него, затем вернись к предыдущему вопросу оператора.
- Если клиент говорит что-то не по теме, отвечает грубо или делает произвольное утверждение, не переходи к следующему шагу.
- Если текущая цель answer_or_react_keep_step, отвечай по смыслу последней реплики клиента, но не раскрывай новые данные сверх доступных фактов.
- Если личность не подтверждена, не раскрывай никаких финансовых деталей.
- После подтверждения личности не проси ФИО, дату рождения, паспорт, последние 4 цифры паспорта или SMS-код.
- После подтверждения личности обращайся: "Петр Петрович" или "уважаемый клиент".
- Если говоришь о суде, говори только нейтрально: "банк может рассмотреть обращение в суд в установленном законом порядке".
- Не угрожай, не дави, не упоминай полицию, уголовное дело, арест или выезд сотрудников.
""".strip()


# ============================================================
# 8. PYQT APP
# ============================================================

class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Создаем state/context ДО init_ui, чтобы init_ui мог их читать.
        self.dialog_state = DialogState()

        self.prompt = (
            "Ты голосовой помощник АО Da банк. "
            "Отвечай кратко, делово и строго по текущей инструкции. "
            "До подтверждения личности запрещено раскрывать причину звонка и любые финансовые детали. "
            "Если клиент ответил 'да', 'это я', 'слушаю' на вопрос о личности, личность считается подтвержденной. "
            "Запрещено просить ФИО, дату рождения, паспортные данные, последние 4 цифры паспорта, код из SMS "
            "или любые дополнительные персональные данные. "
            "Не угрожай, не дави, не упоминай полицию, уголовное дело, арест или выезд сотрудников."
        )

        self.context = [
            {"role": "system", "content": self.prompt},
            {
                "role": "assistant",
                "content": (
                    "Алло, здравствуйте, меня зовут Полина, я сотрудник Da банк. "
                    "Петров Петр Петрович — это вы?"
                ),
            },
        ]

        self.ollama_client = None
        self.model = None

        self.init_ui()
        self.load_model()

    def init_ui(self):
        self.setWindowTitle("Chat bot")
        self.setGeometry(100, 100, 600, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)

        scroll = QScrollArea()
        scroll.setWidget(self.chat_history)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        for c in self.context[1:]:
            self.render_message(c["role"], c["content"])

        self.input_field = QLineEdit()
        self.input_field.returnPressed.connect(self.send_message)
        layout.addWidget(self.input_field)

        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_message)
        layout.addWidget(self.send_btn)

    def load_model(self):
        self.ollama_client = ollama.Client(host=OLLAMA_HOST, trust_env=False)

    def build_messages_for_model(self, dynamic_prompt: str):
        # self.context — история.
        # dynamic_prompt не сохраняем в историю, чтобы не раздувать контекст.
        messages = list(self.context)
        messages.append({
            "role": "user",
            "content": (
                "Текущая инструкция имеет приоритет над историей диалога.\n\n"
                + dynamic_prompt
            ),
        })
        return messages

    def send_message(self):
        user_input = self.input_field.text().strip()
        if not user_input:
            return

        self.input_field.clear()

        # 1. Добавляем реплику клиента в UI и историю.
        self.update_chat_history(f"клиент:{user_input}", user_input, "user")

        # 2. Быстро определяем intent без LLM.
        intent = detect_intent(user_input, self.dialog_state)

        # 3. Запоминаем предыдущую цель.
        previous_goal = self.dialog_state.last_goal

        # 4. FSM выбирает цель и обновляет state.
        goal = choose_goal(intent, self.dialog_state)
        self.dialog_state.last_intent = intent

        # Если клиент задал вопрос / сказал не по теме / грубит — не затираем предыдущую цель.
        # Так модель ответит и вернется к тому, что спрашивала.
        if goal != "answer_or_react_keep_step":
            self.dialog_state.last_goal = goal
        else:
            self.dialog_state.last_goal = previous_goal

        # 5. Создаем dynamic_prompt.
        dynamic_prompt = build_dynamic_prompt(
            user_text=user_input,
            state=self.dialog_state,
            goal=goal,
        )

        # Для отладки можно временно включить:
        print("\n--- DEBUG ---")
        print("intent:", intent)
        print("goal:", goal)
        print("step:", self.dialog_state.step)
        print("identity_confirmed:", self.dialog_state.identity_confirmed)
        print("refusal_stage:", self.dialog_state.refusal_stage)
        print("--- END DEBUG ---\n")

        # 6. Отправляем модели историю + текущую инструкцию.
        messages_for_model = self.build_messages_for_model(dynamic_prompt)

        response = self.ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=messages_for_model,
            options={
                "temperature": 0.1,
                "num_predict": 160,
                "repeat_penalty": 1.1,
                "top_k": 40,
                "top_p": 0.8,
                "min_p": 0.00,
                "num_ctx": 8192,
            },
            think=False,
        )

        bot_text = clean_reply(response["message"]["content"])

        # 7. Добавляем ответ ассистента в UI и историю.
        self.update_chat_history(f"агент:{bot_text}", bot_text, "assistant")

        # 8. Сохраняем последнее сообщение.
        self.dialog_state.last_bot_reply = bot_text

        # 9. Если диалог завершен, блокируем ввод.
        if self.dialog_state.ended:
            self.input_field.setEnabled(False)
            self.send_btn.setEnabled(False)

    def update_chat_history(self, text, text_only, role):
        self.context.append({"role": role, "content": text_only})
        print(text_only)
        self.render_message(role, text_only)

    def render_message(self, role: str, content: str):
        if role == "user":
            role_html = f'<span style="color:blue; font-weight: bold;">{role}:</span>'
        elif role == "assistant":
            role_html = f'<span style="color:green; font-weight: bold;">{role}:</span>'
        else:
            role_html = f'<span style="color:black; font-weight: bold;">{role}:</span>'

        message_html = f'{role_html} <span style="color: black;">{content}</span><br>'
        self.chat_history.insertHtml(message_html)

    def closeEvent(self, event):
        if self.model is not None:
            try:
                import torch
                del self.model
                torch.cuda.empty_cache()
                print("Модель удалена из памяти")
            except Exception:
                pass

        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec_())

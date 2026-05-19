# bank_agent_pyqt_simplified_intent_v1.py
# Упрощённая архитектура:
# - 6 базовых intent: accept, reject, question, contact, other, rude_or_offtopic
# - solution отдельно: current, payment_soon, sources_help, partial_payment, restructuring, vehicle_transfer, unknown
# - phase вместо большого количества step
# - action вместо goal_instruction на каждый маленький шаг
# - final_persuasion: заданное количество свободных реплик, потом ask_of_refuses_all, потом контакты
#
# Установка:
#   pip install PyQt5 ollama
#
# Ollama:
#   ollama serve
#   ollama pull qwen3:4b-instruct
#
# Запуск:
#   python bank_agent_pyqt_simplified_intent_v1.py

import sys
import re
import json
import html
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import ollama

from PyQt5.QtGui import QTextCursor
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


# ============================================================
# 1. НАСТРОЙКИ
# ============================================================

OLLAMA_HOST = "http://127.0.0.1:11434"

AGENT_MODEL = "t-8b-base"
INTENT_MODEL = "qwen3:4b-instruct"

SHOW_DEBUG = True

# Сколько свободных реплик будет в финальном убеждении
# перед вопросом "подтверждаете отказ..."
FINAL_PERSUASION_TURNS = 3


# ============================================================
# 2. ДАННЫЕ
# ============================================================

PUBLIC_DATA = {
    "bank_name": "АО Da банк",
    "agent_name": "Полина",
    "callback_phone": "88005553535",
    "target_name": "Петр Петрович",
    "target_full_name": "Петров Петр Петрович",
}

# Эти данные передаются агенту только после подтверждения личности.
PRIVATE_DATA = {
    "debt_amount": "15000 рублей",
    "loan_type": "автокредит",
    "overdue_days": "10 дней",
    "collateral_value": "2310000 рублей",
}


# ============================================================
# 3. ВАРИАНТЫ УРЕГУЛИРОВАНИЯ
# ============================================================

OFFERS = [
    {
        "id": "payment_soon",
        "title": "оплата в ближайшие три дня",
        "question": "Сможете ли внести платеж в ближайшие три дня?",
        "instruction": (
            "Спроси, сможет ли клиент внести платеж в ближайшие три дня."
        ),
    },
    {
        "id": "sources_help",
        "title": "помощь родственников или друзей, заем средств либо перекредитование",
        "question": (
            "Сможете ли рассмотреть помощь родственников или друзей, "
            "возможность занять средства либо перекредитоваться?"
        ),
        "instruction": (
            "Предложи рассмотреть помощь родственников или друзей, возможность занять средства "
            "или перекредитоваться. Это один общий вариант. Задай один вопрос."
        ),
    },
    {
        "id": "partial_payment",
        "title": "частичная оплата задолженности",
        "question": "Какую часть задолженности сможете внести и когда?",
        "instruction": (
            "Предложи частичную оплату как отдельный вариант, если полной суммы сейчас нет. "
            "Спроси, какую часть клиент сможет внести и когда."
        ),
    },
    {
        "id": "restructuring",
        "title": "реструктуризация кредита",
        "question": "Готовы ли рассмотреть реструктуризацию?",
        "instruction": (
            "Предложи реструктуризацию как отдельный вариант: продление срока кредита "
            "для снижения ежемесячного платежа, ставка от 18,9% годовых. "
            "Спроси, готов ли клиент рассмотреть реструктуризацию."
        ),
    },
    {
        "id": "vehicle_transfer",
        "title": "передача автомобиля для дальнейшего урегулирования",
        "question": "Готовы ли рассмотреть передачу автомобиля как вариант урегулирования?",
        "instruction": (
            "Предложи передачу автомобиля как отдельный вариант урегулирования. "
            "Кратко объясни, что при хорошем состоянии и наличии хода банк может "
            "рассмотреть принятие автомобиля для дальнейшей реализации. Задай один вопрос."
        ),
    },
]

SOLUTION_LABELS = {
    "payment_soon": "оплата в ближайшие три дня",
    "sources_help": "помощь родственников или друзей, заем средств либо перекредитование",
    "partial_payment": "частичная оплата задолженности",
    "restructuring": "реструктуризация кредита",
    "vehicle_transfer": "передача автомобиля для дальнейшего урегулирования",
    "unknown": "добровольное урегулирование",
}


# ============================================================
# 4. STATE
# ============================================================

@dataclass
class DialogState:
    # phases:
    # identity
    # reason
    # offers
    # final_persuasion
    # final_confirm_refusal
    # contacts
    # contact_details
    # ended
    phase: str = "identity"

    identity_confirmed: bool = False
    ended: bool = False

    offer_index: int = 0

    debt_reason: Optional[str] = None

    selected_solution: Optional[str] = None
    selected_solution_detail: Optional[str] = None

    final_persuasion_left: int = 0

    contact_status: Optional[str] = None
    contact_info: Optional[str] = None

    last_action: Optional[str] = None
    last_bot_reply: Optional[str] = None


# ============================================================
# 5. УТИЛИТЫ
# ============================================================

def normalize(text: str) -> str:
    text = str(text).lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9@\.\+\-\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_reply(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"^(assistant|оператор|бот|ответ)\s*:\s*", "", text, flags=re.I)
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip().strip('"').strip()


def safe_json_loads_from_text(text: str) -> Dict[str, Any]:
    text = str(text).strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return default_intent_result("JSON не найден")

    try:
        return json.loads(match.group(0))
    except Exception:
        return default_intent_result("JSON поврежден")


def default_intent_result(reason: str = "fallback") -> Dict[str, Any]:
    return {
        "intent": "other",
        "solution": "unknown",
        "contact_status": "unknown",
        "extracted_value": None,
        "confidence": 0.0,
        "short_reason": reason,
    }


def normalize_intent_result(data: Dict[str, Any]) -> Dict[str, Any]:
    allowed_intents = {
        "accept",
        "reject",
        "question",
        "contact",
        "other",
        "rude_or_offtopic",
    }

    allowed_solutions = {
        "current",
        "payment_soon",
        "sources_help",
        "partial_payment",
        "restructuring",
        "vehicle_transfer",
        "unknown",
    }

    allowed_contact_status = {
        "changed",
        "same",
        "provided",
        "refused",
        "unknown",
    }

    intent = str(data.get("intent", "other")).strip()
    solution = str(data.get("solution", "unknown")).strip()
    contact_status = str(data.get("contact_status", "unknown")).strip()

    if intent not in allowed_intents:
        intent = "other"

    if solution not in allowed_solutions:
        solution = "unknown"

    if contact_status not in allowed_contact_status:
        contact_status = "unknown"

    try:
        confidence = float(data.get("confidence", 0.0) or 0.0)
    except Exception:
        confidence = 0.0

    return {
        "intent": intent,
        "solution": solution,
        "contact_status": contact_status,
        "extracted_value": data.get("extracted_value"),
        "confidence": confidence,
        "short_reason": str(data.get("short_reason", "")),
    }


def contains_contact_details(text: str) -> bool:
    t = normalize(text)

    if "@" in t and "." in t:
        return True

    digits = re.sub(r"\D", "", t)
    if len(digits) >= 7:
        return True

    markers = [
        "звоните", "перезвоните", "пишите", "после", "до ", "вечером",
        "утром", "днем", "днём", "завтра", "сегодня", "в обед",
    ]
    return any(marker in t for marker in markers)


def get_visible_facts(state: DialogState) -> Dict[str, str]:
    facts = dict(PUBLIC_DATA)
    if state.identity_confirmed:
        facts.update(PRIVATE_DATA)
    return facts


def get_current_offer(state: DialogState) -> Optional[Dict[str, str]]:
    if 0 <= state.offer_index < len(OFFERS):
        return OFFERS[state.offer_index]
    return None


def solution_label(solution_id: Optional[str]) -> str:
    if not solution_id:
        return "вариант не выбран"
    return SOLUTION_LABELS.get(solution_id, solution_id)


def resolve_solution(intent_result: Dict[str, Any], state: DialogState) -> str:
    solution = intent_result.get("solution", "unknown")

    if solution == "current":
        current_offer = get_current_offer(state)
        if current_offer:
            return current_offer["id"]
        return "unknown"

    return solution


def has_concrete_solution(solution: str) -> bool:
    return solution in {
        "payment_soon",
        "sources_help",
        "partial_payment",
        "restructuring",
        "vehicle_transfer",
    }


# ============================================================
# 6. INTENT PROMPT
# ============================================================

def build_intent_prompt(state: DialogState, user_text: str) -> str:
    current_offer = get_current_offer(state)

    return f"""
Ты классификатор ответа клиента в банковском диалоге.

Верни строго JSON:
{{
  "intent": "accept | reject | question | contact | other | rude_or_offtopic",
  "solution": "current | payment_soon | sources_help | partial_payment | restructuring | vehicle_transfer | unknown",
  "contact_status": "changed | same | provided | refused | unknown",
  "extracted_value": null,
  "confidence": 0.0,
  "short_reason": "короткое объяснение"
}}

Значение intent:
- accept: клиент согласился. В фазе identity это значит, что клиент подтвердил личность. В фазах offers/final_persuasion/final_confirm_refusal это значит, что клиент согласился на вариант урегулирования или подтверждает что-то.
- reject: клиент отказался. В фазе identity это значит, что это не тот человек или он отказывается подтверждать. В фазах offers/final_persuasion это отказ от предложения.
- question: клиент задаёт вопрос или уточняет условия.
- contact: клиент отвечает на вопрос про контактную информацию.
- other: обычная реплика, не согласие, не отказ, не вопрос.
- rude_or_offtopic: явная грубость или совсем посторонняя тема.

Значение solution:
- current: клиент согласился на текущий предложенный вариант.
- payment_soon: клиент готов оплатить в ближайшие дни / завтра / в течение трех дней / найти деньги к сроку.
- sources_help: клиент согласен попросить родственников/друзей, занять деньги или перекредитоваться.
- partial_payment: клиент согласен на частичную оплату или называет сумму частичной оплаты.
- restructuring: клиент согласен на реструктуризацию.
- vehicle_transfer: клиент согласен на передачу автомобиля.
- unknown: непонятно, на что именно согласился, или решения нет.

Значение contact_status:
- changed: клиент сказал, что контактная информация изменилась, но не назвал сами данные.
- same: клиент сказал, что контактная информация не изменилась.
- provided: клиент дал телефон, email, адрес или удобное время связи.
- refused: клиент отказался уточнять контактную информацию.
- unknown: контактная информация не обсуждается.

Правила:
- Если клиент задаёт вопрос по текущему предложению, intent = question, даже если он сомневается.
- Если клиент просто говорит "да", "хорошо", "согласен", "давайте" в фазе offers, solution = current.
- Если клиент говорит "да" в фазе identity, intent = accept.
- Если клиент в конце говорит "хорошо, попробую найти деньги до завтра", intent = accept, solution = payment_soon.
- Если клиент говорит "давайте реструктуризацию", intent = accept, solution = restructuring.
- Если клиент говорит "могу внести часть", intent = accept, solution = partial_payment.
- Если клиент говорит "попробую у родственников занять", intent = accept, solution = sources_help.
- Если клиент говорит "готов передать авто", intent = accept, solution = vehicle_transfer.
- Если фаза contacts или contact_details, приоритетно классифицируй ответ как contact.

Текущая phase:
{state.phase}

Личность подтверждена:
{state.identity_confirmed}

Текущий предложенный вариант:
{current_offer}

Все варианты:
{OFFERS}

Последняя реплика агента:
{state.last_bot_reply}

Ответ клиента:
{user_text}
""".strip()


# ============================================================
# 7. ROUTER
# ============================================================

def route(state: DialogState, intent_result: Dict[str, Any], user_text: str) -> str:
    intent = intent_result["intent"]
    solution = resolve_solution(intent_result, state)
    contact_status = intent_result.get("contact_status", "unknown")
    extracted = intent_result.get("extracted_value") or user_text

    # Если Qwen не уверен, но в тексте явно контактные данные,
    # не теряем контакт.
    if state.phase in ["contacts", "contact_details"] and contains_contact_details(user_text):
        intent = "contact"
        contact_status = "provided"
        extracted = user_text

    if state.ended:
        return "end_call"

    # ========================================================
    # PHASE: IDENTITY
    # ========================================================

    if state.phase == "identity":
        if intent == "accept":
            state.identity_confirmed = True
            state.phase = "reason"
            return "ask_reason"

        if intent == "question":
            return "privacy_answer"

        if intent == "reject":
            state.ended = True
            state.phase = "ended"
            return "goodbye_no_details"

        if intent == "rude_or_offtopic":
            return "ask_identity_again"

        return "ask_identity_again"

    # ========================================================
    # PHASE: REASON
    # ========================================================

    if state.phase == "reason":
        if intent == "question":
            return "answer_question"

        if intent == "rude_or_offtopic":
            return "return_to_reason"

        # Любая содержательная реплика считается причиной неоплаты.
        state.debt_reason = str(extracted).strip()
        state.phase = "offers"
        state.offer_index = 0
        return "offer"

    # ========================================================
    # ГЛОБАЛЬНЫЙ ACCEPT В ПЕРЕГОВОРНЫХ ФАЗАХ
    # ========================================================

    if state.phase in ["offers", "final_persuasion", "final_confirm_refusal"]:
        # В финальном вопросе "Вы подтверждаете отказ..." простое "да"
        # не должно считаться согласием на вариант. Это подтверждение отказа.
        if state.phase == "final_confirm_refusal":
            if intent == "accept" and not has_concrete_solution(solution):
                state.phase = "contacts"
                state.selected_solution = None
                state.selected_solution_detail = None
                return "ask_contacts_no_agreement"

        if intent == "accept":
            if not has_concrete_solution(solution):
                current_offer = get_current_offer(state)
                solution = current_offer["id"] if current_offer else "unknown"

            state.selected_solution = solution
            state.selected_solution_detail = str(extracted).strip()
            state.phase = "contacts"
            return "ask_contacts_after_agreement"

    # ========================================================
    # PHASE: OFFERS
    # ========================================================

    if state.phase == "offers":
        if intent == "question":
            return "answer_question"

        if intent == "rude_or_offtopic":
            return "return_to_offer"

        if intent == "reject":
            state.offer_index += 1

            if state.offer_index < len(OFFERS):
                return "offer"

            return start_final_persuasion(state)

        return "repeat_offer"

    # ========================================================
    # PHASE: FINAL_PERSUASION
    # ========================================================

    if state.phase == "final_persuasion":
        if intent == "question":
            return "answer_question_free"

        if intent == "rude_or_offtopic":
            return use_final_persuasion_turn(state)

        if intent == "reject":
            return use_final_persuasion_turn(state)

        # other тоже считаем продолжением финального убеждения.
        return use_final_persuasion_turn(state)

    # ========================================================
    # PHASE: FINAL_CONFIRM_REFUSAL
    # ========================================================

    if state.phase == "final_confirm_refusal":
        if intent == "question":
            return "answer_question_final_refusal"

        # Если после финального вопроса клиент не выбрал конкретный вариант,
        # идём к контактам без договорённости.
        state.phase = "contacts"
        state.selected_solution = None
        state.selected_solution_detail = None
        return "ask_contacts_no_agreement"

    # ========================================================
    # PHASE: CONTACTS
    # ========================================================

    if state.phase == "contacts":
        if intent == "contact":
            if contact_status == "changed":
                state.phase = "contact_details"
                state.contact_status = "changed"
                return "ask_contact_details"

            if contact_status == "same":
                state.contact_status = "same"
                state.contact_info = "контактная информация не изменилась"
                state.ended = True
                state.phase = "ended"
                return "summary"

            if contact_status == "provided":
                state.contact_status = "provided"
                state.contact_info = str(extracted).strip()
                state.ended = True
                state.phase = "ended"
                return "summary"

            if contact_status == "refused":
                state.contact_status = "refused"
                state.contact_info = "клиент отказался уточнять контактную информацию"
                state.ended = True
                state.phase = "ended"
                return "summary"

        if contains_contact_details(user_text):
            state.contact_status = "provided"
            state.contact_info = user_text
            state.ended = True
            state.phase = "ended"
            return "summary"

        return "ask_contacts_again"

    # ========================================================
    # PHASE: CONTACT_DETAILS
    # ========================================================

    if state.phase == "contact_details":
        if intent == "contact":
            if contact_status == "provided":
                state.contact_status = "provided"
                state.contact_info = str(extracted).strip()
                state.ended = True
                state.phase = "ended"
                return "summary"

            if contact_status == "same":
                state.contact_status = "same"
                state.contact_info = "клиент уточнил, что контактная информация всё же не изменилась"
                state.ended = True
                state.phase = "ended"
                return "summary"

            if contact_status == "refused":
                state.contact_status = "refused"
                state.contact_info = "клиент отказался назвать актуальные контактные данные"
                state.ended = True
                state.phase = "ended"
                return "summary"

        if contains_contact_details(user_text):
            state.contact_status = "provided"
            state.contact_info = user_text
            state.ended = True
            state.phase = "ended"
            return "summary"

        return "ask_contact_details"

    return "repeat"


def start_final_persuasion(state: DialogState) -> str:
    state.phase = "final_persuasion"
    state.final_persuasion_left = FINAL_PERSUASION_TURNS
    return use_final_persuasion_turn(state)


def use_final_persuasion_turn(state: DialogState) -> str:
    if state.final_persuasion_left > 0:
        state.final_persuasion_left -= 1
        return "free_persuasion"

    state.phase = "final_confirm_refusal"
    return "ask_of_refuses_all"


# ============================================================
# 8. PROMPT ДЛЯ ОСНОВНОГО АГЕНТА
# ============================================================

def build_agent_prompt(
    state: DialogState,
    action: str,
    intent_result: Dict[str, Any],
    user_text: str,
) -> str:
    facts = get_visible_facts(state)
    current_offer = get_current_offer(state)

    if state.identity_confirmed:
        privacy = (
            "Личность подтверждена. Можно обсуждать задолженность, кредит, сумму, "
            "автомобиль и варианты урегулирования."
        )
    else:
        privacy = (
            "Личность НЕ подтверждена. Нельзя раскрывать причину звонка, долг, кредит, "
            "просрочку, сумму, автомобиль, суд, взыскание или любые финансовые детали."
        )

    action_rules = {
        "ask_identity_again": (
            "Повтори просьбу уточнить, разговариваешь ли ты с Петром Петровичем. "
            "Не раскрывай причину звонка."
        ),
        "privacy_answer": (
            "Скажи, что информация предназначена только для Петра Петровича, "
            "и попроси подтвердить, он ли это. Не раскрывай причину звонка."
        ),
        "goodbye_no_details": (
            "Вежливо извинись за беспокойство и попрощайся. Не раскрывай детали."
        ),
        "ask_reason": (
            "Сообщи, что разговор записывается. Назови задолженность, тип кредита "
            "и срок просрочки. Спроси причину неоплаты."
        ),
        "return_to_reason": (
            "Спокойно верни клиента к вопросу о причине неоплаты."
        ),
        "offer": (
            f"Предложи текущий вариант урегулирования: {current_offer}. "
            "Не предлагай другие варианты в этой реплике."
        ),
        "repeat_offer": (
            f"Кратко повтори текущий вопрос по варианту: {current_offer}. "
            "Не переходи к следующему варианту."
        ),
        "return_to_offer": (
            f"Спокойно верни клиента к текущему варианту урегулирования: {current_offer}."
        ),
        "answer_question": (
            "Ответь на вопрос клиента по смыслу. Затем вернись к текущему предложению "
            "или текущему вопросу. Не переходи к следующему варианту."
        ),
        "answer_question_free": (
            "Ответь на вопрос клиента по смыслу. Затем продолжи мягкое свободное убеждение, "
            "без жесткого сценарного вопроса."
        ),
        "free_persuasion": (
            "Это свободный финальный этап переговоров. Естественно и без шаблона попробуй "
            "убедить клиента выбрать добровольный вариант урегулирования. "
            "Не перечисляй все варианты каждый раз. Не задавай жесткий сценарный вопрос. "
            "Не угрожай. Не повторяй предыдущую реплику."
        ),
        "ask_of_refuses_all": (
            "Задай последний контрольный вопрос перед контактами: правильно ли я понимаю, "
            "что вы подтверждаете отказ от оплаты и от всех предложенных вариантов "
            "урегулирования, а также понимаете возможные юридические последствия? "
            "Не спрашивай контакты в этой реплике."
        ),
        "answer_question_final_refusal": (
            "Кратко ответь на вопрос клиента и снова задай последний контрольный вопрос "
            "о подтверждении отказа от оплаты и всех вариантов урегулирования."
        ),
        "ask_contacts_after_agreement": (
            "Кратко зафиксируй предварительную договоренность. Затем обязательно спроси, "
            "изменилась ли контактная информация клиента. Если изменилась, попроси сказать, "
            "что именно изменилось."
        ),
        "ask_contacts_no_agreement": (
            "Кратко зафиксируй, что на данный момент договоренность не достигнута. "
            "Затем обязательно спроси, изменилась ли контактная информация клиента. "
            "Если изменилась, попроси сказать, что именно изменилось."
        ),
        "ask_contacts_again": (
            "Повтори вопрос: изменилась ли контактная информация клиента. "
            "Если изменилась, попроси назвать актуальные данные."
        ),
        "ask_contact_details": (
            "Клиент сказал, что контактная информация изменилась, но не назвал новые данные. "
            "Попроси назвать актуальный номер телефона, email или удобное время связи."
        ),
        "repeat": (
            "Кратко попроси клиента уточнить ответ."
        ),
        "end_call": (
            "Кратко попрощайся."
        ),
    }

    return f"""
Ты банковский агент Полина.

Доступные факты:
{facts}

Режим конфиденциальности:
{privacy}

Текущее состояние:
phase={state.phase}
identity_confirmed={state.identity_confirmed}
offer_index={state.offer_index}
current_offer={current_offer}
debt_reason={state.debt_reason}
selected_solution={state.selected_solution}
selected_solution_detail={state.selected_solution_detail}
final_persuasion_left={state.final_persuasion_left}
contact_status={state.contact_status}
contact_info={state.contact_info}

Последняя реплика клиента:
{user_text}

Intent:
{intent_result}

Action:
{action}

Что сделать:
{action_rules.get(action, "Кратко и вежливо продолжи разговор.")}

Правила:
- Ответь только репликой агента, без JSON и без пояснений.
- 1-2 коротких предложения.
- Не задавай больше одного вопроса.
- Не повторяй предыдущие фразы.
- Не называй себя ботом.
- Не угрожай.
- Не упоминай полицию, уголовное дело, арест или выезд сотрудников.
- Если личность не подтверждена, не раскрывай финансовые детали.
- После подтверждения личности не проси паспорт, дату рождения, код из SMS или другие лишние персональные данные.
""".strip()


def build_free_agent_messages(context: List[Dict[str, str]], state: DialogState, user_text: str) -> List[Dict[str, str]]:
    """
    Для final_persuasion специально не используем большой action prompt.
    Даём модели короткий свободный режим и последние реплики,
    чтобы она меньше повторялась.
    """
    system_prompt = f"""
Ты банковский агент Полина. Сейчас свободный финальный этап переговоров.

Клиент отказался от стандартных вариантов, но ещё можно попытаться договориться.
Разговаривай естественно, как живой оператор.
Отвечай на последнюю реплику клиента по смыслу.
Пытайся мягко убедить выбрать добровольное урегулирование.
Не следуй жёсткому сценарию.
Не повторяй предыдущие фразы.
Не перечисляй все варианты каждый раз.
Не задавай один и тот же вопрос.
Не угрожай и не дави.

Факты, которые можно использовать:
{get_visible_facts(state)}

Причина неоплаты:
{state.debt_reason}

Оставшиеся свободные реплики:
{state.final_persuasion_left}
""".strip()

    messages = [{"role": "system", "content": system_prompt}]

    # Только последняя часть истории, чтобы модель не копировала старые повторы.
    recent_context = [
        m for m in context[-8:]
        if m.get("role") in ["user", "assistant"]
    ]

    messages.extend(recent_context)

    return messages


# ============================================================
# 9. SUMMARY
# ============================================================

def build_summary(state: DialogState) -> str:
    reason = state.debt_reason or "причина неоплаты не указана"
    contacts = state.contact_info or "информация по контактам не уточнена"

    if state.selected_solution:
        return (
            "Хорошо, подытожу договоренность. "
            f"Причина неоплаты: {reason}. "
            f"Согласованный вариант урегулирования: {solution_label(state.selected_solution)}. "
            f"Детали: {state.selected_solution_detail or 'без дополнительных деталей'}. "
            f"Контактная информация: {contacts}. "
            "Информацию зафиксировала, спасибо за разговор."
        )

    return (
        "Хорошо, подытожу разговор. "
        f"Причина неоплаты: {reason}. "
        "На данный момент договоренность по варианту урегулирования не достигнута. "
        f"Контактная информация: {contacts}. "
        "Информацию зафиксировала, спасибо за разговор."
    )


# ============================================================
# 10. PYQT WINDOW
# ============================================================

class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.state = DialogState()
        self.ollama_client = None

        self.context = [
            {
                "role": "system",
                "content": (
                    "Ты банковский агент Полина. До подтверждения личности не раскрывай "
                    "финансовые детали. После подтверждения веди переговоры по задолженности "
                    "кратко, делово и спокойно."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Алло, здравствуйте, меня зовут Полина, я сотрудник Da банк. "
                    "Петров Петр Петрович — это вы?"
                ),
            },
        ]

        self.init_ui()
        self.load_model()

    def init_ui(self):
        self.setWindowTitle("Chat bot simplified")
        self.setGeometry(100, 100, 720, 850)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)

        scroll = QScrollArea()
        scroll.setWidget(self.chat_history)
        scroll.setWidgetResizable(True)

        layout.addWidget(scroll)

        for msg in self.context[1:]:
            self.render_message(msg["role"], msg["content"])

        self.input_field = QLineEdit()
        self.input_field.returnPressed.connect(self.send_message)
        layout.addWidget(self.input_field)

        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_message)
        layout.addWidget(self.send_btn)

    def load_model(self):
        self.ollama_client = ollama.Client(
            host=OLLAMA_HOST,
            trust_env=False,
        )

    def classify_intent(self, user_text: str) -> Dict[str, Any]:
        prompt = build_intent_prompt(self.state, user_text)

        try:
            response = self.ollama_client.chat(
                model=INTENT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Ты строгий JSON-классификатор. "
                            "Отвечай только валидным JSON без markdown."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                options={
                    "temperature": 0,
                    "num_predict": 220,
                    "repeat_penalty": 1.05,
                    "top_k": 20,
                    "top_p": 0.1,
                    "num_ctx": 4096,
                },
                think=False,
            )

            data = safe_json_loads_from_text(response["message"]["content"])
            result = normalize_intent_result(data)

        except Exception as e:
            result = default_intent_result(f"Ошибка intent-модели: {e}")

        # Контактная страховка.
        if self.state.phase in ["contacts", "contact_details"] and contains_contact_details(user_text):
            result["intent"] = "contact"
            result["contact_status"] = "provided"
            result["extracted_value"] = user_text
            result["confidence"] = max(result.get("confidence", 0.0), 0.8)

        return result

    def ask_agent(self, action: str, intent_result: Dict[str, Any], user_text: str) -> str:
        if action == "summary":
            return build_summary(self.state)

        if action == "free_persuasion":
            messages = build_free_agent_messages(
                context=self.context,
                state=self.state,
                user_text=user_text,
            )

            response = self.ollama_client.chat(
                model=AGENT_MODEL,
                messages=messages,
                options={
                    "temperature": 0.75,
                    "num_predict": 220,
                    "repeat_penalty": 1.35,
                    "repeat_last_n": 2048,
                    "top_k": 100,
                    "top_p": 0.95,
                    "min_p": 0.03,
                    "num_ctx": 8192,
                },
                think=False,
            )

            return clean_reply(response["message"]["content"])

        prompt = build_agent_prompt(
            state=self.state,
            action=action,
            intent_result=intent_result,
            user_text=user_text,
        )

        messages = list(self.context)
        messages.append({
            "role": "user",
            "content": (
                "Текущая инструкция имеет приоритет над историей диалога.\n\n"
                + prompt
            ),
        })

        response = self.ollama_client.chat(
            model=AGENT_MODEL,
            messages=messages,
            options={
                "temperature": 0.2,
                "num_predict": 190,
                "repeat_penalty": 1.15,
                "top_k": 60,
                "top_p": 0.85,
                "min_p": 0.0,
                "num_ctx": 8192,
            },
            think=False,
        )

        return clean_reply(response["message"]["content"])

    def send_message(self):
        user_text = self.input_field.text().strip()

        if not user_text:
            return

        self.input_field.clear()
        self.input_field.setEnabled(False)
        self.send_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            self.update_chat_history(user_text, "user")

            intent_result = self.classify_intent(user_text)

            if SHOW_DEBUG:
                self.render_debug(intent_result)

            action = route(self.state, intent_result, user_text)
            self.state.last_action = action

            print("\n--- DEBUG ---")
            print("phase:", self.state.phase)
            print("action:", action)
            print("intent:", intent_result)
            print("offer_index:", self.state.offer_index)
            print("selected_solution:", self.state.selected_solution)
            print("final_persuasion_left:", self.state.final_persuasion_left)
            print("contact_status:", self.state.contact_status)
            print("--- END DEBUG ---\n")

            bot_text = self.ask_agent(action, intent_result, user_text)

            self.update_chat_history(bot_text, "assistant")
            self.state.last_bot_reply = bot_text

            if self.state.ended:
                self.input_field.setEnabled(False)
                self.send_btn.setEnabled(False)
            else:
                self.input_field.setEnabled(True)
                self.send_btn.setEnabled(True)
                self.input_field.setFocus()

        except Exception as e:
            error_text = f"Ошибка обработки сообщения: {e}"
            print(error_text)
            self.update_chat_history(error_text, "system")
            self.input_field.setEnabled(True)
            self.send_btn.setEnabled(True)
            self.input_field.setFocus()

    def update_chat_history(self, content: str, role: str):
        self.context.append({
            "role": role,
            "content": content,
        })

        print(content)
        self.render_message(role, content)

    def render_message(self, role: str, content: str):
        safe_content = html.escape(str(content)).replace("\n", "<br>")

        if role == "user":
            role_html = '<span style="color:blue; font-weight: bold;">user:</span>'
        elif role == "assistant":
            role_html = '<span style="color:green; font-weight: bold;">assistant:</span>'
        else:
            role_html = '<span style="color:black; font-weight: bold;">system:</span>'

        self.chat_history.insertHtml(
            f'{role_html} <span style="color:black;">{safe_content}</span><br>'
        )
        self.chat_history.moveCursor(QTextCursor.End)

    def render_debug(self, intent_result: Dict[str, Any]):
        debug = (
            f'INTENT: {intent_result.get("intent")}, '
            f'SOLUTION: {intent_result.get("solution")}, '
            f'CONTACT: {intent_result.get("contact_status")}, '
            f'CONF: {intent_result.get("confidence")}, '
            f'REASON: {intent_result.get("short_reason")}'
        )

        safe_debug = html.escape(debug)
        self.chat_history.insertHtml(
            f'<span style="color:gray;">{safe_debug}</span><br>'
        )
        self.chat_history.moveCursor(QTextCursor.End)

    def closeEvent(self, event):
        super().closeEvent(event)


# ============================================================
# 11. RUN
# ============================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = ChatWindow()
    window.show()

    sys.exit(app.exec_())

# bank_agent_pyqt_qwen_intent_v3.py
# PyQt + Ollama агент для сценария взыскания с отдельной Qwen intent-моделью.
#
# Что изменено относительно bank_agent_pyqt_full_v2.py:
# - intent больше не определяется большой пачкой if/elif: для этого вызывается Qwen через Ollama;
# - причины неоплаты оставлены как в исходном сценарии: после подтверждения личности агент спрашивает причину, затем спрашивает оплату в 3 дня;
# - варианты урегулирования взяты из исходного файла:
#     step 6: помощь родственников / занять / перекредитоваться;
#     step 7: частичная оплата;
#     step 8: реструктуризация;
#     step 9: передача автомобиля;
# - если клиента удалось убедить на любой вариант урегулирования, агент НЕ завершает сразу;
# - после договоренности агент обязательно спрашивает, изменилась ли контактная информация;
# - после ответа по контактам агент подытоживает договоренность и завершает разговор.
#
# Перед запуском:
#   pip install PyQt5 ollama
#   ollama serve
#   ollama pull qwen3:4b-instruct
#
# Поменяй под себя:
#   OLLAMA_HOST
#   AGENT_MODEL
#   INTENT_MODEL

import sys
import re
import json
import html
from dataclasses import dataclass
from typing import Dict, Optional, Any, List

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


OLLAMA_HOST = "http://127.0.0.1:11434"

# Основная модель, которая формулирует реплики агента.
AGENT_MODEL = "t-8b-base"

# Отдельная модель для классификации intent.
INTENT_MODEL = "qwen3:4b-instruct"

SHOW_DEBUG = True


# ============================================================
# 1. ДАННЫЕ
# ============================================================

PUBLIC_DATA = {
    "bank_name": "АО Da банк",
    "agent_name": "Полина",
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
    # 1 — предложили помощь родственников / занять / перекредитоваться
    # 2 — предложили частичную оплату
    # 3 — предложили реструктуризацию
    # 4 — предложили передачу авто
    # 5 — озвучили последствия
    # 6 — озвучили возможный суд
    refusal_stage: int = 0

    # Данные, которые нужны для итогового подытоживания.
    debt_reason: Optional[str] = None
    selected_solution: Optional[str] = None
    selected_solution_label: Optional[str] = None
    contact_update: Optional[str] = None
    vehicle_owner: Optional[str] = None
    vehicle_condition: Optional[str] = None

    last_goal: Optional[str] = None
    last_intent: Optional[str] = None
    last_bot_reply: Optional[str] = None


SOLUTION_LABELS = {
    "payment_3_days_yes": "полная оплата в ближайшие три дня",
    "agrees_sources_help": "помощь родственников или друзей, заем средств либо перекредитование",
    "agrees_partial_payment": "частичная оплата задолженности",
    "agrees_restructuring": "реструктуризация кредита",
    "vehicle_transfer_ready": "передача автомобиля для дальнейшего урегулирования",
}


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


def clean_reply(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"^(assistant|оператор|бот|ответ)\s*:\s*", "", text, flags=re.I)
    return text.strip().strip('"').strip()


def safe_json_loads_from_text(text: str) -> Dict[str, Any]:
    """
    Qwen иногда может вернуть не чистый JSON, а текст + JSON.
    Поэтому сначала пробуем json.loads, потом вытаскиваем {...}.
    """
    text = str(text).strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {
            "intent": "unclear",
            "confidence": 0.0,
            "is_answer_to_current_question": False,
            "extracted_value": None,
            "short_reason": "JSON не найден",
        }

    try:
        return json.loads(match.group(0))
    except Exception:
        return {
            "intent": "unclear",
            "confidence": 0.0,
            "is_answer_to_current_question": False,
            "extracted_value": None,
            "short_reason": "JSON повреждён",
        }


def normalize_intent_result(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "intent": str(data.get("intent", "unclear")),
        "confidence": float(data.get("confidence", 0.0) or 0.0),
        "is_answer_to_current_question": bool(data.get("is_answer_to_current_question", False)),
        "extracted_value": data.get("extracted_value"),
        "short_reason": str(data.get("short_reason", "")),
    }


# ============================================================
# 4. INTENT ЧЕРЕЗ QWEN
# ============================================================

def allowed_intents_for_state(state: DialogState) -> List[str]:
    """
    Qwen видит только те intent, которые разрешены на текущем шаге.
    Это защищает от перескоков между вариантами урегулирования.
    """
    common = [
        "asks_any_question",
        "rude_or_offtopic",
        "free_client_message",
        "unclear",
    ]

    if not state.identity_confirmed:
        if state.step == 1:
            return [
                "identity_confirmed",
                "identity_denied",
                "third_person_relative",
                "third_person_unknown",
                "asks_reason_before_identity",
                *common,
            ]

        if state.step == 2:
            return [
                "identity_confirmed",
                "identity_denied",
                "third_person_relative",
                "third_person_unknown",
                *common,
            ]

    # После подтверждения личности.
    if state.step == 4:
        return [
            "answer_debt_reason",
            "refuses_payment",
            "needs_time",
            *common,
        ]

    if state.step == 5:
        return [
            "payment_3_days_yes",
            "payment_3_days_no",
            "refuses_payment",
            "needs_time",
            *common,
        ]

    # step 6: помощь родственников / занять / перекредитоваться.
    if state.step == 6:
        return [
            "agrees_sources_help",
            "declines_sources_help",
            *common,
        ]

    # step 7: частичная оплата.
    if state.step == 7:
        return [
            "agrees_partial_payment",
            "declines_partial_payment",
            *common,
        ]

    # step 8: реструктуризация.
    if state.step == 8:
        return [
            "agrees_restructuring",
            "declines_restructuring",
            *common,
        ]

    # step 9: передача автомобиля.
    if state.step == 9:
        return [
            "agrees_vehicle_transfer",
            "declines_vehicle_transfer",
            *common,
        ]

    if state.step == 10:
        return [
            "vehicle_owner_answer",
            *common,
        ]

    if state.step == 11:
        return [
            "vehicle_condition_answer",
            *common,
        ]

    if state.step == 12:
        return [
            "vehicle_transfer_ready",
            "vehicle_transfer_not_ready",
            *common,
        ]

    # step 13: после последствий клиент может внезапно согласиться на один из вариантов.
    if state.step == 13:
        return [
            "agrees_sources_help",
            "agrees_partial_payment",
            "agrees_restructuring",
            "agrees_vehicle_transfer",
            "refuses_payment",
            "needs_time",
            *common,
        ]

    # step 14: обязательная проверка контактной информации.
    if state.step == 14:
        return [
            "contact_info_changed",
            "contact_info_same",
            "contact_info_provided",
            "declines_contact_info",
            *common,
        ]

    return common


def current_question_for_step(state: DialogState) -> str:
    questions = {
        1: "Петров Петр Петрович — это вы?",
        2: "Кем вы приходитесь Петру Петровичу?",
        4: "По какой причине не получилось оплатить задолженность?",
        5: "Сможете ли внести платеж в ближайшие три дня?",
        6: "Сможете ли рассмотреть помощь родственников или друзей, возможность занять средства либо перекредитоваться?",
        7: "Какую часть задолженности сможете внести и когда?",
        8: "Готовы ли рассмотреть реструктуризацию?",
        9: "Готовы ли рассмотреть передачу автомобиля как вариант урегулирования?",
        10: "На кого зарегистрирован автомобиль?",
        11: "В каком состоянии автомобиль?",
        12: "Сможете ли в течение трех дней подписать документы и передать автомобиль на стоянку?",
        13: "Готовы ли выбрать вариант урегулирования без суда?",
        14: "Подскажите, изменилась ли ваша контактная информация? Если да, назовите актуальные данные.",
    }
    return questions.get(state.step, "Уточните, пожалуйста, ваш ответ.")


def build_intent_system_prompt(allowed_intents: List[str]) -> str:
    return f"""
Ты intent-классификатор для диалога банковского агента с клиентом.

Твоя задача:
1. Определить intent последней реплики клиента.
2. Учитывать текущий шаг диалога и последний вопрос агента.
3. Не писать ответ клиенту.
4. Не выбирать следующий шаг.
5. Вернуть только JSON без markdown.

Доступные intent только из этого списка:
{allowed_intents}

Расшифровка intent:
- identity_confirmed: клиент подтвердил, что он Петр Петрович. Например: "да", "это я", "слушаю", "говорите".
- identity_denied: клиент говорит, что это не он.
- third_person_relative: отвечает родственник, друг, коллега или другое третье лицо.
- third_person_unknown: ошиблись номером, такого человека не знают.
- asks_reason_before_identity: до подтверждения личности клиент спрашивает причину звонка.
- asks_any_question: клиент задаёт вопрос по теме или уточняет.
- rude_or_offtopic: грубость, уход от разговора, совсем не по теме.
- free_client_message: обычная реплика, но не ответ на текущий вопрос.

После подтверждения личности:
- answer_debt_reason: клиент назвал причину неоплаты. Например: задержали зарплату, потерял работу, были расходы, болезнь, нет денег, забыл, не успел.
- payment_3_days_yes: клиент согласен внести платеж в ближайшие три дня.
- payment_3_days_no: клиент не может внести платеж в ближайшие три дня.
- refuses_payment: клиент отказывается платить вообще или говорит, что платить нечем.
- needs_time: клиент просит время, просит позже, просит перезвонить.

Варианты урегулирования:
- agrees_sources_help: клиент согласен попросить помощь родственников/друзей, занять деньги или перекредитоваться.
- declines_sources_help: клиент отказывается от помощи родственников/друзей, займа или перекредитования.
- agrees_partial_payment: клиент согласен на частичную оплату или называет сумму частичной оплаты.
- declines_partial_payment: клиент отказывается от частичной оплаты.
- agrees_restructuring: клиент согласен рассмотреть реструктуризацию.
- declines_restructuring: клиент отказывается от реструктуризации.
- agrees_vehicle_transfer: клиент согласен рассмотреть передачу автомобиля.
- declines_vehicle_transfer: клиент отказывается от передачи автомобиля.

Автомобиль:
- vehicle_owner_answer: клиент ответил, на кого зарегистрирован автомобиль.
- vehicle_condition_answer: клиент описал состояние автомобиля.
- vehicle_transfer_ready: клиент готов подписать документы и передать авто в течение трех дней.
- vehicle_transfer_not_ready: клиент не готов передать авто в течение трех дней.

Контакты:
- contact_info_changed: клиент говорит, что контактные данные изменились.
- contact_info_same: клиент говорит, что контактные данные не изменились / те же / актуальны.
- contact_info_provided: клиент дал новый телефон, email, адрес или удобное время связи.
- declines_contact_info: клиент отказался говорить по контактам.

Правила:
- Если клиент просто пишет "да", "угу", "хорошо", классифицируй по текущему вопросу агента.
- Если текущий вопрос про причину неоплаты, содержательный ответ клиента = answer_debt_reason.
- Если текущий вопрос про родственников/занять/перекредитоваться, согласие = agrees_sources_help.
- Если текущий вопрос про частичную оплату, согласие или сумма = agrees_partial_payment.
- Если текущий вопрос про реструктуризацию, согласие = agrees_restructuring.
- Если текущий вопрос про передачу автомобиля, согласие = agrees_vehicle_transfer.
- Если текущий вопрос про контактную информацию, "нет, не изменилась" = contact_info_same.
- Если клиент задаёт вопрос, is_answer_to_current_question = false.
- Если клиент грубит или уходит от темы, is_answer_to_current_question = false.
- Если не уверен, intent = "unclear", confidence ниже 0.6.
- Intent должен быть строго из списка доступных intent.

Формат ответа строго JSON:
{{
  "intent": "one_of_allowed_intents",
  "confidence": 0.0,
  "is_answer_to_current_question": true,
  "extracted_value": null,
  "short_reason": "короткое объяснение"
}}
""".strip()


# ============================================================
# 5. FSM: ВЫБОР ЦЕЛИ
# ============================================================

def choose_goal(intent_result: Dict[str, Any], state: DialogState, user_text: str) -> str:
    intent = intent_result.get("intent", "unclear")
    confidence = float(intent_result.get("confidence", 0.0))
    is_answer = bool(intent_result.get("is_answer_to_current_question", False))
    extracted = intent_result.get("extracted_value") or user_text

    if state.ended:
        return "end_call"

    # Если модель не уверена, не двигаем step.
    if confidence < 0.6 and intent not in ["asks_any_question", "rude_or_offtopic"]:
        return "safe_repeat"

    # До подтверждения личности.
    if not state.identity_confirmed:
        if state.step == 1:
            if intent == "identity_confirmed" and is_answer:
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

            if intent in ["asks_any_question", "rude_or_offtopic", "free_client_message", "unclear"]:
                return "answer_or_react_keep_step"

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

    # После подтверждения личности: вопросы, грубость, оффтопик не двигают step.
    if intent in ["asks_any_question", "rude_or_offtopic", "free_client_message", "unclear"]:
        return "answer_or_react_keep_step"

    # step 4: причина неоплаты.
    if state.step == 4:
        if intent == "answer_debt_reason" and is_answer:
            state.debt_reason = str(extracted).strip()
            state.step = 5
            return "ask_payment_3_days"

        if intent in ["refuses_payment", "needs_time"]:
            return refusal_goal(state)

        return "safe_repeat"

    # step 5: оплата в 3 дня.
    if state.step == 5:
        if intent == "payment_3_days_yes" and is_answer:
            record_agreement(state, "payment_3_days_yes")
            return "ask_contact_change_after_agreement"

        if intent in ["payment_3_days_no", "refuses_payment", "needs_time"]:
            return refusal_goal(state)

        return "safe_repeat"

    # step 6: помощь родственников / занять / перекредитоваться.
    if state.step == 6:
        if intent == "agrees_sources_help" and is_answer:
            record_agreement(state, "agrees_sources_help")
            return "ask_contact_change_after_agreement"

        if intent == "declines_sources_help":
            return decline_current_option(state, minimum_stage=1)

        return "safe_repeat"

    # step 7: частичная оплата.
    if state.step == 7:
        if intent == "agrees_partial_payment" and is_answer:
            detail = str(extracted).strip()
            record_agreement(state, "agrees_partial_payment", extra_detail=detail)
            return "ask_contact_change_after_agreement"

        if intent == "declines_partial_payment":
            return decline_current_option(state, minimum_stage=2)

        return "safe_repeat"

    # step 8: реструктуризация.
    if state.step == 8:
        if intent == "agrees_restructuring" and is_answer:
            record_agreement(state, "agrees_restructuring")
            return "ask_contact_change_after_agreement"

        if intent == "declines_restructuring":
            return decline_current_option(state, minimum_stage=3)

        return "safe_repeat"

    # step 9: передача автомобиля.
    if state.step == 9:
        if intent == "agrees_vehicle_transfer" and is_answer:
            state.step = 10
            state.state = "NEGOTIATION"
            return "ask_vehicle_owner"

        if intent == "declines_vehicle_transfer":
            return decline_current_option(state, minimum_stage=4)

        return "safe_repeat"

    # step 10: собственник авто.
    if state.step == 10:
        if intent == "vehicle_owner_answer" and is_answer:
            state.vehicle_owner = str(extracted).strip()
            state.step = 11
            return "ask_vehicle_condition"

        return "safe_repeat"

    # step 11: состояние авто.
    if state.step == 11:
        if intent == "vehicle_condition_answer" and is_answer:
            state.vehicle_condition = str(extracted).strip()
            state.step = 12
            return "ask_vehicle_transfer_3_days"

        return "safe_repeat"

    # step 12: готовность передать авто.
    if state.step == 12:
        if intent == "vehicle_transfer_ready" and is_answer:
            record_agreement(state, "vehicle_transfer_ready")
            return "ask_contact_change_after_agreement"

        if intent == "vehicle_transfer_not_ready":
            return decline_current_option(state, minimum_stage=4)

        return "safe_repeat"

    # step 13: после последствий клиент может согласиться на один из вариантов.
    if state.step == 13:
        if intent in [
            "agrees_sources_help",
            "agrees_partial_payment",
            "agrees_restructuring",
        ]:
            record_agreement(state, intent)
            return "ask_contact_change_after_agreement"

        if intent == "agrees_vehicle_transfer":
            state.step = 10
            state.state = "NEGOTIATION"
            return "ask_vehicle_owner"

        if intent in ["refuses_payment", "needs_time"]:
            return refusal_goal(state)

        return "answer_or_react_keep_step"

    # step 14: контакты перед итогом.
    if state.step == 14:
        if intent in ["contact_info_changed", "contact_info_same", "contact_info_provided", "declines_contact_info"]:
            if intent == "contact_info_same":
                state.contact_update = "контактная информация не изменилась"
            elif intent == "declines_contact_info":
                state.contact_update = "клиент отказался уточнять контактную информацию"
            else:
                state.contact_update = str(extracted).strip()

            state.ended = True
            state.state = "RESOLVED" if state.selected_solution else "END"
            return "summarize_agreement_and_end" if state.selected_solution else "goodbye_after_contacts"

        return "ask_contacts_again"

    return "safe_repeat"


def record_agreement(state: DialogState, solution_intent: str, extra_detail: Optional[str] = None) -> None:
    label = SOLUTION_LABELS.get(solution_intent, solution_intent)
    if extra_detail and extra_detail != solution_intent:
        label = f"{label}: {extra_detail}"

    state.selected_solution = solution_intent
    state.selected_solution_label = label
    state.state = "AGREEMENT_REACHED"
    state.step = 14


def decline_current_option(state: DialogState, minimum_stage: int) -> str:
    state.refusal_stage = max(state.refusal_stage, minimum_stage)
    return refusal_goal(state)


def refusal_goal(state: DialogState) -> str:
    """
    Лестница убеждения при отказе.
    Все варианты идут по отдельности, как в исходном файле.
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
        state.step = 13
        return "explain_possible_court"

    state.ended = True
    state.state = "END"
    return "close_no_agreement"


# ============================================================
# 6. PROMPT ДЛЯ ОСНОВНОЙ МОДЕЛИ
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
            "Прими причину неоплаты к сведению и спроси, сможет ли клиент внести платеж "
            "в ближайшие три дня."
        ),

        # Варианты урегулирования из исходного файла.
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

        # Авто.
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
            "порядке. Без угроз. После этого предложи выбрать вариант урегулирования."
        ),
        "explain_possible_court": (
            "Нейтрально скажи, что если договориться не получится, банк может рассмотреть "
            "обращение в суд в установленном законом порядке. Предложи выбрать вариант "
            "урегулирования без суда."
        ),

        # Новое обязательное звено после договоренности.
        "ask_contact_change_after_agreement": (
            "Кратко зафиксируй предварительную договоренность по выбранному варианту. "
            "Обязательно спроси, изменилась ли контактная информация клиента. Если изменилась, "
            "попроси назвать актуальный номер, адрес электронной почты или удобное время связи. "
            "Не завершай разговор и не подводи итог, пока клиент не ответит по контактам."
        ),
        "ask_contacts_again": (
            "Повтори вопрос: изменилась ли контактная информация клиента. Если да, попроси "
            "назвать актуальные данные. Не завершай разговор."
        ),
        "goodbye_after_contacts": (
            "Поблагодари за уточнение контактной информации и попрощайся."
        ),

        "answer_or_react_keep_step": (
            "Отреагируй на последнюю реплику клиента естественно и по ситуации. "
            "Если клиент задал вопрос — ответь на вопрос. Если клиент грубит — спокойно не спорь "
            "и верни разговор к текущему вопросу. Если клиент говорит не по теме — кратко верни "
            "разговор к задолженности. Не продвигай сценарий дальше. После ответа вернись к "
            "предыдущему вопросу оператора. Если личность не подтверждена, не раскрывай финансовые детали."
        ),
        "safe_repeat": (
            "Кратко попроси уточнить ответ по текущему вопросу, не добавляя новых деталей."
        ),
        "close_no_agreement": (
            "Кратко зафиксируй, что договоренность не достигнута, банк продолжит работу "
            "в установленном порядке. Попрощайся."
        ),
        "end_call": (
            "Кратко попрощайся."
        ),
    }
    return instructions.get(goal, "Кратко и вежливо продолжи разговор по текущей цели.")


def build_dynamic_prompt(user_text: str, state: DialogState, goal: str, intent_result: Dict[str, Any]) -> str:
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
debt_reason={state.debt_reason}
selected_solution={state.selected_solution_label}
contact_update={state.contact_update}
vehicle_owner={state.vehicle_owner}
vehicle_condition={state.vehicle_condition}

Последняя реплика клиента:
{user_text}

Qwen intent:
{intent_result}

Предыдущая цель/вопрос оператора:
{state.last_goal}

Текущая цель ответа:
{goal}

Текущий вопрос сценария:
{current_question_for_step(state)}

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


def build_summary_message(state: DialogState) -> str:
    reason = state.debt_reason or "причина неоплаты не указана"
    solution = state.selected_solution_label or "вариант урегулирования не выбран"
    contacts = state.contact_update or "информация по контактам не уточнена"

    extra_auto = ""
    if state.selected_solution == "vehicle_transfer_ready":
        owner = state.vehicle_owner or "собственник не уточнён"
        condition = state.vehicle_condition or "состояние не уточнено"
        extra_auto = f" Автомобиль: регистрация — {owner}; состояние — {condition}."

    return (
        "Хорошо, подытожу договоренность. "
        f"Причина неоплаты: {reason}. "
        f"Согласованный вариант урегулирования: {solution}."
        f"{extra_auto} "
        f"Контактная информация: {contacts}. "
        "Информацию зафиксировала, спасибо за разговор."
    )


# ============================================================
# 7. PYQT APP
# ============================================================

class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()

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
        self.setGeometry(100, 100, 700, 850)

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

    def classify_intent_with_qwen(self, user_input: str) -> Dict[str, Any]:
        allowed_intents = allowed_intents_for_state(self.dialog_state)
        system_prompt = build_intent_system_prompt(allowed_intents)

        user_prompt = f"""
Текущий state:
{self.dialog_state.state}

Текущий step:
{self.dialog_state.step}

Личность подтверждена:
{self.dialog_state.identity_confirmed}

Последний вопрос агента:
{current_question_for_step(self.dialog_state)}

Последняя реплика агента:
{self.dialog_state.last_bot_reply}

Ответ клиента:
{user_input}
""".strip()

        try:
            response = self.ollama_client.chat(
                model=INTENT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
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

            raw_content = response["message"]["content"]
            data = normalize_intent_result(safe_json_loads_from_text(raw_content))

        except Exception as e:
            data = {
                "intent": "unclear",
                "confidence": 0.0,
                "is_answer_to_current_question": False,
                "extracted_value": None,
                "short_reason": f"Ошибка intent-модели: {e}",
            }

        if data.get("intent") not in allowed_intents:
            data = {
                "intent": "unclear",
                "confidence": 0.0,
                "is_answer_to_current_question": False,
                "extracted_value": None,
                "short_reason": "Qwen вернул intent, который запрещён на текущем шаге",
            }

        # Небольшая страховка для причины неоплаты: как в исходном файле,
        # на step 4 почти любой содержательный ответ считается причиной,
        # если это не вопрос.
        if self.dialog_state.identity_confirmed and self.dialog_state.step == 4:
            text = normalize(user_input)
            if (
                data["intent"] in ["free_client_message", "unclear"]
                and len(text.split()) >= 2
                and not looks_like_question(user_input, text)
            ):
                data = {
                    "intent": "answer_debt_reason",
                    "confidence": 0.8,
                    "is_answer_to_current_question": True,
                    "extracted_value": user_input,
                    "short_reason": "Содержательный ответ на вопрос о причине неоплаты",
                }

        return data

    def send_message(self):
        user_input = self.input_field.text().strip()
        if not user_input:
            return

        self.input_field.clear()
        self.input_field.setEnabled(False)
        self.send_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            # 1. Добавляем реплику клиента в UI и историю.
            self.update_chat_history(f"клиент:{user_input}", user_input, "user")

            # 2. Qwen определяет intent.
            intent_result = self.classify_intent_with_qwen(user_input)

            if SHOW_DEBUG:
                self.render_debug(intent_result)

            # 3. FSM выбирает цель и обновляет state.
            previous_goal = self.dialog_state.last_goal
            goal = choose_goal(intent_result, self.dialog_state, user_input)
            self.dialog_state.last_intent = intent_result.get("intent")

            # Если клиент задал вопрос / сказал не по теме / грубит — не затираем предыдущую цель.
            if goal != "answer_or_react_keep_step":
                self.dialog_state.last_goal = goal
            else:
                self.dialog_state.last_goal = previous_goal

            print("\n--- DEBUG ---")
            print("intent:", intent_result)
            print("goal:", goal)
            print("step:", self.dialog_state.step)
            print("state:", self.dialog_state.state)
            print("identity_confirmed:", self.dialog_state.identity_confirmed)
            print("refusal_stage:", self.dialog_state.refusal_stage)
            print("debt_reason:", self.dialog_state.debt_reason)
            print("selected_solution:", self.dialog_state.selected_solution_label)
            print("contact_update:", self.dialog_state.contact_update)
            print("--- END DEBUG ---\n")

            # 4. Итог договоренности лучше строить кодом, чтобы модель ничего не потеряла.
            if goal == "summarize_agreement_and_end":
                bot_text = build_summary_message(self.dialog_state)
            else:
                dynamic_prompt = build_dynamic_prompt(
                    user_text=user_input,
                    state=self.dialog_state,
                    goal=goal,
                    intent_result=intent_result,
                )

                messages_for_model = self.build_messages_for_model(dynamic_prompt)

                response = self.ollama_client.chat(
                    model=AGENT_MODEL,
                    messages=messages_for_model,
                    options={
                        "temperature": 0.1,
                        "num_predict": 180,
                        "repeat_penalty": 1.1,
                        "top_k": 40,
                        "top_p": 0.8,
                        "min_p": 0.00,
                        "num_ctx": 8192,
                    },
                    think=False,
                )

                bot_text = clean_reply(response["message"]["content"])

            # 5. Добавляем ответ ассистента в UI и историю.
            self.update_chat_history(f"агент:{bot_text}", bot_text, "assistant")
            self.dialog_state.last_bot_reply = bot_text

            # 6. Если диалог завершен, блокируем ввод.
            if self.dialog_state.ended:
                self.input_field.setEnabled(False)
                self.send_btn.setEnabled(False)
            else:
                self.input_field.setEnabled(True)
                self.send_btn.setEnabled(True)
                self.input_field.setFocus()

        except Exception as e:
            error_text = f"Ошибка обработки сообщения: {e}"
            print(error_text)
            self.update_chat_history(f"system:{error_text}", error_text, "system")
            self.input_field.setEnabled(True)
            self.send_btn.setEnabled(True)
            self.input_field.setFocus()

    def update_chat_history(self, text, text_only, role):
        self.context.append({"role": role, "content": text_only})
        print(text_only)
        self.render_message(role, text_only)

    def render_message(self, role: str, content: str):
        safe_content = html.escape(str(content)).replace("\n", "<br>")

        if role == "user":
            role_html = f'<span style="color:blue; font-weight: bold;">{role}:</span>'
        elif role == "assistant":
            role_html = f'<span style="color:green; font-weight: bold;">{role}:</span>'
        else:
            role_html = f'<span style="color:black; font-weight: bold;">{role}:</span>'

        message_html = f'{role_html} <span style="color: black;">{safe_content}</span><br>'
        self.chat_history.insertHtml(message_html)
        self.chat_history.moveCursor(QTextCursor.End)

    def render_debug(self, intent_result: Dict[str, Any]):
        intent = html.escape(str(intent_result.get("intent")))
        confidence = html.escape(str(intent_result.get("confidence")))
        is_answer = html.escape(str(intent_result.get("is_answer_to_current_question")))
        reason = html.escape(str(intent_result.get("short_reason")))
        extracted = html.escape(str(intent_result.get("extracted_value")))

        debug_html = (
            f'<span style="color:gray;">'
            f'INTENT: {intent}, confidence: {confidence}, is_answer: {is_answer}, '
            f'extracted: {extracted}, reason: {reason}'
            f'</span><br>'
        )
        self.chat_history.insertHtml(debug_html)
        self.chat_history.moveCursor(QTextCursor.End)

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

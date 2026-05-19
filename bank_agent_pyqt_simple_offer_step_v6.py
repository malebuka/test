# bank_agent_pyqt_simple_offer_step_v6.py
# Простая архитектура:
# - мало intent'ов;
# - solution определяется intent-моделью, но зависит от текущего offer/question;
# - step — строка состояния, а не номер;
# - offer хранит id/title/question;
# - два dynamic prompt: обычный и свободный final_persuasion;
# - оба prompt передаются в модель как user message через build_messages_for_model(dynamic_prompt);
# - без regex и без solution="current".

import sys
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
# 1. SETTINGS
# ============================================================

OLLAMA_HOST = "http://127.0.0.1:11434"

AGENT_MODEL = "t-8b-base"
INTENT_MODEL = "qwen3:4b-instruct"

SHOW_DEBUG = True

# Сколько свободных убеждающих реплик даём после отказа от всех offer.
FINAL_PERSUASION_TURNS = 3


# ============================================================
# 2. DATA
# ============================================================

PUBLIC_DATA = {
    "bank_name": "АО Da банк",
    "agent_name": "Полина",
    "target_full_name": "Петров Петр Петрович",
    "target_name": "Петр Петрович",
    "callback_phone": "88005553535",
}

PRIVATE_DATA = {
    "debt_amount": "15000 рублей",
    "loan_type": "автокредит",
    "overdue_days": "10 дней",
    "collateral_value": "2310000 рублей",
}


# step="offer" всегда работает с текущим offer из этого списка.
# Вся информация по предложению хранится здесь.
OFFERS = [
    {
        "id": "payment_soon",
        "title": "оплата в ближайшие дни",
        "question": "Сможете ли внести платеж в ближайшие три дня?",
        "reply_task": (
            "Спроси, сможет ли клиент внести платеж в ближайшие три дня. "
            "Не предлагай другие варианты в этой реплике."
        ),
    },
    {
        "id": "sources_help",
        "title": "помощь родственников или друзей, заем средств либо перекредитование",
        "question": "Сможете ли рассмотреть помощь родственников или друзей, заем средств либо перекредитование?",
        "reply_task": (
            "Предложи помощь родственников или друзей, заем средств либо перекредитование как один общий вариант. "
            "Задай вопрос, сможет ли клиент рассмотреть такой вариант."
        ),
    },
    {
        "id": "partial_payment",
        "title": "частичная оплата задолженности",
        "question": "Какую часть задолженности сможете внести и когда?",
        "reply_task": (
            "Предложи частичную оплату как отдельный вариант, если полной суммы сейчас нет. "
            "Спроси, какую часть задолженности клиент сможет внести и когда."
        ),
    },
    {
        "id": "restructuring",
        "title": "реструктуризация кредита",
        "question": "Готовы ли рассмотреть реструктуризацию?",
        "reply_task": (
            "Предложи реструктуризацию как отдельный вариант. "
            "Кратко объясни, что это может помочь снизить ежемесячную нагрузку. "
            "Спроси, готов ли клиент рассмотреть этот вариант."
        ),
    },
    {
        "id": "vehicle_transfer",
        "title": "передача автомобиля для дальнейшего урегулирования",
        "question": "Готовы ли рассмотреть передачу автомобиля как вариант урегулирования?",
        "reply_task": (
            "Предложи передачу автомобиля как отдельный вариант урегулирования. "
            "Скажи, что это может быть рассмотрено банком в установленном порядке. "
            "Спроси, готов ли клиент рассмотреть этот вариант."
        ),
    },
]


INTENTS = [
    "accept",
    "reject",
    "question",
    "contact",
    "reason",
    "other",
    "rude_or_offtopic",
]

SOLUTIONS = [
    "none",
    "payment_soon",
    "sources_help",
    "partial_payment",
    "restructuring",
    "vehicle_transfer",
]

CONTACT_STATUSES = [
    "none",
    "same",
    "changed",
    "provided",
    "refused",
]


# ============================================================
# 3. STATE
# ============================================================

@dataclass
class DialogState:
    # step — это не число, а простая стадия.
    step: str = "identity"
    offer_index: int = 0

    identity_confirmed: bool = False
    ended: bool = False

    debt_reason: Optional[str] = None

    selected_solution: Optional[str] = None
    selected_solution_title: Optional[str] = None
    selected_solution_detail: Optional[str] = None

    contact_status: Optional[str] = None
    contact_info: Optional[str] = None

    final_persuasion_active: bool = False
    final_persuasion_left: int = 0

    last_action: Optional[str] = None
    last_bot_reply: Optional[str] = None
    last_intent_result: Optional[Dict[str, Any]] = None


# ============================================================
# 4. SMALL HELPERS
# ============================================================

def get_visible_facts(state: DialogState) -> Dict[str, str]:
    facts = dict(PUBLIC_DATA)
    if state.identity_confirmed:
        facts.update(PRIVATE_DATA)
    return facts


def get_current_offer(state: DialogState) -> Optional[Dict[str, str]]:
    if 0 <= state.offer_index < len(OFFERS):
        return OFFERS[state.offer_index]
    return None


def get_offer_by_solution(solution: str) -> Optional[Dict[str, str]]:
    for offer in OFFERS:
        if offer["id"] == solution:
            return offer
    return None


def get_current_question(state: DialogState) -> str:
    if state.step == "identity":
        return "Петров Петр Петрович — это вы?"

    if state.step == "reason":
        return "По какой причине не получилось оплатить задолженность?"

    if state.step == "offer":
        offer = get_current_offer(state)
        return offer["question"] if offer else "Готовы ли выбрать вариант урегулирования?"

    if state.step == "final_persuasion":
        return "Какой добровольный вариант урегулирования вы готовы рассмотреть?"

    if state.step == "final_confirm_refusal":
        return (
            "Правильно ли я понимаю, что вы подтверждаете отказ от оплаты и от всех предложенных "
            "вариантов урегулирования, а также понимаете возможные юридические последствия?"
        )

    if state.step == "contacts":
        return "Подскажите, изменилась ли ваша контактная информация?"

    if state.step == "contact_details":
        return "Подскажите, пожалуйста, актуальный номер телефона, email или удобное время связи."

    return "Уточните, пожалуйста, ваш ответ."


def safe_json(text: str) -> Dict[str, Any]:
    text = str(text).strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    # Без regex: ищем первый { и последний }.
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass

    return {
        "intent": "other",
        "solution": "none",
        "contact_status": "none",
        "extracted_value": None,
        "confidence": 0.0,
        "short_reason": "Не удалось разобрать JSON",
    }


def normalize_result(data: Dict[str, Any]) -> Dict[str, Any]:
    intent = str(data.get("intent", "other"))
    solution = str(data.get("solution", "none"))
    contact_status = str(data.get("contact_status", "none"))

    if intent not in INTENTS:
        intent = "other"

    if solution not in SOLUTIONS:
        solution = "none"

    if contact_status not in CONTACT_STATUSES:
        contact_status = "none"

    try:
        confidence = float(data.get("confidence", 0.0))
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


def solution_title(solution: str) -> str:
    offer = get_offer_by_solution(solution)
    return offer["title"] if offer else "вариант не выбран"


def clean_reply(text: str) -> str:
    text = str(text).strip()
    prefixes = [
        "assistant:",
        "оператор:",
        "бот:",
        "ответ:",
        "полина:",
    ]

    low = text.lower()
    for prefix in prefixes:
        if low.startswith(prefix):
            return text[len(prefix):].strip()

    return text.strip('"').strip()


def start_final_persuasion(state: DialogState) -> str:
    state.step = "final_persuasion"
    state.final_persuasion_active = True
    # Первая свободная реплика будет сейчас, поэтому оставляем ещё N-1.
    state.final_persuasion_left = max(FINAL_PERSUASION_TURNS - 1, 0)
    return "final_persuasion"


def accept_solution(state: DialogState, solution: str, detail: Optional[str]) -> str:
    if solution == "none":
        return "clarify_solution"

    state.selected_solution = solution
    state.selected_solution_title = solution_title(solution)
    state.selected_solution_detail = detail
    state.final_persuasion_active = False
    state.final_persuasion_left = 0
    state.step = "contacts"
    return "ask_contacts"


def move_to_next_offer_or_final(state: DialogState) -> str:
    state.offer_index += 1

    if state.offer_index < len(OFFERS):
        state.step = "offer"
        return "offer"

    return start_final_persuasion(state)


# ============================================================
# 5. INTENT PROMPT
# ============================================================

def build_intent_prompt(state: DialogState, user_text: str) -> str:
    offer = get_current_offer(state)

    offer_text = "нет текущего offer"
    if offer:
        offer_text = json.dumps(
            {
                "id": offer["id"],
                "title": offer["title"],
                "question": offer["question"],
            },
            ensure_ascii=False,
        )

    offers_text = json.dumps(
        [
            {
                "id": offer_item["id"],
                "title": offer_item["title"],
            }
            for offer_item in OFFERS
        ],
        ensure_ascii=False,
    )

    return f"""
Ты классификатор ответа клиента в диалоге банковского агента.

Верни только JSON:
{{
  "intent": "accept | reject | question | contact | reason | other | rude_or_offtopic",
  "solution": "none | payment_soon | sources_help | partial_payment | restructuring | vehicle_transfer",
  "contact_status": "none | same | changed | provided | refused",
  "extracted_value": null,
  "confidence": 0.0,
  "short_reason": "короткое объяснение"
}}

Текущий step:
{state.step}

Текущий вопрос агента:
{get_current_question(state)}

Текущий offer:
{offer_text}

Все offer:
{offers_text}

Последняя реплика агента:
{state.last_bot_reply}

Ответ клиента:
{user_text}

Правила классификации:
1. intent = accept, если клиент соглашается с текущим вопросом или выбирает любой offer.
2. intent = reject, если клиент отказывается, говорит что не может, не будет, не получится, денег нет.
3. intent = question, если клиент задаёт вопрос или уточняет условия.
4. intent = contact, если клиент отвечает про контактную информацию.
5. intent = reason, если текущий step = reason и клиент назвал причину неоплаты.
6. intent = rude_or_offtopic только для явной грубости или полной посторонней темы.
7. intent = other для обычной реплики, которую нельзя считать согласием/отказом/вопросом.

Правила solution:
1. solution всегда должен быть конкретным id из списка offer или "none".
2. НЕЛЬЗЯ возвращать solution="current". Такого значения нет.
3. Если клиент соглашается на текущий offer словами "да", "согласен", "давайте", "попробую", верни id текущего offer.
4. Если клиент явно называет вариант, верни этот solution:
   - "найду деньги", "оплачу завтра", "внесу платеж" → payment_soon;
   - "займу", "родственники", "друзья", "перекредитуюсь" → sources_help;
   - "частично", "часть суммы", "внесу 5000" → partial_payment;
   - "реструктуризация", "реструктурировать" → restructuring;
   - "авто", "машина", "передача автомобиля" → vehicle_transfer.
5. Если intent не accept, обычно solution="none".
6. Если step = final_persuasion или final_confirm_refusal, и клиент вдруг согласился на любой вариант, intent=accept и solution должен быть конкретным.

Правила контактов:
1. Если step = contacts и клиент говорит "нет, не изменилась", intent=contact, contact_status=same.
2. Если step = contacts и клиент говорит "да, изменилась", но не называет новые данные, intent=contact, contact_status=changed.
3. Если клиент дал номер, email, адрес или удобное время связи, intent=contact, contact_status=provided.
4. Если отказался говорить контакты, intent=contact, contact_status=refused.

Важные ограничения:
- Если step = identity и клиент отвечает "да", "это я", "слушаю", это intent=accept.
- Если step = identity и клиент говорит "не я", это intent=reject.
- Если step = final_confirm_refusal и клиент говорит "да, подтверждаю отказ", это intent=reject, solution=none.
- Если клиент говорит просто "давайте", это accept только когда текущий вопрос агента содержит конкретный offer.
- Если клиент говорит "давайте" после общего вопроса без конкретного offer, intent=other, solution=none.
""".strip()


# ============================================================
# 6. ROUTER
# ============================================================

def route(state: DialogState, result: Dict[str, Any]) -> str:
    intent = result["intent"]
    solution = result["solution"]
    contact_status = result["contact_status"]
    extracted = result.get("extracted_value")

    if state.ended:
        return "end"

    # Глобально: если клиент в любой стадии внезапно согласился на конкретный solution.
    if intent == "accept" and solution != "none" and state.step in [
        "offer",
        "final_persuasion",
        "final_confirm_refusal",
    ]:
        return accept_solution(state, solution, extracted)

    if state.step == "identity":
        if intent == "accept":
            state.identity_confirmed = True
            state.step = "reason"
            return "ask_reason"

        if intent == "reject":
            state.ended = True
            state.step = "ended"
            return "end_no_identity"

        if intent == "question":
            return "answer_question"

        return "ask_identity"

    if state.step == "reason":
        if intent == "reason":
            state.debt_reason = str(extracted or "").strip() or "причина не указана"
            state.step = "offer"
            state.offer_index = 0
            return "offer"

        if intent == "question":
            return "answer_question"

        if intent == "reject":
            state.debt_reason = "клиент отказался подробно объяснять причину"
            state.step = "offer"
            state.offer_index = 0
            return "offer"

        return "ask_reason"

    if state.step == "offer":
        if intent == "accept":
            # Если модель сказала accept, но solution не указала, считаем, что речь о текущем offer.
            current = get_current_offer(state)
            if current:
                return accept_solution(state, current["id"], extracted)
            return "clarify_solution"

        if intent == "reject":
            return move_to_next_offer_or_final(state)

        if intent == "question":
            return "answer_question"

        if intent == "rude_or_offtopic":
            return "react_and_repeat_offer"

        return "repeat_offer"

    if state.step == "final_persuasion":
        if intent == "accept" and solution != "none":
            return accept_solution(state, solution, extracted)

        # Вопросы тоже обрабатываем свободным prompt: ответить и убеждать дальше.
        if state.final_persuasion_left > 0:
            state.final_persuasion_left -= 1
            return "final_persuasion"

        state.final_persuasion_active = False
        state.step = "final_confirm_refusal"
        return "ask_of_refuses_all"

    if state.step == "final_confirm_refusal":
        if intent == "accept" and solution != "none":
            return accept_solution(state, solution, extracted)

        # Подтвердил отказ или уклоняется — всё равно дальше контакты.
        state.step = "contacts"
        return "ask_contacts_no_agreement"

    if state.step == "contacts":
        if intent == "contact":
            if contact_status == "changed":
                state.contact_status = "changed"
                state.step = "contact_details"
                return "ask_contact_details"

            if contact_status == "same":
                state.contact_status = "same"
                state.contact_info = "контактная информация не изменилась"
                state.step = "ended"
                state.ended = True
                return "summary"

            if contact_status == "provided":
                state.contact_status = "provided"
                state.contact_info = str(extracted or "").strip() or "клиент предоставил актуальные контакты"
                state.step = "ended"
                state.ended = True
                return "summary"

            if contact_status == "refused":
                state.contact_status = "refused"
                state.contact_info = "клиент отказался уточнять контактную информацию"
                state.step = "ended"
                state.ended = True
                return "summary"

        return "ask_contacts"

    if state.step == "contact_details":
        if intent == "contact" and contact_status in ["provided", "same", "refused"]:
            state.contact_status = contact_status

            if contact_status == "provided":
                state.contact_info = str(extracted or "").strip() or "клиент предоставил актуальные контакты"
            elif contact_status == "same":
                state.contact_info = "клиент уточнил, что контактная информация не изменилась"
            else:
                state.contact_info = "клиент отказался назвать актуальные контактные данные"

            state.step = "ended"
            state.ended = True
            return "summary"

        return "ask_contact_details"

    return "repeat"


# ============================================================
# 7. REPLY PROMPTS
# ============================================================

def build_dynamic_prompt(state: DialogState, action: str, result: Dict[str, Any], user_text: str) -> str:
    facts = get_visible_facts(state)
    offer = get_current_offer(state)

    if offer:
        offer_block = json.dumps(
            {
                "id": offer["id"],
                "title": offer["title"],
                "question": offer["question"],
                "reply_task": offer["reply_task"],
            },
            ensure_ascii=False,
        )
    else:
        offer_block = "нет текущего offer"

    selected = state.selected_solution_title or "нет"

    action_tasks = {
        "ask_identity": (
            "Уточни, разговариваешь ли ты с Петровым Петром Петровичем. "
            "Не раскрывай причину звонка."
        ),
        "answer_question": (
            "Ответь на вопрос клиента по доступным фактам. Затем вернись к текущему вопросу. "
            "Если личность не подтверждена, не раскрывай финансовые детали."
        ),
        "ask_reason": (
            "Личность подтверждена. Сообщи, что разговор записывается. "
            "Назови сумму, тип кредита и срок просрочки. Спроси причину неоплаты."
        ),
        "offer": (
            "Предложи текущий offer. Используй question/reply_task текущего offer. "
            "Не предлагай другие offer в этой реплике."
        ),
        "repeat_offer": (
            "Кратко верни клиента к текущему offer и повтори вопрос по нему."
        ),
        "react_and_repeat_offer": (
            "Спокойно отреагируй на резкую или постороннюю реплику клиента и верни разговор к текущему offer."
        ),
        "clarify_solution": (
            "Клиент вроде согласился, но непонятно на какой вариант. "
            "Попроси уточнить, какой вариант он готов рассмотреть."
        ),
        "ask_of_refuses_all": (
            "Задай последний контрольный вопрос: правильно ли я понимаю, что клиент подтверждает отказ "
            "от оплаты и от всех предложенных вариантов урегулирования, а также понимает возможные юридические последствия. "
            "Не спрашивай контакты в этой реплике."
        ),
        "ask_contacts": (
            "Кратко зафиксируй договорённость и спроси, изменилась ли контактная информация клиента. "
            "Если изменилась, попроси назвать актуальные данные."
        ),
        "ask_contacts_no_agreement": (
            "Кратко зафиксируй, что договорённость не достигнута. "
            "Затем всё равно спроси, изменилась ли контактная информация клиента."
        ),
        "ask_contact_details": (
            "Клиент сказал, что контактная информация изменилась, но ещё не назвал новые данные. "
            "Попроси назвать актуальный номер телефона, email или удобное время связи."
        ),
        "end_no_identity": (
            "Вежливо извинись за беспокойство и заверши разговор. "
            "Не раскрывай причину звонка."
        ),
        "end": (
            "Вежливо заверши разговор."
        ),
    }

    task = action_tasks.get(action, "Кратко и вежливо продолжи разговор по текущему состоянию.")

    privacy = (
        "Личность подтверждена. Можно обсуждать задолженность и варианты урегулирования."
        if state.identity_confirmed
        else "Личность не подтверждена. Нельзя раскрывать долг, кредит, просрочку, сумму, авто, взыскание, суд."
    )

    return f"""
Ты банковский агент Полина.

Доступные факты:
{facts}

Конфиденциальность:
{privacy}

Текущий step:
{state.step}

Текущий action:
{action}

Текущий offer:
{offer_block}

Текущий вопрос:
{get_current_question(state)}

Причина неоплаты:
{state.debt_reason}

Выбранный вариант:
{selected}

Последняя реплика клиента:
{user_text}

Intent result:
{result}

Твоя задача:
{task}

Правила ответа:
- Ответь только репликой агента.
- 1-2 коротких предложения.
- Не задавай больше одного вопроса.
- Не перечисляй все offer сразу.
- Не добавляй новых шагов.
- Если action=offer, предложи только текущий offer.
- Если action=answer_question, сначала ответь на вопрос, потом вернись к текущему вопросу.
- Если личность не подтверждена, не раскрывай финансовые детали.
- Не проси паспорт, дату рождения, код из SMS.
- Не угрожай, не дави, не упоминай полицию, арест, уголовное дело.
""".strip()


def build_dynamic_prompt_free(state: DialogState, result: Dict[str, Any], user_text: str) -> str:
    facts = get_visible_facts(state)

    offers = json.dumps(
        [
            {
                "id": offer["id"],
                "title": offer["title"],
            }
            for offer in OFFERS
        ],
        ensure_ascii=False,
    )

    return f"""
Ты банковский агент Полина. Сейчас свободный этап убеждения клиента.

Доступные факты:
{facts}

Текущий step:
{state.step}

final_persuasion_active:
{state.final_persuasion_active}

Осталось свободных реплик после этой:
{state.final_persuasion_left}

Доступные варианты урегулирования:
{offers}

Причина неоплаты:
{state.debt_reason}

Последняя реплика клиента:
{user_text}

Intent result:
{result}

Главная цель:
Свободно и естественно убедить клиента выбрать оплату или добровольный вариант урегулирования.

Как отвечать:
- Ответь именно на последнюю реплику клиента.
- Убеждай оплатить или выбрать добровольный вариант.
- Можно говорить про оплату в ближайшие дни, частичную оплату, помощь близких/заём, реструктуризацию или автомобильный вариант.
- Не говори шаблонно.
- Не повторяй предыдущую реплику.
- Не перечисляй все варианты каждый раз.
- Не завершай разговор.
- Не спрашивай контактную информацию.
- Не подводи итог.
- Не говори "ждём вашего звонка".
- Не говори "когда примете решение".
- Не говори "обратитесь в банк".
- Не угрожай, не дави, не упоминай полицию, арест, уголовное дело.

Ответ:
1-3 предложения, живым разговорным стилем, но делово.
""".strip()


def build_summary(state: DialogState) -> str:
    reason = state.debt_reason or "причина неоплаты не указана"
    contact = state.contact_info or "контактная информация не уточнена"

    if state.selected_solution:
        solution = state.selected_solution_title or solution_title(state.selected_solution)
        detail = f" Детали: {state.selected_solution_detail}." if state.selected_solution_detail else ""
        return (
            "Хорошо, подытожу договорённость. "
            f"Причина неоплаты: {reason}. "
            f"Согласованный вариант: {solution}.{detail} "
            f"Контактная информация: {contact}. "
            "Информацию зафиксировала, спасибо за разговор."
        )

    return (
        "Хорошо, подытожу разговор. "
        f"Причина неоплаты: {reason}. "
        "Договорённость по варианту урегулирования на данный момент не достигнута. "
        f"Контактная информация: {contact}. "
        "Информацию зафиксировала, спасибо за разговор."
    )


# ============================================================
# 8. PYQT
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
                    "Ты голосовой помощник АО Da банк. Отвечай кратко и строго по текущей инструкции. "
                    "До подтверждения личности не раскрывай финансовые детали. "
                    "Не угрожай и не дави."
                ),
            },
            {
                "role": "assistant",
                "content": "Алло, здравствуйте, меня зовут Полина, я сотрудник Da банк. Петров Петр Петрович — это вы?",
            },
        ]

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

        for message in self.context[1:]:
            self.render_message(message["role"], message["content"])

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

    def build_messages_for_model(self, dynamic_prompt: str) -> List[Dict[str, str]]:
        messages = list(self.context)
        messages.append(
            {
                "role": "user",
                "content": (
                    "Текущая инструкция имеет приоритет над историей диалога.\n\n"
                    + dynamic_prompt
                ),
            }
        )
        return messages

    def classify_intent(self, user_text: str) -> Dict[str, Any]:
        prompt = build_intent_prompt(self.state, user_text)

        try:
            response = self.ollama_client.chat(
                model=INTENT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Ты возвращаешь только JSON без markdown и без пояснений.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                options={
                    "temperature": 0,
                    "num_predict": 220,
                    "top_p": 0.1,
                    "repeat_penalty": 1.05,
                    "num_ctx": 4096,
                },
                think=False,
            )

            data = safe_json(response["message"]["content"])
            return normalize_result(data)

        except Exception as error:
            return {
                "intent": "other",
                "solution": "none",
                "contact_status": "none",
                "extracted_value": None,
                "confidence": 0.0,
                "short_reason": f"Ошибка intent-модели: {error}",
            }

    def call_agent(self, dynamic_prompt: str, free_mode: bool = False) -> str:
        if free_mode:
            options = {
                "temperature": 0.85,
                "num_predict": 220,
                "repeat_penalty": 1.35,
                "repeat_last_n": 2048,
                "top_k": 100,
                "top_p": 0.95,
                "min_p": 0.03,
                "num_ctx": 8192,
            }
        else:
            options = {
                "temperature": 0.2,
                "num_predict": 180,
                "repeat_penalty": 1.15,
                "top_k": 40,
                "top_p": 0.8,
                "min_p": 0.0,
                "num_ctx": 8192,
            }

        response = self.ollama_client.chat(
            model=AGENT_MODEL,
            messages=self.build_messages_for_model(dynamic_prompt),
            options=options,
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
            self.update_chat_history("user", user_text)

            result = self.classify_intent(user_text)
            self.state.last_intent_result = result

            if SHOW_DEBUG:
                self.render_debug(result)

            action = route(self.state, result)
            self.state.last_action = action

            print("\n--- DEBUG ---")
            print("step:", self.state.step)
            print("offer_index:", self.state.offer_index)
            print("current_offer:", get_current_offer(self.state))
            print("action:", action)
            print("intent_result:", result)
            print("final_persuasion_active:", self.state.final_persuasion_active)
            print("final_persuasion_left:", self.state.final_persuasion_left)
            print("selected_solution:", self.state.selected_solution)
            print("--- END DEBUG ---\n")

            if action == "summary":
                bot_text = build_summary(self.state)

            elif action == "final_persuasion" and self.state.final_persuasion_active:
                dynamic_prompt = build_dynamic_prompt_free(
                    state=self.state,
                    result=result,
                    user_text=user_text,
                )
                bot_text = self.call_agent(dynamic_prompt, free_mode=True)

            else:
                dynamic_prompt = build_dynamic_prompt(
                    state=self.state,
                    action=action,
                    result=result,
                    user_text=user_text,
                )
                bot_text = self.call_agent(dynamic_prompt, free_mode=False)

            self.update_chat_history("assistant", bot_text)
            self.state.last_bot_reply = bot_text

            if self.state.ended:
                self.input_field.setEnabled(False)
                self.send_btn.setEnabled(False)
            else:
                self.input_field.setEnabled(True)
                self.send_btn.setEnabled(True)
                self.input_field.setFocus()

        except Exception as error:
            error_text = f"Ошибка обработки сообщения: {error}"
            print(error_text)

            self.update_chat_history("system", error_text)

            self.input_field.setEnabled(True)
            self.send_btn.setEnabled(True)
            self.input_field.setFocus()

    def update_chat_history(self, role: str, content: str):
        self.context.append(
            {
                "role": role,
                "content": content,
            }
        )

        print(content)
        self.render_message(role, content)

    def render_message(self, role: str, content: str):
        safe_content = html.escape(str(content)).replace("\n", "<br>")

        if role == "user":
            role_html = '<span style="color:blue; font-weight: bold;">user:</span>'
        elif role == "assistant":
            role_html = '<span style="color:green; font-weight: bold;">assistant:</span>'
        else:
            role_html = f'<span style="color:black; font-weight: bold;">{html.escape(role)}:</span>'

        self.chat_history.insertHtml(
            f'{role_html} <span style="color: black;">{safe_content}</span><br>'
        )
        self.chat_history.moveCursor(QTextCursor.End)

    def render_debug(self, result: Dict[str, Any]):
        debug_text = (
            f'INTENT: {result.get("intent")}, '
            f'SOLUTION: {result.get("solution")}, '
            f'CONTACT: {result.get("contact_status")}, '
            f'CONF: {result.get("confidence")}, '
            f'REASON: {result.get("short_reason")}'
        )

        safe = html.escape(debug_text)
        self.chat_history.insertHtml(f'<span style="color:gray;">{safe}</span><br>')
        self.chat_history.moveCursor(QTextCursor.End)

    def closeEvent(self, event):
        super().closeEvent(event)


# ============================================================
# 9. RUN
# ============================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = ChatWindow()
    window.show()

    sys.exit(app.exec_())

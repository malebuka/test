# bank_agent_pyqt_simple_step_intent_v3.py
# Упрощённая версия:
# - step вместо phase
# - мало intent
# - solution всегда конкретный, без "current"
# - без regex
# - Qwen сам определяет intent и solution с учётом текущего вопроса агента
# - Python только маршрутизирует step -> action
#
# Установка:
#   pip install PyQt5 ollama
#
# Ollama:
#   ollama serve
#   ollama pull qwen3:4b-instruct
#
# Запуск:
#   python bank_agent_pyqt_simple_step_intent_v3.py

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
# НАСТРОЙКИ
# ============================================================

OLLAMA_HOST = "http://127.0.0.1:11434"

AGENT_MODEL = "t-8b-base"
INTENT_MODEL = "qwen3:4b-instruct"

FINAL_PERSUASION_TURNS = 3

SHOW_DEBUG = True


# ============================================================
# ДАННЫЕ
# ============================================================

PUBLIC_DATA = {
    "bank_name": "АО Da банк",
    "agent_name": "Полина",
    "target_name": "Петр Петрович",
    "target_full_name": "Петров Петр Петрович",
    "callback_phone": "88005553535",
}

PRIVATE_DATA = {
    "debt_amount": "15000 рублей",
    "loan_type": "автокредит",
    "overdue_days": "10 дней",
    "collateral_value": "2310000 рублей",
}


# Варианты идут по порядку.
# Больше не нужны отдельные agrees_restructuring / declines_restructuring.
# Qwen возвращает intent=accept/reject и конкретный solution.
OFFERS = [
    {
        "solution": "payment_soon",
        "title": "оплата в ближайшие три дня",
        "question": "Сможете ли внести платеж в ближайшие три дня?",
        "details": "Клиент вносит платеж в ближайшие три дня.",
    },
    {
        "solution": "sources_help",
        "title": "помощь родственников или друзей, заем средств либо перекредитование",
        "question": "Сможете ли рассмотреть помощь родственников или друзей, заем средств либо перекредитование?",
        "details": "Клиент пытается найти деньги через близких, заем или перекредитование.",
    },
    {
        "solution": "partial_payment",
        "title": "частичная оплата",
        "question": "Какую часть задолженности сможете внести и когда?",
        "details": "Клиент вносит часть задолженности и называет срок.",
    },
    {
        "solution": "restructuring",
        "title": "реструктуризация",
        "question": "Готовы ли рассмотреть реструктуризацию кредита?",
        "details": "Реструктуризация может снизить ежемесячный платеж за счет изменения условий.",
    },
    {
        "solution": "vehicle_transfer",
        "title": "передача автомобиля",
        "question": "Готовы ли рассмотреть передачу автомобиля как вариант урегулирования?",
        "details": "Автомобиль может быть рассмотрен как способ урегулирования по автокредиту.",
    },
]

SOLUTION_TITLES = {item["solution"]: item["title"] for item in OFFERS}
ALL_SOLUTIONS = list(SOLUTION_TITLES.keys()) + ["none"]


# ============================================================
# STATE
# ============================================================

@dataclass
class DialogState:
    # Основные шаги:
    # identity -> reason -> offers -> final_persuasion -> final_confirm_refusal
    # -> contacts -> contact_details -> ended
    step: str = "identity"

    identity_confirmed: bool = False
    ended: bool = False

    offer_index: int = 0
    final_persuasion_left: int = 0

    debt_reason: Optional[str] = None
    selected_solution: Optional[str] = None
    selected_solution_detail: Optional[str] = None

    contact_status: Optional[str] = None
    contact_info: Optional[str] = None

    last_agent_question: str = "Петров Петр Петрович — это вы?"
    last_action: Optional[str] = None
    last_intent: Optional[str] = None


# ============================================================
# МИНИМАЛЬНЫЕ HELPERS
# ============================================================

def current_offer(state: DialogState) -> Optional[Dict[str, str]]:
    if 0 <= state.offer_index < len(OFFERS):
        return OFFERS[state.offer_index]
    return None


def visible_facts(state: DialogState) -> Dict[str, str]:
    data = dict(PUBLIC_DATA)
    if state.identity_confirmed:
        data.update(PRIVATE_DATA)
    return data


def parse_json_answer(text: str) -> Dict[str, Any]:
    """
    Без regex. Просто вытаскиваем JSON между первой { и последней }.
    """
    text = str(text).strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return {
            "intent": "other",
            "solution": "none",
            "contact_status": "unknown",
            "extracted_value": None,
            "confidence": 0.0,
            "reason": "JSON не найден",
        }

    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return {
            "intent": "other",
            "solution": "none",
            "contact_status": "unknown",
            "extracted_value": None,
            "confidence": 0.0,
            "reason": "JSON повреждён",
        }


def normalize_result(data: Dict[str, Any]) -> Dict[str, Any]:
    intent = str(data.get("intent", "other"))
    solution = str(data.get("solution", "none"))
    contact_status = str(data.get("contact_status", "unknown"))

    allowed_intents = [
        "identity_yes",
        "identity_no",
        "third_person",
        "reason",
        "accept",
        "reject",
        "question",
        "contact",
        "other",
        "rude_or_offtopic",
    ]

    allowed_contact_status = [
        "same",
        "changed",
        "provided",
        "refused",
        "unknown",
    ]

    if intent not in allowed_intents:
        intent = "other"

    if solution not in ALL_SOLUTIONS:
        solution = "none"

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
        "reason": str(data.get("reason", "")),
    }


def solution_title(solution: Optional[str]) -> str:
    if not solution:
        return "вариант не выбран"
    return SOLUTION_TITLES.get(solution, solution)


# ============================================================
# INTENT PROMPT
# ============================================================

def build_intent_prompt(state: DialogState, user_text: str) -> str:
    offer = current_offer(state)

    if offer:
        current_offer_text = {
            "solution": offer["solution"],
            "title": offer["title"],
            "question": offer["question"],
            "details": offer["details"],
        }
    else:
        current_offer_text = None

    return f"""
Ты классификатор ответа клиента в банковском диалоге.

Верни только JSON:
{{
  "intent": "identity_yes | identity_no | third_person | reason | accept | reject | question | contact | other | rude_or_offtopic",
  "solution": "payment_soon | sources_help | partial_payment | restructuring | vehicle_transfer | none",
  "contact_status": "same | changed | provided | refused | unknown",
  "extracted_value": null,
  "confidence": 0.0,
  "reason": "коротко"
}}

Текущий step:
{state.step}

Текущий вопрос агента:
{state.last_agent_question}

Текущий предложенный вариант:
{current_offer_text}

Все варианты:
{OFFERS}

Реплика клиента:
{user_text}

Правила:
1. Если клиент подтверждает, что он Петр Петрович, intent = identity_yes.
2. Если говорит, что это не он, intent = identity_no.
3. Если отвечает другое лицо, intent = third_person.
4. Если step = reason и клиент объясняет причину неоплаты, intent = reason.
5. Если клиент согласился с текущим конкретным вопросом агента, intent = accept, а solution поставь равным solution текущего предложенного варианта.
6. Если клиент согласился на конкретный вариант из текста, intent = accept, solution = этот конкретный вариант.
7. Если клиент говорит просто "давайте", "ну давайте", "можно", но текущий вопрос агента НЕ содержит конкретного варианта, intent = other, solution = none.
8. Если клиент отказывается, intent = reject, solution = none.
9. Если клиент задаёт вопрос, intent = question, solution = none, кроме случая когда в вопросе он явно выбирает вариант.
10. Если step = contacts или contact_details, классифицируй ответ как intent = contact и заполни contact_status.
11. Если контакты не изменились, contact_status = same.
12. Если клиент сказал, что контакты изменились, но не дал новые данные, contact_status = changed.
13. Если клиент дал номер, email, адрес или удобное время связи, contact_status = provided.
14. Если отказался говорить контакты, contact_status = refused.
15. rude_or_offtopic используй только для явной грубости или полностью посторонней темы.
16. Не используй solution = current. Такого значения нет. Всегда верни конкретный solution или none.
""".strip()


# ============================================================
# ROUTER: INTENT + STEP -> ACTION
# ============================================================

def route(state: DialogState, result: Dict[str, Any]) -> str:
    intent = result["intent"]
    solution = result["solution"]
    extracted = result.get("extracted_value")

    state.last_intent = intent

    # ---------------------------
    # identity
    # ---------------------------
    if state.step == "identity":
        if intent == "identity_yes":
            state.identity_confirmed = True
            state.step = "reason"
            return "ask_reason"

        if intent in ["identity_no", "third_person"]:
            state.ended = True
            state.step = "ended"
            return "privacy_goodbye"

        if intent == "question":
            return "answer_question"

        return "ask_identity_again"

    # ---------------------------
    # reason
    # ---------------------------
    if state.step == "reason":
        if intent == "reason":
            state.debt_reason = str(extracted or "")
            state.step = "offers"
            state.offer_index = 0
            return "offer"

        if intent == "question":
            return "answer_question"

        return "ask_reason_again"

    # ---------------------------
    # offers
    # ---------------------------
    if state.step == "offers":
        offer = current_offer(state)
        current_solution = offer["solution"] if offer else "none"

        if intent == "accept":
            # Если Qwen почему-то не указал solution, берём текущий конкретный offer.
            # Но только в offers, потому что тут всегда был задан конкретный вопрос.
            if solution == "none":
                solution = current_solution

            state.selected_solution = solution
            state.selected_solution_detail = str(extracted or "")
            state.step = "contacts"
            return "ask_contacts"

        if intent == "reject":
            state.offer_index += 1

            if state.offer_index < len(OFFERS):
                return "offer"

            state.step = "final_persuasion"
            state.final_persuasion_left = max(FINAL_PERSUASION_TURNS - 1, 0)
            return "final_persuasion"

        if intent == "question":
            return "answer_question"

        return "repeat_offer"

    # ---------------------------
    # final_persuasion
    # ---------------------------
    if state.step == "final_persuasion":
        if intent == "accept" and solution != "none":
            state.selected_solution = solution
            state.selected_solution_detail = str(extracted or "")
            state.step = "contacts"
            return "ask_contacts"

        if intent == "question":
            return "answer_question"

        if state.final_persuasion_left > 0:
            state.final_persuasion_left -= 1
            return "final_persuasion"

        state.step = "final_confirm_refusal"
        return "ask_of_refuses_all"

    # ---------------------------
    # final_confirm_refusal
    # ---------------------------
    if state.step == "final_confirm_refusal":
        if intent == "accept" and solution != "none":
            state.selected_solution = solution
            state.selected_solution_detail = str(extracted or "")
            state.step = "contacts"
            return "ask_contacts"

        if intent == "question":
            return "answer_question"

        # Любой отказ / подтверждение отказа / неопределённый ответ после финального вопроса
        # ведёт к контактам.
        state.step = "contacts"
        return "ask_contacts_no_agreement"

    # ---------------------------
    # contacts
    # ---------------------------
    if state.step == "contacts":
        if intent == "contact":
            if result["contact_status"] == "changed":
                state.step = "contact_details"
                state.contact_status = "changed"
                return "ask_contact_details"

            if result["contact_status"] == "provided":
                state.contact_status = "provided"
                state.contact_info = str(extracted or "")
                state.step = "ended"
                state.ended = True
                return "summary"

            if result["contact_status"] == "same":
                state.contact_status = "same"
                state.contact_info = "контактная информация не изменилась"
                state.step = "ended"
                state.ended = True
                return "summary"

            if result["contact_status"] == "refused":
                state.contact_status = "refused"
                state.contact_info = "клиент отказался уточнять контактную информацию"
                state.step = "ended"
                state.ended = True
                return "summary"

        if intent == "question":
            return "answer_question"

        return "ask_contacts_again"

    # ---------------------------
    # contact_details
    # ---------------------------
    if state.step == "contact_details":
        if intent == "contact":
            if result["contact_status"] == "provided":
                state.contact_status = "provided"
                state.contact_info = str(extracted or "")
                state.step = "ended"
                state.ended = True
                return "summary"

            if result["contact_status"] == "same":
                state.contact_status = "same"
                state.contact_info = "клиент уточнил, что контактная информация не изменилась"
                state.step = "ended"
                state.ended = True
                return "summary"

            if result["contact_status"] == "refused":
                state.contact_status = "refused"
                state.contact_info = "клиент отказался назвать новые контактные данные"
                state.step = "ended"
                state.ended = True
                return "summary"

        if intent == "question":
            return "answer_question"

        return "ask_contact_details"

    return "end"


# ============================================================
# REPLY PROMPT
# ============================================================

def build_reply_prompt(state: DialogState, action: str, result: Dict[str, Any], user_text: str) -> str:
    offer = current_offer(state)
    facts = visible_facts(state)

    if action == "ask_reason":
        instruction = (
            "Сообщи, что разговор записывается. Назови сумму задолженности, тип кредита и срок просрочки. "
            "Спроси причину неоплаты."
        )

    elif action == "offer":
        instruction = (
            f"Предложи текущий вариант урегулирования: {offer['title']}. "
            f"Кратко поясни: {offer['details']} "
            f"Задай этот вопрос: {offer['question']}"
        )

    elif action == "repeat_offer":
        instruction = (
            f"Кратко верни клиента к текущему варианту: {offer['title']}. "
            f"Задай один вопрос: {offer['question']}"
        )

    elif action == "final_persuasion":
        instruction = (
            "Свободно и естественно убеждай клиента выбрать добровольный вариант урегулирования. "
            "Не иди по жёсткому сценарию. Не перечисляй все варианты сухим списком. "
            "Отвечай на последнюю реплику клиента по смыслу. Не угрожай. "
            "Можно мягко показать, что добровольное решение лучше юридического сценария."
        )

    elif action == "ask_of_refuses_all":
        instruction = (
            "Задай последний контрольный вопрос: правильно ли я понимаю, что клиент подтверждает отказ "
            "от оплаты и от всех предложенных вариантов урегулирования, а также понимает возможные "
            "юридические последствия? Не спрашивай контакты в этой реплике."
        )

    elif action in ["ask_contacts", "ask_contacts_no_agreement"]:
        if state.selected_solution:
            instruction = (
                "Кратко зафиксируй выбранный вариант. Затем обязательно спроси, изменилась ли контактная "
                "информация клиента. Если изменилась, попроси назвать актуальные данные."
            )
        else:
            instruction = (
                "Кратко зафиксируй, что договорённость пока не достигнута. Затем обязательно спроси, "
                "изменилась ли контактная информация клиента. Если изменилась, попроси назвать актуальные данные."
            )

    elif action == "ask_contact_details":
        instruction = (
            "Клиент сказал, что контактная информация изменилась, но не назвал новые данные. "
            "Попроси назвать актуальный номер телефона, email или удобное время связи."
        )

    elif action == "ask_contacts_again":
        instruction = (
            "Повтори вопрос: изменилась ли контактная информация клиента. Если изменилась, попроси назвать актуальные данные."
        )

    elif action == "answer_question":
        if state.identity_confirmed:
            instruction = (
                "Ответь на вопрос клиента по смыслу в рамках доступных фактов. После ответа вернись к текущему вопросу агента. "
                "Не продвигай сценарий дальше."
            )
        else:
            instruction = (
                "Ответь без раскрытия финансовых деталей. До подтверждения личности нельзя говорить о долге, кредите, сумме, просрочке, суде или автомобиле. "
                "Вернись к вопросу, Петр Петрович ли это."
            )

    elif action == "ask_reason_again":
        instruction = "Кратко попроси назвать причину неоплаты."

    elif action == "ask_identity_again":
        instruction = "Кратко уточни, Петров Петр Петрович ли это. Не называй причину звонка."

    elif action == "privacy_goodbye":
        instruction = (
            "Не раскрывай финансовые детали. Вежливо скажи, что информация предназначена только для Петра Петровича, "
            "попроси передать ему просьбу связаться с банком и попрощайся."
        )

    else:
        instruction = "Кратко и вежливо ответь клиенту."

    return f"""
Ты банковский агент Полина.

Доступные факты:
{facts}

step:
{state.step}

action:
{action}

Текущий вариант:
{offer}

Причина неоплаты:
{state.debt_reason}

Выбранное решение:
{solution_title(state.selected_solution)}

Последний вопрос агента:
{state.last_agent_question}

Реплика клиента:
{user_text}

Intent JSON:
{result}

Что сделать:
{instruction}

Правила:
- Ответь только репликой агента.
- 1-2 коротких предложения.
- Не задавай больше одного вопроса.
- Не повторяй дословно прошлые формулировки.
- До подтверждения личности не раскрывай финансовые детали.
- После подтверждения личности не проси паспорт, дату рождения, SMS-код или ФИО.
- Не угрожай, не дави, не упоминай полицию или уголовное дело.
""".strip()


def build_summary(state: DialogState) -> str:
    reason = state.debt_reason or "причина неоплаты не указана"

    if state.selected_solution:
        agreement = f"согласованный вариант урегулирования — {solution_title(state.selected_solution)}"
        if state.selected_solution_detail:
            agreement += f" ({state.selected_solution_detail})"
    else:
        agreement = "договорённость по варианту урегулирования не достигнута"

    contact = state.contact_info or "контактная информация не уточнена"

    return (
        "Хорошо, подытожу разговор. "
        f"Причина неоплаты: {reason}. "
        f"{agreement}. "
        f"Контактная информация: {contact}. "
        "Информацию зафиксировала, спасибо за разговор."
    )


# ============================================================
# PYQT
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
                    "Ты банковский агент Полина. Говори кратко, спокойно и по делу. "
                    "До подтверждения личности не раскрывай финансовые детали."
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
        self.setWindowTitle("Simple Step Intent Bot")
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
        self.ollama_client = ollama.Client(host=OLLAMA_HOST, trust_env=False)

    def classify(self, user_text: str) -> Dict[str, Any]:
        prompt = build_intent_prompt(self.state, user_text)

        try:
            response = self.ollama_client.chat(
                model=INTENT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Ты классификатор. Верни только JSON.",
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

            return normalize_result(parse_json_answer(response["message"]["content"]))

        except Exception as error:
            return {
                "intent": "other",
                "solution": "none",
                "contact_status": "unknown",
                "extracted_value": None,
                "confidence": 0.0,
                "reason": f"Ошибка intent-модели: {error}",
            }

    def generate_reply(self, action: str, result: Dict[str, Any], user_text: str) -> str:
        if action == "summary":
            return build_summary(self.state)

        prompt = build_reply_prompt(self.state, action, result, user_text)

        if action == "final_persuasion":
            temperature = 0.75
            repeat_penalty = 1.3
            top_p = 0.95
        else:
            temperature = 0.2
            repeat_penalty = 1.1
            top_p = 0.85

        try:
            messages = [
                self.context[0],
                *self.context[-8:],
                {
                    "role": "user",
                    "content": prompt,
                },
            ]

            response = self.ollama_client.chat(
                model=AGENT_MODEL,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": 220,
                    "top_k": 60,
                    "top_p": top_p,
                    "repeat_penalty": repeat_penalty,
                    "num_ctx": 8192,
                },
                think=False,
            )

            return str(response["message"]["content"]).strip()

        except Exception as error:
            return f"Ошибка генерации ответа: {error}"

    def send_message(self):
        user_text = self.input_field.text().strip()

        if not user_text:
            return

        self.input_field.clear()
        self.input_field.setEnabled(False)
        self.send_btn.setEnabled(False)
        QApplication.processEvents()

        self.add_to_chat("user", user_text)

        result = self.classify(user_text)
        action = route(self.state, result)
        self.state.last_action = action

        if SHOW_DEBUG:
            self.render_debug(result, action)

        reply = self.generate_reply(action, result, user_text)

        self.add_to_chat("assistant", reply)
        self.state.last_agent_question = reply

        if self.state.ended:
            self.input_field.setEnabled(False)
            self.send_btn.setEnabled(False)
        else:
            self.input_field.setEnabled(True)
            self.send_btn.setEnabled(True)
            self.input_field.setFocus()

    def add_to_chat(self, role: str, content: str):
        self.context.append({"role": role, "content": content})
        self.render_message(role, content)
        print(f"{role}: {content}")

    def render_message(self, role: str, content: str):
        safe_content = html.escape(str(content)).replace("\n", "<br>")

        if role == "user":
            role_html = '<span style="color:blue; font-weight:bold;">user:</span>'
        elif role == "assistant":
            role_html = '<span style="color:green; font-weight:bold;">assistant:</span>'
        else:
            role_html = f'<span style="color:black; font-weight:bold;">{html.escape(role)}:</span>'

        self.chat_history.insertHtml(
            f'{role_html} <span style="color:black;">{safe_content}</span><br>'
        )
        self.chat_history.moveCursor(QTextCursor.End)

    def render_debug(self, result: Dict[str, Any], action: str):
        safe = html.escape(json.dumps(result, ensure_ascii=False))
        safe_action = html.escape(action)

        self.chat_history.insertHtml(
            f'<span style="color:gray;">DEBUG action={safe_action}; intent={safe}</span><br>'
        )
        self.chat_history.moveCursor(QTextCursor.End)

    def closeEvent(self, event):
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = ChatWindow()
    window.show()

    sys.exit(app.exec_())

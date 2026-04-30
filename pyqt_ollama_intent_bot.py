import sys
import json
import re
import html
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

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
from PyQt5.QtGui import QTextCursor


# ============================================================
# НАСТРОЙКИ
# ============================================================

OLLAMA_HOST = "http://127.0.0.1:11434"

# Основная модель агента.
# Если хочешь, можешь заменить на свою:
# "agent_8b_0_6_layer_scales", "agent_8b_0_6_v3_dpo" и т.д.
AGENT_MODEL = "t-8b-base"

# Модель для intent-классификации.
# Перед запуском:
# ollama pull qwen3:4b-instruct
INTENT_MODEL = "qwen3:4b-instruct"

# Если True, в чате будет показываться серый debug intent'а.
SHOW_DEBUG_INTENT = True

# По умолчанию ответы агента формируются жёстко через step-логику.
# Так безопаснее: бот не перескакивает между шагами.
# Если поставить True, основной AGENT_MODEL будет переформулировать готовый ответ.
USE_MAIN_AGENT_FOR_REPLY = False


prompt = """
Ты банковский агент Полина. Ты ведёшь диалог с клиентом по вопросу просроченной задолженности.

Правила:
- Говори спокойно и профессионально.
- Не угрожай.
- Не спорь.
- Не перескакивай между вариантами урегулирования.
- Если клиент не ответил на текущий вопрос, верни его к текущему вопросу.
- Не придумывай данные клиента.
- Всегда двигайся по сценарию.
"""


# ============================================================
# INTENT И STEP
# ============================================================

class Intent(str, Enum):
    CONFIRM_IDENTITY = "confirm_identity"
    NOT_CLIENT = "not_client"

    ANSWER_REASON = "answer_reason"

    AGREE_RELATIVE_HELP = "agree_relative_help"
    AGREE_RESTRUCTURING = "agree_restructuring"
    AGREE_CAR_COLLATERAL = "agree_car_collateral"
    AGREE_REFINANCE = "agree_refinance"

    PROVIDE_CONTACT_INFO = "provide_contact_info"

    REJECT = "reject"
    QUESTION = "question"
    OFFTOPIC = "offtopic"
    INSULT = "insult"
    UNCLEAR = "unclear"


class Step(str, Enum):
    CONFIRM_IDENTITY = "confirm_identity"
    ASK_REASON = "ask_reason"

    OFFER_RELATIVE_HELP = "offer_relative_help"
    OFFER_RESTRUCTURING = "offer_restructuring"
    OFFER_CAR_COLLATERAL = "offer_car_collateral"
    OFFER_REFINANCE = "offer_refinance"

    ASK_CONTACTS = "ask_contacts"
    FINISH = "finish"


@dataclass
class DialogueState:
    step: Step = Step.CONFIRM_IDENTITY
    reason: Optional[str] = None
    selected_solution: Optional[str] = None
    contact_info: Optional[str] = None
    is_client_confirmed: bool = False


SOLUTION_TITLES = {
    Intent.AGREE_RELATIVE_HELP.value: "помощь родственников или знакомых",
    Intent.AGREE_RESTRUCTURING.value: "частичная реструктуризация платежа",
    Intent.AGREE_CAR_COLLATERAL.value: "вариант с залогом автомобиля",
    Intent.AGREE_REFINANCE.value: "перекредитование в другой организации",
}


STEP_CONFIG = {
    Step.CONFIRM_IDENTITY: {
        "agent_question": "Петров Петр Петрович - это вы?",
        "allowed_intents": [
            Intent.CONFIRM_IDENTITY,
            Intent.NOT_CLIENT,
            Intent.QUESTION,
            Intent.OFFTOPIC,
            Intent.INSULT,
            Intent.UNCLEAR,
        ],
        "success_intent": Intent.CONFIRM_IDENTITY,
        "next_step": Step.ASK_REASON,
    },

    Step.ASK_REASON: {
        "agent_question": "По какой причине не получилось оплатить задолженность?",
        "allowed_intents": [
            Intent.ANSWER_REASON,
            Intent.QUESTION,
            Intent.OFFTOPIC,
            Intent.INSULT,
            Intent.UNCLEAR,
        ],
        "success_intent": Intent.ANSWER_REASON,
        "next_step": Step.OFFER_RELATIVE_HELP,
    },

    Step.OFFER_RELATIVE_HELP: {
        "agent_question": "Можете попросить помощь у родственников или знакомых, чтобы закрыть оплату?",
        "allowed_intents": [
            Intent.AGREE_RELATIVE_HELP,
            Intent.REJECT,
            Intent.QUESTION,
            Intent.OFFTOPIC,
            Intent.INSULT,
            Intent.UNCLEAR,
        ],
        "success_intent": Intent.AGREE_RELATIVE_HELP,
        "next_step": Step.ASK_CONTACTS,
        "reject_next_step": Step.OFFER_RESTRUCTURING,
    },

    Step.OFFER_RESTRUCTURING: {
        "agent_question": "Тогда можем рассмотреть частичную реструктуризацию платежа. Вам подходит такой вариант?",
        "allowed_intents": [
            Intent.AGREE_RESTRUCTURING,
            Intent.REJECT,
            Intent.QUESTION,
            Intent.OFFTOPIC,
            Intent.INSULT,
            Intent.UNCLEAR,
        ],
        "success_intent": Intent.AGREE_RESTRUCTURING,
        "next_step": Step.ASK_CONTACTS,
        "reject_next_step": Step.OFFER_CAR_COLLATERAL,
    },

    Step.OFFER_CAR_COLLATERAL: {
        "agent_question": "Можно рассмотреть вариант с залогом автомобиля. Такой вариант вам подходит?",
        "allowed_intents": [
            Intent.AGREE_CAR_COLLATERAL,
            Intent.REJECT,
            Intent.QUESTION,
            Intent.OFFTOPIC,
            Intent.INSULT,
            Intent.UNCLEAR,
        ],
        "success_intent": Intent.AGREE_CAR_COLLATERAL,
        "next_step": Step.ASK_CONTACTS,
        "reject_next_step": Step.OFFER_REFINANCE,
    },

    Step.OFFER_REFINANCE: {
        "agent_question": "Можно попробовать перекредитоваться в другой организации. Готовы рассмотреть этот вариант?",
        "allowed_intents": [
            Intent.AGREE_REFINANCE,
            Intent.REJECT,
            Intent.QUESTION,
            Intent.OFFTOPIC,
            Intent.INSULT,
            Intent.UNCLEAR,
        ],
        "success_intent": Intent.AGREE_REFINANCE,
        "next_step": Step.ASK_CONTACTS,
        "reject_next_step": Step.ASK_CONTACTS,
    },

    Step.ASK_CONTACTS: {
        "agent_question": "Подскажите, пожалуйста, актуальный контактный номер и удобное время для связи, чтобы мы зафиксировали договорённость.",
        "allowed_intents": [
            Intent.PROVIDE_CONTACT_INFO,
            Intent.REJECT,
            Intent.QUESTION,
            Intent.OFFTOPIC,
            Intent.INSULT,
            Intent.UNCLEAR,
        ],
        "success_intent": Intent.PROVIDE_CONTACT_INFO,
        "next_step": Step.FINISH,
    },
}


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def get_question_for_step(step: Step) -> str:
    if step == Step.FINISH:
        return "Хорошо, зафиксировала информацию. Спасибо за разговор."

    return STEP_CONFIG[step]["agent_question"]


def safe_json_loads_from_text(text: str) -> Dict[str, Any]:
    """
    Qwen иногда может вернуть не чистый JSON, а текст + JSON.
    Поэтому сначала пробуем json.loads, потом вытаскиваем {...}.
    """
    text = text.strip()

    # Убираем возможные markdown-блоки
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {
            "intent": Intent.UNCLEAR.value,
            "confidence": 0.0,
            "is_answer_to_current_question": False,
            "extracted_value": None,
            "short_reason": "JSON не найден",
        }

    try:
        return json.loads(match.group(0))
    except Exception:
        return {
            "intent": Intent.UNCLEAR.value,
            "confidence": 0.0,
            "is_answer_to_current_question": False,
            "extracted_value": None,
            "short_reason": "JSON повреждён",
        }


def normalize_intent_result(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Приводим ответ модели к безопасному формату.
    """
    return {
        "intent": str(data.get("intent", Intent.UNCLEAR.value)),
        "confidence": float(data.get("confidence", 0.0) or 0.0),
        "is_answer_to_current_question": bool(data.get("is_answer_to_current_question", False)),
        "extracted_value": data.get("extracted_value"),
        "short_reason": str(data.get("short_reason", "")),
    }


# ============================================================
# ОСНОВНОЕ ОКНО
# ============================================================

class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.state = DialogueState()

        self.agent_model = AGENT_MODEL
        self.intent_model = INTENT_MODEL

        self.ollama_client = None
        self.model = None  # оставил, чтобы closeEvent был похож на твой старый код

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

        self.prompt = prompt

        self.context = [
            {
                "role": "system",
                "content": self.prompt
            },
            {
                "role": "assistant",
                "content": "Алло, здравствуйте, меня зовут Полина, я сотрудник Da банк. Петров Петр Петрович - это вы?"
            },
        ]

        for c in self.context[1:]:
            self.render_message(c["role"], c["content"])

        self.input_field = QLineEdit()
        self.input_field.returnPressed.connect(self.send_message)
        layout.addWidget(self.input_field)

        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_message)
        layout.addWidget(self.send_btn)

    def load_model(self):
        self.ollama_client = ollama.Client(
            host=OLLAMA_HOST,
            trust_env=False
        )

        self.model = None

    def send_message(self):
        user_input = self.input_field.text().strip()

        if not user_input:
            return

        self.input_field.clear()

        self.send_btn.setEnabled(False)
        self.input_field.setEnabled(False)
        QApplication.processEvents()

        self.update_chat_history(
            text=f"клиент:{user_input}",
            text_only=user_input,
            role="user"
        )

        try:
            # 1. Сначала Qwen определяет intent
            intent_result = self.classify_intent(user_input)

            if SHOW_DEBUG_INTENT:
                self.render_debug_intent(intent_result)

            # 2. Python step-логика решает, что делать дальше
            planned_reply = self.process_intent(user_input, intent_result)

            # 3. По умолчанию используем жёсткий ответ.
            # Если хочешь, можно включить переформулировку через основного агента.
            if USE_MAIN_AGENT_FOR_REPLY:
                agent_reply = self.ask_agent_model_to_rephrase(planned_reply)
            else:
                agent_reply = planned_reply

            self.update_chat_history(
                text=f"агент:{agent_reply}",
                text_only=agent_reply,
                role="assistant"
            )

        except Exception as e:
            error_text = f"Ошибка обработки сообщения: {e}"
            print(error_text)
            self.update_chat_history(
                text=f"system:{error_text}",
                text_only=error_text,
                role="system"
            )

        finally:
            self.send_btn.setEnabled(True)
            self.input_field.setEnabled(True)
            self.input_field.setFocus()

    def classify_intent(self, user_input: str) -> Dict[str, Any]:
        current_step = self.state.step

        if current_step == Step.FINISH:
            return {
                "intent": Intent.UNCLEAR.value,
                "confidence": 1.0,
                "is_answer_to_current_question": False,
                "extracted_value": None,
                "short_reason": "Диалог уже завершён",
            }

        step_config = STEP_CONFIG[current_step]

        allowed_intents = [
            intent.value for intent in step_config["allowed_intents"]
        ]

        system_prompt = f"""
Ты intent-классификатор для диалога банковского агента с клиентом.

Твоя задача:
1. Определить intent сообщения клиента.
2. Учитывать текущий шаг диалога.
3. Учитывать последний вопрос агента.
4. Не писать ответ клиенту.
5. Не выбирать следующий шаг.
6. Вернуть только JSON без markdown.

Доступные intent только из этого списка:
{allowed_intents}

Правила:
- Если клиент отвечает по смыслу на последний вопрос агента, is_answer_to_current_question = true.
- Если клиент ушёл в сторону, ругается, спорит не по теме или пишет не ответ на вопрос, is_answer_to_current_question = false.
- Если клиент просто пишет "да", "угу", "верно", "это я", на шаге confirm_identity это intent = "confirm_identity".
- Если клиент говорит, что это не он, intent = "not_client".
- Если вопрос был про причину неоплаты, нормальный ответ с причиной = "answer_reason".
- Если вопрос был про помощь родственников, согласие = "agree_relative_help".
- Если вопрос был про реструктуризацию, согласие = "agree_restructuring".
- Если вопрос был про залог автомобиля, согласие = "agree_car_collateral".
- Если вопрос был про перекредитование, согласие = "agree_refinance".
- Если вопрос был про контактные данные и клиент дал телефон, email, время связи или другой контакт, intent = "provide_contact_info".
- Если клиент отказывается, говорит "нет", "не получится", "не могу", intent = "reject".
- Если клиент задаёт уточняющий вопрос, intent = "question".
- Если клиент оскорбляет агента, intent = "insult".
- Если клиент говорит совсем не по теме, intent = "offtopic".
- Если не уверен, intent = "unclear", confidence ниже 0.6.

Важно:
- Intent должен быть только из списка доступных intent.
- Не добавляй пояснения вне JSON.
- Не используй markdown.
- Не пиши ```json.

Формат ответа строго JSON:
{{
  "intent": "one_of_allowed_intents",
  "confidence": 0.0,
  "is_answer_to_current_question": true,
  "extracted_value": null,
  "short_reason": "короткое объяснение"
}}
"""

        user_prompt = f"""
Текущий шаг:
{current_step.value}

Последний вопрос агента:
{step_config["agent_question"]}

Ответ клиента:
{user_input}
"""

        try:
            response = self.ollama_client.chat(
                model=self.intent_model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt.strip()
                    },
                    {
                        "role": "user",
                        "content": user_prompt.strip()
                    },
                ],
                options={
                    "temperature": 0,
                    "num_predict": 200,
                    "top_p": 0.1,
                    "repeat_penalty": 1.05,
                    "num_ctx": 4096,
                },
                think=False
            )

            raw_content = response["message"]["content"]
            data = safe_json_loads_from_text(raw_content)
            data = normalize_intent_result(data)

        except Exception as e:
            data = {
                "intent": Intent.UNCLEAR.value,
                "confidence": 0.0,
                "is_answer_to_current_question": False,
                "extracted_value": None,
                "short_reason": f"Ошибка intent-модели: {e}",
            }

        # Защита: если модель вернула intent не из разрешённых для текущего шага
        if data.get("intent") not in allowed_intents:
            data = {
                "intent": Intent.UNCLEAR.value,
                "confidence": 0.0,
                "is_answer_to_current_question": False,
                "extracted_value": None,
                "short_reason": "Модель вернула intent, который запрещён на текущем шаге",
            }

        return data

    def process_intent(self, user_input: str, intent_result: Dict[str, Any]) -> str:
        if self.state.step == Step.FINISH:
            return "Диалог уже завершён. Ожидаем выполнение договорённости."

        current_step = self.state.step
        step_config = STEP_CONFIG[current_step]

        intent = intent_result.get("intent", Intent.UNCLEAR.value)
        confidence = float(intent_result.get("confidence", 0.0))
        is_answer = bool(intent_result.get("is_answer_to_current_question", False))

        # 1. Если Qwen не уверен — step не двигаем
        if confidence < 0.65:
            return self.build_reask_message(intent_result)

        # 2. Если клиент не ответил на текущий вопрос — step не двигаем
        if not is_answer:
            return self.build_reask_message(intent_result)

        # 3. Подтверждение личности
        if current_step == Step.CONFIRM_IDENTITY:
            if intent == Intent.CONFIRM_IDENTITY.value:
                self.state.is_client_confirmed = True
                self.state.step = step_config["next_step"]
                return get_question_for_step(self.state.step)

            if intent == Intent.NOT_CLIENT.value:
                self.state.step = Step.FINISH
                return "Поняла вас. В таком случае информацию зафиксирую. Спасибо."

            return self.build_reask_message(intent_result)

        # 4. Причина неоплаты
        if current_step == Step.ASK_REASON:
            if intent == Intent.ANSWER_REASON.value:
                self.state.reason = intent_result.get("extracted_value") or user_input
                self.state.step = step_config["next_step"]
                return get_question_for_step(self.state.step)

            return self.build_reask_message(intent_result)

        # 5. Варианты урегулирования
        if current_step in [
            Step.OFFER_RELATIVE_HELP,
            Step.OFFER_RESTRUCTURING,
            Step.OFFER_CAR_COLLATERAL,
            Step.OFFER_REFINANCE,
        ]:
            success_intent = step_config["success_intent"].value

            # Согласился именно с текущим вариантом
            if intent == success_intent:
                self.state.selected_solution = intent
                self.state.step = step_config["next_step"]
                return get_question_for_step(self.state.step)

            # Отказался — идём к следующему варианту
            if intent == Intent.REJECT.value:
                self.state.step = step_config.get("reject_next_step", Step.ASK_CONTACTS)
                return get_question_for_step(self.state.step)

            return self.build_reask_message(intent_result)

        # 6. Контактные данные
        if current_step == Step.ASK_CONTACTS:
            if intent == Intent.PROVIDE_CONTACT_INFO.value:
                self.state.contact_info = intent_result.get("extracted_value") or user_input
                self.state.step = Step.FINISH
                return self.build_summary_message()

            if intent == Intent.REJECT.value:
                self.state.contact_info = "Клиент отказался предоставить контактные данные"
                self.state.step = Step.FINISH
                return self.build_summary_message()

            return self.build_reask_message(intent_result)

        return self.build_reask_message(intent_result)

    def build_reask_message(self, intent_result: Dict[str, Any]) -> str:
        current_step = self.state.step
        question = get_question_for_step(current_step)

        intent = intent_result.get("intent", Intent.UNCLEAR.value)

        if intent == Intent.INSULT.value:
            return f"Я понимаю, что ситуация может быть неприятной. Давайте вернёмся к вопросу: {question}"

        if intent == Intent.OFFTOPIC.value:
            return f"Давайте всё-таки решим вопрос по задолженности. {question}"

        if intent == Intent.QUESTION.value:
            return f"Поясню: мне нужно зафиксировать ответ по текущему вопросу. {question}"

        if current_step == Step.CONFIRM_IDENTITY:
            return "Подскажите, пожалуйста, Петров Петр Петрович - это вы?"

        return f"Не совсем понял ваш ответ. {question}"

    def build_summary_message(self) -> str:
        reason = self.state.reason or "причина не указана"

        if self.state.selected_solution:
            solution = SOLUTION_TITLES.get(
                self.state.selected_solution,
                self.state.selected_solution
            )
        else:
            solution = "вариант урегулирования не выбран"

        contact = self.state.contact_info or "контактные данные не указаны"

        return (
            "Хорошо, подытожу разговор. "
            f"Причина неоплаты: {reason}. "
            f"Вариант урегулирования: {solution}. "
            f"Контактная информация: {contact}. "
            "Информацию зафиксировала. Спасибо за разговор."
        )

    def ask_agent_model_to_rephrase(self, planned_reply: str) -> str:
        """
        Опционально: основной агент переформулирует готовый ответ,
        но не меняет смысл и не выбирает следующий шаг.
        По умолчанию USE_MAIN_AGENT_FOR_REPLY = False.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "Ты банковский агент Полина. "
                    "Тебе дают готовый смысл ответа. "
                    "Твоя задача — только переформулировать его естественно и кратко. "
                    "Не добавляй новые вопросы. "
                    "Не меняй смысл. "
                    "Не выбирай следующий шаг."
                )
            },
            {
                "role": "user",
                "content": f"Переформулируй этот ответ без изменения смысла:\n{planned_reply}"
            }
        ]

        response = self.ollama_client.chat(
            model=self.agent_model,
            messages=messages,
            options={
                "temperature": 0.3,
                "num_predict": 250,
                "repeat_penalty": 1.1,
                "top_k": 50,
                "top_p": 0.8,
                "min_p": 0.0,
                "num_ctx": 4096,
            },
            think=False
        )

        return response["message"]["content"].strip()

    def update_chat_history(self, text: str, text_only: str, role: str):
        self.context.append({
            "role": role,
            "content": text_only
        })

        print(text_only)
        self.render_message(role, text_only)

    def render_message(self, role: str, content: str):
        safe_content = html.escape(str(content)).replace("\n", "<br>")

        if role == "user":
            role_html = f'<span style="color:blue; font-weight: bold;">{role}:</span>'
        elif role == "assistant":
            role_html = f'<span style="color:green; font-weight: bold;">{role}:</span>'
        else:
            role_html = f'<span style="color:gray; font-weight: bold;">{role}:</span>'

        message_html = f'{role_html} <span style="color: black;">{safe_content}</span><br>'
        self.chat_history.insertHtml(message_html)
        self.chat_history.moveCursor(QTextCursor.End)

    def render_debug_intent(self, intent_result: Dict[str, Any]):
        intent = html.escape(str(intent_result.get("intent")))
        confidence = html.escape(str(intent_result.get("confidence")))
        is_answer = html.escape(str(intent_result.get("is_answer_to_current_question")))
        reason = html.escape(str(intent_result.get("short_reason")))

        debug_html = (
            f'<span style="color:gray;">'
            f'INTENT: {intent}, confidence: {confidence}, '
            f'is_answer: {is_answer}, reason: {reason}'
            f'</span><br>'
        )

        self.chat_history.insertHtml(debug_html)
        self.chat_history.moveCursor(QTextCursor.End)

    def closeEvent(self, event):
        if self.model is not None:
            del self.model

        print("Модель удалена из памяти")
        super().closeEvent(event)


# ============================================================
# ЗАПУСК
# ============================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = ChatWindow()
    window.show()

    sys.exit(app.exec_())

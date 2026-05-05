import sys
import re
import html
from dataclasses import dataclass
from typing import Optional, Dict

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
# 0. НАСТРОЙКИ
# ============================================================

OLLAMA_HOST = "http://127.0.0.1:11434"

# Основная модель, которая отвечает клиенту.
AGENT_MODEL = "t-8b-base"

# Маленькая Qwen-модель, которая определяет только крупный блок:
# INTRO / SETTLEMENT / SUMMARY / END / CURRENT.
# Если у тебя в Ollama модель называется иначе, поменяй здесь.
BLOCK_ROUTER_MODEL = "qwen3:4b-instruct"

# Если True, маленькая Qwen-модель будет помогать выбирать крупный блок.
# Если False, блоки будут переключаться только по служебным меткам основной модели.
USE_BLOCK_ROUTER = True

# Показывать debug в консоли.
DEBUG = True


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

# Эти данные нельзя раскрывать до подтверждения личности.
PRIVATE_DATA = {
    "debt_amount": "15000 рублей",
    "loan_type": "автокредит",
    "overdue_days": "10 дней",
    "collateral_value": "2310000 рублей",
}


# ============================================================
# 2. СОСТОЯНИЕ БЕЗ STAGE ПО УРЕГУЛИРОВАНИЮ
# ============================================================

@dataclass
class DialogState:
    # Крупный блок разговора.
    # INTRO      — приветствие, подтверждение личности, причина неоплаты.
    # SETTLEMENT — варианты урегулирования.
    # SUMMARY    — контакты и подытоживание.
    # END        — завершение.
    block: str = "INTRO"

    identity_confirmed: bool = False
    agreement_reached: bool = False
    contact_question_asked: bool = False
    contact_checked: bool = False
    ended: bool = False

    # Эти поля заполняются не intent'ами, а служебными метками модели.
    debt_reason: Optional[str] = None
    agreement_summary: Optional[str] = None
    contact_info: Optional[str] = None

    # Для отладки.
    last_router_block: Optional[str] = None
    last_bot_raw: Optional[str] = None


def get_visible_facts(state: DialogState) -> Dict[str, str]:
    facts = dict(PUBLIC_DATA)
    if state.identity_confirmed:
        facts.update(PRIVATE_DATA)
    return facts


# ============================================================
# 3. PROMPT'Ы БЛОКОВ
# ============================================================

BASE_SYSTEM_PROMPT = """
Ты голосовой помощник АО Da банк. Тебя зовут Полина.

Общие правила:
- Отвечай кратко, делово и спокойно.
- Не угрожай и не дави.
- Не упоминай полицию, уголовное дело, арест, выезд сотрудников.
- Не проси паспорт, дату рождения, код из SMS, последние цифры паспорта.
- Не раскрывай финансовые детали до подтверждения личности.
- Не задавай больше одного вопроса за раз.
- Не пиши роль "assistant:" или "оператор:".
- Если нужно добавить служебную метку, добавляй ее в самом конце ответа.
"""


INTRO_PROMPT = """
БЛОК: INTRO — начало разговора, подтверждение личности, причина неоплаты.

Цели блока:
1. Поздороваться и представиться.
2. Подтвердить, что разговариваешь именно с Петровым Петром Петровичем.
3. До подтверждения личности НЕ раскрывать причину звонка.
4. Если собеседник спрашивает, по какому вопросу звонок, до подтверждения личности скажи:
   "Информация предназначена только для Петра Петровича".
5. После подтверждения личности сообщи:
   - что разговор записывается;
   - сумма задолженности: 15000 рублей;
   - тип кредита: автокредит;
   - срок просрочки: 10 дней.
6. Спроси причину неоплаты.
7. После ответа с причиной спроси, сможет ли клиент внести оплату в ближайшие 3 дня.
8. Если клиент говорит, что не сможет оплатить в ближайшие 3 дня, переходи к урегулированию.
9. Если клиент говорит, что сможет оплатить в ближайшие 3 дня, зафиксируй договоренность и сразу спроси,
   изменилась ли контактная информация.

Строгие запреты до подтверждения личности:
- не говорить про долг;
- не говорить про кредит;
- не говорить про просрочку;
- не говорить сумму;
- не говорить про автомобиль;
- не говорить про суд;
- не говорить про взыскание.

Служебные метки:
- Если клиент подтвердил личность, добавь: [IDENTITY_CONFIRMED]
- Если клиент назвал причину неоплаты, добавь: [DEBT_REASON=краткая причина]
- Если клиент не сможет оплатить в ближайшие 3 дня и нужно перейти к вариантам урегулирования, добавь: [GO_SETTLEMENT]
- Если клиент согласился оплатить в ближайшие 3 дня, добавь:
  [AGREEMENT_REACHED=полная оплата в ближайшие 3 дня] [CONTACT_QUESTION_ASKED]
- Если ты спросила, изменилась ли контактная информация, добавь: [CONTACT_QUESTION_ASKED]

Важно:
- Не добавляй метку, если событие не произошло.
- Не раскрывай финансовые детали, пока личность не подтверждена.
- Ответ максимум 1-2 предложения.
"""


SETTLEMENT_PROMPT = """
БЛОК: SETTLEMENT — урегулирование задолженности.

Личность клиента уже подтверждена. Можно обсуждать задолженность, кредит, просрочку,
автомобиль и варианты урегулирования.

Твоя задача:
Убедить клиента выбрать реалистичный вариант урегулирования задолженности.

Доступные варианты урегулирования:

1. Полная оплата в ближайшие 3 дня.
   Если клиент говорит, что сможет оплатить полностью, зафиксируй это
   и сразу спроси, изменилась ли контактная информация.

2. Помощь родственников или друзей.
   Предложи клиенту обратиться к родственникам, друзьям или знакомым,
   если у него сейчас нет полной суммы.

3. Занять средства.
   Можно предложить клиенту занять деньги временно, чтобы не допустить
   дальнейшего ухудшения ситуации по просрочке.

4. Перекредитоваться или рефинансироваться.
   Можно предложить клиенту рассмотреть другой источник финансирования,
   если это поможет закрыть текущую просрочку.

5. Частичная оплата.
   Если полной суммы нет, предложи внести хотя бы часть задолженности.
   Уточни, какую сумму клиент сможет внести и когда.

6. Реструктуризация.
   Если клиенту тяжело платить по текущему графику, предложи реструктуризацию:
   продление срока кредита для снижения ежемесячного платежа.
   Условия: ставка от 18,9% годовых.
   Уточни, готов ли клиент рассмотреть этот вариант.

7. Передача автомобиля.
   Так как кредит связан с автомобилем, можно предложить передачу автомобиля
   для дальнейшего рассмотрения банком.
   Если клиент готов обсуждать этот вариант, уточни:
   - на кого зарегистрирован автомобиль;
   - в каком он состоянии;
   - сможет ли клиент подписать документы и передать автомобиль в ближайшие 3 дня.

8. Последствия отсутствия договоренности.
   Если клиент отказывается от всех вариантов, нейтрально объясни:
   - задолженность может увеличиваться из-за начислений;
   - кредитная история может ухудшиться;
   - банк может продолжить работу по договору в установленном порядке.

9. Возможное обращение в суд.
   Если договориться не получается, можно нейтрально сказать:
   банк может рассмотреть обращение в суд в установленном законом порядке.
   Не угрожай и не дави.

Правила ведения блока:
- Не перечисляй все варианты сразу.
- Предлагай варианты последовательно и естественно.
- Ориентируйся на историю диалога.
- Не повторяй вариант, который клиент уже явно отклонил.
- Если клиент согласился на любой вариант, не предлагай следующие варианты.
- После согласия сразу спроси, изменилась ли контактная информация.
- Если клиент задал вопрос, сначала кратко ответь, затем вернись к выбору варианта урегулирования.
- Если клиент грубит или уходит от темы, спокойно верни его к вопросу урегулирования.
- Не задавай больше одного вопроса за раз.
- Ответ должен быть коротким: 1-2 предложения.

Служебные метки:
- Если договоренность достигнута, добавь:
  [AGREEMENT_REACHED=кратко какой вариант согласован]
- Если ты спросила, изменилась ли контактная информация, добавь:
  [CONTACT_QUESTION_ASKED]

Пример метки:
[AGREEMENT_REACHED=реструктуризация кредита] [CONTACT_QUESTION_ASKED]

Важно:
- Если договоренность не достигнута, не добавляй [AGREEMENT_REACHED].
- Не используй JSON.
"""


SUMMARY_PROMPT = """
БЛОК: SUMMARY — проверка контактов и подытоживание договоренности.

Перед завершением разговора обязательно нужно проверить контактную информацию.

Если вопрос про контактную информацию ЕЩЕ НЕ был задан:
Спроси ровно это:
"Подскажите, изменилась ли ваша контактная информация? Если да, назовите актуальные данные."
И добавь метку:
[CONTACT_QUESTION_ASKED]

Если клиент УЖЕ ответил по контактной информации:
- если контакты не изменились, зафиксируй это;
- если клиент назвал новый номер, адрес, email или удобное время связи, зафиксируй это;
- затем подытожь договоренность.

В итоговой реплике обязательно укажи:
1. что договоренность зафиксирована;
2. какой вариант урегулирования согласован;
3. что контактная информация проверена;
4. что банк ожидает выполнение договоренности;
5. благодарность и завершение разговора.

Не предлагай новые варианты урегулирования.
Не возвращайся к спору.
Не задавай новые вопросы после итоговой реплики.

Служебные метки:
- Если контактная информация проверена, добавь:
  [CONTACT_CHECKED=кратко что с контактами]
- Если разговор завершён, добавь:
  [END_DIALOG]

Пример:
[CONTACT_CHECKED=номер не изменился] [END_DIALOG]

Не используй JSON.
"""


END_PROMPT = """
БЛОК: END — завершение разговора.

Кратко и вежливо попрощайся.
Не предлагай новые варианты.
Не раскрывай новые данные.
Добавь метку [END_DIALOG].
"""


# ============================================================
# 4. УТИЛИТЫ ДЛЯ МЕТОК
# ============================================================

MARKER_NAMES = [
    "IDENTITY_CONFIRMED",
    "GO_SETTLEMENT",
    "CONTACT_QUESTION_ASKED",
    "END_DIALOG",
]

VALUE_MARKER_NAMES = [
    "DEBT_REASON",
    "AGREEMENT_REACHED",
    "CONTACT_CHECKED",
]


def clean_reply(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"^(assistant|оператор|бот|ответ)\s*:\s*", "", text, flags=re.I)
    return text.strip().strip('"').strip()


def extract_value_marker(text: str, name: str) -> Optional[str]:
    pattern = r"\[" + re.escape(name) + r"\s*=\s*(.*?)\]"
    match = re.search(pattern, text, flags=re.S)
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def has_marker(text: str, name: str) -> bool:
    return re.search(r"\[" + re.escape(name) + r"\]", text) is not None


def remove_service_markers(text: str) -> str:
    # Удаляем метки вида [NAME]
    for name in MARKER_NAMES:
        text = re.sub(r"\[" + re.escape(name) + r"\]", "", text)

    # Удаляем метки вида [NAME=value]
    for name in VALUE_MARKER_NAMES:
        text = re.sub(r"\[" + re.escape(name) + r"\s*=\s*.*?\]", "", text, flags=re.S)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def apply_markers_to_state(text: str, state: DialogState):
    if has_marker(text, "IDENTITY_CONFIRMED"):
        state.identity_confirmed = True
        state.block = "INTRO"

    debt_reason = extract_value_marker(text, "DEBT_REASON")
    if debt_reason:
        state.debt_reason = debt_reason

    if has_marker(text, "GO_SETTLEMENT"):
        state.block = "SETTLEMENT"

    agreement = extract_value_marker(text, "AGREEMENT_REACHED")
    if agreement:
        state.agreement_reached = True
        state.agreement_summary = agreement
        state.block = "SUMMARY"

    if has_marker(text, "CONTACT_QUESTION_ASKED"):
        state.contact_question_asked = True
        if state.agreement_reached:
            state.block = "SUMMARY"

    contact_info = extract_value_marker(text, "CONTACT_CHECKED")
    if contact_info:
        state.contact_checked = True
        state.contact_info = contact_info

    if has_marker(text, "END_DIALOG"):
        state.ended = True
        state.block = "END"


# ============================================================
# 5. ROUTER ДЛЯ КРУПНОГО БЛОКА
# ============================================================

def extract_block(raw_text: str) -> str:
    raw = raw_text.upper()

    for block in ["INTRO", "SETTLEMENT", "SUMMARY", "END", "CURRENT"]:
        if block in raw:
            return block

    return "CURRENT"


# ============================================================
# 6. PYQT APP
# ============================================================

class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.dialog_state = DialogState()
        self.ollama_client = None
        self.model = None

        self.context = [
            {
                "role": "system",
                "content": BASE_SYSTEM_PROMPT.strip(),
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
        self.ollama_client = ollama.Client(
            host=OLLAMA_HOST,
            trust_env=False
        )

    # ------------------------------------------------------------
    # BLOCK ROUTING
    # ------------------------------------------------------------

    def choose_block(self, user_input: str) -> str:
        state = self.dialog_state

        # Жёсткие переходы. Здесь Qwen не нужен.
        if state.ended:
            return "END"

        if not state.identity_confirmed:
            return "INTRO"

        if state.agreement_reached and not state.contact_checked:
            return "SUMMARY"

        if state.contact_checked:
            return "END"

        if not USE_BLOCK_ROUTER:
            return state.block

        # Qwen выбирает только крупный блок, не intent.
        routed = self.route_block_with_qwen(user_input)
        state.last_router_block = routed

        if routed == "CURRENT":
            return state.block

        return routed

    def route_block_with_qwen(self, user_input: str) -> str:
        state = self.dialog_state

        router_prompt = f"""
Ты классификатор крупного блока банковского диалога.

Верни только одно слово из списка:
INTRO
SETTLEMENT
SUMMARY
END
CURRENT

Никакого JSON.
Никаких пояснений.

Значения:
INTRO — приветствие, подтверждение личности, причина неоплаты, вопрос об оплате в 3 дня.
SETTLEMENT — варианты урегулирования задолженности.
SUMMARY — проверка контактной информации и подытоживание договоренности.
END — разговор завершён.
CURRENT — остаться в текущем блоке.

Текущее состояние:
block={state.block}
identity_confirmed={state.identity_confirmed}
agreement_reached={state.agreement_reached}
contact_question_asked={state.contact_question_asked}
contact_checked={state.contact_checked}
ended={state.ended}

Последняя реплика клиента:
{user_input}
""".strip()

        try:
            response = self.ollama_client.chat(
                model=BLOCK_ROUTER_MODEL,
                messages=[
                    {"role": "user", "content": router_prompt}
                ],
                options={
                    "temperature": 0,
                    "num_predict": 8,
                    "num_ctx": 1024,
                },
                think=False,
            )

            return extract_block(response["message"]["content"])

        except Exception as e:
            print("Router error:", e)
            return "CURRENT"

    # ------------------------------------------------------------
    # PROMPT BUILDING
    # ------------------------------------------------------------

    def get_block_prompt(self, block: str) -> str:
        if block == "INTRO":
            return INTRO_PROMPT.strip()

        if block == "SETTLEMENT":
            return SETTLEMENT_PROMPT.strip()

        if block == "SUMMARY":
            return SUMMARY_PROMPT.strip()

        if block == "END":
            return END_PROMPT.strip()

        return INTRO_PROMPT.strip()

    def build_dynamic_system_prompt(self, block: str, user_input: str) -> str:
        state = self.dialog_state
        facts = get_visible_facts(state)

        if state.identity_confirmed:
            privacy_mode = (
                "Личность подтверждена. Можно обсуждать задолженность, кредит, просрочку, "
                "автомобиль и варианты урегулирования."
            )
        else:
            privacy_mode = (
                "Личность НЕ подтверждена. Запрещено раскрывать причину звонка, долг, кредит, "
                "просрочку, сумму, автомобиль, залог, взыскание, суд, кредитную историю и любые "
                "финансовые детали."
            )

        block_prompt = self.get_block_prompt(block)

        return f"""
{BASE_SYSTEM_PROMPT.strip()}

Доступные факты:
{facts}

Режим конфиденциальности:
{privacy_mode}

Текущее крупное состояние:
block={state.block}
identity_confirmed={state.identity_confirmed}
agreement_reached={state.agreement_reached}
contact_question_asked={state.contact_question_asked}
contact_checked={state.contact_checked}
ended={state.ended}
debt_reason={state.debt_reason}
agreement_summary={state.agreement_summary}
contact_info={state.contact_info}

Выбранный блок для ответа:
{block}

Инструкция выбранного блока:
{block_prompt}

Последняя реплика клиента:
{user_input}

Дополнительные правила:
- История диалога уже дана в messages. Используй ее, чтобы не повторять отклоненные варианты.
- Не используй JSON.
- Служебные метки пиши только если произошло соответствующее событие.
- Служебные метки должны быть в квадратных скобках и в конце ответа.
- Ответь только репликой оператора.
""".strip()

    def build_messages_for_agent(self, dynamic_system_prompt: str):
        # В историю не сохраняем dynamic prompt, чтобы не раздувать context.
        # Каждый раз собираем system заново + живую историю.
        return [
            {"role": "system", "content": dynamic_system_prompt}
        ] + self.context[1:]

    # ------------------------------------------------------------
    # MAIN SEND
    # ------------------------------------------------------------

    def send_message(self):
        user_input = self.input_field.text().strip()
        if not user_input:
            return

        self.input_field.clear()
        self.send_btn.setEnabled(False)
        self.input_field.setEnabled(False)
        QApplication.processEvents()

        try:
            # 1. Добавляем реплику клиента в историю и UI.
            self.update_chat_history(user_input, "user")

            # 2. Выбираем только крупный блок.
            block = self.choose_block(user_input)
            self.dialog_state.block = block

            # 3. Собираем prompt выбранного блока.
            dynamic_system_prompt = self.build_dynamic_system_prompt(
                block=block,
                user_input=user_input,
            )

            # 4. Основной агент формирует ответ.
            response = self.ollama_client.chat(
                model=AGENT_MODEL,
                messages=self.build_messages_for_agent(dynamic_system_prompt),
                options={
                    "temperature": 0.1,
                    "num_predict": 260,
                    "repeat_penalty": 1.1,
                    "top_k": 40,
                    "top_p": 0.8,
                    "min_p": 0.00,
                    "num_ctx": 8192,
                },
                think=False,
            )

            raw_bot_text = clean_reply(response["message"]["content"])
            self.dialog_state.last_bot_raw = raw_bot_text

            # 5. Считываем служебные метки и обновляем только крупное состояние.
            apply_markers_to_state(raw_bot_text, self.dialog_state)

            # 6. Убираем метки перед показом клиенту.
            bot_text = remove_service_markers(raw_bot_text)

            # 7. Если модель по какой-то причине вернула пустой ответ.
            if not bot_text:
                bot_text = "Уточните, пожалуйста, ваш ответ."

            # 8. Добавляем ответ агента в историю и UI.
            self.update_chat_history(bot_text, "assistant")

            if DEBUG:
                self.print_debug(user_input, block, raw_bot_text)

            # 9. Если диалог завершён — блокируем ввод.
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
            self.update_chat_history(error_text, "system")
            self.input_field.setEnabled(True)
            self.send_btn.setEnabled(True)
            self.input_field.setFocus()

    # ------------------------------------------------------------
    # UI HELPERS
    # ------------------------------------------------------------

    def update_chat_history(self, text_only: str, role: str):
        self.context.append({
            "role": role,
            "content": text_only
        })

        print(f"{role}: {text_only}")
        self.render_message(role, text_only)

    def render_message(self, role: str, content: str):
        safe_content = html.escape(str(content)).replace("\n", "<br>")

        if role == "user":
            role_html = '<span style="color:blue; font-weight:bold;">user:</span>'
        elif role == "assistant":
            role_html = '<span style="color:green; font-weight:bold;">assistant:</span>'
        else:
            role_html = f'<span style="color:gray; font-weight:bold;">{html.escape(role)}:</span>'

        message_html = f'{role_html} <span style="color:black;">{safe_content}</span><br>'
        self.chat_history.insertHtml(message_html)
        self.chat_history.moveCursor(QTextCursor.End)

    def print_debug(self, user_input: str, chosen_block: str, raw_bot_text: str):
        state = self.dialog_state

        print("\n--- DEBUG ---")
        print("user_input:", user_input)
        print("router_block:", state.last_router_block)
        print("chosen_block:", chosen_block)
        print("state.block:", state.block)
        print("identity_confirmed:", state.identity_confirmed)
        print("agreement_reached:", state.agreement_reached)
        print("contact_question_asked:", state.contact_question_asked)
        print("contact_checked:", state.contact_checked)
        print("ended:", state.ended)
        print("debt_reason:", state.debt_reason)
        print("agreement_summary:", state.agreement_summary)
        print("contact_info:", state.contact_info)
        print("raw_bot_text:", raw_bot_text)
        print("--- END DEBUG ---\n")

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


# ============================================================
# 7. ЗАПУСК
# ============================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = ChatWindow()
    window.show()

    sys.exit(app.exec_())

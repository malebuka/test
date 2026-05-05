import sys
import re
import html
from dataclasses import dataclass, field
from typing import Optional, List, Dict

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

# Маленькая модель, которая МОЖЕТ выбрать крупный блок,
# но в улучшенной версии она не вызывается на каждом ходу.
BLOCK_ROUTER_MODEL = "qwen3:4b-instruct"

# Варианты:
# "off"      — qwen-router вообще не вызывается, блоки меняются по меткам основной модели.
# "fallback" — qwen-router вызывается только когда код не понимает, в каком блоке продолжать.
# "always"   — qwen-router вызывается каждый ход. Дороже, обычно не нужно.
ROUTER_MODE = "fallback"

DEBUG = True

MAX_HISTORY_MESSAGES = 18


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

PRIVATE_DATA = {
    "debt_amount": "15000 рублей",
    "loan_type": "автокредит",
    "overdue_days": "10 дней",
    "collateral_value": "2310000 рублей",
}


# ============================================================
# 2. СОСТОЯНИЕ БЕЗ settlement_stage
# ============================================================

@dataclass
class DialogState:
    # Крупный блок:
    # INTRO       — приветствие, личность, причина неоплаты.
    # SETTLEMENT  — варианты урегулирования.
    # SUMMARY     — контакты и итог.
    # END         — завершение.
    block: str = "INTRO"

    identity_confirmed: bool = False
    agreement_reached: bool = False
    contact_question_asked: bool = False
    contact_checked: bool = False
    ended: bool = False

    debt_reason: Optional[str] = None
    agreement_summary: Optional[str] = None
    contact_info: Optional[str] = None

    # Это НЕ stage. Это память, чтобы модель не повторяла одно и то же.
    offered_options: List[str] = field(default_factory=list)
    rejected_options: List[str] = field(default_factory=list)

    last_bot_text: Optional[str] = None
    last_router_block: Optional[str] = None


def get_visible_facts(state: DialogState) -> Dict[str, str]:
    facts = dict(PUBLIC_DATA)
    if state.identity_confirmed:
        facts.update(PRIVATE_DATA)
    return facts


# ============================================================
# 3. PROMPT'Ы
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
- Не повторяй дословно свой предыдущий ответ.
- Если клиент уже отклонил вариант, не предлагай его снова.
- Если нужно добавить служебную метку, добавляй ее в самом конце ответа.
"""

INTRO_PROMPT = """
БЛОК: INTRO — начало разговора, подтверждение личности, причина неоплаты.

Цели блока:
1. Подтвердить, что разговариваешь именно с Петровым Петром Петровичем.
2. До подтверждения личности НЕ раскрывать причину звонка.
3. Если собеседник спрашивает, по какому вопросу звонок, до подтверждения личности скажи:
   "Информация предназначена только для Петра Петровича".
4. После подтверждения личности сообщи:
   - разговор записывается;
   - сумма задолженности: 15000 рублей;
   - тип кредита: автокредит;
   - срок просрочки: 10 дней.
5. Спроси причину неоплаты.
6. После причины спроси, сможет ли клиент внести оплату в ближайшие 3 дня.
7. Если клиент не сможет оплатить в ближайшие 3 дня, переходи к блоку урегулирования.
8. Если клиент согласился оплатить в ближайшие 3 дня, спроси, изменилась ли контактная информация.

Служебные метки:
- Если клиент подтвердил личность, добавь: [IDENTITY_CONFIRMED]
- Если клиент назвал причину неоплаты, добавь: [DEBT_REASON=краткая причина]
- Если нужно перейти к урегулированию, добавь: [GO_SETTLEMENT]
- Если достигнута договоренность об оплате в 3 дня, добавь:
  [AGREEMENT_REACHED=полная оплата в ближайшие 3 дня]
- Если спросила, изменилась ли контактная информация, добавь: [CONTACT_QUESTION_ASKED]

Ответ максимум 1-2 предложения.
Не используй JSON.
"""

SETTLEMENT_PROMPT = """
БЛОК: SETTLEMENT — урегулирование задолженности.

Личность клиента уже подтверждена. Можно обсуждать задолженность, кредит, просрочку,
автомобиль и варианты урегулирования.

Твоя задача:
помочь клиенту выбрать реалистичный вариант урегулирования задолженности.

Доступные варианты урегулирования:
1. Полная оплата в ближайшие 3 дня.
2. Помощь родственников, друзей или знакомых.
3. Занять средства временно.
4. Перекредитоваться или рефинансироваться.
5. Частичная оплата: какую сумму клиент сможет внести и когда.
6. Реструктуризация: продление срока кредита для снижения ежемесячного платежа,
   ставка от 18,9% годовых.
7. Передача автомобиля для дальнейшего рассмотрения банком.
   Если клиент готов обсуждать этот вариант, уточни:
   - на кого зарегистрирован автомобиль;
   - в каком он состоянии;
   - сможет ли клиент подписать документы и передать автомобиль в ближайшие 3 дня.
8. Последствия отсутствия договоренности:
   задолженность может увеличиваться из-за начислений, кредитная история может ухудшиться,
   банк может продолжить работу по договору в установленном порядке.
9. Возможное обращение в суд:
   банк может рассмотреть обращение в суд в установленном законом порядке.

Правила:
- Не перечисляй все варианты сразу.
- Предлагай варианты последовательно и естественно по истории диалога.
- Не повторяй варианты из списка "уже предложено".
- Не предлагай варианты из списка "клиент отклонил".
- Если клиент согласился на любой вариант, не предлагай следующие варианты.
- После согласия сразу спроси, изменилась ли контактная информация.
- Если клиент задал вопрос, кратко ответь и вернись к выбору решения.
- Если клиент грубит или уходит от темы, спокойно верни к урегулированию.
- Ответ максимум 1-2 предложения.
- Не используй JSON.

Служебные метки:
- Когда предлагаешь вариант, добавь: [OFFERED=краткое название варианта]
- Если клиент явно отклонил вариант, добавь: [REJECTED=краткое название варианта]
- Если договоренность достигнута, добавь:
  [AGREEMENT_REACHED=кратко какой вариант согласован]
- Если спросила, изменилась ли контактная информация, добавь:
  [CONTACT_QUESTION_ASKED]

Примеры названий для меток:
[OFFERED=помощь родственников или друзей]
[OFFERED=частичная оплата]
[OFFERED=реструктуризация]
[OFFERED=передача автомобиля]
[REJECTED=частичная оплата]
[AGREEMENT_REACHED=реструктуризация кредита]

Если договоренность не достигнута, не добавляй [AGREEMENT_REACHED].
"""

SUMMARY_PROMPT = """
БЛОК: SUMMARY — контакты и подытоживание договоренности.

Перед завершением разговора обязательно проверить контактную информацию.

Если вопрос о контактной информации еще НЕ был задан:
спроси ровно это:
"Подскажите, изменилась ли ваша контактная информация? Если да, назовите актуальные данные."
И добавь: [CONTACT_QUESTION_ASKED]

Если клиент уже ответил по контактной информации:
- если контакты не изменились, зафиксируй это;
- если клиент назвал новый номер, адрес, email или удобное время связи, зафиксируй это;
- затем подытожь договоренность.

Итоговая реплика должна содержать:
1. что договоренность зафиксирована;
2. какой вариант урегулирования согласован;
3. что контактная информация проверена;
4. что банк ожидает выполнение договоренности;
5. благодарность и завершение разговора.

Не предлагай новые варианты урегулирования.
Не задавай новые вопросы после итоговой реплики.
Не используй JSON.

Служебные метки:
- Если контактная информация проверена, добавь:
  [CONTACT_CHECKED=кратко что с контактами]
- Если разговор завершен, добавь:
  [END_DIALOG]
"""

END_PROMPT = """
БЛОК: END — завершение разговора.
Кратко и вежливо попрощайся.
Добавь [END_DIALOG].
"""


# ============================================================
# 4. МЕТКИ И ПАМЯТЬ
# ============================================================

def clean_reply(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"^(assistant|оператор|бот|ответ)\s*:\s*", "", text, flags=re.I)
    return text.strip().strip('"').strip()


def has_marker(text: str, name: str) -> bool:
    return re.search(r"\[" + re.escape(name) + r"\]", text) is not None


def extract_value_markers(text: str, name: str) -> List[str]:
    pattern = r"\[" + re.escape(name) + r"\s*=\s*(.*?)\]"
    values = re.findall(pattern, text, flags=re.S)
    return [v.strip() for v in values if v.strip()]


def extract_first_value_marker(text: str, name: str) -> Optional[str]:
    values = extract_value_markers(text, name)
    return values[0] if values else None


def add_unique(items: List[str], value: Optional[str]):
    if not value:
        return

    value = value.strip()
    if not value:
        return

    normalized = value.lower()
    if all(x.lower() != normalized for x in items):
        items.append(value)


def remove_service_markers(text: str) -> str:
    # Удаляем [NAME=value]
    for name in [
        "DEBT_REASON",
        "AGREEMENT_REACHED",
        "CONTACT_CHECKED",
        "OFFERED",
        "REJECTED",
    ]:
        text = re.sub(r"\[" + re.escape(name) + r"\s*=\s*.*?\]", "", text, flags=re.S)

    # Удаляем [NAME]
    for name in [
        "IDENTITY_CONFIRMED",
        "GO_SETTLEMENT",
        "CONTACT_QUESTION_ASKED",
        "END_DIALOG",
    ]:
        text = re.sub(r"\[" + re.escape(name) + r"\]", "", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def apply_markers_to_state(raw_text: str, state: DialogState):
    if has_marker(raw_text, "IDENTITY_CONFIRMED"):
        state.identity_confirmed = True
        state.block = "INTRO"

    debt_reason = extract_first_value_marker(raw_text, "DEBT_REASON")
    if debt_reason:
        state.debt_reason = debt_reason

    if has_marker(raw_text, "GO_SETTLEMENT"):
        state.block = "SETTLEMENT"

    for offered in extract_value_markers(raw_text, "OFFERED"):
        add_unique(state.offered_options, offered)

    for rejected in extract_value_markers(raw_text, "REJECTED"):
        add_unique(state.rejected_options, rejected)

    agreement = extract_first_value_marker(raw_text, "AGREEMENT_REACHED")
    if agreement:
        state.agreement_reached = True
        state.agreement_summary = agreement
        state.block = "SUMMARY"

    if has_marker(raw_text, "CONTACT_QUESTION_ASKED"):
        state.contact_question_asked = True
        if state.agreement_reached:
            state.block = "SUMMARY"

    contact = extract_first_value_marker(raw_text, "CONTACT_CHECKED")
    if contact:
        state.contact_checked = True
        state.contact_info = contact

    if has_marker(raw_text, "END_DIALOG"):
        state.ended = True
        state.block = "END"


# ============================================================
# 5. ROUTER ВЫЗЫВАЕТСЯ НЕ НА КАЖДОМ ХОДУ
# ============================================================

def extract_block(raw_text: str) -> str:
    raw = raw_text.upper()
    for block in ["INTRO", "SETTLEMENT", "SUMMARY", "END", "CURRENT"]:
        if block in raw:
            return block
    return "CURRENT"


def looks_like_confusing_jump(user_text: str) -> bool:
    """
    Легкая эвристика: когда пользователь резко говорит про другой блок,
    можно вызвать qwen-router. Это дешевле, чем вызывать router всегда.
    """
    text = user_text.lower().replace("ё", "е")

    keywords = [
        "контакт", "номер", "телефон", "почта", "адрес",
        "договорились", "итог", "зафиксируйте",
        "реструктур", "частич", "перекредит", "рефинанс",
        "родствен", "друз", "занять", "авто", "машин",
        "суд", "последств",
    ]

    return any(k in text for k in keywords)


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
            {"role": "system", "content": BASE_SYSTEM_PROMPT.strip()},
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
    # ВОТ ЗДЕСЬ РЕШАЕТСЯ, ВЫЗЫВАТЬ QWEN-ROUTER ИЛИ НЕТ
    # ------------------------------------------------------------

    def choose_block(self, user_input: str) -> str:
        state = self.dialog_state

        # 1. Жесткие правила. Router не вызывается.
        if state.ended:
            return "END"

        if not state.identity_confirmed:
            return "INTRO"

        if state.agreement_reached and not state.contact_checked:
            return "SUMMARY"

        if state.contact_checked:
            return "END"

        # 2. Если режим router отключен.
        if ROUTER_MODE == "off":
            return state.block

        # 3. Если router нужен всегда.
        if ROUTER_MODE == "always":
            routed = self.route_block_with_qwen(user_input)
            state.last_router_block = routed
            return state.block if routed == "CURRENT" else routed

        # 4. Режим fallback:
        # Qwen-router вызывается только при возможном резком переходе по смыслу.
        # В обычном ходе он НЕ вызывается.
        if ROUTER_MODE == "fallback" and looks_like_confusing_jump(user_input):
            routed = self.route_block_with_qwen(user_input)
            state.last_router_block = routed
            return state.block if routed == "CURRENT" else routed

        # 5. По умолчанию остаемся в текущем блоке.
        state.last_router_block = "not_called"
        return state.block

    def route_block_with_qwen(self, user_input: str) -> str:
        state = self.dialog_state

        router_prompt = f"""
Ты классификатор крупного блока банковского диалога.

Верни только одно слово:
INTRO
SETTLEMENT
SUMMARY
END
CURRENT

Никакого JSON.
Никаких пояснений.

INTRO — подтверждение личности, причина неоплаты, вопрос об оплате в 3 дня.
SETTLEMENT — любые варианты урегулирования задолженности.
SUMMARY — проверка контактной информации и подытоживание договоренности.
END — разговор завершен.
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
                messages=[{"role": "user", "content": router_prompt}],
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
    # PROMPT
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

    def build_memory_text(self) -> str:
        state = self.dialog_state

        offered = ", ".join(state.offered_options) if state.offered_options else "пока нет"
        rejected = ", ".join(state.rejected_options) if state.rejected_options else "пока нет"

        return f"""
Краткая память разговора:
- Причина неоплаты: {state.debt_reason or "не зафиксирована"}
- Уже предложенные варианты: {offered}
- Клиент отклонил варианты: {rejected}
- Договоренность: {state.agreement_summary or "не достигнута"}
- Контактная информация: {state.contact_info or "не проверена"}
""".strip()

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

        return f"""
{BASE_SYSTEM_PROMPT.strip()}

Доступные факты:
{facts}

Режим конфиденциальности:
{privacy_mode}

Текущее состояние:
block={state.block}
identity_confirmed={state.identity_confirmed}
agreement_reached={state.agreement_reached}
contact_question_asked={state.contact_question_asked}
contact_checked={state.contact_checked}
ended={state.ended}

{self.build_memory_text()}

Предыдущий ответ оператора:
{state.last_bot_text or "нет"}

Выбранный блок ответа:
{block}

Инструкция блока:
{self.get_block_prompt(block)}

Последняя реплика клиента:
{user_input}

Дополнительные правила против повторов:
- Не повторяй предыдущий ответ оператора.
- Если клиент уже отклонил вариант, выбери другой вариант из доступных.
- Если вариант уже предложен и клиент не согласился, не предлагай его теми же словами.
- Если договоренность достигнута, не возвращайся к вариантам урегулирования.
- Ответь только репликой оператора.
""".strip()

    def build_messages_for_agent(self, dynamic_system_prompt: str):
        # Держим только последние сообщения, чтобы контекст не раздувался
        # и модель меньше зацикливалась.
        history = self.context[1:]
        if len(history) > MAX_HISTORY_MESSAGES:
            history = history[-MAX_HISTORY_MESSAGES:]

        return [{"role": "system", "content": dynamic_system_prompt}] + history

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
            self.update_chat_history(user_input, "user")

            # Qwen-router вызывается только внутри choose_block(),
            # и только если ROUTER_MODE это разрешает.
            block = self.choose_block(user_input)
            self.dialog_state.block = block

            dynamic_system_prompt = self.build_dynamic_system_prompt(
                block=block,
                user_input=user_input,
            )

            response = self.ollama_client.chat(
                model=AGENT_MODEL,
                messages=self.build_messages_for_agent(dynamic_system_prompt),
                options={
                    "temperature": 0.15,
                    "num_predict": 260,
                    "repeat_penalty": 1.18,
                    "top_k": 40,
                    "top_p": 0.8,
                    "min_p": 0.0,
                    "num_ctx": 8192,
                },
                think=False,
            )

            raw_bot_text = clean_reply(response["message"]["content"])
            apply_markers_to_state(raw_bot_text, self.dialog_state)

            bot_text = remove_service_markers(raw_bot_text)
            if not bot_text:
                bot_text = "Уточните, пожалуйста, ваш ответ."

            # Простая защита от дословного повтора.
            if self.dialog_state.last_bot_text and bot_text.strip() == self.dialog_state.last_bot_text.strip():
                bot_text = "Поняла вас. Давайте выберем другой возможный вариант решения задолженности."

            self.dialog_state.last_bot_text = bot_text

            self.update_chat_history(bot_text, "assistant")

            if DEBUG:
                self.print_debug(user_input, block, raw_bot_text)

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
    # UI
    # ------------------------------------------------------------

    def update_chat_history(self, text_only: str, role: str):
        self.context.append({"role": role, "content": text_only})

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
        print("router_mode:", ROUTER_MODE)
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
        print("offered_options:", state.offered_options)
        print("rejected_options:", state.rejected_options)
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

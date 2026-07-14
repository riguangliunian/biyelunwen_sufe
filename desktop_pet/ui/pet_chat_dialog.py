from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QTextCursor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QTextEdit, QFrame, QInputDialog, QMessageBox,
)

from core import chat_client
from ui.pet_widget import DIALOG_STYLE


class _Worker(QThread):
    """Background thread for one API call."""
    done  = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, msg, pet, history):
        super().__init__()
        self.msg     = msg
        self.pet     = pet
        self.history = history

    def run(self):
        try:
            reply = chat_client.chat(self.msg, self.pet, self.history)
            self.done.emit(reply)
        except Exception as e:
            self.error.emit(str(e))


QUICK_MSGS = ["你好呀~", "今天怎么样？", "肚子饿了吗？", "加油！", "任务完成了！", "陪我玩~"]

BUBBLE_CSS_USER = (
    "display:inline-block; background:#1e3a5f; color:#89b4fa;"
    "border-radius:10px; padding:6px 12px; font-size:13px; font-family:system-ui;"
)
BUBBLE_CSS_PET = (
    "display:inline-block; background:#2a1f3a; color:#cdd6f4;"
    "border-left:3px solid #cba6f7; border-radius:0 10px 10px 0;"
    "padding:6px 12px; font-size:13px; font-family:system-ui;"
)
BUBBLE_CSS_SYS = (
    "color:#585b70; font-size:11px; font-family:monospace;"
)


class PetChatDialog(QWidget):
    def __init__(self, pet, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.pet     = pet
        self.history: list[dict] = []
        self._worker = None

        self.setWindowTitle(f"和 {pet.name} 聊聊")
        self.setStyleSheet(DIALOG_STYLE)
        self.resize(400, 540)
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(8)

        # Header
        hdr = QHBoxLayout()
        title = QLabel(f"🐱  {self.pet.name}")
        title.setStyleSheet("font-size:15px; font-weight:bold; color:#cba6f7;")
        hdr.addWidget(title)
        hdr.addStretch()
        key_btn = QPushButton("🔑 设置 Key")
        key_btn.setStyleSheet(
            "QPushButton{background:#313244;color:#6c7086;"
            "border:1px solid #45475a;border-radius:5px;padding:3px 10px;font-size:11px;}"
            "QPushButton:hover{color:#cdd6f4;}"
        )
        key_btn.clicked.connect(self._set_key)
        hdr.addWidget(key_btn)
        root.addLayout(hdr)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#313244;"); root.addWidget(sep)

        # Chat display
        self.display = QTextEdit()
        self.display.setReadOnly(True)
        self.display.setStyleSheet(
            "QTextEdit{background:#11111b; border:none;"
            "color:#cdd6f4; font-size:13px;}"
        )
        root.addWidget(self.display)

        # Quick message buttons
        quick_row = QHBoxLayout(); quick_row.setSpacing(5)
        for q in QUICK_MSGS:
            b = QPushButton(q)
            b.setStyleSheet(
                "QPushButton{background:#1e1e2e;color:#6c7086;"
                "border:1px solid #313244;border-radius:12px;"
                "padding:3px 8px;font-size:11px;}"
                "QPushButton:hover{color:#cdd6f4;border-color:#585b70;}"
            )
            b.clicked.connect(lambda _, t=q: self._send_text(t))
            quick_row.addWidget(b)
        root.addLayout(quick_row)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color:#313244;"); root.addWidget(sep2)

        # Input
        inp_row = QHBoxLayout(); inp_row.setSpacing(8)
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("说点什么…")
        self.input_edit.setStyleSheet(
            "QLineEdit{background:#181825;color:#cdd6f4;"
            "border:1px solid #45475a;border-radius:8px;padding:7px 11px;"
            "font-size:13px;}"
        )
        self.input_edit.returnPressed.connect(self._send)
        send_btn = QPushButton("发送 ↵")
        send_btn.setStyleSheet(
            "QPushButton{background:#313244;color:#cba6f7;"
            "border:1px solid #cba6f7;border-radius:7px;padding:7px 14px;}"
            "QPushButton:hover{background:#2a1f3a;}"
        )
        send_btn.clicked.connect(self._send)
        self.send_btn = send_btn
        inp_row.addWidget(self.input_edit)
        inp_row.addWidget(send_btn)
        root.addLayout(inp_row)

        # Initial greeting
        self._append_pet(self.pet.speech)

    # ── Logic ─────────────────────────────────────────────────────────────

    def _send(self):
        self._send_text(self.input_edit.text().strip())
        self.input_edit.clear()

    def _send_text(self, text: str):
        if not text or self._worker is not None:
            return
        self._append_user(text)
        self._set_busy(True)
        self._append_sys("思考中…")

        self._worker = _Worker(text, self.pet, self.history)
        self._worker.done.connect(self._on_reply)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(lambda: self._set_busy(False))
        self._worker.start()

    def _on_reply(self, text: str):
        self._remove_last_sys()
        self._append_pet(text)
        # Update history
        self.history.append({"role": "user",      "content": self._last_user})
        self.history.append({"role": "assistant",  "content": text})
        if len(self.history) > 12:
            self.history = self.history[-12:]
        self._worker = None

    def _on_error(self, err: str):
        self._remove_last_sys()
        self._append_sys(f"⚠ {err}")
        self._worker = None

    def _set_busy(self, busy: bool):
        self.input_edit.setEnabled(not busy)
        self.send_btn.setEnabled(not busy)

    def _set_key(self):
        current = chat_client.get_api_key()
        key, ok = QInputDialog.getText(
            self, "设置 Anthropic API Key",
            "粘贴你的 API Key（以 sk-ant- 开头）：",
            text=current,
        )
        if ok and key.strip():
            chat_client.save_api_key(key.strip())

    # ── Display helpers ───────────────────────────────────────────────────

    def _append_user(self, text: str):
        self._last_user = text
        escaped = text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        self._insert(
            f'<div style="margin:6px 0; text-align:right;">'
            f'<span style="{BUBBLE_CSS_USER}">🧑 {escaped}</span></div>'
        )

    def _append_pet(self, text: str):
        escaped = text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        self._insert(
            f'<div style="margin:6px 0;">'
            f'<span style="{BUBBLE_CSS_PET}">🐱 {escaped}</span></div>'
        )

    def _append_sys(self, text: str):
        escaped = text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        self._insert(
            f'<div id="sys" style="margin:3px 8px; {BUBBLE_CSS_SYS}">{escaped}</div>'
        )

    def _remove_last_sys(self):
        # Cheap approach: re-render without last sys block (use marker in text)
        html = self.display.toHtml()
        # Remove the last "思考中…" block by truncating at its marker
        marker = '<div id="sys"'
        idx = html.rfind(marker)
        if idx >= 0:
            self.display.setHtml(html[:idx])

    def _insert(self, html: str):
        cur = self.display.textCursor()
        cur.movePosition(QTextCursor.MoveOperation.End)
        cur.insertHtml(html)
        self.display.setTextCursor(cur)
        self.display.ensureCursorVisible()

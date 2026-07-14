import os
import re
import pty
import threading
import select
import subprocess

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QColor, QFont, QTextCursor, QKeyEvent
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QTextEdit, QFrame, QSplitter,
)

from ui.pet_widget import DIALOG_STYLE

ANSI_RE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
SPINNER_CHARS = set("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏◐◓◑◒")
BOX_RE = re.compile(r'[╭╮╰╯─│╠╬╣╔╗╚╝═║┌┐└┘├┤┬┴┼▲▼◆◇]')


def strip_output(text: str) -> str:
    text = ANSI_RE.sub('', text)
    text = BOX_RE.sub('', text)
    lines = []
    for line in text.splitlines():
        line = ''.join(c for c in line if c not in SPINNER_CHARS)
        line = line.rstrip()
        if line:
            lines.append(line)
    return '\n'.join(lines)


# ── PTY reader thread ─────────────────────────────────────────────────────────

class _Emitter(QObject):
    data = pyqtSignal(str)
    finished = pyqtSignal()


class PtyReader(threading.Thread):
    def __init__(self, master_fd: int, emitter: _Emitter):
        super().__init__(daemon=True)
        self.master_fd = master_fd
        self.emitter = emitter
        self._stop = False

    def run(self):
        buf = b''
        while not self._stop:
            try:
                r, _, _ = select.select([self.master_fd], [], [], 0.1)
                if r:
                    chunk = os.read(self.master_fd, 4096)
                    if not chunk:
                        break
                    buf += chunk
                    # flush whole lines (or after 200 ms implicit by timer)
                    try:
                        text = buf.decode('utf-8', errors='replace')
                        buf = b''
                        self.emitter.data.emit(text)
                    except Exception:
                        buf = b''
            except OSError:
                break
        self.emitter.finished.emit()

    def stop(self):
        self._stop = True


# ── Main dialog ───────────────────────────────────────────────────────────────

class ClaudeDialog(QWidget):
    """Embedded Claude Code terminal — chat with claude CLI via PTY."""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Claude Code")
        self.setStyleSheet(DIALOG_STYLE)
        self.resize(680, 680)

        self._master_fd: int | None = None
        self._process: subprocess.Popen | None = None
        self._reader: PtyReader | None = None
        self._emitter = _Emitter()
        self._emitter.data.connect(self._on_output)
        self._emitter.finished.connect(self._on_exit)
        self._current_assistant_block = ""
        self._pending_flush = QTimer(self)
        self._pending_flush.setSingleShot(True)
        self._pending_flush.timeout.connect(self._flush_block)

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # header
        hdr = QHBoxLayout()
        title = QLabel("🤖  Claude Code")
        title.setStyleSheet("font-size:15px; font-weight:bold; color:#89b4fa;")
        hdr.addWidget(title)
        hdr.addStretch()

        self.status_lbl = QLabel("未启动")
        self.status_lbl.setStyleSheet("color:#6c7086; font-size:11px;")
        hdr.addWidget(self.status_lbl)
        root.addLayout(hdr)

        # control buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        def btn(text, color, fn):
            b = QPushButton(text)
            b.setStyleSheet(
                f"QPushButton{{background:#313244;color:{color};"
                f"border:1px solid {color};border-radius:6px;padding:5px 12px;}}"
                f"QPushButton:hover{{background:#2a2a3e;}}"
            )
            b.clicked.connect(fn)
            return b

        self.btn_new    = btn("▶ 新对话",     "#a6e3a1", self._start_new)
        self.btn_resume = btn("↩ 继续上次",   "#89b4fa", self._start_resume)
        self.btn_stop   = btn("■ 停止",       "#f38ba8", self._stop)
        self.btn_stop.setEnabled(False)

        for b in (self.btn_new, self.btn_resume, self.btn_stop):
            btn_row.addWidget(b)
        btn_row.addStretch()

        self.btn_clear = btn("🗑  清空",      "#6c7086", self._clear)
        btn_row.addWidget(self.btn_clear)
        root.addLayout(btn_row)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#313244;"); root.addWidget(sep)

        # chat display
        self.display = QTextEdit()
        self.display.setReadOnly(True)
        self.display.setStyleSheet(
            "QTextEdit{background:#11111b;border:none;"
            "color:#cdd6f4;font-family:Menlo,Monaco,monospace;font-size:12px;}"
        )
        root.addWidget(self.display)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color:#313244;"); root.addWidget(sep2)

        # input row
        inp_row = QHBoxLayout(); inp_row.setSpacing(8)
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("输入消息，按 Enter 发送…")
        self.input_edit.setStyleSheet(
            "QLineEdit{background:#181825;color:#cdd6f4;"
            "border:1px solid #45475a;border-radius:8px;padding:8px 12px;"
            "font-size:13px;}"
        )
        self.input_edit.returnPressed.connect(self._send)
        send_btn = btn("发送 ↵", "#a6e3a1", self._send)
        inp_row.addWidget(self.input_edit)
        inp_row.addWidget(send_btn)
        root.addLayout(inp_row)

        hint = QLabel("Tip：启动后直接在输入框打字即可与 Claude 对话")
        hint.setStyleSheet("color:#585b70;font-size:10px;")
        root.addWidget(hint)

    # ── Process management ────────────────────────────────────────────────

    def _start(self, args: list[str]):
        self._stop()

        claude_bin = self._find_claude()
        if not claude_bin:
            self._append_system("❌ 找不到 claude 命令。请先安装 Claude Code CLI。")
            return

        try:
            master, slave = pty.openpty()
        except Exception as e:
            self._append_system(f"❌ PTY 创建失败：{e}")
            return

        env = {**os.environ, 'TERM': 'xterm-256color', 'COLUMNS': '100', 'LINES': '40'}

        try:
            self._process = subprocess.Popen(
                [claude_bin] + args,
                stdin=slave, stdout=slave, stderr=slave,
                close_fds=True, env=env,
            )
        except Exception as e:
            os.close(slave); os.close(master)
            self._append_system(f"❌ 启动失败：{e}")
            return

        os.close(slave)
        self._master_fd = master

        self._reader = PtyReader(master, self._emitter)
        self._reader.start()

        self.status_lbl.setText("● 运行中")
        self.status_lbl.setStyleSheet("color:#a6e3a1; font-size:11px;")
        self.btn_stop.setEnabled(True)
        self.btn_new.setEnabled(False)
        self.btn_resume.setEnabled(False)

    def _start_new(self):
        self._clear()
        self._append_system("▶ 启动新对话…")
        self._start([])

    def _start_resume(self):
        self._clear()
        self._append_system("↩ 继续上次对话…")
        self._start(["--resume"])

    def _stop(self):
        if self._reader:
            self._reader.stop()
            self._reader = None
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=2)
            except Exception:
                try: self._process.kill()
                except Exception: pass
            self._process = None
        if self._master_fd is not None:
            try: os.close(self._master_fd)
            except OSError: pass
            self._master_fd = None
        self._flush_block()
        self.status_lbl.setText("已停止")
        self.status_lbl.setStyleSheet("color:#f38ba8; font-size:11px;")
        self.btn_stop.setEnabled(False)
        self.btn_new.setEnabled(True)
        self.btn_resume.setEnabled(True)

    @staticmethod
    def _find_claude() -> str | None:
        for path in [
            "claude",
            os.path.expanduser("~/.local/bin/claude"),
            "/usr/local/bin/claude",
            "/opt/homebrew/bin/claude",
        ]:
            try:
                result = subprocess.run(
                    ["which", path] if "/" not in path else ["test", "-x", path],
                    capture_output=True
                )
                if result.returncode == 0:
                    return path
            except Exception:
                pass
        # try 'which' directly
        try:
            r = subprocess.run(["which", "claude"], capture_output=True, text=True)
            if r.returncode == 0:
                return r.stdout.strip()
        except Exception:
            pass
        return None

    # ── I/O ───────────────────────────────────────────────────────────────

    def _send(self):
        text = self.input_edit.text()
        if not text:
            return
        self.input_edit.clear()

        # show user message
        self._flush_block()
        self._append_user(text)

        if self._master_fd is not None:
            try:
                os.write(self._master_fd, (text + "\n").encode())
            except OSError as e:
                self._append_system(f"⚠ 发送失败：{e}")
        else:
            self._append_system("⚠ 未连接，请先启动对话。")

    def _on_output(self, raw: str):
        cleaned = strip_output(raw)
        if not cleaned:
            return
        self._current_assistant_block += cleaned
        # debounce flush: wait 150ms for more data
        self._pending_flush.stop()
        self._pending_flush.start(150)

    def _flush_block(self):
        if self._current_assistant_block.strip():
            self._append_assistant(self._current_assistant_block)
        self._current_assistant_block = ""

    def _on_exit(self):
        self._flush_block()
        self._append_system("── 进程结束 ──")
        self.status_lbl.setText("已结束")
        self.status_lbl.setStyleSheet("color:#6c7086; font-size:11px;")
        self.btn_stop.setEnabled(False)
        self.btn_new.setEnabled(True)
        self.btn_resume.setEnabled(True)

    # ── Display helpers ───────────────────────────────────────────────────

    def _append_user(self, text: str):
        escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html = (
            f'<div style="margin:8px 0; text-align:right;">'
            f'<span style="display:inline-block; background:#1e3a5f; color:#89b4fa;'
            f'border-radius:10px; padding:6px 12px; max-width:80%; '
            f'font-family:system-ui; font-size:13px; text-align:left;">'
            f'🧑 {escaped}</span></div>'
        )
        self._insert_html(html)

    def _append_assistant(self, text: str):
        text = text.strip()
        if not text:
            return
        escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        escaped = escaped.replace("\n", "<br>")
        html = (
            f'<div style="margin:8px 0;">'
            f'<span style="display:inline-block; background:#1a1a2e; color:#cdd6f4;'
            f'border-left:3px solid #89b4fa; border-radius:0 10px 10px 0;'
            f'padding:6px 12px; max-width:95%;'
            f'font-family:Menlo,Monaco,monospace; font-size:12px;">'
            f'🤖 {escaped}</span></div>'
        )
        self._insert_html(html)

    def _append_system(self, text: str):
        escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html = (
            f'<div style="margin:4px 0; color:#585b70; font-size:11px;'
            f'font-family:monospace; padding:2px 8px;">{escaped}</div>'
        )
        self._insert_html(html)

    def _insert_html(self, html: str):
        cur = self.display.textCursor()
        cur.movePosition(QTextCursor.MoveOperation.End)
        cur.insertHtml(html)
        self.display.setTextCursor(cur)
        self.display.ensureCursorVisible()

    def _clear(self):
        self.display.clear()

    def closeEvent(self, event):
        self._stop()
        event.accept()

    def keyPressEvent(self, event: QKeyEvent):
        # Ctrl+C → send interrupt
        if (event.modifiers() & Qt.KeyboardModifier.ControlModifier
                and event.key() == Qt.Key.Key_C):
            if self._master_fd is not None:
                try: os.write(self._master_fd, b'\x03')
                except OSError: pass
        super().keyPressEvent(event)

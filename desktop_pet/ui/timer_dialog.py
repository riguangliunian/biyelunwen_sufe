import math

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush, QFont, QFontMetrics
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QSpinBox, QFrame, QScrollArea,
)

from core.timer_manager import TimerManager, TimerSession
from core.pet_state import PetState
from ui.pet_widget import DIALOG_STYLE


class _Ring(QWidget):
    """Small circular progress ring for one timer."""
    def __init__(self, session: TimerSession, parent=None):
        super().__init__(parent)
        self.session = session
        self.setFixedSize(64, 64)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # bg ring
        p.setPen(QPen(QColor(50, 50, 70), 5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(6, 6, 52, 52)

        # progress arc
        prog = self.session.progress
        color = (QColor(130, 200, 255) if self.session.is_running
                 else QColor(180, 140, 255))
        p.setPen(QPen(color, 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawArc(6, 6, 52, 52, 90 * 16, int(-prog * 360 * 16))

        # time text
        p.setPen(QColor(205, 214, 244))
        p.setFont(QFont("Courier", 10, QFont.Weight.Bold))
        fm = QFontMetrics(p.font())
        t = self.session.display_time
        p.drawText(
            int(32 - fm.horizontalAdvance(t) / 2),
            int(32 + fm.ascent() / 2 - 2),
            t,
        )


class _TimerCard(QFrame):
    """One row for a single active timer."""
    removed = pyqtSignal(int)

    def __init__(self, session: TimerSession, timer_mgr: TimerManager, parent=None):
        super().__init__(parent)
        self.session   = session
        self.timer_mgr = timer_mgr
        self.setStyleSheet(
            "QFrame{background:#181825;border:1px solid #313244;"
            "border-radius:10px;padding:4px;}"
        )
        self._build()

    def _build(self):
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(12)

        self.ring = _Ring(self.session)
        row.addWidget(self.ring)

        info = QVBoxLayout(); info.setSpacing(2)
        self.name_lbl = QLabel(self.session.name)
        self.name_lbl.setStyleSheet("font-size:13px; font-weight:bold; color:#cdd6f4;")
        self.state_lbl = QLabel()
        self.state_lbl.setStyleSheet("font-size:10px; color:#6c7086;")
        info.addWidget(self.name_lbl)
        info.addWidget(self.state_lbl)
        row.addLayout(info)
        row.addStretch()

        self.pause_btn = QPushButton("⏸")
        self.stop_btn  = QPushButton("⏹")
        for b in (self.pause_btn, self.stop_btn):
            b.setFixedSize(32, 32)
            b.setStyleSheet(
                "QPushButton{background:#313244;color:#cdd6f4;"
                "border:1px solid #45475a;border-radius:6px;font-size:14px;}"
                "QPushButton:hover{background:#45475a;}"
            )
        self.pause_btn.clicked.connect(self._toggle)
        self.stop_btn.clicked.connect(self._stop)
        row.addWidget(self.pause_btn)
        row.addWidget(self.stop_btn)

        self._update_state_label()

    def refresh(self):
        self.ring.update()
        self._update_state_label()

    def _update_state_label(self):
        s = self.session
        if s.finished:
            self.state_lbl.setText("✅ 完成！")
            self.pause_btn.setEnabled(False)
        elif s.is_paused:
            self.state_lbl.setText("⏸ 已暂停")
            self.pause_btn.setText("▶")
        elif s.is_running:
            pct = int(s.progress * 100)
            self.state_lbl.setText(f"⏱ 进行中 {pct}%")
            self.pause_btn.setText("⏸")
        else:
            self.state_lbl.setText("停止")

    def _toggle(self):
        self.timer_mgr.toggle_pause(self.session.id)
        self._update_state_label()

    def _stop(self):
        self.timer_mgr.remove(self.session.id)
        self.removed.emit(self.session.id)


class TimerDialog(QWidget):
    def __init__(self, timer: TimerManager, pet: PetState, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.timer = timer
        self.pet   = pet
        self.setWindowTitle("⏱  专注计时")
        self.setStyleSheet(DIALOG_STYLE)
        self.setFixedWidth(360)
        self._cards: dict[int, _TimerCard] = {}
        self._build_ui()

        self._refresh_tmr = QTimer(self)
        self._refresh_tmr.timeout.connect(self._refresh)
        self._refresh_tmr.start(500)

    # ── Build ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        title = QLabel("⏱  专注计时")
        title.setStyleSheet("font-size:15px; font-weight:bold; color:#89b4fa;")
        root.addWidget(title)

        # Presets
        pre_row = QHBoxLayout(); pre_row.setSpacing(5)
        for label, secs in self.timer.PRESETS:
            b = QPushButton(label)
            b.setFixedHeight(32)
            b.clicked.connect(lambda _, n=label, s=secs: self._add_timer(n, s))
            pre_row.addWidget(b)
        root.addLayout(pre_row)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#313244;"); root.addWidget(sep)

        # Active timers (scrollable)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        self.cards_widget = QWidget()
        self.cards_widget.setStyleSheet("background:transparent;")
        self.cards_layout = QVBoxLayout(self.cards_widget)
        self.cards_layout.setSpacing(6)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.addStretch()
        scroll.setWidget(self.cards_widget)
        scroll.setMinimumHeight(160)
        root.addWidget(scroll)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color:#313244;"); root.addWidget(sep2)

        # Custom new timer
        custom_title = QLabel("自定义计时")
        custom_title.setStyleSheet("font-size:12px; color:#6c7086;")
        root.addWidget(custom_title)

        c1 = QHBoxLayout()
        self.name_in = QLineEdit(); self.name_in.setPlaceholderText("任务名称…")
        c1.addWidget(QLabel("名称：")); c1.addWidget(self.name_in)
        root.addLayout(c1)

        c2 = QHBoxLayout()
        self.min_spin = QSpinBox()
        self.min_spin.setRange(1, 180); self.min_spin.setValue(25)
        self.min_spin.setStyleSheet(DIALOG_STYLE)
        add_btn = QPushButton("▶ 开始")
        add_btn.setStyleSheet(
            "QPushButton{background:#313244;color:#a6e3a1;"
            "border:1px solid #a6e3a1;border-radius:6px;padding:6px 14px;}"
            "QPushButton:hover{background:#2c3a2c;}"
        )
        add_btn.clicked.connect(self._add_custom)
        c2.addWidget(QLabel("时长(分)："))
        c2.addWidget(self.min_spin)
        c2.addWidget(add_btn)
        root.addLayout(c2)

    # ── Cards ─────────────────────────────────────────────────────────────

    def _add_card(self, session: TimerSession):
        card = _TimerCard(session, self.timer, self.cards_widget)
        card.removed.connect(self._remove_card)
        # Insert before stretch
        self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)
        self._cards[session.id] = card

    def _remove_card(self, sid: int):
        card = self._cards.pop(sid, None)
        if card:
            self.cards_layout.removeWidget(card)
            card.deleteLater()

    def _add_timer(self, name: str, seconds: int):
        s = self.timer.add(name, seconds)
        if self.pet.mood.value not in ('sleeping',):
            from core.pet_state import PetMood
            self.pet.start_working()
        self._add_card(s)

    def _add_custom(self):
        name = self.name_in.text().strip() or "专注"
        mins = self.min_spin.value()
        self._add_timer(name, mins * 60)
        self.name_in.clear()

    def _refresh(self):
        for card in self._cards.values():
            card.refresh()

    def _on_timer_finished(self, session: TimerSession):
        card = self._cards.get(session.id)
        if card:
            card.refresh()

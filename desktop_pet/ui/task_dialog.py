from datetime import date, datetime, timedelta

from PyQt6.QtCore import Qt, QDate, QTime
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QComboBox, QListWidget, QListWidgetItem,
    QDateEdit, QTimeEdit, QFrame, QCheckBox, QMenu,
)

from core.task_manager import Task, TaskManager
from core.pet_state import PetState
from ui.pet_widget import DIALOG_STYLE

CAT_COLORS = {
    "工作": "#89b4fa",
    "学习": "#a6e3a1",
    "生活": "#fab387",
    "其他": "#cba6f7",
}

OVERDUE_COLOR  = "#f38ba8"
TODAY_COLOR    = "#fab387"
FUTURE_COLOR   = "#a6e3a1"

BTN_QUICK = (
    "QPushButton{background:#313244;color:#cba6f7;"
    "border:1px solid #45475a;border-radius:5px;padding:3px 9px;font-size:11px;}"
    "QPushButton:hover{background:#3a2f4a;border-color:#cba6f7;}"
    "QPushButton:checked{background:#45365a;border-color:#cba6f7;color:#fff;}"
)


class TaskDialog(QWidget):
    def __init__(self, tasks: TaskManager, pet: PetState, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.tasks = tasks
        self.pet   = pet
        self.setWindowTitle("任务规划")
        self.setStyleSheet(DIALOG_STYLE)
        self.setMinimumSize(400, 560)
        self._build_ui()
        self.refresh()

    # ── Build ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # Header
        hdr = QHBoxLayout()
        title = QLabel("📋  今日任务")
        title.setStyleSheet("font-size:16px; font-weight:bold; color:#89b4fa;")
        hdr.addWidget(title)
        hdr.addStretch()
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color:#a6e3a1; font-size:12px;")
        hdr.addWidget(self.progress_label)
        layout.addLayout(hdr)

        date_label = QLabel(date.today().strftime("%Y年%m月%d日 · 今天"))
        date_label.setStyleSheet("color:#6c7086; font-size:11px;")
        layout.addWidget(date_label)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#313244;"); layout.addWidget(sep)

        # Task list — signal connected ONCE here
        self.task_list = QListWidget()
        self.task_list.setSpacing(2)
        self.task_list.setStyleSheet(DIALOG_STYLE + "QListWidget::item{padding:8px 6px;}")
        self.task_list.itemDoubleClicked.connect(self._toggle_task)
        layout.addWidget(self.task_list)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color:#313244;"); layout.addWidget(sep2)

        # ── Add-task area ──────────────────────────────────────────────

        add_title = QLabel("➕  新建任务")
        add_title.setStyleSheet("font-size:13px; font-weight:bold; color:#cdd6f4;")
        layout.addWidget(add_title)

        # Name + category row
        row1 = QHBoxLayout(); row1.setSpacing(8)
        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("任务名称…")
        self.title_input.returnPressed.connect(self._add_task)
        self.cat_combo = QComboBox()
        self.cat_combo.setStyleSheet(DIALOG_STYLE)
        for c in self.tasks.CATEGORIES:
            self.cat_combo.addItem(c)
        self.cat_combo.setFixedWidth(80)
        row1.addWidget(self.title_input)
        row1.addWidget(self.cat_combo)
        layout.addLayout(row1)

        # Deadline row
        dl_row = QHBoxLayout(); dl_row.setSpacing(6)
        self.dl_check = QCheckBox("截止时间")
        self.dl_check.setStyleSheet("color:#6c7086; font-size:12px;")
        self.dl_check.toggled.connect(self._on_deadline_toggle)
        dl_row.addWidget(self.dl_check)
        dl_row.addStretch()
        layout.addLayout(dl_row)

        # Quick-select buttons
        quick_row = QHBoxLayout(); quick_row.setSpacing(5)
        self._quick_btns = []
        for label, fn in [
            ("今天",   self._quick_today),
            ("明天",   self._quick_tomorrow),
            ("本周五", self._quick_friday),
            ("一周后", self._quick_week),
        ]:
            b = QPushButton(label)
            b.setStyleSheet(BTN_QUICK)
            b.setEnabled(False)
            b.clicked.connect(fn)
            self._quick_btns.append(b)
            quick_row.addWidget(b)
        quick_row.addStretch()
        layout.addLayout(quick_row)

        # Date + time pickers
        picker_row = QHBoxLayout(); picker_row.setSpacing(8)
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDate(QDate.currentDate())
        self.date_edit.setStyleSheet(DIALOG_STYLE)
        self.date_edit.setEnabled(False)
        self.time_edit = QTimeEdit()
        self.time_edit.setDisplayFormat("HH:mm")
        self.time_edit.setTime(QTime(17, 0))
        self.time_edit.setStyleSheet(DIALOG_STYLE)
        self.time_edit.setEnabled(False)
        picker_row.addWidget(QLabel("日期:"))
        picker_row.addWidget(self.date_edit)
        picker_row.addWidget(QLabel("时间:"))
        picker_row.addWidget(self.time_edit)
        picker_row.addStretch()
        layout.addLayout(picker_row)

        # Add button
        add_btn = QPushButton("➕ 添加任务")
        add_btn.setStyleSheet(
            "QPushButton{background:#313244;color:#a6e3a1;"
            "border:1px solid #a6e3a1;border-radius:6px;padding:7px 18px;}"
            "QPushButton:hover{background:#2c3a2c;}"
        )
        add_btn.clicked.connect(self._add_task)
        layout.addWidget(add_btn, alignment=Qt.AlignmentFlag.AlignRight)

    # ── Deadline helpers ──────────────────────────────────────────────────

    def _on_deadline_toggle(self, checked: bool):
        for b in self._quick_btns:
            b.setEnabled(checked)
        self.date_edit.setEnabled(checked)
        self.time_edit.setEnabled(checked)

    def _set_quick(self, target: date, hour: int, minute: int = 0):
        self.dl_check.setChecked(True)
        self.date_edit.setDate(QDate(target.year, target.month, target.day))
        self.time_edit.setTime(QTime(hour, minute))

    def _quick_today(self):
        self._set_quick(date.today(), 17)

    def _quick_tomorrow(self):
        self._set_quick(date.today() + timedelta(days=1), 9)

    def _quick_friday(self):
        today = date.today()
        days_ahead = (4 - today.weekday()) % 7 or 7   # 4 = Friday
        self._set_quick(today + timedelta(days=days_ahead), 18)

    def _quick_week(self):
        self._set_quick(date.today() + timedelta(weeks=1), 9)

    # ── Task list ─────────────────────────────────────────────────────────

    def refresh(self):
        self.task_list.clear()
        today_tasks = self.tasks.get_today_tasks()
        done  = sum(1 for t in today_tasks if t.completed)
        total = len(today_tasks)
        self.progress_label.setText(f"{done}/{total} 完成")

        if not today_tasks:
            item = QListWidgetItem("今天还没有任务，快去添加吧！")
            item.setForeground(QColor("#6c7086"))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.task_list.addItem(item)
            return

        today = date.today()
        for task in today_tasks:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, task.id)

            check   = "✅" if task.completed else "⬜"
            cat_col = CAT_COLORS.get(task.category, "#cdd6f4")
            dl_text = ""
            dl_col  = cat_col

            dt = task.deadline_dt
            if dt:
                dl_text = f"  📅 {task.deadline_label}"
                delta   = (dt.date() - today).days
                if dt < datetime.now() and not task.completed:
                    dl_col = OVERDUE_COLOR
                elif delta == 0:
                    dl_col = TODAY_COLOR
                else:
                    dl_col = FUTURE_COLOR

            display = f"{check} [{task.category}] {task.title}{dl_text}"
            item.setText(display)

            if task.completed:
                item.setForeground(QColor("#6c7086"))
                f = QFont(); f.setStrikeOut(True); item.setFont(f)
            else:
                item.setForeground(QColor(dl_col if dt else cat_col))

            self.task_list.addItem(item)

    def _toggle_task(self, item: QListWidgetItem):
        task_id = item.data(Qt.ItemDataRole.UserRole)
        if not task_id:
            return
        task = next((t for t in self.tasks.tasks if t.id == task_id), None)
        if task is None:
            return
        if task.completed:
            self.tasks.uncomplete_task(task_id)
        else:
            self.tasks.complete_task(task_id)
            self.pet.complete_task()
        self.refresh()

    def _add_task(self):
        title = self.title_input.text().strip()
        if not title:
            return
        cat = self.cat_combo.currentText()
        reminder = None
        if self.dl_check.isChecked():
            qd = self.date_edit.date()
            qt = self.time_edit.time()
            d  = date(qd.year(), qd.month(), qd.day())
            reminder = f"{d.isoformat()} {qt.hour():02d}:{qt.minute():02d}"
        self.tasks.add_task(title, cat, reminder)
        self.title_input.clear()
        self.dl_check.setChecked(False)
        self.refresh()

    # ── Context menu ──────────────────────────────────────────────────────

    def contextMenuEvent(self, event):
        pos  = event.globalPos()
        item = self.task_list.itemAt(self.task_list.mapFromGlobal(pos))
        if not item:
            return
        task_id = item.data(Qt.ItemDataRole.UserRole)
        if not task_id:
            return
        task = next((t for t in self.tasks.tasks if t.id == task_id), None)
        if not task:
            return
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu{background:#1e1e2e;color:#cdd6f4;border:1px solid #313244;"
            "border-radius:6px;padding:4px;}"
            "QMenu::item{padding:5px 16px;border-radius:4px;}"
            "QMenu::item:selected{background:#313244;}"
        )
        if task.completed:
            menu.addAction("↩ 标记未完成", lambda: self._set_done(task_id, False))
        else:
            menu.addAction("✅ 标记完成",   lambda: self._set_done(task_id, True))
        menu.addAction("🗑  删除",          lambda: self._delete(task_id))
        menu.exec(pos)

    def _set_done(self, task_id: str, done: bool):
        if done:
            self.tasks.complete_task(task_id)
            self.pet.complete_task()
        else:
            self.tasks.uncomplete_task(task_id)
        self.refresh()

    def _delete(self, task_id: str):
        self.tasks.remove_task(task_id)
        self.refresh()

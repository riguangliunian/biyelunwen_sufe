from PyQt6.QtCore import Qt, QTimer, QRect, QPoint
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush, QFont, QFontMetrics
from PyQt6.QtWidgets import QWidget, QApplication
from core.macos_utils import boost_window_level

CAT_COLORS = {
    "工作": QColor(137, 180, 250),
    "学习": QColor(166, 227, 161),
    "生活": QColor(250, 179, 135),
    "其他": QColor(203, 166, 247),
}


class TaskMiniPanel(QWidget):
    """Floating task list that sits next to the pet widget."""

    def __init__(self, tasks, parent_pet: QWidget):
        super().__init__(None)
        self.tasks = tasks
        self.parent_pet = parent_pet
        self._items: list[dict] = []

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedWidth(230)

        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self.refresh)
        self._refresh_timer.start(30_000)

    # ── Public ────────────────────────────────────────────────────────────

    def refresh(self):
        today = self.tasks.get_today_tasks()
        self._items = [
            {
                "title": t.title,
                "cat": t.category,
                "done": t.completed,
                "remind": t.reminder_time or "",
            }
            for t in today
        ]
        h = self._calc_height()
        self.setFixedHeight(h)
        self.reposition()
        self.update()

    def reposition(self):
        pet = self.parent_pet
        pw, ph = pet.width(), pet.height()
        gx, gy = pet.x(), pet.y()
        screen = QApplication.primaryScreen().geometry()
        # Try placing to the left
        x = gx - self.width() - 8
        if x < 0:
            x = gx + pw + 8
        # Clamp vertically
        y = max(0, min(gy, screen.height() - self.height()))
        self.move(x, y)

    # ── Drawing ────────────────────────────────────────────────────────────

    def _calc_height(self):
        return max(60, 44 + len(self._items) * 28 + 10)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        # clear
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        painter.fillRect(self.rect(), Qt.GlobalColor.transparent)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        w, h = self.width(), self.height()

        # panel bg
        painter.setBrush(QBrush(QColor(22, 22, 36, 215)))
        painter.setPen(QPen(QColor(80, 80, 120, 160), 1.5))
        painter.drawRoundedRect(1, 1, w - 2, h - 2, 12, 12)

        # title
        painter.setFont(QFont("", 11, QFont.Weight.Bold))
        painter.setPen(QColor(137, 180, 250, 230))
        painter.drawText(QRect(12, 8, w - 50, 22),
                         Qt.AlignmentFlag.AlignVCenter, "📋 今日任务")

        # × close button (top-right)
        painter.setFont(QFont("", 11))
        painter.setPen(QColor(120, 120, 150, 180))
        painter.drawText(QRect(w - 22, 5, 18, 18),
                         Qt.AlignmentFlag.AlignCenter, "×")

        # progress chip
        total = len(self._items)
        done = sum(1 for it in self._items if it["done"])
        chip_text = f"{done}/{total}"
        fm = QFontMetrics(QFont("", 10))
        cw = fm.horizontalAdvance(chip_text) + 14
        cx = w - cw - 30
        painter.setBrush(QBrush(QColor(166, 227, 161, 60)))
        painter.setPen(QPen(QColor(166, 227, 161, 150), 1))
        painter.drawRoundedRect(cx, 9, cw, 18, 9, 9)
        painter.setFont(QFont("", 10))
        painter.setPen(QColor(166, 227, 161, 220))
        painter.drawText(QRect(cx, 9, cw, 18), Qt.AlignmentFlag.AlignCenter, chip_text)

        # separator
        painter.setPen(QPen(QColor(80, 80, 120, 100), 1))
        painter.drawLine(10, 34, w - 10, 34)

        # items
        if not self._items:
            painter.setFont(QFont("", 10))
            painter.setPen(QColor(100, 100, 130, 200))
            painter.drawText(QRect(0, 42, w, 24),
                             Qt.AlignmentFlag.AlignCenter, "还没有任务~")
        else:
            for i, item in enumerate(self._items):
                iy = 42 + i * 28
                self._draw_item(painter, item, iy, w)

    def _draw_item(self, painter: QPainter, item: dict, y: int, pw: int):
        done = item["done"]
        cat_color = CAT_COLORS.get(item["cat"], QColor(200, 200, 220))

        # category dot
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(cat_color))
        painter.drawEllipse(12, y + 8, 7, 7)

        # checkbox
        cb_color = QColor(166, 227, 161, 200) if done else QColor(100, 100, 130, 180)
        painter.setPen(QPen(cb_color, 1.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(24, y + 6, 12, 12, 3, 3)
        if done:
            painter.setPen(QPen(QColor(166, 227, 161), 1.5))
            painter.drawLine(26, y + 12, 29, y + 15)
            painter.drawLine(29, y + 15, 34, y + 8)

        # title
        painter.setFont(QFont("", 10))
        col = QColor(100, 100, 120) if done else QColor(205, 214, 244)
        painter.setPen(col)
        title = item["title"]
        fm = QFontMetrics(painter.font())
        max_w = pw - 60 - (40 if item["remind"] else 0)
        if fm.horizontalAdvance(title) > max_w:
            while fm.horizontalAdvance(title + "…") > max_w and title:
                title = title[:-1]
            title += "…"

        flags = Qt.AlignmentFlag.AlignVCenter
        if done:
            # draw strikethrough manually
            tx = 42
            ty = y + 4
            tw = fm.horizontalAdvance(title)
            th = 20
            painter.drawText(QRect(tx, ty, tw + 4, th), flags, title)
            painter.setPen(QPen(QColor(100, 100, 120, 180), 1))
            painter.drawLine(tx, ty + th // 2, tx + tw, ty + th // 2)
        else:
            painter.drawText(QRect(42, y + 4, max_w, 20), flags, title)

        # reminder badge
        if item["remind"]:
            painter.setFont(QFont("", 8))
            painter.setPen(QColor(166, 227, 161, 160))
            painter.drawText(QRect(pw - 46, y + 6, 40, 16),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                             "⏰" + item["remind"])

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(200, lambda: boost_window_level(self))

    def mousePressEvent(self, event):
        # Close only when the × button area is clicked (top-right 22×22 px)
        if event.pos().x() >= self.width() - 24 and event.pos().y() <= 24:
            self.hide()
        event.accept()

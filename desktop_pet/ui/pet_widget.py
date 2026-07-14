import math
import random

from PyQt6.QtCore import Qt, QTimer, QPoint, QRect
from PyQt6.QtGui import (
    QColor, QPainter, QPen, QBrush,
    QFont, QFontMetrics, QPixmap, QIcon,
)
from PyQt6.QtWidgets import (
    QWidget, QApplication, QMenu, QSystemTrayIcon, QMessageBox,
)

from core.pet_state import PetMood, PetState
from core.task_manager import TaskManager
from core.timer_manager import TimerManager
from ui import pixel_cat


# ── Styles ────────────────────────────────────────────────────────────────────

DIALOG_STYLE = """
QWidget { background:#1e1e2e; color:#cdd6f4; font-size:13px; }
QLabel  { color:#cdd6f4; }
QPushButton {
    background:#313244; color:#cdd6f4;
    border:1px solid #45475a; border-radius:6px; padding:6px 14px;
}
QPushButton:hover  { background:#45475a; }
QPushButton:pressed{ background:#585b70; }
QLineEdit, QComboBox, QTimeEdit, QSpinBox {
    background:#313244; color:#cdd6f4;
    border:1px solid #45475a; border-radius:5px; padding:4px 8px;
}
QListWidget {
    background:#181825; border:1px solid #313244; border-radius:6px;
}
QListWidget::item { padding:6px; border-radius:4px; }
QListWidget::item:selected { background:#313244; }
QListWidget::item:hover    { background:#2a2a3e; }
QScrollBar:vertical { background:#1e1e2e; width:8px; border-radius:4px; }
QScrollBar::handle:vertical {
    background:#45475a; border-radius:4px; min-height:20px;
}
"""

MENU_STYLE = """
QMenu {
    background:rgba(28,28,44,235); border:1px solid rgba(100,100,140,160);
    border-radius:10px; padding:5px; color:#cdd6f4; font-size:13px;
}
QMenu::item { padding:6px 22px 6px 14px; border-radius:5px; }
QMenu::item:selected { background:rgba(255,255,255,35); }
QMenu::separator { height:1px; background:rgba(255,255,255,28); margin:3px 8px; }
QMenu::item:disabled { color:rgba(205,214,244,90); }
"""

# ── Idle speech & behaviour ───────────────────────────────────────────────────

IDLE_SPEECHES = [
    "摸摸我~", "今天也要加油！", "有什么任务要做吗？",
    "你盯着我干嘛~", "我在思考…", "喵~",
    "加油！你是最棒的~", "记得定时休息哦~", "有什么想和我说的吗？",
    "今天天气真好呀~", "我会一直陪着你！", "要不要来个番茄钟？",
]

IDLE_ACTIONS = (
    ['none'] * 4 +
    ['look_left', 'look_right', 'scratch', 'yawn',
     'walk_left', 'walk_right', 'jump']
)

ACTION_DURATION = {
    'none': (20, 35), 'look_left': (10, 18), 'look_right': (10, 18),
    'scratch': (22, 38), 'yawn': (28, 35), 'jump': (22, 30),
    'walk_left': (28, 55), 'walk_right': (28, 55),
}


class PetWidget(QWidget):
    def __init__(self, pet: PetState, tasks: TaskManager, timer: TimerManager,
                 parent=None):
        super().__init__(parent)
        self.pet = pet
        self.tasks = tasks
        self.timer = timer

        # animation state
        self.anim_frame = 0
        self.speech_text: str | None = None
        self.speech_visible = False
        self.eating_frames = 0
        self.happy_frames = 0
        self._drag_pos: QPoint | None = None

        # idle behaviour
        self.idle_action = 'none'
        self.idle_frames_left = 0
        self._orig_pos: QPoint | None = None
        self._jump_max = 20

        # child dialogs (lazy)
        self.task_dialog = None
        self.timer_dialog = None
        self.claude_dialog = None
        self.task_panel = None

        # mini-game state
        self.minigame: str | None = None   # 'ball' | 'teaser' | None
        self.minigame_frames = 0
        self.ball_x = 80.0;  self.ball_y = 90.0
        self.ball_vx = 4.2;  self.ball_vy = -3.1
        self.teaser_t = 0.0

        self._setup_window()
        self._setup_timers()
        self._setup_tray()

        self.show_speech(self.pet.speech, 4000)
        # Auto-show task panel after window is fully ready
        QTimer.singleShot(800, self._init_task_panel)

    # ── Setup ─────────────────────────────────────────────────────────────

    def _setup_window(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowDoesNotAcceptFocus   # never steal keyboard focus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(160, 200)
        screen = QApplication.primaryScreen().geometry()
        self.move(screen.width() - 175, screen.height() - 225)

    def showEvent(self, event):
        super().showEvent(event)
        from core.macos_utils import boost_window_level
        QTimer.singleShot(200, lambda: boost_window_level(self))

    def _setup_timers(self):
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._anim_tick)
        self.anim_timer.start(200)

        self.second_timer = QTimer(self)
        self.second_timer.timeout.connect(self._second_tick)
        self.second_timer.start(1000)

        self.minute_timer = QTimer(self)
        self.minute_timer.timeout.connect(self._minute_tick)
        self.minute_timer.start(60_000)

        self.reminder_timer = QTimer(self)
        self.reminder_timer.timeout.connect(self._check_reminders)
        self.reminder_timer.start(30_000)

        self.behavior_timer = QTimer(self)
        self.behavior_timer.timeout.connect(self._pick_behavior)
        self.behavior_timer.start(10_000)

        self.speech_hide = QTimer(self)
        self.speech_hide.setSingleShot(True)
        self.speech_hide.timeout.connect(self._hide_speech)

    def _setup_tray(self):
        px = QPixmap(32, 32)
        px.fill(Qt.GlobalColor.transparent)
        p = QPainter(px)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(216, 140, 58)))
        p.drawEllipse(4, 4, 24, 24)
        p.setBrush(QBrush(QColor(36, 22, 8)))
        p.drawEllipse(10, 14, 5, 5)
        p.drawEllipse(17, 14, 5, 5)
        p.end()

        self.tray = QSystemTrayIcon(QIcon(px), self)
        m = QMenu(); m.setStyleSheet(MENU_STYLE)
        m.addAction("📍 找到我 (回到角落)", self._bring_to_corner)
        m.addAction("显示/隐藏", self.toggle_visibility)
        m.addSeparator()
        m.addAction("退出", QApplication.instance().quit)
        self.tray.setContextMenu(m)
        self.tray.activated.connect(
            lambda r: self._bring_to_corner()
            if r == QSystemTrayIcon.ActivationReason.Trigger else None
        )
        self.tray.show()

    def _bring_to_corner(self):
        screen = QApplication.primaryScreen().geometry()
        self.move(screen.width() - self.width() - 20,
                  screen.height() - self.height() - 20)
        self.show()
        self.raise_()



    # ── Ticks ─────────────────────────────────────────────────────────────

    def _anim_tick(self):
        self.anim_frame = (self.anim_frame + 1) % 240

        # eating → happy → idle transition
        if self.eating_frames > 0:
            self.eating_frames -= 1
            if self.eating_frames == 0:
                self.pet.mood = PetMood.HAPPY
                self.happy_frames = 18
                self.show_speech("好吃！心满意足~", 3000)
        if self.happy_frames > 0:
            self.happy_frames -= 1
            if self.happy_frames == 0 and self.pet.mood == PetMood.HAPPY:
                self.pet.mood = PetMood.IDLE

        # behaviour countdown
        if self.idle_frames_left > 0:
            self.idle_frames_left -= 1
            if self.idle_frames_left == 0:
                self._end_behavior()

        # walk / jump movement
        self._apply_movement()

        # mini-game tick
        if self.minigame == 'ball':
            self._tick_ball()
        elif self.minigame == 'teaser':
            self._tick_teaser()

        self.update()

    def _second_tick(self):
        for s in self.timer.tick_all():
            self._on_timer_done(s)
        if self.timer.active_sessions:
            self.update()

    def _minute_tick(self):
        old_lv = self.pet.level
        self.pet.tick_minute()
        if self.pet.level > old_lv:
            days = self.pet.companionship_days + 1
            self.show_speech(f"新的一天！陪伴第 {days} 天 🎉", 8000)
            self.tray.showMessage(
                f"陪伴第 {days} 天",
                f"{self.pet.name} 升到 Lv.{self.pet.level}，谢谢你的陪伴！",
                QSystemTrayIcon.MessageIcon.Information, 5000)
        elif self.pet.hunger < 20 and not self.speech_visible:
            self.show_speech(self.pet.speech)
        elif self.pet.energy < 20 and not self.speech_visible:
            self.show_speech(self.pet.speech)
        self._check_stats_alert()
        if self.task_panel and self.task_panel.isVisible():
            self.task_panel.refresh()

    def _check_reminders(self):
        for t in self.tasks.get_due_reminders():
            self.tray.showMessage("⏰ 任务提醒",
                f"该做「{t.title}」了！",
                QSystemTrayIcon.MessageIcon.Information, 5000)
            self.show_speech(f"记得：{t.title}", 5000)

    def _on_timer_done(self, session):
        if not self.timer.any_running:
            self.pet.finish_working()
        self.happy_frames = 22
        self.tray.showMessage("计时结束 🎉", f"「{session.name}」完成！辛苦了~",
                              QSystemTrayIcon.MessageIcon.Information, 5000)
        self.show_speech("完成了！好棒~", 5000)

    # ── Idle behaviour ────────────────────────────────────────────────────

    def _pick_behavior(self):
        if self.idle_frames_left > 0:
            return
        if self.pet.mood not in (PetMood.IDLE, PetMood.HAPPY):
            return
        action = random.choice(IDLE_ACTIONS)
        lo, hi = ACTION_DURATION.get(action, (15, 30))
        self.idle_action = action
        self.idle_frames_left = random.randint(lo, hi)
        self._orig_pos = self.pos()

        if random.random() < 0.28:
            self.show_speech(random.choice(IDLE_SPEECHES))
        if action == 'yawn' and not self.speech_visible:
            self.show_speech(random.choice(["哈欠…", "好困~", "伸个懒腰"]))

    def _end_behavior(self):
        if self.idle_action in ('walk_left', 'walk_right', 'jump'):
            if self._orig_pos:
                self.move(self._orig_pos)
        self.idle_action = 'none'
        self._orig_pos = None
        if self.task_panel and self.task_panel.isVisible():
            self.task_panel.reposition()

    def _apply_movement(self):
        if self.idle_action == 'walk_left':
            new_x = max(0, self.x() - 2)
            self.move(new_x, self.y())
            if self.task_panel and self.task_panel.isVisible():
                self.task_panel.reposition()
        elif self.idle_action == 'walk_right':
            sw = QApplication.primaryScreen().geometry().width()
            new_x = min(sw - self.width(), self.x() + 2)
            self.move(new_x, self.y())
            if self.task_panel and self.task_panel.isVisible():
                self.task_panel.reposition()
        elif self.idle_action == 'jump' and self._orig_pos:
            progress = self.idle_frames_left
            off = -int(abs(math.sin(progress * 0.28)) * self._jump_max)
            self.move(self._orig_pos.x(), self._orig_pos.y() + off)

    # ── Speech ────────────────────────────────────────────────────────────

    def show_speech(self, text: str, duration: int = 4000):
        self.speech_text = text
        self.speech_visible = True
        self.speech_hide.stop()
        self.speech_hide.start(duration)
        self.update()

    def _hide_speech(self):
        self.speech_visible = False
        self.update()

    def toggle_visibility(self):
        self.setVisible(not self.isVisible())

    # ── Paint ─────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        painter.fillRect(self.rect(), Qt.GlobalColor.transparent)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        # cat
        state = self.pet.mood.value
        idle_action = self.idle_action if state == 'idle' else 'none'
        pixel_cat.draw(
            painter,
            cx=80, cy=163,
            state=state,
            frame=self.anim_frame,
            idle_action=idle_action,
        )

        # timer badge
        if self.timer.active_sessions:
            self._draw_timer_badge(painter)

        # mini-game overlay
        if self.minigame:
            self._draw_minigame(painter)

        # speech bubble drawn last so it's on top
        if self.speech_visible and self.speech_text:
            self._draw_speech(painter, self.speech_text)

        # status bar
        self._draw_status(painter)

    def _draw_speech(self, painter: QPainter, text: str):
        font = QFont("", 11)
        painter.setFont(font)
        fm = QFontMetrics(font)
        max_w, line_h = 155, fm.height() + 2

        lines, cur = [], ""
        for ch in text:
            if fm.horizontalAdvance(cur + ch) > max_w - 22:
                if cur: lines.append(cur)
                cur = ch
            else:
                cur += ch
        if cur: lines.append(cur)

        bw = min(max_w, max((fm.horizontalAdvance(l) for l in lines), default=50) + 22)
        bh = len(lines) * line_h + 16
        bx = max(2, min(self.width() - bw - 5, 8))
        by = max(4, 45 - bh)

        painter.setPen(QPen(QColor(180, 180, 200, 180), 1.5))
        painter.setBrush(QBrush(QColor(255, 255, 255, 230)))
        painter.drawRoundedRect(QRect(int(bx), int(by), int(bw), int(bh)), 10, 10)

        from PyQt6.QtGui import QPolygon
        from PyQt6.QtCore import QPoint
        tail = QPolygon([
            QPoint(int(bx + 18), int(by + bh)),
            QPoint(int(bx + 32), int(by + bh)),
            QPoint(int(bx + 24), int(by + bh + 10)),
        ])
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(255, 255, 255, 230)))
        painter.drawPolygon(tail)

        painter.setPen(QColor(45, 45, 60))
        for i, line in enumerate(lines):
            painter.drawText(
                QRect(int(bx + 9), int(by + 8 + i * line_h), int(bw - 18), line_h),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                line,
            )

    def _draw_timer_badge(self, painter: QPainter):
        active = self.timer.active_sessions
        if not active:
            return
        running = [s for s in active if s.is_running]
        badge = QRect(88, 115, 60, 22)
        painter.setBrush(QBrush(QColor(80, 60, 20, 215) if running
                                else QColor(55, 55, 90, 215)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(badge, 8, 8)
        painter.setPen(QColor(255, 220, 80) if running else QColor(180, 180, 220))
        painter.setFont(QFont("Courier", 11, QFont.Weight.Bold))
        if len(active) == 1:
            painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, active[0].display_time)
        else:
            painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, f"⏱×{len(active)}")

    def _draw_status(self, painter: QPainter):
        stats = [
            (10,  self.pet.hunger / 100,   QColor(255, 108, 78),  "🍖"),
            (57,  self.pet.energy / 100,   QColor(78, 178, 255),  "⚡"),
            (104, self.pet.happiness / 100, QColor(255, 192, 60), "❤"),
        ]
        bw, bh, by = 40, 7, 178

        for bx, ratio, color, icon in stats:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(38, 38, 52, 130)))
            painter.drawRoundedRect(QRect(bx, by, bw, bh), 3, 3)
            fw = max(3, int(bw * ratio))
            painter.setBrush(QBrush(color))
            painter.drawRoundedRect(QRect(bx, by, fw, bh), 3, 3)
            painter.setFont(QFont("", 8))
            painter.setPen(QColor(255, 255, 255, 200))
            painter.drawText(QRect(bx, by - 13, bw, 13),
                             Qt.AlignmentFlag.AlignCenter, icon)

        painter.setFont(QFont("", 8, QFont.Weight.Bold))
        painter.setPen(QColor(255, 220, 100, 220))
        painter.drawText(QRect(0, 190, 160, 18),
                         Qt.AlignmentFlag.AlignCenter,
                         f"{self.pet.name}  Lv.{self.pet.level}")

    # ── Mouse ─────────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.pos()
            # Click in speech bubble area → quick replies
            if self.speech_visible and pos.y() < 95:
                self._show_quick_replies(event.globalPosition().toPoint())
            else:
                self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                self.show_speech(self.pet.speech)
        event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton and self._drag_pos is not None:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            if self.task_panel and self.task_panel.isVisible():
                self.task_panel.reposition()
        event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        event.accept()

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.setStyleSheet(MENU_STYLE)

        hdr = menu.addAction(f"🐱 {self.pet.name}  Lv.{self.pet.level}")
        hdr.setEnabled(False)
        menu.addSeparator()

        feed = menu.addMenu("🍖 喂食")
        feed.setStyleSheet(MENU_STYLE)
        feed.addAction("普通食物 (+30)", lambda: self._feed(30, 5))
        feed.addAction("美味零食 (+20 +15心情)", lambda: self._feed(20, 15))
        feed.addAction("特制餐食 (+50 +25心情)", lambda: self._feed(50, 25))

        if self.pet.mood == PetMood.SLEEPING:
            menu.addAction("☀️ 叫醒", self._wake)
        else:
            menu.addAction("💤 休息", self._sleep)

        play = menu.addMenu("🎮 互动")
        play.setStyleSheet(MENU_STYLE)
        play.addAction("😊 摸摸 (+心情)",      self._interact_pet)
        play.addAction("⚽ 玩球 (12秒+心情)",  self._start_ball_game)
        play.addAction("🎣 逗猫棒 (10秒+心情)", self._start_teaser)
        play.addAction("💪 加油鼓励",           self._interact_cheer)

        menu.addSeparator()
        menu.addAction("⏱  专注计时",  self._open_timer)
        menu.addAction("📋 任务规划",  self._toggle_tasks)
        menu.addAction("💬 和小咪聊聊", self._open_claude)
        menu.addSeparator()
        menu.addAction("📊 宠物状态", self._show_stats)
        menu.addSeparator()
        menu.addAction("退出", QApplication.instance().quit)

        menu.exec(event.globalPos())

    # ── Actions ───────────────────────────────────────────────────────────

    def _feed(self, h, hp):
        self.pet.feed(h, hp)
        self.eating_frames = 15
        self.show_speech("啊呜啊呜~", 3000)

    def _sleep(self):
        self.pet.rest(); self.show_speech("晚安~ Zzz", 3000)

    def _wake(self):
        self.pet.wake_up(); self.show_speech("起来啦！精神满满~", 3000)

    # ── Quick replies ─────────────────────────────────────────────────────

    def _show_quick_replies(self, global_pos: QPoint):
        """Show preset quick-reply menu near the speech bubble."""
        menu = QMenu(self)
        menu.setStyleSheet(MENU_STYLE)
        replies = [
            ("😊 摸摸你~",    self._interact_pet),
            ("🍖 给你零食",   lambda: self._feed(20, 15)),
            ("⚽ 一起玩球",   self._start_ball_game),
            ("🎣 逗猫棒",     self._start_teaser),
            ("💪 加油加油！", self._interact_cheer),
            ("🌙 去休息吧",   self._sleep),
        ]
        for label, fn in replies:
            menu.addAction(label, fn)
        menu.exec(global_pos)

    # ── Interactive actions ───────────────────────────────────────────────

    def _interact_pet(self):
        """Stroke the cat: +happiness, happy animation."""
        self.pet.happiness = min(100, self.pet.happiness + 15)
        self.pet.energy    = min(100, self.pet.energy    + 3)
        self.pet.mood = PetMood.HAPPY
        self.happy_frames = 22
        self.pet.gain_exp(2)
        self.show_speech(random.choice([
            "呼噜呼噜~ 好舒服！", "喵~ 继续摸！", "嘿嘿嘿~",
            "软软的~ 好喜欢！", "呜呜~ 爱你~",
        ]), 3000)

    def _interact_cheer(self):
        """Cheer the cat: +happiness +exp."""
        self.pet.happiness = min(100, self.pet.happiness + 10)
        self.pet.mood = PetMood.HAPPY
        self.happy_frames = 16
        self.pet.gain_exp(3)
        self.show_speech(random.choice([
            "谢谢你！我会加油的！", "嗷嗷！元气满满~",
            "有你在我不怕！", "冲冲冲！！",
        ]), 3000)

    # ── Mini-games ────────────────────────────────────────────────────────

    def _start_ball_game(self):
        if self.minigame:
            return
        self.minigame = 'ball'
        self.minigame_frames = 70        # 70 × 200ms ≈ 14 s
        self.ball_x = 80.0;  self.ball_y = 80.0
        self.ball_vx = random.choice([-1, 1]) * random.uniform(3.5, 5.0)
        self.ball_vy = random.choice([-1, 1]) * random.uniform(2.5, 4.0)
        self.pet.mood = PetMood.HAPPY
        self.show_speech("球球！抓到你了！", 2500)

    def _tick_ball(self):
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        # Bounce inside widget (reserve top 20px and bottom 165px)
        if self.ball_x < 12 or self.ball_x > 148:
            self.ball_vx *= -1
            self.ball_x = max(12.0, min(148.0, self.ball_x))
        if self.ball_y < 20 or self.ball_y > 155:
            self.ball_vy *= -1
            self.ball_y = max(20.0, min(155.0, self.ball_y))
        # Cat looks toward ball
        self.idle_action = 'look_left' if self.ball_x < 75 else 'look_right'
        self.minigame_frames -= 1
        if self.minigame_frames <= 0:
            self._end_minigame()

    def _start_teaser(self):
        if self.minigame:
            return
        self.minigame = 'teaser'
        self.minigame_frames = 60        # 60 × 200ms = 12 s
        self.teaser_t = 0.0
        self.pet.mood = PetMood.HAPPY
        self.show_speech("那是什么？！抓住它！", 2500)

    def _tick_teaser(self):
        self.teaser_t += 0.18
        tx = 80 + math.sin(self.teaser_t) * 46
        # Cat tracks the teaser
        self.idle_action = 'look_left' if tx < 72 else 'look_right'
        self.minigame_frames -= 1
        if self.minigame_frames <= 0:
            self._end_minigame()

    def _end_minigame(self):
        game = self.minigame
        self.minigame = None
        self.idle_action = 'none'
        self.pet.happiness = min(100, self.pet.happiness + 25)
        self.pet.energy    = max(0,   self.pet.energy    - (5 if game == 'ball' else 8))
        self.pet.gain_exp(5)
        self.pet.mood = PetMood.HAPPY
        self.happy_frames = 20
        self.show_speech(random.choice([
            "好好玩！再来一次！", "哈哈哈太开心了！",
            "呼~ 累了但好满足~", "下次还要玩！",
        ]), 3500)

    def _draw_minigame(self, painter: QPainter):
        if self.minigame == 'ball':
            bx, by = int(self.ball_x), int(self.ball_y)
            # glow
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(255, 110, 40, 80)))
            painter.drawEllipse(bx - 10, by - 10, 20, 20)
            # ball
            painter.setBrush(QBrush(QColor(255, 120, 40)))
            painter.drawEllipse(bx - 7, by - 7, 14, 14)
            # shine
            painter.setBrush(QBrush(QColor(255, 255, 220, 180)))
            painter.drawEllipse(bx - 4, by - 5, 4, 4)

        elif self.minigame == 'teaser':
            tx = int(80 + math.sin(self.teaser_t) * 46)
            ty = int(75 + math.sin(self.teaser_t * 1.7) * 28)
            # stick
            painter.setPen(QPen(QColor(160, 100, 40), 2))
            painter.drawLine(tx, ty, tx + 14, ty + 18)
            # feather puffs
            for i, (dx, dy, c) in enumerate([
                (-5, -5, QColor(255, 80, 210, 240)),
                ( 0, -8, QColor(255, 140, 220, 200)),
                ( 5, -5, QColor(200, 60, 255, 220)),
            ]):
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(c))
                painter.drawEllipse(tx + dx - 4, ty + dy - 4, 9, 9)

    def _open_timer(self):
        if self.timer_dialog is None:
            from ui.timer_dialog import TimerDialog
            self.timer_dialog = TimerDialog(self.timer, self.pet, self)
        self.timer_dialog.show(); self.timer_dialog.raise_()

    def _init_task_panel(self):
        """Create and permanently show the task mini-panel beside the cat."""
        if self.task_panel is None:
            from ui.task_panel import TaskMiniPanel
            self.task_panel = TaskMiniPanel(self.tasks, self)
        self.task_panel.refresh()
        self.task_panel.show()

    def _toggle_tasks(self):
        """Right-click menu: open the full task dialog; panel stays visible."""
        if self.task_panel is None:
            self._init_task_panel()
        elif not self.task_panel.isVisible():
            self.task_panel.refresh()
            self.task_panel.show()

        if self.task_dialog is None:
            from ui.task_dialog import TaskDialog
            self.task_dialog = TaskDialog(self.tasks, self.pet, self)
        self.task_dialog.refresh()
        self.task_dialog.show(); self.task_dialog.raise_()

    def _open_claude(self):
        if self.claude_dialog is None:
            from ui.pet_chat_dialog import PetChatDialog
            self.claude_dialog = PetChatDialog(self.pet, self)
        self.claude_dialog.show(); self.claude_dialog.raise_()

    def _check_stats_alert(self):
        """Show an alert dialog when any stat is critically low (< 15)."""
        if not hasattr(self, '_alert_cooldown'):
            self._alert_cooldown = 0
        # Only nag every ~5 minutes at most
        self._alert_cooldown = max(0, self._alert_cooldown - 1)
        if self._alert_cooldown > 0:
            return

        crits = []
        if self.pet.hunger    < 15: crits.append(("🍖", "饱食度很低", "快来喂食！"))
        if self.pet.energy    < 15: crits.append(("⚡", "精力快耗尽", "让她休息！"))
        if self.pet.happiness < 15: crits.append(("❤", "心情很低落", "和她聊聊天~"))
        if not crits:
            return

        self._alert_cooldown = 5   # skip next 4 minute-ticks
        self._show_stats_alert(crits)

    def _show_stats_alert(self, crits):
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
        dlg = QDialog(self)
        dlg.setWindowTitle(f"{self.pet.name} 需要你！")
        dlg.setStyleSheet(DIALOG_STYLE)
        dlg.setFixedWidth(280)

        root = QVBoxLayout(dlg)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(10)

        hdr = QLabel(f"🐱 {self.pet.name} 需要你的关注！")
        hdr.setStyleSheet("font-size:14px; font-weight:bold; color:#f38ba8;")
        hdr.setWordWrap(True)
        root.addWidget(hdr)

        for icon, label, hint in crits:
            row = QHBoxLayout(); row.setSpacing(8)
            row.addWidget(QLabel(icon))
            info = QLabel(f"<b>{label}</b> — {hint}")
            info.setStyleSheet("color:#cdd6f4; font-size:12px;")
            row.addWidget(info); row.addStretch()
            root.addLayout(row)

        root.addSpacing(6)
        btns = QHBoxLayout(); btns.setSpacing(8)
        feed_btn = QPushButton("🍖 喂食")
        rest_btn = QPushButton("💤 休息")
        chat_btn = QPushButton("💬 聊聊")
        ok_btn   = QPushButton("知道了")
        feed_btn.clicked.connect(lambda: (self._feed(30, 10), dlg.accept()))
        rest_btn.clicked.connect(lambda: (self._sleep(),      dlg.accept()))
        chat_btn.clicked.connect(lambda: (self._open_claude(), dlg.accept()))
        ok_btn.clicked.connect(dlg.accept)
        for b in (feed_btn, rest_btn, chat_btn, ok_btn):
            btns.addWidget(b)
        root.addLayout(btns)

        dlg.show()

    def _show_stats(self):
        td   = self.tasks.today_total_count
        done = self.tasks.today_done_count
        days = self.pet.companionship_days + 1
        box  = QMessageBox(self)
        box.setWindowTitle("宠物状态")
        box.setText(
            f"🐱  {self.pet.name}\n\n"
            f"🗓  陪伴了第 {days} 天   Lv.{self.pet.level}\n\n"
            f"🍖 饱食度：{self.pet.hunger:.0f}/100\n"
            f"⚡ 精力：  {self.pet.energy:.0f}/100\n"
            f"❤  心情：  {self.pet.happiness:.0f}/100\n\n"
            f"📋 今日任务：{done}/{td} 完成"
        )
        box.setStyleSheet(
            "QMessageBox{background:#1e1e2e;color:#cdd6f4;}"
            "QLabel{color:#cdd6f4;font-size:13px;}"
            "QPushButton{background:#313244;color:#cdd6f4;"
            "border:1px solid #45475a;border-radius:5px;padding:5px 16px;}"
        )
        box.exec()

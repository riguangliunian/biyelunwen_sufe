import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, time
from typing import List, Optional


@dataclass
class Task:
    id: str
    title: str
    category: str = "工作"          # 工作/学习/生活/其他
    reminder_time: Optional[str] = None   # "HH:MM" or None
    completed: bool = False
    date: str = field(default_factory=lambda: date.today().isoformat())
    reminded: bool = False

    def is_today(self) -> bool:
        return self.date == date.today().isoformat()

    @property
    def deadline_dt(self) -> Optional[datetime]:
        """Parse reminder_time as a datetime. Supports 'HH:MM' and 'YYYY-MM-DD HH:MM'."""
        if not self.reminder_time:
            return None
        try:
            if " " in self.reminder_time:
                return datetime.strptime(self.reminder_time, "%Y-%m-%d %H:%M")
            else:
                h, m = map(int, self.reminder_time.split(":"))
                return datetime.combine(date.fromisoformat(self.date), time(h, m))
        except Exception:
            return None

    @property
    def deadline_label(self) -> str:
        """Human-readable deadline for display."""
        dt = self.deadline_dt
        if dt is None:
            return ""
        today = date.today()
        delta = (dt.date() - today).days
        if delta < 0:
            prefix = "已过期"
        elif delta == 0:
            prefix = "今天"
        elif delta == 1:
            prefix = "明天"
        else:
            prefix = dt.strftime("%-m月%-d日")
        return f"{prefix} {dt.strftime('%H:%M')}"

    def reminder_due(self) -> bool:
        if not self.reminder_time or self.reminded or self.completed:
            return False
        dt = self.deadline_dt
        return dt is not None and datetime.now() >= dt


class TaskManager:
    CATEGORIES = ["工作", "学习", "生活", "其他"]

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.filepath = os.path.join(data_dir, "tasks.json")
        self.tasks: List[Task] = []
        self.load()

    def load(self):
        if not os.path.exists(self.filepath):
            return
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.tasks = [Task(**t) for t in raw]
        except Exception:
            self.tasks = []

    def save(self):
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump([asdict(t) for t in self.tasks], f, ensure_ascii=False, indent=2)

    def add_task(self, title: str, category: str = "工作",
                 reminder_time: Optional[str] = None) -> Task:
        task = Task(
            id=str(uuid.uuid4()),
            title=title,
            category=category,
            reminder_time=reminder_time,
            date=date.today().isoformat(),
        )
        self.tasks.append(task)
        self.save()
        return task

    def remove_task(self, task_id: str):
        self.tasks = [t for t in self.tasks if t.id != task_id]
        self.save()

    def complete_task(self, task_id: str):
        for t in self.tasks:
            if t.id == task_id:
                t.completed = True
        self.save()

    def uncomplete_task(self, task_id: str):
        for t in self.tasks:
            if t.id == task_id:
                t.completed = False
        self.save()

    def get_today_tasks(self) -> List[Task]:
        today = date.today().isoformat()
        return [t for t in self.tasks if t.date == today]

    def get_due_reminders(self) -> List[Task]:
        """Check ALL tasks (not just today's) for overdue reminders."""
        due = []
        for t in self.tasks:
            if t.reminder_due():
                t.reminded = True
                due.append(t)
        if due:
            self.save()
        return due

    def cleanup_old_tasks(self):
        """Keep only last 7 days of tasks"""
        today = date.today()
        self.tasks = [
            t for t in self.tasks
            if (today - date.fromisoformat(t.date)).days <= 7
        ]
        self.save()

    @property
    def today_done_count(self) -> int:
        return sum(1 for t in self.get_today_tasks() if t.completed)

    @property
    def today_total_count(self) -> int:
        return len(self.get_today_tasks())

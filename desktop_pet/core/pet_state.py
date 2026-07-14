import json
import os
from enum import Enum
from datetime import date, timedelta


class PetMood(Enum):
    IDLE = "idle"
    HAPPY = "happy"
    HUNGRY = "hungry"
    SLEEPY = "sleepy"
    SLEEPING = "sleeping"
    EATING = "eating"
    WORKING = "working"


class PetState:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.filepath = os.path.join(data_dir, "pet_state.json")

        self.hunger = 80.0
        self.energy = 80.0
        self.happiness = 70.0
        self.name = "小咪"
        self.created_at = date.today().isoformat()
        self._mood = PetMood.IDLE

        self.load()

    @property
    def mood(self) -> PetMood:
        return self._mood

    @mood.setter
    def mood(self, value: PetMood):
        self._mood = value

    # ── Level / companionship (computed from created_at) ──────────────────

    @property
    def companionship_days(self) -> int:
        """Days since first launch (0 on day 1 → display as day 1)."""
        try:
            return max(0, (date.today() - date.fromisoformat(self.created_at)).days)
        except Exception:
            return 0

    @property
    def level(self) -> int:
        """Level = companionship day number (day 1 = Lv.1)."""
        return self.companionship_days + 1

    # ── Persistence ───────────────────────────────────────────────────────

    def load(self):
        if not os.path.exists(self.filepath):
            return
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                d = json.load(f)
            self.hunger     = float(d.get("hunger", 80))
            self.energy     = float(d.get("energy", 80))
            self.happiness  = float(d.get("happiness", 70))
            self.name       = d.get("name", "小咪")
            self.created_at = d.get("created_at", date.today().isoformat())
        except Exception:
            pass

    def save(self):
        data = {
            "hunger":     round(self.hunger, 2),
            "energy":     round(self.energy, 2),
            "happiness":  round(self.happiness, 2),
            "name":       self.name,
            "created_at": self.created_at,
        }
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def tick_minute(self):
        if self._mood == PetMood.SLEEPING:
            self.energy = min(100, self.energy + 5)
            self.hunger = max(0, self.hunger - 0.5)
        elif self._mood == PetMood.WORKING:
            self.energy = max(0, self.energy - 1.5)
            self.hunger = max(0, self.hunger - 1.5)
        else:
            self.energy = max(0, self.energy - 0.5)
            self.hunger = max(0, self.hunger - 1.0)

        self.happiness = max(0, self.happiness - 0.3)

        if self._mood not in (PetMood.EATING, PetMood.SLEEPING, PetMood.WORKING):
            if self.hunger < 20:
                self._mood = PetMood.HUNGRY
            elif self.energy < 20:
                self._mood = PetMood.SLEEPY
            elif self._mood in (PetMood.HUNGRY, PetMood.SLEEPY):
                if self.hunger > 30 and self.energy > 30:
                    self._mood = PetMood.IDLE

        self.save()

    def feed(self, hunger_amt: int = 30, happy_amt: int = 5):
        self.hunger = min(100, self.hunger + hunger_amt)
        self.happiness = min(100, self.happiness + happy_amt)
        self._mood = PetMood.EATING
        self.gain_exp(10)
        self.save()

    def rest(self):
        self._mood = PetMood.SLEEPING
        self.save()

    def wake_up(self):
        self._mood = PetMood.IDLE
        self.save()

    def start_working(self):
        self._mood = PetMood.WORKING
        self.save()

    def finish_working(self):
        self.happiness = min(100, self.happiness + 20)
        self._mood = PetMood.HAPPY
        self.gain_exp(30)
        self.save()

    def complete_task(self):
        self.happiness = min(100, self.happiness + 10)
        self.gain_exp(20)
        self.save()

    def gain_exp(self, amount: int):
        """Legacy call sites kept; gives a small happiness nudge instead of XP level-up."""
        self.happiness = min(100, self.happiness + amount * 0.05)
        self.save()

    @property
    def speech(self) -> str:
        if self._mood == PetMood.HUNGRY:
            return "好饿~快喂我！"
        if self._mood == PetMood.SLEEPY:
            return "好困...想睡觉"
        if self._mood == PetMood.SLEEPING:
            return "Zzz..."
        if self._mood == PetMood.EATING:
            return "啊呜啊呜~好吃！"
        if self._mood == PetMood.WORKING:
            return "专注中！加油~"
        if self._mood == PetMood.HAPPY:
            return "好开心！嘻嘻~"
        if self.happiness > 80:
            return "今天也很棒哦~"
        days = self.companionship_days
        if days > 0 and days % 7 == 0:
            return f"陪伴 {days} 天了，感谢你！"
        return "摸摸我~"

    @property
    def age_text(self) -> str:
        days = self.companionship_days
        if days == 0:
            return "第1天"
        return f"第{days + 1}天"

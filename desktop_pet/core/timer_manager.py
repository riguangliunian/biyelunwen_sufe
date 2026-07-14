from dataclasses import dataclass, field


@dataclass
class TimerSession:
    id: int
    name: str = "专注"
    total_seconds: int = 25 * 60
    remaining_seconds: int = 0
    is_running: bool = False
    is_paused: bool = False
    finished: bool = False

    def __post_init__(self):
        if self.remaining_seconds == 0:
            self.remaining_seconds = self.total_seconds

    @property
    def progress(self) -> float:
        if self.total_seconds == 0:
            return 0.0
        return 1.0 - self.remaining_seconds / self.total_seconds

    @property
    def display_time(self) -> str:
        m = self.remaining_seconds // 60
        s = self.remaining_seconds % 60
        return f"{m:02d}:{s:02d}"


class TimerManager:
    PRESETS = [
        ("🍅 番茄", 25 * 60),
        ("☕ 短休",  5 * 60),
        ("🌿 长休", 15 * 60),
        ("⚡ 冲刺", 45 * 60),
    ]

    def __init__(self):
        self.sessions: list[TimerSession] = []
        self._next_id = 1

    # ── CRUD ──────────────────────────────────────────────────────────────

    def add(self, name: str, seconds: int) -> TimerSession:
        s = TimerSession(id=self._next_id, name=name,
                         total_seconds=seconds, remaining_seconds=seconds,
                         is_running=True)
        self._next_id += 1
        self.sessions.append(s)
        return s

    def remove(self, session_id: int):
        self.sessions = [s for s in self.sessions if s.id != session_id]

    def toggle_pause(self, session_id: int):
        for s in self.sessions:
            if s.id == session_id:
                if s.is_running:
                    s.is_running = False
                    s.is_paused = True
                elif s.is_paused:
                    s.is_running = True
                    s.is_paused = False

    def cleanup_finished(self):
        self.sessions = [s for s in self.sessions if not s.finished]

    # ── Tick ──────────────────────────────────────────────────────────────

    def tick_all(self) -> list[TimerSession]:
        """Decrement all running sessions by 1 s. Returns newly-finished ones."""
        done = []
        for s in self.sessions:
            if s.is_running:
                s.remaining_seconds -= 1
                if s.remaining_seconds <= 0:
                    s.remaining_seconds = 0
                    s.is_running = False
                    s.finished = True
                    done.append(s)
        return done

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def any_running(self) -> bool:
        return any(s.is_running for s in self.sessions)

    @property
    def active_sessions(self) -> list[TimerSession]:
        return [s for s in self.sessions if not s.finished]

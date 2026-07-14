import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from PyQt6.QtWidgets import QApplication

from core.pet_state import PetState
from core.task_manager import TaskManager
from core.timer_manager import TimerManager
from ui.pet_widget import PetWidget

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName("桌宠小咪")

    pet   = PetState(DATA_DIR)
    tasks = TaskManager(DATA_DIR)
    timer = TimerManager()
    tasks.cleanup_old_tasks()

    widget = PetWidget(pet, tasks, timer)
    widget.show()   # showEvent → _boost_ns_level fires after 200 ms

    sys.exit(app.exec())


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()

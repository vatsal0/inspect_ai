import sys
from contextvars import ContextVar

import rich

from inspect_ai.util._display import display_type

from ..log.display import LogDisplay
from ..plain.display import PlainDisplay
from ..rich.display import RichDisplay
from ..textual.display import TextualDisplay
from .display import Display, TaskScreen

_active_display: Display | None = None


def display() -> Display:
    global _active_display
    if _active_display is None:
        if display_type() == "plain":
            _active_display = PlainDisplay()
        elif (
            display_type() == "full"
            and sys.stdout.isatty()
            and not rich.get_console().is_jupyter
        ):
            _active_display = TextualDisplay()
        elif display_type() == "log":
            _active_display = LogDisplay()
        else:
            _active_display = RichDisplay()

    return _active_display


def task_screen() -> TaskScreen:
    screen = _active_task_screen.get(None)
    if screen is None:
        raise RuntimeError(
            "console input function called outside of running evaluation."
        )
    return screen


def init_task_screen(screen: TaskScreen) -> None:
    _active_task_screen.set(screen)


def clear_task_screen() -> None:
    _active_task_screen.set(None)


_active_task_screen: ContextVar[TaskScreen | None] = ContextVar(
    "task_screen", default=None
)

"""Terminal-only presentation. No configuration, credentials or network operations."""

import os
import re
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from io import StringIO
from time import monotonic
from typing import Any

import questionary
import typer
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.output import ColorDepth
from rich.cells import cell_len
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text


class Cancelled(Exception):
    """Leave the current task without committing a draft."""


class Back(Exception):
    """Return to the previous draft step."""


def clean(value: object) -> str:
    return re.sub(r"[\x00-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]", " ", str(value))


def interactive_terminal(*, plain: bool = False) -> bool:
    ci = os.getenv("CI", "").lower() not in {"", "0", "false", "no"}
    return (
        sys.stdin.isatty()
        and sys.stdout.isatty()
        and not ci
        and (plain or os.getenv("TERM", "").lower() != "dumb")
    )


class Terminal:
    def __init__(self, *, plain: bool = False, console: Console | None = None) -> None:
        self.plain = plain
        self.no_color = plain or "NO_COLOR" in os.environ
        self.console = console or Console(
            no_color=self.no_color,
            highlight=False,
            markup=False,
            force_terminal=False if plain else None,
        )

    def tell(self, message: object, *, error: bool = False) -> None:
        self.console.print(Text(clean(message), style="red" if error else ""))

    def heading(self, title: str, subtitle: str = "") -> None:
        text = Text(clean(title), style="bold green")
        if subtitle:
            text.append("\n" + clean(subtitle), style="not bold")
        self.console.print(text if self.plain else Panel(text, border_style="cyan", padding=(0, 1)))

    def _ask(self, question: Any) -> str:
        bindings = KeyBindings()

        @bindings.add("escape", eager=True)
        def go_back(event: Any) -> None:
            event.app.exit(exception=Back())

        @bindings.add("c-d", eager=True)
        def eof(event: Any) -> None:
            event.app.exit(exception=Cancelled())

        original = question.application.key_bindings
        question.application.key_bindings = merge_key_bindings(
            [original, bindings] if original is not None else [bindings]
        )
        try:
            answer = question.unsafe_ask()
        except (KeyboardInterrupt, EOFError) as exc:
            raise Cancelled() from exc
        if answer is None:
            raise Cancelled()
        return str(answer)

    def select(self, label: str, choices: list[tuple[str, str]], default: str | None = None) -> str:
        if self.plain:
            self.tell(label)
            for number, (_, title) in enumerate(choices, 1):
                self.tell(f"  {number}. {title}")
            selected = next((i for i, (key, _) in enumerate(choices, 1) if key == default), 1)
            while True:
                value = self.text("Choice", str(selected))
                if value.isdigit() and 1 <= int(value) <= len(choices):
                    return choices[int(value) - 1][0]
                self.tell("Choose a listed number.", error=True)
        titles = []
        limit = max(12, self.console.width - 8)
        for key, title in choices:
            text = Text(clean(title))
            if cell_len(text.plain) > limit:
                text.truncate(limit - 3, overflow="crop")
                text.append("...")
            titles.append(questionary.Choice(text.plain, value=key))
        return self._ask(
            questionary.select(
                clean(label),
                choices=titles,
                default=default,
                qmark="",
                pointer=">",
                use_shortcuts=len(choices) <= 9,
                use_jk_keys=False,
                color_depth=ColorDepth.DEPTH_1_BIT if self.no_color else None,
            )
        )

    def path(
        self,
        label: str,
        default: str = "",
        *,
        validate: Callable[[str], bool | str] | None = None,
    ) -> str:
        if self.plain:
            return self.text(label, default, validate=validate)
        return self._ask(
            questionary.path(
                clean(label),
                default=clean(default),
                qmark="",
                validate=validate or (lambda v: True),
                color_depth=ColorDepth.DEPTH_1_BIT if self.no_color else None,
            )
        )

    def text(
        self,
        label: str,
        default: str = "",
        *,
        secret: bool = False,
        validate: Callable[[str], bool | str] | None = None,
    ) -> str:
        if self.plain:
            while True:
                try:
                    value = str(
                        typer.prompt(
                            clean(label),
                            default="" if secret else default,
                            hide_input=secret,
                            show_default=not secret,
                        )
                    )
                except (typer.Abort, KeyboardInterrupt, EOFError) as exc:
                    raise Cancelled() from exc
                if value == ":back" and not secret:
                    raise Back()
                if value == ":cancel" and not secret:
                    raise Cancelled()
                result = validate(value) if validate else True
                if result is True:
                    return value
                self.tell(result if isinstance(result, str) else "Invalid value.", error=True)
        factory = questionary.password if secret else questionary.text
        return self._ask(
            factory(
                clean(label),
                default="" if secret else clean(default),
                qmark="",
                validate=validate or (lambda value: True),
                color_depth=ColorDepth.DEPTH_1_BIT if self.no_color else None,
            )
        )

    def view(self, title: str, content: str, *, markdown: bool = False) -> None:
        self.heading(title)
        buffer = StringIO()
        renderer = Console(
            file=buffer,
            width=max(20, self.console.width),
            color_system=None,
            highlight=False,
            markup=False,
        )
        # Preserve newlines while removing terminal control characters from every line.
        content = "\n".join(clean(line) for line in content.splitlines())
        renderer.print(Markdown(content) if markdown and not self.plain else Text(content))
        lines = buffer.getvalue().splitlines()
        size = max(5, self.console.height - 9)
        pages = [lines[i : i + size] for i in range(0, len(lines), size)] or [["No entries."]]
        page = 0
        while True:
            self.console.print(Text("\n".join(pages[page])))
            options = []
            if page + 1 < len(pages):
                options.append(("next", "Next page"))
            if page:
                options.append(("previous", "Previous page"))
            options.append(("back", "Back"))
            try:
                action = self.select(f"Page {page + 1}/{len(pages)}", options)
            except Back:
                return
            if action == "back":
                return
            page += 1 if action == "next" else -1

    @contextmanager
    def progress(self, run_id: str) -> Iterator[Callable[[str], None]]:
        started = monotonic()
        operation = "Preparing research"
        console = Console(stderr=True, no_color=self.no_color, markup=False, highlight=False)

        def render() -> Text:
            return Text(
                f"Run {run_id}\n{operation}\nElapsed {monotonic() - started:.0f}s", style="cyan"
            )

        def update(message: str) -> None:
            nonlocal operation
            operation = clean(message)
            if self.plain or not console.is_terminal:
                console.print(Text(operation))

        if self.plain or not console.is_terminal:
            yield update
        else:
            with Live(
                console=console,
                get_renderable=render,
                refresh_per_second=2,
                redirect_stdout=False,
                redirect_stderr=False,
            ):
                yield update

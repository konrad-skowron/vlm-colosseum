from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import queue
import re
import tkinter as tk
from tkinter import scrolledtext


MODEL_LOG_RE = re.compile(
    r"^(?P<model>.+?): (?P<action>.*?)"
    r" \| latency=(?P<latency>[0-9.]+)ms"
    r"(?P<hallucination> \| hallucination)?"
    r"(?: \| (?P<summary>.*))?$"
)

BG = "#111827"
SURFACE = "#182235"
SURFACE_ALT = "#0f172a"
TEXT = "#e5edf7"
MUTED = "#94a3b8"
BORDER = "#2d3b52"
P1_ACCENT = "#60a5fa"
P2_ACCENT = "#f59e0b"
STATUS_ACCENT = "#34d399"
ERROR = "#f87171"
WARNING = "#fbbf24"


@dataclass(slots=True)
class LogMessage:
    channel: str
    text: str


class SplitLogWindow:
    def __init__(self) -> None:
        self._root = tk.Tk()
        self._root.title("LLM Arena Logs")
        self._root.geometry("1320x780")
        self._root.minsize(980, 620)
        self._root.configure(bg=BG)
        self._root.protocol("WM_DELETE_WINDOW", self.close)

        self._queue: queue.Queue[LogMessage] = queue.Queue()
        self._closed = False

        container = tk.Frame(self._root, padx=14, pady=14, bg=BG)
        container.pack(fill="both", expand=True)

        title_row = tk.Frame(container, bg=BG)
        title_row.pack(fill="x", pady=(0, 12))
        tk.Label(
            title_row,
            text="LLM Arena telemetry",
            font=("Segoe UI Semibold", 18),
            fg=TEXT,
            bg=BG,
        ).pack(side="left")
        tk.Label(
            title_row,
            text="model decisions, latency and command stream",
            font=("Segoe UI", 10),
            fg=MUTED,
            bg=BG,
        ).pack(side="left", padx=(14, 0), pady=(5, 0))

        status_frame = self._make_card(container)
        status_frame.pack(fill="x", pady=(0, 12))
        tk.Label(
            status_frame,
            text="STATUS",
            font=("Segoe UI Semibold", 9),
            fg=STATUS_ACCENT,
            bg=SURFACE,
        ).pack(anchor="w", padx=12, pady=(10, 4))
        self._status_box = scrolledtext.ScrolledText(
            status_frame,
            height=4,
            wrap="word",
            font=("Cascadia Mono", 10),
            bg=SURFACE_ALT,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=8,
        )
        self._status_box.pack(fill="x", expand=False, padx=10, pady=(0, 10))
        self._configure_text_box(self._status_box, STATUS_ACCENT)

        columns = tk.Frame(container, bg=BG)
        columns.pack(fill="both", expand=True)
        columns.grid_columnconfigure(0, weight=1)
        columns.grid_columnconfigure(1, weight=1)
        columns.grid_rowconfigure(0, weight=1)

        self._p1_box = self._make_player_panel(columns, "P1", P1_ACCENT, 0)
        self._p2_box = self._make_player_panel(columns, "P2", P2_ACCENT, 1)

    def _make_card(self, parent: tk.Widget) -> tk.Frame:
        return tk.Frame(
            parent,
            bg=SURFACE,
            highlightbackground=BORDER,
            highlightthickness=1,
            bd=0,
        )

    def _make_player_panel(
        self,
        parent: tk.Widget,
        title: str,
        accent: str,
        column: int,
    ) -> scrolledtext.ScrolledText:
        panel = self._make_card(parent)
        panel.grid(row=0, column=column, sticky="nsew", padx=6)
        panel.grid_rowconfigure(1, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        header = tk.Frame(panel, bg=SURFACE)
        header.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 6))
        tk.Label(
            header,
            text=title,
            font=("Segoe UI Semibold", 14),
            fg=accent,
            bg=SURFACE,
        ).pack(side="left")
        tk.Label(
            header,
            text="decision log",
            font=("Segoe UI", 9),
            fg=MUTED,
            bg=SURFACE,
        ).pack(side="left", padx=(10, 0), pady=(3, 0))

        box = scrolledtext.ScrolledText(
            panel,
            wrap="word",
            font=("Cascadia Mono", 10),
            bg=SURFACE_ALT,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            borderwidth=0,
            padx=12,
            pady=10,
        )
        box.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self._configure_text_box(box, accent)
        return box

    def _configure_text_box(
        self,
        box: scrolledtext.ScrolledText,
        accent: str,
    ) -> None:
        box.configure(state="disabled")
        box.tag_configure("time", foreground=MUTED)
        box.tag_configure("label", foreground=MUTED, font=("Cascadia Mono", 9, "bold"))
        box.tag_configure("accent", foreground=accent, font=("Cascadia Mono", 10, "bold"))
        box.tag_configure("action", foreground=TEXT, font=("Cascadia Mono", 10, "bold"))
        box.tag_configure("summary", foreground="#cbd5e1")
        box.tag_configure("muted", foreground=MUTED)
        box.tag_configure("warning", foreground=WARNING, font=("Cascadia Mono", 10, "bold"))
        box.tag_configure("error", foreground=ERROR, font=("Cascadia Mono", 10, "bold"))
        box.tag_configure("spacer", spacing3=8)

    def _timestamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _insert(self, box: scrolledtext.ScrolledText, text: str, tag: str | None = None) -> None:
        if tag is None:
            box.insert("end", text)
        else:
            box.insert("end", text, tag)

    def _append_event(self, box: scrolledtext.ScrolledText, text: str) -> None:
        tag = "error" if "error" in text.lower() else "muted"
        self._insert(box, f"[{self._timestamp()}] ", "time")
        self._insert(box, text + "\n", tag)

    def _append_model_log(
        self,
        box: scrolledtext.ScrolledText,
        text: str,
    ) -> None:
        match = MODEL_LOG_RE.match(text)
        if match is None:
            self._append_event(box, text)
            return

        model = match.group("model")
        action = match.group("action")
        latency = float(match.group("latency"))
        summary = match.group("summary") or ""
        hallucination = match.group("hallucination") is not None
        latency_tag = "warning" if latency >= 1500 else "accent"

        self._insert(box, f"[{self._timestamp()}] ", "time")
        self._insert(box, model + "\n", "accent")
        self._insert(box, "ACTION   ", "label")
        self._insert(box, action + "\n", "action")
        self._insert(box, "LATENCY  ", "label")
        self._insert(box, f"{latency:.1f} ms", latency_tag)
        if hallucination:
            self._insert(box, "  HALLUCINATION", "error")
        self._insert(box, "\n")
        if summary:
            self._insert(box, "REASON   ", "label")
            self._insert(box, summary + "\n", "summary")
        self._insert(box, "\n", "spacer")

    def _append_to_box(
        self,
        box: scrolledtext.ScrolledText,
        text: str,
        *,
        is_player_box: bool,
    ) -> None:
        box.configure(state="normal")
        if is_player_box:
            self._append_model_log(box, text)
        else:
            self._append_event(box, text)
        box.see("end")
        box.configure(state="disabled")

    def log_status(self, message: str) -> None:
        self._queue.put(LogMessage(channel="status", text=message))

    def log_p1(self, message: str) -> None:
        self._queue.put(LogMessage(channel="p1", text=message))

    def log_p2(self, message: str) -> None:
        self._queue.put(LogMessage(channel="p2", text=message))

    def log(self, channel: str, message: str) -> None:
        self._queue.put(LogMessage(channel=channel, text=message))

    def pump(self) -> None:
        if self._closed:
            return

        try:
            self._root.update_idletasks()
            self._root.update()
        except tk.TclError:
            self._closed = True
            return

        self._drain_queue()

    def _drain_queue(self) -> None:
        while True:
            try:
                message = self._queue.get_nowait()
            except queue.Empty:
                break
            self._append(message.channel, message.text)

    def _append(self, channel: str, text: str) -> None:
        if channel == "p1":
            box = self._p1_box
            is_player_box = True
        elif channel == "p2":
            box = self._p2_box
            is_player_box = True
        else:
            box = self._status_box
            is_player_box = False

        self._append_to_box(box, text, is_player_box=is_player_box)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._root.destroy()
        except tk.TclError:
            pass

    @property
    def closed(self) -> bool:
        return self._closed

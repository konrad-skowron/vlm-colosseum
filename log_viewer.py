from __future__ import annotations

from dataclasses import dataclass
import queue
import tkinter as tk
from tkinter import scrolledtext


@dataclass(slots=True)
class LogMessage:
    channel: str
    text: str


class SplitLogWindow:
    def __init__(self) -> None:
        self._root = tk.Tk()
        self._root.title("LLM Arena Logs")
        self._root.geometry("1200x700")
        self._root.protocol("WM_DELETE_WINDOW", self.close)

        self._queue: queue.Queue[LogMessage] = queue.Queue()
        self._closed = False

        container = tk.Frame(self._root, padx=10, pady=10)
        container.pack(fill="both", expand=True)

        status_frame = tk.Frame(container)
        status_frame.pack(fill="x", pady=(0, 10))
        tk.Label(
            status_frame,
            text="Status",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w")
        self._status_box = scrolledtext.ScrolledText(
            status_frame,
            height=5,
            wrap="word",
            font=("Consolas", 10),
        )
        self._status_box.pack(fill="x", expand=False)
        self._status_box.configure(state="disabled")

        columns = tk.Frame(container)
        columns.pack(fill="both", expand=True)
        columns.grid_columnconfigure(0, weight=1)
        columns.grid_columnconfigure(1, weight=1)
        columns.grid_rowconfigure(0, weight=1)

        self._p1_box = self._make_player_panel(columns, "P1", 0)
        self._p2_box = self._make_player_panel(columns, "P2", 1)

    def _make_player_panel(self, parent: tk.Widget, title: str, column: int) -> scrolledtext.ScrolledText:
        panel = tk.Frame(parent)
        panel.grid(row=0, column=column, sticky="nsew", padx=5)
        panel.grid_rowconfigure(1, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        tk.Label(panel, text=title, font=("Segoe UI", 11, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )
        box = scrolledtext.ScrolledText(
            panel,
            wrap="word",
            font=("Consolas", 10),
        )
        box.grid(row=1, column=0, sticky="nsew")
        box.configure(state="disabled")
        return box

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
        elif channel == "p2":
            box = self._p2_box
        else:
            box = self._status_box

        box.configure(state="normal")
        box.insert("end", text + "\n")
        box.see("end")
        box.configure(state="disabled")

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

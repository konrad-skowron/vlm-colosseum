from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile


SNAPSHOT_LUA_TEMPLATE = """local interval_seconds = {interval_seconds}
local next_snapshot_time = interval_seconds

snapshot_frame_subscription = emu.add_machine_frame_notifier(function ()
    if manager.machine.paused or manager.machine.exit_pending then
        return
    end

    local now = manager.machine.time:as_double()
    if now < next_snapshot_time then
        return
    end

    manager.machine.video:snapshot()

    while now >= next_snapshot_time do
        next_snapshot_time = next_snapshot_time + interval_seconds
    end
end)
"""


@dataclass(slots=True)
class SnapshotLoopConfig:
    output_dir: Path
    script_path: Path

    def mame_args(self) -> list[str]:
        return [
            "-autoboot_delay",
            "0",
            "-autoboot_script",
            str(self.script_path),
            "-snapshot_directory",
            str(self.output_dir),
            "-snapname",
            "latest_frame",
            "-snapview",
            "native",
        ]

    def cleanup(self) -> None:
        if self.script_path.exists():
            self.script_path.unlink()


def create_snapshot_loop(
    output_dir: Path | str = "captures",
    interval_seconds: float = 0.5,
    extra_lua: str = "",
) -> SnapshotLoopConfig:
    """Prepare a MAME Lua script that saves a snapshot at a fixed interval."""
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than 0")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    script_contents = SNAPSHOT_LUA_TEMPLATE.format(
        interval_seconds=repr(interval_seconds),
    )
    if extra_lua.strip():
        script_contents = script_contents + "\n" + extra_lua.strip() + "\n"
    script_path = Path(tempfile.gettempdir()) / "mame_snapshot_loop.lua"
    script_path.write_text(script_contents, encoding="utf-8")

    return SnapshotLoopConfig(
        output_dir=output_dir,
        script_path=script_path,
    )


def delete_screenshots(output_dir: Path | str = "captures") -> None:
    """Delete PNG snapshots created for the current capture session."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return

    for screenshot_path in output_dir.rglob("*.png"):
        screenshot_path.unlink(missing_ok=True)

    for directory in sorted(
        (path for path in output_dir.rglob("*") if path.is_dir()),
        reverse=True,
    ):
        if not any(directory.iterdir()):
            directory.rmdir()

    if output_dir.exists() and not any(output_dir.iterdir()):
        output_dir.rmdir()

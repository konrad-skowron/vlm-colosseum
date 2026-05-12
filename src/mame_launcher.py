from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Sequence


MAME_DIR = Path(r"C:\Emulators\mame")
DEFAULT_ROM = "sfiii3n"


def open_sfiii3n(
    rom_name: str = DEFAULT_ROM,
    mame_dir: Path = MAME_DIR,
    extra_args: Sequence[str] | None = None,
) -> subprocess.Popen[bytes]:
    """Launch a MAME ROM, defaulting to Street Fighter III 3rd Strike NO CD."""
    mame_dir = Path(mame_dir)
    mame_exe = mame_dir / "mame.exe"

    if not mame_exe.exists():
        raise FileNotFoundError(f"MAME executable was not found at {mame_exe}")

    command = [str(mame_exe)]
    if extra_args:
        command.extend(extra_args)
    command.append(rom_name)

    return subprocess.Popen(
        command,
        cwd=mame_dir,
    )


if __name__ == "__main__":
    open_sfiii3n()

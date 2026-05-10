import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from subprocess import TimeoutExpired
import threading
import time

from fight_starter import FightStartConfig, build_fight_start_lua
from llm_arena import (
    build_arena_config,
    build_match_state_lua,
    build_move_bridge_lua,
    initialize_command_files,
    load_dotenv,
    start_llm_workers,
    wait_for_fight_start,
    wait_for_screenshot_warmup,
)
from mame_launcher import open_sfiii3n
from log_viewer import SplitLogWindow
from screenshot_loop import create_snapshot_loop


ENABLE_LLM_ARENA = True
ENABLE_LOG_WINDOW = True
CAPTURES_DIR = "captures"
EXPERIMENT_MATCH_COUNT = 1
MATCH_MAX_SECONDS = 230.0
LLM_ROUND_START_BUFFER_SECONDS = 12.0
LLM_SCREENSHOT_WARMUP_UPDATES = 4


def _terminate_process(process) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except TimeoutExpired:
        process.kill()
        process.wait()


def _append_experiment_summary(
    *,
    summary_path: Path,
    row: dict[str, str],
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "match_index",
        "started_at",
        "ended_at",
        "duration_seconds",
        "status",
        "result",
        "model_p1",
        "model_p2",
        "match_dir",
    )
    should_write_header = not summary_path.exists() or summary_path.stat().st_size == 0
    with summary_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if should_write_header:
            writer.writeheader()
        writer.writerow(row)


def _read_match_state(state_path: Path) -> dict[str, object] | None:
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _run_single_match(
    *,
    match_index: int,
    fight_start: FightStartConfig,
    match_dir: Path,
    log_window: SplitLogWindow | None,
) -> dict[str, str]:
    stop_event = threading.Event()
    process = None
    snapshot_loop = None
    started_at = datetime.now(timezone.utc)
    match_started_perf = time.perf_counter()
    status = "unknown"
    result = "not_detected"

    def emit_log(channel: str, message: str) -> None:
        if log_window is None:
            return
        log_window.log(channel, f"[match {match_index}] {message}")

    extra_lua_parts = [build_fight_start_lua(fight_start).strip()]

    load_dotenv()
    arena_config = build_arena_config(
        fight_start=fight_start,
        captures_dir=match_dir,
        round_start_buffer_seconds=LLM_ROUND_START_BUFFER_SECONDS,
        screenshot_warmup_updates=LLM_SCREENSHOT_WARMUP_UPDATES,
    )
    initialize_command_files(arena_config)
    state_path = match_dir / "match_state.json"
    extra_lua_parts.append(
        build_move_bridge_lua(
            arena_config.command_path_p1,
            arena_config.command_path_p2,
        ).strip()
    )
    extra_lua_parts.append(build_match_state_lua(state_path).strip())

    if log_window is not None:
        log_window.log_status(f"Starting match {match_index}.")

    try:
        snapshot_loop = create_snapshot_loop(
            output_dir=match_dir,
            interval_seconds=0.5,
            extra_lua="\n".join(extra_lua_parts),
        )
        process = open_sfiii3n(
            extra_args=snapshot_loop.mame_args() + ["-skip_gameinfo"]
        )

        wait_for_fight_start(arena_config, emit_log if log_window else None)
        wait_for_screenshot_warmup(
            arena_config.screenshot_path,
            arena_config.screenshot_warmup_updates,
            emit_log if log_window else None,
        )
        llm_workers = start_llm_workers(
            arena_config,
            stop_event,
            emit_log if log_window else None,
        )

        match_started_perf = time.perf_counter()
        while process.poll() is None:
            elapsed_seconds = time.perf_counter() - match_started_perf
            match_state = _read_match_state(state_path)
            if match_state and match_state.get("match_over") is True:
                status = "match_over"
                result = str(match_state.get("winner", "unknown"))
                break
            if elapsed_seconds >= MATCH_MAX_SECONDS:
                status = "timeout"
                if match_state:
                    result = (
                        f"unfinished "
                        f"wins={match_state.get('wins_p1')}-"
                        f"{match_state.get('wins_p2')}"
                    )
                break
            if all(not worker.is_alive() for worker in llm_workers):
                status = "workers_stopped"
                raise RuntimeError("All LLM workers stopped unexpectedly.")
            if log_window is not None:
                log_window.pump()
            time.sleep(0.5)
        else:
            status = f"process_exited_{process.returncode}"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except Exception as exc:
        status = f"error: {exc}"
        if log_window is not None:
            log_window.log_status(f"Match {match_index} error: {exc}")
    finally:
        stop_event.set()
        if process is not None:
            _terminate_process(process)
        if snapshot_loop is not None:
            snapshot_loop.cleanup()

    ended_at = datetime.now(timezone.utc)
    duration_seconds = time.perf_counter() - match_started_perf
    return {
        "match_index": str(match_index),
        "started_at": started_at.isoformat(timespec="seconds"),
        "ended_at": ended_at.isoformat(timespec="seconds"),
        "duration_seconds": f"{duration_seconds:.3f}",
        "status": status,
        "result": result,
        "model_p1": arena_config.model_p1,
        "model_p2": arena_config.model_p2,
        "match_dir": str(match_dir),
    }


def main() -> None:
    fight_start = FightStartConfig(
        p1_character='dudley',
        p2_character='dudley',
        p1_super_art=1,
        p2_super_art=1,
    )
    log_window = SplitLogWindow() if (ENABLE_LLM_ARENA and ENABLE_LOG_WINDOW) else None

    if not ENABLE_LLM_ARENA:
        raise RuntimeError("Batch experiments require ENABLE_LLM_ARENA = True.")

    captures_root = Path(CAPTURES_DIR)
    summary_path = captures_root / "experiment_summary.csv"

    try:
        for match_index in range(1, EXPERIMENT_MATCH_COUNT + 1):
            match_dir = captures_root / f"match_{match_index:03d}"
            row = _run_single_match(
                match_index=match_index,
                fight_start=fight_start,
                match_dir=match_dir,
                log_window=log_window,
            )
            _append_experiment_summary(summary_path=summary_path, row=row)
            if log_window is not None:
                log_window.log_status(
                    f"Finished match {match_index}: "
                    f"{row['status']} ({row['duration_seconds']}s)."
                )
    except KeyboardInterrupt:
        if log_window is not None:
            log_window.log_status("Experiment interrupted.")
    finally:
        if log_window is not None:
            log_window.log_status("Experiment finished. Close this window to exit.")

    if log_window is not None:
        while not log_window.closed:
            log_window.pump()
            time.sleep(0.05)


if __name__ == "__main__":
    main()

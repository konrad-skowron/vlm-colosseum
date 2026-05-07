import threading
import time

from fight_starter import FightStartConfig, build_fight_start_lua
from llm_arena import (
    build_arena_config,
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
LLM_ROUND_START_BUFFER_SECONDS = 15.0
LLM_SCREENSHOT_WARMUP_UPDATES = 4


def main() -> None:
    fight_start = FightStartConfig(
        p1_character='dudley',
        p2_character='dudley',
        p1_super_art=1,
        p2_super_art=1,
    )
    stop_event = threading.Event()
    log_window = SplitLogWindow() if (ENABLE_LLM_ARENA and ENABLE_LOG_WINDOW) else None

    def emit_log(channel: str, message: str) -> None:
        if log_window is None:
            return
        log_window.log(channel, message)

    extra_lua_parts = [build_fight_start_lua(fight_start).strip()]

    if ENABLE_LLM_ARENA:
        load_dotenv()
        arena_config = build_arena_config(
            fight_start=fight_start,
            round_start_buffer_seconds=LLM_ROUND_START_BUFFER_SECONDS,
            screenshot_warmup_updates=LLM_SCREENSHOT_WARMUP_UPDATES,
        )
        initialize_command_files(arena_config)
        extra_lua_parts.append(
            build_move_bridge_lua(
                arena_config.command_path_p1,
                arena_config.command_path_p2,
            ).strip()
        )
        if log_window is not None:
            log_window.log_status("LLM arena enabled.")
    else:
        arena_config = None

    snapshot_loop = create_snapshot_loop(
        output_dir=CAPTURES_DIR,
        interval_seconds=0.5,
        extra_lua="\n".join(extra_lua_parts),
    )
    process = open_sfiii3n(
        extra_args=snapshot_loop.mame_args() + ["-skip_gameinfo"]
    )

    try:
        if arena_config is not None:
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
            while process.poll() is None:
                if all(not worker.is_alive() for worker in llm_workers):
                    raise RuntimeError("All LLM workers stopped unexpectedly.")
                if log_window is not None:
                    log_window.pump()
                time.sleep(0.5)
        else:
            process.wait()
    except KeyboardInterrupt:
        stop_event.set()
        process.terminate()
        process.wait()
    except Exception as exc:
        stop_event.set()
        if log_window is not None:
            log_window.log_status(f"Error: {exc}")
    finally:
        stop_event.set()
        snapshot_loop.cleanup()

    if log_window is not None:
        log_window.log_status("Game ended. Close this window to exit.")
        while not log_window.closed:
            log_window.pump()
            time.sleep(0.05)


if __name__ == "__main__":
    main()

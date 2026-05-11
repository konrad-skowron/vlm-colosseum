import csv
import json
from datetime import datetime, timezone
import os
from pathlib import Path
from subprocess import TimeoutExpired
import threading
import time

import agent_arena
from fight_starter import FightStartConfig, build_fight_start_lua
import llm_arena
from mame_launcher import open_sfiii3n
from log_viewer import SplitLogWindow
from screenshot_loop import create_snapshot_loop
from tensorboard_logger import TensorboardRunLogger


ENABLE_LLM_ARENA = True
ENABLE_LOG_WINDOW = True
ENABLE_TENSORBOARD = True
CAPTURES_DIR = "captures"
EXPERIMENT_MATCH_COUNT = 1
MATCH_MAX_SECONDS = 230.0
AI_PLAYERS = (1, 2)
FIGHT_MODE = "agent"
USE_ON_DEMAND_SCREENSHOTS = True
MAME_WINDOW_ARGS: list[str] = ["-resolution", "960x720"]
LLM_ROUND_START_BUFFER_SECONDS = 12.0
LLM_SCREENSHOT_WARMUP_UPDATES = 4
ELO_INITIAL_RATING = 1500.0
ELO_K_FACTOR = 32.0


def _select_arena_module():
    if FIGHT_MODE == "text":
        return llm_arena
    if FIGHT_MODE == "agent":
        return agent_arena
    raise ValueError(f"Unsupported FIGHT_MODE: {FIGHT_MODE}")


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
        "wins_p1",
        "wins_p2",
        "health_p1",
        "health_p2",
        "p1_actions",
        "p2_actions",
        "p1_avg_latency_ms",
        "p2_avg_latency_ms",
        "p1_hallucinations",
        "p2_hallucinations",
        "p1_estimated_damage",
        "p2_estimated_damage",
        "p1_estimated_hits",
        "p2_estimated_hits",
        "p1_estimated_hit_rate",
        "p2_estimated_hit_rate",
        "p1_elo_before",
        "p2_elo_before",
        "p1_elo_after",
        "p2_elo_after",
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


def _safe_float(value: object) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def _summarize_player_actions(rows: list[dict[str, str]], player_label: str) -> dict[str, str]:
    player_rows = [row for row in rows if row.get("player_label") == player_label]
    latencies = [
        latency
        for row in player_rows
        if (latency := _safe_float(row.get("latency_ms"))) is not None
    ]
    actions = len(player_rows)
    hits = sum(1 for row in player_rows if row.get("estimated_hit") == "true")
    damage = sum(_safe_int(row.get("estimated_opponent_damage")) for row in player_rows)
    hallucinations = sum(1 for row in player_rows if row.get("is_hallucination") == "true")
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    hit_rate = hits / actions if actions else 0.0
    return {
        "actions": str(actions),
        "avg_latency_ms": f"{avg_latency:.3f}",
        "hallucinations": str(hallucinations),
        "estimated_damage": str(damage),
        "estimated_hits": str(hits),
        "estimated_hit_rate": f"{hit_rate:.3f}",
    }


def _summarize_fight_log(match_dir: Path) -> dict[str, str]:
    log_path = match_dir / "fight_log.csv"
    if not log_path.exists():
        return {}

    with log_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    p1 = _summarize_player_actions(rows, "P1")
    p2 = _summarize_player_actions(rows, "P2")
    return {
        "p1_actions": p1["actions"],
        "p2_actions": p2["actions"],
        "p1_avg_latency_ms": p1["avg_latency_ms"],
        "p2_avg_latency_ms": p2["avg_latency_ms"],
        "p1_hallucinations": p1["hallucinations"],
        "p2_hallucinations": p2["hallucinations"],
        "p1_estimated_damage": p1["estimated_damage"],
        "p2_estimated_damage": p2["estimated_damage"],
        "p1_estimated_hits": p1["estimated_hits"],
        "p2_estimated_hits": p2["estimated_hits"],
        "p1_estimated_hit_rate": p1["estimated_hit_rate"],
        "p2_estimated_hit_rate": p2["estimated_hit_rate"],
    }


def _expected_elo_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + (10 ** ((rating_b - rating_a) / 400.0)))


def _apply_elo_update(
    *,
    ratings: dict[str, float],
    row: dict[str, str],
) -> None:
    model_p1 = row["model_p1"]
    model_p2 = row["model_p2"]
    ratings.setdefault(model_p1, ELO_INITIAL_RATING)
    ratings.setdefault(model_p2, ELO_INITIAL_RATING)

    p1_before = ratings[model_p1]
    p2_before = ratings[model_p2]
    row["p1_elo_before"] = f"{p1_before:.3f}"
    row["p2_elo_before"] = f"{p2_before:.3f}"

    if row["result"] == "P1":
        score_p1 = 1.0
    elif row["result"] == "P2":
        score_p1 = 0.0
    else:
        score_p1 = 0.5

    if model_p1 != model_p2 and row["result"] in {"P1", "P2", "draw_or_unknown"}:
        expected_p1 = _expected_elo_score(p1_before, p2_before)
        expected_p2 = 1.0 - expected_p1
        score_p2 = 1.0 - score_p1
        ratings[model_p1] = p1_before + ELO_K_FACTOR * (score_p1 - expected_p1)
        ratings[model_p2] = p2_before + ELO_K_FACTOR * (score_p2 - expected_p2)

    row["p1_elo_after"] = f"{ratings[model_p1]:.3f}"
    row["p2_elo_after"] = f"{ratings[model_p2]:.3f}"


def _write_elo_ratings(path: Path, ratings: dict[str, float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("model_name", "elo_rating"))
        writer.writeheader()
        for model_name, rating in sorted(ratings.items()):
            writer.writerow(
                {
                    "model_name": model_name,
                    "elo_rating": f"{rating:.3f}",
                }
            )


def _build_run_config(run_id: str, fight_start: FightStartConfig) -> dict[str, object]:
    return {
        "run_id": run_id,
        "fight_mode": FIGHT_MODE,
        "experiment_match_count": EXPERIMENT_MATCH_COUNT,
        "match_max_seconds": MATCH_MAX_SECONDS,
        "ai_players": list(AI_PLAYERS),
        "use_on_demand_screenshots": USE_ON_DEMAND_SCREENSHOTS,
        "mame_window_args": MAME_WINDOW_ARGS,
        "round_start_buffer_seconds": LLM_ROUND_START_BUFFER_SECONDS,
        "screenshot_warmup_updates": LLM_SCREENSHOT_WARMUP_UPDATES,
        "enable_log_window": ENABLE_LOG_WINDOW,
        "enable_tensorboard": ENABLE_TENSORBOARD,
        "elo_initial_rating": ELO_INITIAL_RATING,
        "elo_k_factor": ELO_K_FACTOR,
        "p1_character": fight_start.p1_character,
        "p2_character": fight_start.p2_character,
        "p1_super_art": fight_start.p1_super_art,
        "p2_super_art": fight_start.p2_super_art,
        "active_players": list(fight_start.active_players),
        "model_p1": os.environ.get("OPENROUTER_MODEL_P1", ""),
        "model_p2": os.environ.get("OPENROUTER_MODEL_P2", ""),
    }


def _run_single_match(
    *,
    match_index: int,
    fight_start: FightStartConfig,
    match_dir: Path,
    log_window: SplitLogWindow | None,
    tensorboard_logger: TensorboardRunLogger | None,
) -> dict[str, str]:
    stop_event = threading.Event()
    process = None
    snapshot_loop = None
    started_at = datetime.now(timezone.utc)
    match_started_perf = time.perf_counter()
    status = "unknown"
    result = "not_detected"
    arena_module = _select_arena_module()

    def emit_log(channel: str, message: str) -> None:
        if log_window is None:
            return
        log_window.log(channel, f"[match {match_index}] {message}")

    extra_lua_parts = [build_fight_start_lua(fight_start).strip()]

    arena_module.load_dotenv()
    arena_config = arena_module.build_arena_config(
        fight_start=fight_start,
        captures_dir=match_dir,
        round_start_buffer_seconds=LLM_ROUND_START_BUFFER_SECONDS,
        screenshot_warmup_updates=LLM_SCREENSHOT_WARMUP_UPDATES,
        snapshot_request_path=(
            match_dir / "snapshot_request.txt"
            if USE_ON_DEMAND_SCREENSHOTS
            else None
        ),
        ai_players=AI_PLAYERS,
    )
    arena_module.initialize_command_files(arena_config)
    state_path = arena_config.match_state_path
    extra_lua_parts.append(
        arena_module.build_move_bridge_lua(
            arena_config.command_path_p1,
            arena_config.command_path_p2,
        ).strip()
    )
    extra_lua_parts.append(arena_module.build_match_state_lua(state_path).strip())

    if log_window is not None:
        log_window.log_status(f"Starting match {match_index}.")
    print(
        f"Starting match {match_index}/{EXPERIMENT_MATCH_COUNT} "
        f"-> {match_dir}"
    )

    try:
        snapshot_loop = create_snapshot_loop(
            output_dir=match_dir,
            interval_seconds=0.5,
            extra_lua="\n".join(extra_lua_parts),
            on_demand=USE_ON_DEMAND_SCREENSHOTS,
        )
        process = open_sfiii3n(
            extra_args=snapshot_loop.mame_args() + MAME_WINDOW_ARGS + ["-skip_gameinfo"]
        )

        arena_module.wait_for_fight_start(arena_config, emit_log if log_window else None)
        if USE_ON_DEMAND_SCREENSHOTS:
            for warmup_index in range(arena_config.screenshot_warmup_updates):
                arena_module.request_fresh_screenshot(
                    screenshot_path=arena_config.screenshot_path,
                    request_path=arena_config.snapshot_request_path,
                )
                emit_log(
                    "status",
                    "Observed on-demand screenshot warmup update "
                    f"{warmup_index + 1}/{arena_config.screenshot_warmup_updates}.",
                )
        else:
            arena_module.wait_for_screenshot_warmup(
                arena_config.screenshot_path,
                arena_config.screenshot_warmup_updates,
                emit_log if log_window else None,
            )
        llm_workers = arena_module.start_llm_workers(
            arena_config,
            stop_event,
            emit_log if log_window else None,
            tensorboard_logger=tensorboard_logger,
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
    final_state = _read_match_state(state_path) or {}
    row = {
        "match_index": str(match_index),
        "started_at": started_at.isoformat(timespec="seconds"),
        "ended_at": ended_at.isoformat(timespec="seconds"),
        "duration_seconds": f"{duration_seconds:.3f}",
        "status": status,
        "result": result,
        "model_p1": arena_config.model_p1 or "non_ai_p1",
        "model_p2": arena_config.model_p2 or "non_ai_p2",
        "wins_p1": str(final_state.get("wins_p1", "")),
        "wins_p2": str(final_state.get("wins_p2", "")),
        "health_p1": str(final_state.get("health_p1", "")),
        "health_p2": str(final_state.get("health_p2", "")),
        "match_dir": str(match_dir),
    }
    row.update(_summarize_fight_log(match_dir))
    return row


def main() -> None:
    llm_arena.load_dotenv()
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    fight_start = FightStartConfig(
        p1_character='dudley',
        p2_character='dudley',
        p1_super_art=1,
        p2_super_art=1,
        active_players=AI_PLAYERS,
    )
    log_window = SplitLogWindow() if (ENABLE_LLM_ARENA and ENABLE_LOG_WINDOW) else None

    if not ENABLE_LLM_ARENA:
        raise RuntimeError("Batch experiments require ENABLE_LLM_ARENA = True.")

    captures_root = Path(CAPTURES_DIR) / run_id
    summary_path = captures_root / "experiment_summary.csv"
    elo_path = captures_root / "elo_ratings.csv"
    elo_ratings: dict[str, float] = {}
    captures_root.mkdir(parents=True, exist_ok=True)
    tensorboard_logger = TensorboardRunLogger(
        captures_root / "tensorboard",
        enabled=ENABLE_TENSORBOARD,
    )
    print(f"Starting experiment run: {captures_root}")
    print(f"Fight mode: {FIGHT_MODE}")
    print(f"TensorBoard: {tensorboard_logger.status_message}")
    if log_window is not None:
        log_window.log_status(f"Starting experiment run: {captures_root}")
        log_window.log_status(f"Fight mode: {FIGHT_MODE}")
        log_window.log_status(f"TensorBoard: {tensorboard_logger.status_message}")
    tensorboard_logger.log_run_config(_build_run_config(run_id, fight_start))

    try:
        for match_index in range(1, EXPERIMENT_MATCH_COUNT + 1):
            match_dir = captures_root / f"match_{match_index:03d}"
            row = _run_single_match(
                match_index=match_index,
                fight_start=fight_start,
                match_dir=match_dir,
                log_window=log_window,
                tensorboard_logger=tensorboard_logger,
            )
            _apply_elo_update(ratings=elo_ratings, row=row)
            _append_experiment_summary(summary_path=summary_path, row=row)
            _write_elo_ratings(elo_path, elo_ratings)
            tensorboard_logger.log_match_row(match_index, row)
            if log_window is not None:
                log_window.log_status(
                    f"Finished match {match_index}: "
                    f"{row['status']} ({row['duration_seconds']}s)."
                )
    except KeyboardInterrupt:
        if log_window is not None:
            log_window.log_status("Experiment interrupted.")
    finally:
        tensorboard_logger.close()
        if log_window is not None:
            log_window.log_status("Experiment finished. Close this window to exit.")

    if log_window is not None:
        while not log_window.closed:
            log_window.pump()
            time.sleep(0.05)


if __name__ == "__main__":
    main()

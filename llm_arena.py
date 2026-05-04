from __future__ import annotations

import json
import os
import re
import threading
import time
from base64 import b64encode
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib import error, request

from fight_starter import (
    FightStartConfig,
    estimate_fight_start_frame,
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
CAPTURES_DIR = Path("captures")
SCREENSHOT_PATH = CAPTURES_DIR / "latest_frame.png"
COMMAND_PATH_P1 = CAPTURES_DIR / "llm_moves_p1.txt"
COMMAND_PATH_P2 = CAPTURES_DIR / "llm_moves_p2.txt"
DEFAULT_FPS = 60.0
POLL_SECONDS = 1.5
REQUEST_TIMEOUT_SECONDS = 45.0
REQUEST_RETRY_SECONDS = 1.0
STEP_PRESS_FRAMES = 4
STEP_GAP_FRAMES = 1
MAX_STEPS = 4
MAX_TOKENS_PER_STEP = 3
MAX_HOLD_FRAMES = 60
SCREENSHOT_READ_RETRIES = 5
SCREENSHOT_READ_RETRY_SECONDS = 0.1

ALLOWED_MOVE_TOKENS = (
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "LP",
    "MP",
    "HP",
    "LK",
    "MK",
    "HK",
    "NONE",
)

TOKEN_TO_MAME_SUFFIX = {
    "UP": "JOYSTICK_UP",
    "DOWN": "JOYSTICK_DOWN",
    "LEFT": "JOYSTICK_LEFT",
    "RIGHT": "JOYSTICK_RIGHT",
    "LP": "BUTTON1",
    "MP": "BUTTON2",
    "HP": "BUTTON3",
    "LK": "BUTTON4",
    "MK": "BUTTON5",
    "HK": "BUTTON6",
}

SYSTEM_PROMPT = """You are controlling a player in Street Fighter III: 3rd Strike.
You will receive a single gameplay screenshot.
Return only JSON with this shape:
{"steps":[{"tokens":["TOKEN"],"hold_frames":4}],"summary":"short text"}

Rules:
- Allowed tokens: UP, DOWN, LEFT, RIGHT, LP, MP, HP, LK, MK, HK, NONE
- steps must be a list with 0 to 4 entries
- each step must contain 1 to 3 tokens
- each step must include hold_frames from 1 to 60
- simultaneous inputs go in the same step
- sequential inputs go in separate steps
- your objective is to win the round
- choose the move sequence you believe best improves your chance to win
- do not force a fixed style such as always attacking, always retreating, or always waiting
- use larger hold_frames when you want to keep holding a direction or button
- if no useful action is clear, return {"steps":[{"tokens":["NONE"],"hold_frames":4}],"summary":"..."}
- do not add markdown
"""

PLAYER_PROMPT_TEMPLATE = """You are Player {player_number}.
Choose the next short input sequence for Player {player_number} only.
Play to win using your own judgment.
Keep it brief and usually executable in under one second.
"""


@dataclass(slots=True)
class ArenaConfig:
    fight_start: FightStartConfig
    model_p1: str
    model_p2: str
    api_key: str
    round_start_buffer_seconds: float
    screenshot_warmup_updates: int
    screenshot_path: Path = SCREENSHOT_PATH
    command_path_p1: Path = COMMAND_PATH_P1
    command_path_p2: Path = COMMAND_PATH_P2
    captures_dir: Path = CAPTURES_DIR
    poll_seconds: float = POLL_SECONDS


@dataclass(slots=True)
class ParsedMove:
    steps: list["ParsedStep"]
    summary: str


@dataclass(slots=True)
class ParsedStep:
    tokens: list[str]
    hold_frames: int


LogFn = Callable[[str, str], None]


def load_dotenv(dotenv_path: Path = Path(".env")) -> None:
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _lua_path(path: Path) -> str:
    return path.resolve().as_posix()


def _emit_log(log_fn: LogFn | None, channel: str, message: str) -> None:
    if log_fn is not None:
        log_fn(channel, message)
    else:
        print(message)


def build_move_bridge_lua(command_path_p1: Path, command_path_p2: Path) -> str:
    lua_command_path_p1 = _lua_path(command_path_p1)
    lua_command_path_p2 = _lua_path(command_path_p2)
    return f"""
local llm_command_poll_mod = 6
local llm_step_press_frames = {STEP_PRESS_FRAMES}
local llm_step_gap_frames = {STEP_GAP_FRAMES}
local llm_frame_counter = 0

local llm_fields_by_token = {{}}
local llm_active_fields = {{}}
local llm_players = {{
    p1 = {{
        steps = {{}},
        step_index = nil,
        step_frame = 0,
        command_path = "{lua_command_path_p1}",
        last_command_id = nil,
        prefix = "P1_",
        label = "P1",
    }},
    p2 = {{
        steps = {{}},
        step_index = nil,
        step_frame = 0,
        command_path = "{lua_command_path_p2}",
        last_command_id = nil,
        prefix = "P2_",
        label = "P2",
    }},
}}

local function llm_split(input, separator)
    local result = {{}}
    for part in string.gmatch(input, "([^" .. separator .. "]+)") do
        table.insert(result, part)
    end
    return result
end

local function llm_find_field(token)
    if llm_fields_by_token[token] then
        return llm_fields_by_token[token]
    end

    local input_type, player = manager.machine.ioport:token_to_input_type(token)
    if input_type == nil then
        return nil
    end

    for _, port in pairs(manager.machine.ioport.ports) do
        for _, field in pairs(port.fields) do
            if field.type == input_type and field.player == player then
                llm_fields_by_token[token] = field
                return field
            end
        end
    end

    return nil
end

local function llm_parse_steps(raw_steps, prefix)
    local steps = {{}}
    if raw_steps == nil or raw_steps == "" or raw_steps == "NONE" then
        return steps
    end

    for _, raw_step in ipairs(llm_split(raw_steps, ";")) do
        local raw_step_tokens = raw_step
        local hold_frames = llm_step_press_frames
        local tokens_part, hold_part = raw_step:match("^(.-):(%d+)$")
        if tokens_part ~= nil then
            raw_step_tokens = tokens_part
            hold_frames = tonumber(hold_part) or llm_step_press_frames
        end
        local step_fields = {{}}
        for _, token in ipairs(llm_split(raw_step_tokens, "+")) do
            local trimmed = token:match("^%s*(.-)%s*$")
            if trimmed ~= "" and trimmed ~= "NONE" then
                local full_token = prefix .. trimmed
                local field = llm_find_field(full_token)
                if field then
                    table.insert(step_fields, {{ token = full_token, field = field }})
                else
                    emu.print_error("llm_move: missing input token " .. full_token)
                end
            end
        end
        if #step_fields > 0 then
            table.insert(steps, {{
                fields = step_fields,
                hold_frames = hold_frames,
            }})
        end
    end

    return steps
end

local function llm_load_player_steps(player_state, raw_steps)
    player_state.steps = llm_parse_steps(raw_steps, player_state.prefix)
    player_state.step_index = (#player_state.steps > 0) and 1 or nil
    player_state.step_frame = 0
end

local function llm_load_command(player_state, command_id, raw_steps)
    llm_load_player_steps(player_state, raw_steps)
    player_state.last_command_id = command_id
    emu.print_info(
        "llm_move: loaded " .. player_state.label .. " command " .. command_id
    )
end

local function llm_read_command_file(player_state)
    local file = io.open(player_state.command_path, "r")
    if not file then
        return
    end

    local command_id = file:read("*l")
    local raw_steps = file:read("*l") or "NONE"
    file:close()

    if command_id == nil or command_id == player_state.last_command_id then
        return
    end

    llm_load_command(player_state, command_id, raw_steps)
end

local function llm_process_player(player_state, next_active)
    if player_state.step_index == nil then
        return
    end

    local step = player_state.steps[player_state.step_index]
    if step == nil then
        player_state.step_index = nil
        player_state.step_frame = 0
        return
    end

    player_state.step_frame = player_state.step_frame + 1

    if player_state.step_frame <= step.hold_frames then
        for _, entry in ipairs(step.fields) do
            next_active[entry.token] = entry.field
        end
        return
    end

    if player_state.step_frame > (step.hold_frames + llm_step_gap_frames) then
        player_state.step_index = player_state.step_index + 1
        player_state.step_frame = 0
        if player_state.steps[player_state.step_index] == nil then
            player_state.step_index = nil
        end
    end
end

llm_move_subscription = emu.add_machine_frame_notifier(function ()
    if manager.machine.paused or manager.machine.exit_pending then
        return
    end

    llm_frame_counter = llm_frame_counter + 1
    if llm_frame_counter % llm_command_poll_mod == 0 then
        llm_read_command_file(llm_players.p1)
        llm_read_command_file(llm_players.p2)
    end

    local next_active = {{}}
    llm_process_player(llm_players.p1, next_active)
    llm_process_player(llm_players.p2, next_active)

    for token, field in pairs(next_active) do
        field:set_value(1)
    end

    for token, field in pairs(llm_active_fields) do
        if not next_active[token] then
            field:clear_value()
        end
    end

    llm_active_fields = next_active
end)
"""


def _extract_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("Model response JSON was not an object")
    return data


def _normalize_tokens(raw_tokens: Any) -> list[str]:
    if not isinstance(raw_tokens, list):
        raise ValueError("step tokens must be a list")

    normalized_tokens: list[str] = []
    for raw_token in raw_tokens[:MAX_TOKENS_PER_STEP]:
        if not isinstance(raw_token, str):
            raise ValueError("each token must be a string")
        token = raw_token.strip().upper().replace(" ", "_")
        if token not in ALLOWED_MOVE_TOKENS:
            raise ValueError(f"Unsupported move token: {token}")
        if token != "NONE":
            normalized_tokens.append(token)

    return normalized_tokens


def _normalize_hold_frames(raw_hold_frames: Any) -> int:
    if not isinstance(raw_hold_frames, int):
        raise ValueError("hold_frames must be an integer")
    if raw_hold_frames < 1 or raw_hold_frames > MAX_HOLD_FRAMES:
        raise ValueError(
            f"hold_frames must be between 1 and {MAX_HOLD_FRAMES}"
        )
    return raw_hold_frames


def _normalize_steps(raw_steps: Any) -> list[ParsedStep]:
    if not isinstance(raw_steps, list):
        raise ValueError("steps must be a list")

    normalized_steps: list[ParsedStep] = []
    for step in raw_steps[:MAX_STEPS]:
        if isinstance(step, list):
            normalized_tokens = _normalize_tokens(step)
            hold_frames = STEP_PRESS_FRAMES
        elif isinstance(step, dict):
            normalized_tokens = _normalize_tokens(step.get("tokens", []))
            hold_frames = _normalize_hold_frames(
                step.get("hold_frames", STEP_PRESS_FRAMES)
            )
        else:
            raise ValueError("each step must be a list or object")

        if normalized_tokens:
            normalized_steps.append(
                ParsedStep(
                    tokens=normalized_tokens,
                    hold_frames=hold_frames,
                )
            )

    return normalized_steps


def parse_model_move(response_text: str) -> ParsedMove:
    data = _extract_json_object(response_text)
    steps = _normalize_steps(data.get("steps", []))
    summary = data.get("summary", "")
    if not isinstance(summary, str):
        summary = ""
    return ParsedMove(steps=steps, summary=summary.strip())


def _steps_to_command_line(parsed_move: ParsedMove) -> str:
    if not parsed_move.steps:
        return "NONE"
    return ";".join(
        "+".join(TOKEN_TO_MAME_SUFFIX[token] for token in step.tokens)
        + f":{step.hold_frames}"
        for step in parsed_move.steps
    )


def _move_duration_seconds(parsed_move: ParsedMove) -> float:
    if not parsed_move.steps:
        return 0.0

    total_frames = sum(step.hold_frames for step in parsed_move.steps)
    total_frames += STEP_GAP_FRAMES * max(len(parsed_move.steps) - 1, 0)
    return total_frames / DEFAULT_FPS


def _read_screenshot_bytes(path: Path) -> bytes:
    last_error: OSError | None = None
    for _ in range(SCREENSHOT_READ_RETRIES):
        try:
            return path.read_bytes()
        except OSError as exc:
            last_error = exc
            time.sleep(SCREENSHOT_READ_RETRY_SECONDS)
    assert last_error is not None
    raise last_error


def _encode_image_as_data_url(path: Path) -> str:
    image_bytes = _read_screenshot_bytes(path)
    encoded = b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def call_openrouter_model(
    *,
    api_key: str,
    model: str,
    screenshot_path: Path,
    player_number: int,
) -> ParsedMove:
    screenshot_url = _encode_image_as_data_url(screenshot_path)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PLAYER_PROMPT_TEMPLATE.format(
                            player_number=player_number
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": screenshot_url},
                    },
                ],
            },
        ],
        "temperature": 0.2,
        "max_tokens": 200,
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        OPENROUTER_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenRouter request failed with HTTP {exc.code}: {error_body}"
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

    payload_data = json.loads(body)
    raw_content = (
        payload_data["choices"][0]["message"]["content"]
        if payload_data.get("choices")
        else ""
    )
    if not isinstance(raw_content, str):
        raise RuntimeError("OpenRouter response content was not text")
    return parse_model_move(raw_content)


def write_player_command_file(
    command_path: Path,
    *,
    command_id: int,
    player_move: ParsedMove,
) -> None:
    command_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = command_path.with_suffix(".tmp")
    contents = "\n".join(
        [
            str(command_id),
            _steps_to_command_line(player_move),
        ]
    )
    temp_path.write_text(contents, encoding="utf-8")
    temp_path.replace(command_path)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def build_arena_config(
    *,
    fight_start: FightStartConfig | None = None,
    captures_dir: Path = CAPTURES_DIR,
    round_start_buffer_seconds: float = 6.0,
    screenshot_warmup_updates: int = 3,
) -> ArenaConfig:
    return ArenaConfig(
        fight_start=fight_start or FightStartConfig(),
        model_p1=_require_env("OPENROUTER_MODEL_P1"),
        model_p2=_require_env("OPENROUTER_MODEL_P2"),
        api_key=_require_env("OPENROUTER_API_KEY"),
        round_start_buffer_seconds=round_start_buffer_seconds,
        screenshot_warmup_updates=screenshot_warmup_updates,
        captures_dir=captures_dir,
        screenshot_path=captures_dir / "latest_frame.png",
        command_path_p1=captures_dir / "llm_moves_p1.txt",
        command_path_p2=captures_dir / "llm_moves_p2.txt",
    )


def initialize_command_files(config: ArenaConfig) -> None:
    write_player_command_file(
        config.command_path_p1,
        command_id=0,
        player_move=ParsedMove(steps=[], summary=""),
    )
    write_player_command_file(
        config.command_path_p2,
        command_id=0,
        player_move=ParsedMove(steps=[], summary=""),
    )


def wait_for_fight_start(config: ArenaConfig, log_fn: LogFn | None = None) -> None:
    final_frame = estimate_fight_start_frame(config.fight_start)
    delay_seconds = (
        final_frame / DEFAULT_FPS
    ) + config.round_start_buffer_seconds
    _emit_log(
        log_fn,
        "status",
        f"Waiting {delay_seconds:.1f}s for fight start sequence to finish.",
    )
    time.sleep(delay_seconds)


def wait_for_screenshot_exists(path: Path) -> None:
    while True:
        if path.exists():
            return
        time.sleep(0.1)


def wait_for_screenshot_warmup(
    path: Path,
    update_count: int,
    log_fn: LogFn | None = None,
) -> None:
    if update_count <= 0:
        return

    wait_for_screenshot_exists(path)
    last_mtime_ns = path.stat().st_mtime_ns
    seen_updates = 0

    _emit_log(
        log_fn,
        "status",
        "Waiting for "
        f"{update_count} fresh screenshot update(s) before starting LLM workers.",
    )

    while seen_updates < update_count:
        time.sleep(0.1)
        try:
            current_mtime_ns = path.stat().st_mtime_ns
        except FileNotFoundError:
            continue

        if current_mtime_ns != last_mtime_ns:
            last_mtime_ns = current_mtime_ns
            seen_updates += 1
            _emit_log(
                log_fn,
                "status",
                "Observed screenshot warmup update "
                f"{seen_updates}/{update_count}.",
            )


def llm_worker(
    *,
    stop_event: threading.Event,
    api_key: str,
    model: str,
    screenshot_path: Path,
    command_path: Path,
    player_number: int,
    poll_seconds: float,
    log_fn: LogFn | None = None,
) -> None:
    command_id = 0
    channel = f"p{player_number}"
    _emit_log(log_fn, channel, f"worker starting for model {model}")
    wait_for_screenshot_exists(screenshot_path)
    _emit_log(
        log_fn,
        channel,
        f"worker found screenshot source at {screenshot_path}",
    )

    while not stop_event.is_set():
        try:
            _emit_log(
                log_fn,
                channel,
                f"requesting next move from {model}",
            )
            player_move = call_openrouter_model(
                api_key=api_key,
                model=model,
                screenshot_path=screenshot_path,
                player_number=player_number,
            )
            _emit_log(
                log_fn,
                channel,
                f"{model}: {_steps_to_command_line(player_move)}"
                + (f" | {player_move.summary}" if player_move.summary else ""),
            )

            command_id += 1
            write_player_command_file(
                command_path,
                command_id=command_id,
                player_move=player_move,
            )
            _emit_log(
                log_fn,
                channel,
                f"Wrote command {command_id} to {command_path}",
            )
        except Exception as exc:
            _emit_log(log_fn, channel, f"worker error: {exc}")
            if stop_event.wait(REQUEST_RETRY_SECONDS):
                return
            continue

        wait_seconds = max(
            poll_seconds,
            _move_duration_seconds(player_move),
        )
        if stop_event.wait(wait_seconds):
            return


def start_llm_workers(
    config: ArenaConfig,
    stop_event: threading.Event,
    log_fn: LogFn | None = None,
) -> list[threading.Thread]:
    workers = [
        threading.Thread(
            target=llm_worker,
            kwargs={
                "stop_event": stop_event,
                "api_key": config.api_key,
                "model": config.model_p1,
                "screenshot_path": config.screenshot_path,
                "command_path": config.command_path_p1,
                "player_number": 1,
                "poll_seconds": config.poll_seconds,
                "log_fn": log_fn,
            },
            daemon=True,
            name="llm-p1",
        ),
        threading.Thread(
            target=llm_worker,
            kwargs={
                "stop_event": stop_event,
                "api_key": config.api_key,
                "model": config.model_p2,
                "screenshot_path": config.screenshot_path,
                "command_path": config.command_path_p2,
                "player_number": 2,
                "poll_seconds": config.poll_seconds,
                "log_fn": log_fn,
            },
            daemon=True,
            name="llm-p2",
        ),
    ]

    for worker in workers:
        worker.start()

    return workers

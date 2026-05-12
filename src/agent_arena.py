from __future__ import annotations

import json
from collections import deque
from typing import Any
from urllib import error, request

from src.llm_arena import (
    ALLOWED_MOVE_TOKENS,
    OPENROUTER_URL,
    PLAYER_PROMPT_TEMPLATE,
    REQUEST_RETRY_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    ArenaConfig,
    ExperimentLogger,
    FightStartConfig,
    LogFn,
    ModelCallResult,
    ParsedMove,
    ParsedStep,
    _emit_log,
    _fallback_move,
    _normalize_summary,
    _move_duration_seconds,
    _read_screenshot_bytes,
    _steps_to_command_line,
    _super_art_context,
    build_arena_config,
    build_match_state_lua,
    build_move_bridge_lua,
    format_decision_details,
    initialize_command_files,
    load_dotenv,
    read_match_state,
    request_fresh_screenshot,
    wait_for_fight_start,
    wait_for_screenshot_exists,
    wait_for_screenshot_warmup,
    write_player_command_file,
)
from base64 import b64encode
import threading
import time


AGENT_SYSTEM_PROMPT = """You control one player in Street Fighter III: 3rd Strike.
You receive one gameplay screenshot and must choose the next controller input sequence using tools only.
Track your own character by Player number and visual identity, especially character color, not by screen side alone. Characters can switch sides after jumps, throws, cross-ups, or Super Arts.
The game continues in real time while you are deciding. It does not pause waiting for your response, so the state may change before your move is executed.

Controller semantics:
- UP, DOWN, LEFT, RIGHT are joystick directions.
- LP=light punch, MP=medium punch, HP=heavy punch.
- LK=light kick, MK=medium kick, HK=heavy kick.
- The bottom meter is the Super Art meter. When it shows MAX, a Super Art is available.
- Throw is LP+LK when close.
- Universal overhead is MP+MK.
- EX special moves use two punch buttons or two kick buttons and consume Super Art meter.
- Dash is two quick forward inputs; backdash is two quick back inputs.
- High parry is a quick forward tap; low parry is a quick DOWN tap.
- Charge moves require holding a charge direction before the release direction/button.

Tool usage rules:
- Use only tool calls. Do not answer with normal text.
- Also call set_reason once with one very short summary of your immediate intent.
- Build a sequence of 0 to 16 steps.
- Tools that share the same step_index happen at the same time.
- Increasing step_index values happen in order.
- hold_frames must be between 1 and 60.
- LEFT and RIGHT are physical joystick directions, not semantic forward/back.
- Never use placeholder values such as FORWARD or BACK.
- After a side switch, re-identify which visible character is yours before choosing directions.
- If no useful action is clear, call the no_input tool.
- Keep the summary to at most 12 words.

Choose whatever tactic and input sequence you judge most likely to win the round.
"""

TOOL_NAME_TO_TOKEN = {
    "press_up": "UP",
    "press_down": "DOWN",
    "press_left": "LEFT",
    "press_right": "RIGHT",
    "press_lp": "LP",
    "press_mp": "MP",
    "press_hp": "HP",
    "press_lk": "LK",
    "press_mk": "MK",
    "press_hk": "HK",
}

BUTTON_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": f"Press {token} during one step of the input sequence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "step_index": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 16,
                        "description": "1-based step number. Same step_index means simultaneous input.",
                    },
                    "hold_frames": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 60,
                        "description": "How many frames to hold this step.",
                    },
                },
                "required": ["step_index", "hold_frames"],
                "additionalProperties": False,
            },
        },
    }
    for tool_name, token in TOOL_NAME_TO_TOKEN.items()
]

NO_INPUT_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "no_input",
        "description": "Choose no useful action for one step.",
        "parameters": {
            "type": "object",
            "properties": {
                "step_index": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 16,
                },
                "hold_frames": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 60,
                },
            },
            "required": ["step_index", "hold_frames"],
            "additionalProperties": False,
        },
    },
}

REASON_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "set_reason",
        "description": "Provide one very short summary of the immediate intent behind this action sequence.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Very short reason, at most 12 words.",
                },
            },
            "required": ["summary"],
            "additionalProperties": False,
        },
    },
}

AGENT_TOOLS = BUTTON_TOOL_DEFINITIONS + [NO_INPUT_TOOL_DEFINITION, REASON_TOOL_DEFINITION]


def _encode_image_as_data_url(path) -> str:
    image_bytes = _read_screenshot_bytes(path)
    encoded = b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _normalize_tool_int(value: Any, *, min_value: int, max_value: int) -> int:
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if not isinstance(value, int):
        raise ValueError("tool argument must be an integer")
    if value < min_value or value > max_value:
        raise ValueError(f"tool argument must be between {min_value} and {max_value}")
    return value


def _tool_trace(tool_calls: list[dict[str, Any]]) -> str:
    entries: list[str] = []
    for tool_call in tool_calls:
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        tool_name = function.get("name")
        raw_arguments = function.get("arguments", "{}")
        if not isinstance(tool_name, str):
            continue
        if tool_name == "set_reason":
            continue
        try:
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else {}
        except json.JSONDecodeError:
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        step_index = arguments.get("step_index", "?")
        hold_frames = arguments.get("hold_frames", "?")
        entries.append(f"{tool_name}(step={step_index},hold={hold_frames})")
    return ", ".join(entries)


def _extract_message_summary(message: dict[str, Any]) -> str:
    raw_content = message.get("content")
    if isinstance(raw_content, str):
        return _normalize_summary(raw_content)
    if isinstance(raw_content, list):
        fragments: list[str] = []
        for item in raw_content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                fragments.append(text.strip())
        return _normalize_summary(" ".join(fragments))
    return ""


def _extract_reason_from_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    for tool_call in tool_calls:
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        if function.get("name") != "set_reason":
            continue
        raw_arguments = function.get("arguments", "{}")
        if not isinstance(raw_arguments, str):
            continue
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            continue
        if not isinstance(arguments, dict):
            continue
        return _normalize_summary(arguments.get("summary", ""))
    return ""


def _parse_tool_move(tool_calls: list[dict[str, Any]]) -> ParsedMove:
    steps_by_index: dict[int, ParsedStep] = {}

    for tool_call in tool_calls:
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue

        tool_name = function.get("name")
        raw_arguments = function.get("arguments", "{}")
        if not isinstance(tool_name, str):
            continue
        if not isinstance(raw_arguments, str):
            raise ValueError("tool arguments must be a JSON string")

        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid tool arguments JSON") from exc
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must decode to an object")

        if tool_name == "set_reason":
            continue

        step_index = _normalize_tool_int(
            arguments.get("step_index"),
            min_value=1,
            max_value=16,
        )
        hold_frames = _normalize_tool_int(
            arguments.get("hold_frames"),
            min_value=1,
            max_value=60,
        )

        step = steps_by_index.setdefault(
            step_index,
            ParsedStep(tokens=[], hold_frames=hold_frames),
        )
        step.hold_frames = max(step.hold_frames, hold_frames)

        if tool_name == "no_input":
            continue

        token = TOOL_NAME_TO_TOKEN.get(tool_name)
        if token is None or token not in ALLOWED_MOVE_TOKENS:
            raise ValueError(f"unsupported tool: {tool_name}")
        if token not in step.tokens and len(step.tokens) < 3:
            step.tokens.append(token)

    ordered_steps = [
        steps_by_index[index]
        for index in sorted(steps_by_index)
        if steps_by_index[index].tokens
    ]
    return ParsedMove(steps=ordered_steps, summary="", trace=_tool_trace(tool_calls))


def call_openrouter_model(
    *,
    api_key: str,
    model: str,
    screenshot_path,
    snapshot_request_path,
    snapshot_request_lock=None,
    player_number: int,
    fight_start: FightStartConfig,
    action_history=None,
) -> ModelCallResult:
    request_fresh_screenshot(
        screenshot_path=screenshot_path,
        request_path=snapshot_request_path,
        snapshot_request_lock=snapshot_request_lock,
    )
    screenshot_url = _encode_image_as_data_url(screenshot_path)
    player_prompt = PLAYER_PROMPT_TEMPLATE.format(player_number=player_number)
    player_prompt += _super_art_context(fight_start, player_number)
    if action_history:
        history = ", ".join(action_history)
        player_prompt += (
            "\nYour recent actions (oldest to newest): "
            f"[{history}]"
        )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": player_prompt},
                    {"type": "image_url", "image_url": {"url": screenshot_url}},
                ],
            },
        ],
        "tools": AGENT_TOOLS,
        "tool_choice": "auto",
        "temperature": 0.2,
        "max_tokens": 300,
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

    request_start = time.perf_counter()
    try:
        with request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenRouter tool request failed with HTTP {exc.code}: {error_body}"
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(f"OpenRouter tool request failed: {exc}") from exc

    latency_ms = (time.perf_counter() - request_start) * 1000.0

    try:
        payload_data = json.loads(body)
        message = payload_data["choices"][0]["message"]
        tool_calls = message["tool_calls"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return ModelCallResult(
            player_move=_fallback_move(),
            latency_ms=latency_ms,
            is_hallucination=True,
        )

    if not isinstance(tool_calls, list) or not tool_calls:
        return ModelCallResult(
            player_move=_fallback_move(),
            latency_ms=latency_ms,
            is_hallucination=True,
        )

    try:
        player_move = _parse_tool_move(tool_calls)
    except ValueError:
        return ModelCallResult(
            player_move=_fallback_move(),
            latency_ms=latency_ms,
            is_hallucination=True,
        )

    player_move.summary = (
        _extract_reason_from_tool_calls(tool_calls)
        or _extract_message_summary(message)
    )

    return ModelCallResult(
        player_move=player_move,
        latency_ms=latency_ms,
        is_hallucination=False,
    )


def llm_worker(
    *,
    stop_event: threading.Event,
    api_key: str,
    model: str,
    screenshot_path,
    snapshot_request_path,
    snapshot_request_lock=None,
    match_state_path,
    command_path,
    player_number: int,
    poll_seconds: float,
    fight_start: FightStartConfig,
    use_action_history: bool = False,
    experiment_logger: ExperimentLogger | None = None,
    log_fn: LogFn | None = None,
) -> None:
    command_id = 0
    channel = f"p{player_number}"
    player_label = f"P{player_number}"
    recent_actions: deque[str] = deque(maxlen=5)
    _emit_log(log_fn, channel, f"agent worker starting for model {model}")
    wait_for_screenshot_exists(screenshot_path)
    _emit_log(
        log_fn,
        channel,
        f"agent worker found screenshot source at {screenshot_path}",
    )

    while not stop_event.is_set():
        try:
            _emit_log(
                log_fn,
                channel,
                f"requesting next tool sequence from {model}",
            )
            result = call_openrouter_model(
                api_key=api_key,
                model=model,
                screenshot_path=screenshot_path,
                snapshot_request_path=snapshot_request_path,
                snapshot_request_lock=snapshot_request_lock,
                player_number=player_number,
                fight_start=fight_start,
                action_history=list(recent_actions) if use_action_history else None,
            )
            player_move = result.player_move
            parsed_action = _steps_to_command_line(player_move)
            state_before = read_match_state(match_state_path)
            decision_details = format_decision_details(player_move)
            _emit_log(
                log_fn,
                channel,
                f"{model}: {parsed_action}"
                + f" | latency={result.latency_ms:.1f}ms"
                + (" | hallucination" if result.is_hallucination else "")
                + (f" | {decision_details}" if decision_details else ""),
            )
            if use_action_history:
                recent_actions.append(parsed_action)

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
            wait_seconds = max(
                poll_seconds,
                _move_duration_seconds(player_move),
            )
            was_stopped = stop_event.wait(wait_seconds)
            state_after = read_match_state(match_state_path)
            if experiment_logger is not None:
                experiment_logger.log_action(
                    player_label=player_label,
                    model_name=model,
                    command_id=command_id,
                    parsed_action=parsed_action,
                    decision_details=decision_details,
                    latency_ms=result.latency_ms,
                    is_hallucination=result.is_hallucination,
                    state_before=state_before,
                    state_after=state_after,
                )
            if was_stopped:
                return
        except Exception as exc:
            _emit_log(log_fn, channel, f"agent worker error: {exc}")
            if experiment_logger is not None:
                experiment_logger.log_action(
                    player_label=player_label,
                    model_name=model,
                    command_id=command_id,
                    parsed_action="NONE",
                    decision_details="",
                    latency_ms=0.0,
                    is_hallucination=True,
                )
            if stop_event.wait(REQUEST_RETRY_SECONDS):
                return
            continue


def start_llm_workers(
    config: ArenaConfig,
    stop_event: threading.Event,
    log_fn: LogFn | None = None,
    tensorboard_logger=None,
) -> list[threading.Thread]:
    experiment_logger = ExperimentLogger(
        config.captures_dir / "fight_log.csv",
        tensorboard_logger=tensorboard_logger,
    )
    workers: list[threading.Thread] = []

    if 1 in config.ai_players:
        workers.append(threading.Thread(
            target=llm_worker,
            kwargs={
                "stop_event": stop_event,
                "api_key": config.api_key,
                "model": config.model_p1,
                "screenshot_path": config.screenshot_path,
                "snapshot_request_path": config.snapshot_request_path,
                "snapshot_request_lock": config.snapshot_request_lock,
                "match_state_path": config.match_state_path,
                "command_path": config.command_path_p1,
                "player_number": 1,
                "poll_seconds": config.poll_seconds,
                "fight_start": config.fight_start,
                "use_action_history": config.use_action_history,
                "experiment_logger": experiment_logger,
                "log_fn": log_fn,
            },
            daemon=True,
            name="agent-p1",
        ))

    if 2 in config.ai_players:
        workers.append(threading.Thread(
            target=llm_worker,
            kwargs={
                "stop_event": stop_event,
                "api_key": config.api_key,
                "model": config.model_p2,
                "screenshot_path": config.screenshot_path,
                "snapshot_request_path": config.snapshot_request_path,
                "snapshot_request_lock": config.snapshot_request_lock,
                "match_state_path": config.match_state_path,
                "command_path": config.command_path_p2,
                "player_number": 2,
                "poll_seconds": config.poll_seconds,
                "fight_start": config.fight_start,
                "use_action_history": config.use_action_history,
                "experiment_logger": experiment_logger,
                "log_fn": log_fn,
            },
            daemon=True,
            name="agent-p2",
        ))

    for worker in workers:
        worker.start()

    return workers

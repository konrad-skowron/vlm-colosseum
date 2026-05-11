from __future__ import annotations

import csv
import json
import os
import re
import threading
import time
from collections import deque
from base64 import b64encode
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence
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
FIGHT_LOG_PATH = CAPTURES_DIR / "fight_log.csv"
MATCH_STATE_PATH = CAPTURES_DIR / "match_state.json"
SNAPSHOT_REQUEST_PATH = CAPTURES_DIR / "snapshot_request.txt"
DEFAULT_FPS = 60.0
POLL_SECONDS = 1.5
REQUEST_TIMEOUT_SECONDS = 45.0
REQUEST_RETRY_SECONDS = 1.0
SNAPSHOT_REQUEST_TIMEOUT_SECONDS = 2.0
STEP_PRESS_FRAMES = 4
STEP_GAP_FRAMES = 1
MAX_STEPS = 16
MAX_TOKENS_PER_STEP = 3
MAX_HOLD_FRAMES = 60
SCREENSHOT_READ_RETRIES = 20
SCREENSHOT_READ_RETRY_SECONDS = 0.05
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_IEND_CHUNK = b"\x00\x00\x00\x00IEND\xaeB`\x82"
_SNAPSHOT_REQUEST_LOCK = threading.Lock()

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

TOKEN_ALIASES = {
    "LIGHT_PUNCH": "LP",
    "MEDIUM_PUNCH": "MP",
    "HEAVY_PUNCH": "HP",
    "LIGHT_KICK": "LK",
    "MEDIUM_KICK": "MK",
    "HEAVY_KICK": "HK",
    "PUNCH": "HP",
    "KICK": "HK",
    "NO_INPUT": "NONE",
    "WAIT": "NONE",
}

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

SFIII3N_MEMORY_ADDRESSES = {
    "fighting": 0x0200EE44,
    "wins_p1": 0x02011383,
    "wins_p2": 0x02011385,
    "health_p1": 0x02068D0B,
    "health_p2": 0x020691A3,
}

SemanticStep = tuple[str, ...]


def _qcf(final_button: str) -> tuple[SemanticStep, ...]:
    return (
        ("DOWN",),
        ("DOWN", "F"),
        ("F",),
        ("DOWN",),
        ("DOWN", "F"),
        ("F", final_button),
    )


def _qcb(final_button: str) -> tuple[SemanticStep, ...]:
    return (
        ("DOWN",),
        ("DOWN", "B"),
        ("B",),
        ("DOWN",),
        ("DOWN", "B"),
        ("B", final_button),
    )


def _qcf_then(final_button: str, followup_button: str) -> tuple[SemanticStep, ...]:
    return _qcf(final_button) + ((followup_button,),)


def _qcf_then_qcf(final_button: str, followup_button: str) -> tuple[SemanticStep, ...]:
    return _qcf(final_button) + (
        ("DOWN",),
        ("DOWN", "F"),
        ("F", followup_button),
    )


def _triple_down(final_buttons: tuple[str, ...]) -> tuple[SemanticStep, ...]:
    return (
        ("DOWN",),
        ("DOWN",),
        ("DOWN",) + final_buttons,
    )


def _shun_goku_satsu() -> tuple[SemanticStep, ...]:
    return (
        ("LP",),
        ("LP",),
        ("F",),
        ("LK",),
        ("HP",),
    )


def _circle(final_button: str) -> tuple[SemanticStep, ...]:
    return (
        ("F",),
        ("DOWN", "F"),
        ("DOWN",),
        ("DOWN", "B"),
        ("B",),
        ("UP", "B"),
        ("UP",),
        ("UP", "F", final_button),
    )


def _double_circle(final_button: str) -> tuple[SemanticStep, ...]:
    return _circle("")[:-1] + (
        ("UP", "F"),
        ("F",),
        ("DOWN", "F"),
        ("DOWN",),
        ("DOWN", "B"),
        ("B",),
        ("UP", "B"),
        ("UP",),
        ("UP", "F", final_button),
    )


SUPER_ART_GUIDE = {
    "alex": {
        1: {
            "name": "Hyper Bomb",
            "motion": "360 + punch, close range grab",
            "steps": _circle("HP"),
        },
        2: {
            "name": "Boomerang Raid",
            "motion": "qcf, qcf + punch",
            "steps": _qcf("HP"),
        },
        3: {
            "name": "Stun Gun Headbutt",
            "motion": "qcf, qcf + punch",
            "steps": _qcf("HP"),
        },
    },
    "akuma": {
        1: {
            "name": "Messatsu Gou Hadou",
            "motion": "qcf, qcf + punch, air or ground",
            "steps": _qcf("HP"),
            "extra": "Max meter extra options: Shun Goku Satsu = LP; LP; forward; LK; HP, Kongou Kokuretsu Zan = DOWN; DOWN; DOWN+LP+MP+HP.",
        },
        2: {
            "name": "Messatsu Gou Shoryu",
            "motion": "qcf, qcf + punch",
            "steps": _qcf("HP"),
            "extra": "Max meter extra options: Shun Goku Satsu = LP; LP; forward; LK; HP, Kongou Kokuretsu Zan = DOWN; DOWN; DOWN+LP+MP+HP.",
        },
        3: {
            "name": "Messatsu Gou Rasen",
            "motion": "qcf, qcf + kick, air or ground",
            "steps": _qcf("HK"),
            "extra": "Max meter extra options: Shun Goku Satsu = LP; LP; forward; LK; HP, Kongou Kokuretsu Zan = DOWN; DOWN; DOWN+LP+MP+HP.",
        },
    },
    "chun_li": {
        1: {"name": "Kikou Shou", "motion": "qcf, qcf + punch", "steps": _qcf("HP")},
        2: {"name": "Houyoku Sen", "motion": "qcf, qcf + kick", "steps": _qcf("HK")},
        3: {"name": "Tensei Ranka", "motion": "qcf, qcf + kick", "steps": _qcf("HK")},
    },
    "dudley": {
        1: {
            "name": "Rocket Uppercut",
            "motion": "qcf, qcf + punch",
            "steps": _qcf("HP"),
        },
        2: {
            "name": "Rolling Thunder",
            "motion": "qcf, qcf + punch repeatedly",
            "steps": _qcf("HP"),
            "extra": "Press punch repeatedly after activation if continuing the Super Art is useful.",
        },
        3: {
            "name": "Corkscrew Blow",
            "motion": "qcf, qcf + punch",
            "steps": _qcf("HP"),
        },
    },
    "elena": {
        1: {"name": "Spinning Beat", "motion": "qcf, qcf + kick", "steps": _qcf("HK")},
        2: {"name": "Brave Dance", "motion": "qcf, qcf + kick", "steps": _qcf("HK")},
        3: {
            "name": "Healing",
            "motion": "qcf, qcf + hold punch",
            "steps": _qcf("HP"),
            "extra": "Hold the punch button longer to continue healing when safe.",
        },
    },
    "hugo": {
        1: {
            "name": "Gigas Breaker",
            "motion": "720 + punch, close range grab",
            "steps": _double_circle("HP"),
        },
        2: {"name": "Megaton Press", "motion": "qcf, qcf + kick", "steps": _qcf("HK")},
        3: {
            "name": "Hammer Frenzy",
            "motion": "qcf, qcf + hold punch",
            "steps": _qcf("HP"),
        },
    },
    "ibuki": {
        1: {
            "name": "Kasumi Suzaku",
            "motion": "qcf, qcf + punch repeatedly, while jumping",
            "steps": _qcf("HP"),
            "extra": "This Super Art must be done while airborne.",
        },
        2: {"name": "Yoroi Doushi", "motion": "qcf, qcf + punch", "steps": _qcf("HP")},
        3: {"name": "Yami Shigure", "motion": "qcf, qcf + punch", "steps": _qcf("HP")},
    },
    "ken": {
        1: {"name": "Shoryu Reppa", "motion": "qcf, qcf + punch", "steps": _qcf("HP")},
        2: {
            "name": "Shinryu Ken",
            "motion": "qcf, qcf + kick repeatedly",
            "steps": _qcf("HK"),
            "extra": "Press kick repeatedly after activation if useful.",
        },
        3: {"name": "Shippu Jinrai Kyaku", "motion": "qcf, qcf + kick", "steps": _qcf("HK")},
    },
    "makoto": {
        1: {"name": "Seichusen Godanzuki", "motion": "qcf, qcf + punch", "steps": _qcf("HP")},
        2: {"name": "Abare Tosanami", "motion": "qcf, qcf + kick", "steps": _qcf("HK")},
        3: {"name": "Tanden Renki", "motion": "qcf, qcf + punch", "steps": _qcf("HP")},
    },
    "necro": {
        1: {
            "name": "Magnetic Storm",
            "motion": "qcf, qcf + punch repeatedly",
            "steps": _qcf("HP"),
            "extra": "Press punch repeatedly after activation if useful.",
        },
        2: {"name": "Slam Dance", "motion": "qcf, qcf + punch, close range grab", "steps": _qcf("HP")},
        3: {"name": "Electric Snake", "motion": "qcf, qcf + punch", "steps": _qcf("HP")},
    },
    "oro": {
        1: {
            "name": "Kishin Riki",
            "motion": "qcf, qcf + punch, then punch, close range grab",
            "steps": _qcf_then("HP", "HP"),
            "extra": "At max meter, EX Kishin Riki can use qcf, qcf + LP+MP+HP.",
        },
        2: {
            "name": "Yagyou Dama",
            "motion": "qcf, qcf + punch",
            "steps": _qcf("HP"),
            "extra": "At max meter, EX Yagyou Dama can use qcf, qcf + LP+MP+HP.",
        },
        3: {
            "name": "Tengu Stone",
            "motion": "qcf, qcf + punch",
            "steps": _qcf("HP"),
            "extra": "At max meter, EX Tengu Stone can use qcf, qcf + LP+MP+HP.",
        },
    },
    "q": {
        1: {"name": "Critical Combo Attack", "motion": "qcf, qcf + punch", "steps": _qcf("HP")},
        2: {"name": "Deadly Double Combination", "motion": "qcf, qcf + punch", "steps": _qcf("HP")},
        3: {
            "name": "Total Destruction",
            "motion": "qcf, qcf + punch, then qcf + punch or kick",
            "steps": _qcf_then_qcf("HP", "HP"),
        },
    },
    "remy": {
        1: {"name": "Light of Justice", "motion": "qcf, qcf + punch", "steps": _qcf("HP")},
        2: {"name": "Supreme Rising Rage Flash", "motion": "qcf, qcf + kick", "steps": _qcf("HK")},
        3: {"name": "Blue Nocturne", "motion": "qcf, qcf + kick", "steps": _qcf("HK")},
    },
    "ryu": {
        1: {"name": "Shinkuu Hadouken", "motion": "qcf, qcf + punch", "steps": _qcf("HP")},
        2: {"name": "Shin Shoryuken", "motion": "qcf, qcf + punch", "steps": _qcf("HP")},
        3: {
            "name": "Denjin Hadouken",
            "motion": "qcf, qcf + hold punch",
            "steps": _qcf("HP"),
            "extra": "Hold punch longer to charge Denjin when safe.",
        },
    },
    "sean": {
        1: {"name": "Hado Burst", "motion": "qcf, qcf + punch", "steps": _qcf("HP")},
        2: {
            "name": "Shoryu Cannon",
            "motion": "qcf, qcf + punch repeatedly",
            "steps": _qcf("HP"),
            "extra": "Press punch repeatedly after activation if useful.",
        },
        3: {"name": "Hyper Tornado", "motion": "qcf, qcf + punch", "steps": _qcf("HP")},
    },
    "twelve": {
        1: {"name": "X.N.D.L.", "motion": "qcf, qcf + punch", "steps": _qcf("HP")},
        2: {
            "name": "X.F.L.A.T.",
            "motion": "qcf, qcf + kick, while jumping",
            "steps": _qcf("HK"),
            "extra": "This Super Art must be done while airborne.",
        },
        3: {"name": "X.C.O.P.Y.", "motion": "qcf, qcf + punch", "steps": _qcf("HP")},
    },
    "urien": {
        1: {"name": "Tyrant Slaughter", "motion": "qcf, qcf + punch", "steps": _qcf("HP")},
        2: {"name": "Temporal Thunder", "motion": "qcf, qcf + punch", "steps": _qcf("HP")},
        3: {
            "name": "Aegis Reflector",
            "motion": "qcf, qcf + punch or LP+MP+HP",
            "steps": _qcf("HP"),
            "extra": "LP+MP+HP version is available if a three-punch version is useful.",
        },
    },
    "yang": {
        1: {"name": "Raishin Mahhaken", "motion": "qcf, qcf + punch", "steps": _qcf("HP")},
        2: {"name": "Tenshin Senkyutai", "motion": "qcf, qcf + kick", "steps": _qcf("HK")},
        3: {"name": "Seiei Enbu", "motion": "qcf, qcf + punch", "steps": _qcf("HP")},
    },
    "yun": {
        1: {"name": "You Hou", "motion": "qcf, qcf + punch", "steps": _qcf("HP")},
        2: {"name": "Sourai Rengeki", "motion": "qcf, qcf + punch", "steps": _qcf("HP")},
        3: {"name": "Genei Jin", "motion": "qcf, qcf + punch", "steps": _qcf("HP")},
    },
}

SYSTEM_PROMPT = """You control one player in Street Fighter III: 3rd Strike.
You receive one gameplay screenshot and must output only the next controller input sequence.
Track your own character by Player number and visual identity, especially character color, not by screen side alone. Characters can switch sides after jumps, throws, cross-ups, or Super Arts.

Return exactly one JSON object and nothing else.
Example valid response:
{"steps":[{"tokens":["RIGHT"],"hold_frames":4}]}

Controller tokens:
- UP, DOWN, LEFT, RIGHT are joystick directions.
- LP=light punch, MP=medium punch, HP=heavy punch.
- LK=light kick, MK=medium kick, HK=heavy kick.
- NONE means no input for that step.
- The bottom meter is the Super Art meter. When it shows MAX, a Super Art is available.
- Throw is LP+LK when close.
- Universal overhead is MP+MK.
- EX special moves use two punch buttons or two kick buttons and consume Super Art meter.
- Dash is two quick forward inputs; backdash is two quick back inputs.
- High parry is a quick forward tap; low parry is a quick DOWN tap.
- Charge moves require holding a charge direction before the release direction/button.

Input rules:
- steps must contain 0 to 16 entries.
- each step must contain 1 to 3 tokens.
- each step must include hold_frames from 1 to 60.
- tokens in the same step are pressed at the same time, e.g. ["DOWN","RIGHT"].
- steps are executed in order, so use multiple steps for combos, motions, cancels, blocks, movement, or attacks.
- hold_frames is how long to hold all tokens in that step; use larger values to keep holding a direction or button.
- LEFT and RIGHT are physical joystick directions, not semantic forward/back. Infer forward, backward, and blocking direction from the screenshot.
- Never output placeholder values such as TOKEN, FORWARD, or BACK. Convert every direction to UP, DOWN, LEFT, or RIGHT.
- Special moves and Super Arts are input as ordered direction/button steps, e.g. quarter-circle motions followed by punch or kick.
- After a side switch, re-identify which visible character is yours before choosing directions. Do not assume you are still on the same side as in the previous screenshot.
- If no useful action is clear, return {"steps":[{"tokens":["NONE"],"hold_frames":4}]}.
- Do not include explanations, summaries, markdown, comments, or any text outside the JSON object.

Choose whatever tactic and input sequence you judge most likely to win the round.
"""

PLAYER_PROMPT_TEMPLATE = """You are Player {player_number}.
Identify Player {player_number}'s character visually before acting, including color and current side.
Choose only the next short input sequence for Player {player_number}.
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
    snapshot_request_path: Path | None = None
    match_state_path: Path = MATCH_STATE_PATH
    command_path_p1: Path = COMMAND_PATH_P1
    command_path_p2: Path = COMMAND_PATH_P2
    captures_dir: Path = CAPTURES_DIR
    poll_seconds: float = POLL_SECONDS
    use_action_history: bool = False
    ai_players: tuple[int, ...] = (1, 2)


@dataclass(slots=True)
class ParsedMove:
    steps: list["ParsedStep"]
    summary: str


@dataclass(slots=True)
class ParsedStep:
    tokens: list[str]
    hold_frames: int


@dataclass(slots=True)
class ModelCallResult:
    player_move: ParsedMove
    latency_ms: float
    is_hallucination: bool


LogFn = Callable[[str, str], None]


def read_match_state(state_path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _state_str(state: dict[str, Any] | None, key: str) -> str:
    if state is None:
        return ""
    value = state.get(key)
    return "" if value is None else str(value)


def _state_int(state: dict[str, Any] | None, key: str) -> int | None:
    if state is None:
        return None
    value = state.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _valid_health(value: int | None) -> int | None:
    if value is None or value == 255 or value < 0 or value > 200:
        return None
    return value


def _estimated_damage(before: int | None, after: int | None) -> int:
    before = _valid_health(before)
    after = _valid_health(after)
    if before is None or after is None:
        return 0
    return max(before - after, 0)


def _estimate_action_metrics(
    *,
    player_number: int,
    state_before: dict[str, Any] | None,
    state_after: dict[str, Any] | None,
) -> dict[str, str]:
    opponent_number = 2 if player_number == 1 else 1
    opponent_damage = _estimated_damage(
        _state_int(state_before, f"health_p{opponent_number}"),
        _state_int(state_after, f"health_p{opponent_number}"),
    )
    self_damage = _estimated_damage(
        _state_int(state_before, f"health_p{player_number}"),
        _state_int(state_after, f"health_p{player_number}"),
    )
    wins_before = _state_int(state_before, f"wins_p{player_number}") or 0
    wins_after = _state_int(state_after, f"wins_p{player_number}") or 0
    round_win_delta = max(wins_after - wins_before, 0)
    estimated_hit = opponent_damage > 0 or round_win_delta > 0
    return {
        "estimated_opponent_damage": str(opponent_damage),
        "estimated_self_damage": str(self_damage),
        "estimated_hit": str(estimated_hit).lower(),
        "round_win_delta": str(round_win_delta),
    }


class ExperimentLogger:
    _fieldnames = (
        "timestamp",
        "player_label",
        "model_name",
        "command_id",
        "parsed_action",
        "latency_ms",
        "is_hallucination",
        "state_before_frame",
        "state_after_frame",
        "wins_p1_before",
        "wins_p2_before",
        "wins_p1_after",
        "wins_p2_after",
        "health_p1_before",
        "health_p2_before",
        "health_p1_after",
        "health_p2_after",
        "estimated_opponent_damage",
        "estimated_self_damage",
        "estimated_hit",
        "round_win_delta",
    )

    def __init__(self, log_path: Path = FIGHT_LOG_PATH) -> None:
        self.log_path = log_path
        self._lock = threading.Lock()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    def _ensure_header(self) -> None:
        with self._lock:
            should_write_header = (
                not self.log_path.exists() or self.log_path.stat().st_size == 0
            )
            if not should_write_header:
                return
            with self.log_path.open("a", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=self._fieldnames)
                writer.writeheader()

    def log_action(
        self,
        *,
        player_label: str,
        model_name: str,
        command_id: int,
        parsed_action: str,
        latency_ms: float,
        is_hallucination: bool,
        state_before: dict[str, Any] | None = None,
        state_after: dict[str, Any] | None = None,
    ) -> None:
        player_number = 1 if player_label == "P1" else 2
        metrics = _estimate_action_metrics(
            player_number=player_number,
            state_before=state_before,
            state_after=state_after,
        )
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "player_label": player_label,
            "model_name": model_name,
            "command_id": str(command_id),
            "parsed_action": parsed_action,
            "latency_ms": f"{latency_ms:.3f}",
            "is_hallucination": str(is_hallucination).lower(),
            "state_before_frame": _state_str(state_before, "frame"),
            "state_after_frame": _state_str(state_after, "frame"),
            "wins_p1_before": _state_str(state_before, "wins_p1"),
            "wins_p2_before": _state_str(state_before, "wins_p2"),
            "wins_p1_after": _state_str(state_after, "wins_p1"),
            "wins_p2_after": _state_str(state_after, "wins_p2"),
            "health_p1_before": _state_str(state_before, "health_p1"),
            "health_p2_before": _state_str(state_before, "health_p2"),
            "health_p1_after": _state_str(state_after, "health_p1"),
            "health_p2_after": _state_str(state_after, "health_p2"),
            "estimated_opponent_damage": metrics["estimated_opponent_damage"],
            "estimated_self_damage": metrics["estimated_self_damage"],
            "estimated_hit": metrics["estimated_hit"],
            "round_win_delta": metrics["round_win_delta"],
        }
        with self._lock:
            with self.log_path.open("a", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=self._fieldnames)
                writer.writerow(row)


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


def _normalize_character_key(character_name: str | None) -> str | None:
    if character_name is None:
        return None
    return character_name.strip().lower().replace(" ", "_").replace("-", "_")


def _physical_step_text(step: SemanticStep, *, facing_right: bool) -> str:
    direction_map = {
        "F": "RIGHT" if facing_right else "LEFT",
        "B": "LEFT" if facing_right else "RIGHT",
    }
    tokens = [direction_map.get(token, token) for token in step if token]
    return "+".join(tokens)


def _physical_sequence_text(
    steps: Sequence[SemanticStep],
    *,
    facing_right: bool,
) -> str:
    return "; ".join(
        step_text
        for step in steps
        if (step_text := _physical_step_text(step, facing_right=facing_right))
    )


def _super_art_context(config: FightStartConfig, player_number: int) -> str:
    if player_number == 1:
        character = _normalize_character_key(config.p1_character)
        super_art = config.p1_super_art
    else:
        character = _normalize_character_key(config.p2_character)
        super_art = config.p2_super_art

    if character is None:
        return ""

    guide = SUPER_ART_GUIDE.get(character, {}).get(super_art)
    if guide is None:
        return (
            f"\nSelected character: {character}. "
            f"Selected Super Art: SA{super_art}. "
            "No exact Super Art motion is configured for this character."
        )

    return (
        f"\nSelected character: {character}. "
        f"Selected Super Art: SA{super_art} - {guide['name']} "
        f"({guide['motion']}). "
        "When facing right, use: "
        f"{_physical_sequence_text(guide['steps'], facing_right=True)}. "
        "When facing left, use: "
        f"{_physical_sequence_text(guide['steps'], facing_right=False)}."
        + (f" {guide['extra']}" if guide.get("extra") else "")
    )


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


def build_match_state_lua(state_path: Path = MATCH_STATE_PATH) -> str:
    lua_state_path = _lua_path(state_path)
    lua_temp_path = _lua_path(state_path.with_suffix(".tmp"))
    return f"""
local sfiii_state_path = "{lua_state_path}"
local sfiii_state_temp_path = "{lua_temp_path}"
local sfiii_state_poll_mod = 30
local sfiii_state_frame = 0
local sfiii_state_space = nil

local sfiii_addr_fighting = {SFIII3N_MEMORY_ADDRESSES["fighting"]}
local sfiii_addr_wins_p1 = {SFIII3N_MEMORY_ADDRESSES["wins_p1"]}
local sfiii_addr_wins_p2 = {SFIII3N_MEMORY_ADDRESSES["wins_p2"]}
local sfiii_addr_health_p1 = {SFIII3N_MEMORY_ADDRESSES["health_p1"]}
local sfiii_addr_health_p2 = {SFIII3N_MEMORY_ADDRESSES["health_p2"]}

local function sfiii_state_get_space()
    if sfiii_state_space ~= nil then
        return sfiii_state_space
    end

    local cpu = manager.machine.devices[":maincpu"]
    if cpu == nil then
        return nil
    end

    sfiii_state_space = cpu.spaces["program"]
    return sfiii_state_space
end

local function sfiii_state_read_u8(space, address)
    local ok, value = pcall(function()
        return space:read_u8(address)
    end)
    if ok then
        return value
    end
    return -1
end

local function sfiii_state_write()
    local space = sfiii_state_get_space()
    if space == nil then
        return
    end

    local fighting = sfiii_state_read_u8(space, sfiii_addr_fighting)
    local wins_p1 = sfiii_state_read_u8(space, sfiii_addr_wins_p1)
    local wins_p2 = sfiii_state_read_u8(space, sfiii_addr_wins_p2)
    local health_p1 = sfiii_state_read_u8(space, sfiii_addr_health_p1)
    local health_p2 = sfiii_state_read_u8(space, sfiii_addr_health_p2)
    local match_over = (wins_p1 >= 2) or (wins_p2 >= 2)
    local winner = "unknown"

    if wins_p1 >= 2 and wins_p1 > wins_p2 then
        winner = "P1"
    elseif wins_p2 >= 2 and wins_p2 > wins_p1 then
        winner = "P2"
    elseif match_over then
        winner = "draw_or_unknown"
    end

    local file = io.open(sfiii_state_temp_path, "w")
    if file == nil then
        return
    end

    file:write(string.format(
        '{{"frame":%d,"fighting":%d,"wins_p1":%d,"wins_p2":%d,"health_p1":%d,"health_p2":%d,"match_over":%s,"winner":"%s"}}',
        sfiii_state_frame,
        fighting,
        wins_p1,
        wins_p2,
        health_p1,
        health_p2,
        tostring(match_over),
        winner
    ))
    file:close()
    os.remove(sfiii_state_path)
    os.rename(sfiii_state_temp_path, sfiii_state_path)
end

sfiii_state_subscription = emu.add_machine_frame_notifier(function ()
    if manager.machine.paused or manager.machine.exit_pending then
        return
    end

    sfiii_state_frame = sfiii_state_frame + 1
    if sfiii_state_frame % sfiii_state_poll_mod == 0 then
        sfiii_state_write()
    end
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
    if isinstance(raw_tokens, str):
        raw_tokens = [raw_tokens]
    if not isinstance(raw_tokens, list):
        raise ValueError("step tokens must be a list or string")

    normalized_tokens: list[str] = []
    seen_tokens: set[str] = set()
    for raw_token in raw_tokens:
        if not isinstance(raw_token, str):
            raise ValueError("each token must be a string")
        raw_parts = re.split(r"[+,/]", raw_token)
        for raw_part in raw_parts:
            token = raw_part.strip().upper().replace(" ", "_")
            token = TOKEN_ALIASES.get(token, token)
            if not token:
                continue
            if token not in ALLOWED_MOVE_TOKENS:
                raise ValueError(f"Unsupported move token: {token}")
            if token == "NONE" or token in seen_tokens:
                continue
            if len(normalized_tokens) >= MAX_TOKENS_PER_STEP:
                return normalized_tokens
            normalized_tokens.append(token)
            seen_tokens.add(token)

    return normalized_tokens


def _normalize_hold_frames(raw_hold_frames: Any) -> int:
    if isinstance(raw_hold_frames, str) and raw_hold_frames.strip().isdigit():
        raw_hold_frames = int(raw_hold_frames.strip())
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
    if "steps" not in data:
        raise KeyError("steps")
    steps = _normalize_steps(data.get("steps", []))
    return ParsedMove(steps=steps, summary="")


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


def _fallback_move() -> ParsedMove:
    return ParsedMove(steps=[], summary="")


def _looks_like_complete_png(image_bytes: bytes) -> bool:
    return (
        image_bytes.startswith(PNG_SIGNATURE)
        and image_bytes.endswith(PNG_IEND_CHUNK)
    )


def _read_screenshot_bytes(path: Path) -> bytes:
    last_error: OSError | RuntimeError | None = None
    for _ in range(SCREENSHOT_READ_RETRIES):
        try:
            stat_before = path.stat()
            image_bytes = path.read_bytes()
            stat_after = path.stat()
        except OSError as exc:
            last_error = exc
            time.sleep(SCREENSHOT_READ_RETRY_SECONDS)
            continue

        is_stable_file = (
            stat_before.st_size == stat_after.st_size
            and stat_before.st_mtime_ns == stat_after.st_mtime_ns
        )
        if is_stable_file and _looks_like_complete_png(image_bytes):
            return image_bytes

        last_error = RuntimeError(
            f"Screenshot file is not a stable complete PNG yet: {path}"
        )
        time.sleep(SCREENSHOT_READ_RETRY_SECONDS)

    assert last_error is not None
    raise last_error


def _encode_image_as_data_url(path: Path) -> str:
    image_bytes = _read_screenshot_bytes(path)
    encoded = b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def request_fresh_screenshot(
    *,
    screenshot_path: Path,
    request_path: Path | None,
) -> None:
    if request_path is None:
        return

    with _SNAPSHOT_REQUEST_LOCK:
        previous_mtime = screenshot_path.stat().st_mtime_ns if screenshot_path.exists() else None
        request_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = request_path.with_suffix(".tmp")
        request_id = str(time.time_ns())
        temp_path.write_text(request_id, encoding="utf-8")
        temp_path.replace(request_path)

        deadline = time.perf_counter() + SNAPSHOT_REQUEST_TIMEOUT_SECONDS
        while time.perf_counter() < deadline:
            if screenshot_path.exists():
                current_mtime = screenshot_path.stat().st_mtime_ns
                if previous_mtime is None or current_mtime != previous_mtime:
                    _read_screenshot_bytes(screenshot_path)
                    return
            time.sleep(SCREENSHOT_READ_RETRY_SECONDS)

        raise TimeoutError(f"Timed out waiting for fresh screenshot: {screenshot_path}")


def call_openrouter_model(
    *,
    api_key: str,
    model: str,
    screenshot_path: Path,
    snapshot_request_path: Path | None,
    player_number: int,
    fight_start: FightStartConfig,
    action_history: Sequence[str] | None = None,
) -> ModelCallResult:
    request_fresh_screenshot(
        screenshot_path=screenshot_path,
        request_path=snapshot_request_path,
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
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": player_prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": screenshot_url},
                    },
                ],
            },
        ],
        "temperature": 0.2,
        "max_tokens": 420,
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
            f"OpenRouter request failed with HTTP {exc.code}: {error_body}"
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

    latency_ms = (time.perf_counter() - request_start) * 1000.0

    try:
        payload_data = json.loads(body)
    except json.JSONDecodeError:
        return ModelCallResult(
            player_move=_fallback_move(),
            latency_ms=latency_ms,
            is_hallucination=True,
        )

    try:
        raw_content = payload_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ModelCallResult(
            player_move=_fallback_move(),
            latency_ms=latency_ms,
            is_hallucination=True,
        )
    if not isinstance(raw_content, str):
        return ModelCallResult(
            player_move=_fallback_move(),
            latency_ms=latency_ms,
            is_hallucination=True,
        )

    try:
        player_move = parse_model_move(raw_content)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return ModelCallResult(
            player_move=_fallback_move(),
            latency_ms=latency_ms,
            is_hallucination=True,
        )

    return ModelCallResult(
        player_move=player_move,
        latency_ms=latency_ms,
        is_hallucination=False,
    )


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
    use_action_history: bool = False,
    snapshot_request_path: Path | None = None,
    ai_players: Sequence[int] = (1, 2),
) -> ArenaConfig:
    ai_player_set = set(ai_players)
    normalized_ai_players = tuple(
        player for player in (1, 2) if player in ai_player_set
    )
    if not normalized_ai_players:
        raise ValueError("At least one AI player must be enabled.")

    model_p1 = _require_env("OPENROUTER_MODEL_P1") if 1 in normalized_ai_players else ""
    model_p2 = _require_env("OPENROUTER_MODEL_P2") if 2 in normalized_ai_players else ""

    return ArenaConfig(
        fight_start=fight_start or FightStartConfig(),
        model_p1=model_p1,
        model_p2=model_p2,
        api_key=_require_env("OPENROUTER_API_KEY"),
        round_start_buffer_seconds=round_start_buffer_seconds,
        screenshot_warmup_updates=screenshot_warmup_updates,
        captures_dir=captures_dir,
        screenshot_path=captures_dir / "latest_frame.png",
        snapshot_request_path=snapshot_request_path,
        match_state_path=captures_dir / "match_state.json",
        command_path_p1=captures_dir / "llm_moves_p1.txt",
        command_path_p2=captures_dir / "llm_moves_p2.txt",
        use_action_history=use_action_history,
        ai_players=normalized_ai_players,
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
    snapshot_request_path: Path | None,
    match_state_path: Path,
    command_path: Path,
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
            result = call_openrouter_model(
                api_key=api_key,
                model=model,
                screenshot_path=screenshot_path,
                snapshot_request_path=snapshot_request_path,
                player_number=player_number,
                fight_start=fight_start,
                action_history=list(recent_actions) if use_action_history else None,
            )
            player_move = result.player_move
            parsed_action = _steps_to_command_line(player_move)
            state_before = read_match_state(match_state_path)
            _emit_log(
                log_fn,
                channel,
                f"{model}: {parsed_action}"
                + f" | latency={result.latency_ms:.1f}ms"
                + (" | hallucination" if result.is_hallucination else "")
                + (f" | {player_move.summary}" if player_move.summary else ""),
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
                    latency_ms=result.latency_ms,
                    is_hallucination=result.is_hallucination,
                    state_before=state_before,
                    state_after=state_after,
                )
            if was_stopped:
                return
        except Exception as exc:
            _emit_log(log_fn, channel, f"worker error: {exc}")
            if experiment_logger is not None:
                experiment_logger.log_action(
                    player_label=player_label,
                    model_name=model,
                    command_id=command_id,
                    parsed_action="NONE",
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
) -> list[threading.Thread]:
    experiment_logger = ExperimentLogger(config.captures_dir / "fight_log.csv")
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
            name="llm-p1",
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
            name="llm-p2",
        ))

    for worker in workers:
        worker.start()

    return workers

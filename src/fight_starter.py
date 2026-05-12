from __future__ import annotations

from dataclasses import dataclass


PRESS_FRAMES = 2
BOOT_WAIT_FRAMES = 360
POST_COIN_WAIT_FRAMES = 60
SECOND_COIN_WAIT_FRAMES = 60
BETWEEN_STARTS_WAIT_FRAMES = 60
POST_START_WAIT_FRAMES = 60
BETWEEN_PLAYER_ACTIONS_FRAMES = 12
POST_CHARACTER_WAIT_FRAMES = 45

MOVE_TOKENS = {
    "up": "JOYSTICK_UP",
    "down": "JOYSTICK_DOWN",
    "left": "JOYSTICK_LEFT",
    "right": "JOYSTICK_RIGHT",
}

CHARACTER_ORDER = (
    "alex",
    "twelve",
    "hugo",
    "sean",
    "makoto",
    "elena",
    "ibuki",
    "chun_li",
    "dudley",
    "necro",
    "q",
    "oro",
    "urien",
    "remy",
    "ryu",
    "akuma",
    "yun",
    "yang",
    "ken",
)

CHARACTER_ALIASES = {
    "chunli": "chun_li",
    "chun-li": "chun_li",
    "gouki": "akuma",
    "qouky": "akuma",
}

CHARACTER_INDEX_BY_NAME = {
    character_name: index for index, character_name in enumerate(CHARACTER_ORDER)
}
CURSOR_START_CHARACTER = {
    "P1": "alex",
    "P2": "ryu",
}


@dataclass(slots=True)
class FightStartConfig:
    p1_character: str | None = None
    p2_character: str | None = None
    p1_moves: tuple[str, ...] = ()
    p2_moves: tuple[str, ...] = ()
    p1_super_art: int = 1
    p2_super_art: int = 1
    active_players: tuple[int, ...] = (1, 2)


def _normalize_character_name(character_name: str) -> str:
    normalized = character_name.strip().lower().replace(" ", "_")
    return CHARACTER_ALIASES.get(normalized, normalized)


def _moves_for_character(character_name: str, player_label: str) -> tuple[str, ...]:
    normalized = _normalize_character_name(character_name)
    normalized_player_label = player_label.strip().upper()
    try:
        target_index = CHARACTER_INDEX_BY_NAME[normalized]
    except KeyError as exc:
        supported = ", ".join(CHARACTER_ORDER)
        raise ValueError(
            f"{player_label} character '{character_name}' is unsupported. "
            f"Supported characters: {supported}"
        ) from exc

    cursor_start_name = CURSOR_START_CHARACTER.get(normalized_player_label, CHARACTER_ORDER[0])
    start_index = CHARACTER_INDEX_BY_NAME[cursor_start_name]
    right_moves = (target_index - start_index) % len(CHARACTER_ORDER)
    return ("right",) * right_moves


def _validate_moves(moves: tuple[str, ...], player_label: str) -> None:
    invalid_moves = [move for move in moves if move not in MOVE_TOKENS]
    if invalid_moves:
        raise ValueError(
            f"{player_label} has unsupported moves: {', '.join(invalid_moves)}"
        )


def _validate_super_art(super_art: int, player_label: str) -> None:
    if super_art not in (1, 2, 3):
        raise ValueError(f"{player_label} super art must be 1, 2, or 3")


def _active_players(config: FightStartConfig) -> tuple[int, ...]:
    players = tuple(player for player in (1, 2) if player in set(config.active_players))
    if not players:
        raise ValueError("At least one active player must be configured")
    return players


def _build_player_schedule(
    player_number: int,
    moves: tuple[str, ...],
    super_art: int,
    start_frame: int,
) -> tuple[list[tuple[int, str, int]], int]:
    token_prefix = f"P{player_number}_"
    schedule: list[tuple[int, str, int]] = []
    current_frame = start_frame

    for move in moves:
        schedule.append(
            (
                current_frame,
                token_prefix + MOVE_TOKENS[move],
                PRESS_FRAMES,
            )
        )
        current_frame += PRESS_FRAMES + BETWEEN_PLAYER_ACTIONS_FRAMES

    schedule.append((current_frame, token_prefix + "BUTTON1", PRESS_FRAMES))
    current_frame += POST_CHARACTER_WAIT_FRAMES
    schedule.append(
        (current_frame, token_prefix + f"BUTTON{super_art}", PRESS_FRAMES)
    )
    current_frame += PRESS_FRAMES

    return schedule, current_frame


def _build_schedule(config: FightStartConfig) -> list[tuple[int, str, int]]:
    active_players = _active_players(config)
    p1_moves = (
        config.p1_moves
        if config.p1_moves
        else _moves_for_character(config.p1_character, "P1")
        if config.p1_character
        else ()
    )
    p2_moves = (
        config.p2_moves
        if config.p2_moves
        else _moves_for_character(config.p2_character, "P2")
        if config.p2_character
        else ()
    )

    _validate_moves(p1_moves, "P1")
    _validate_moves(p2_moves, "P2")
    _validate_super_art(config.p1_super_art, "P1")
    _validate_super_art(config.p2_super_art, "P2")

    schedule: list[tuple[int, str, int]] = []
    current_frame = BOOT_WAIT_FRAMES

    schedule.append((current_frame, "COIN1", PRESS_FRAMES))
    current_frame += POST_COIN_WAIT_FRAMES
    schedule.append((current_frame, "COIN1", PRESS_FRAMES))
    current_frame += SECOND_COIN_WAIT_FRAMES

    for player in active_players:
        schedule.append((current_frame, f"START{player}", PRESS_FRAMES))
        current_frame += BETWEEN_STARTS_WAIT_FRAMES

    current_frame += POST_START_WAIT_FRAMES

    if 1 in active_players:
        p1_schedule, current_frame = _build_player_schedule(
            player_number=1,
            moves=p1_moves,
            super_art=config.p1_super_art,
            start_frame=current_frame,
        )
        schedule.extend(p1_schedule)
        current_frame += BETWEEN_PLAYER_ACTIONS_FRAMES

    if 2 in active_players:
        p2_schedule, current_frame = _build_player_schedule(
            player_number=2,
            moves=p2_moves,
            super_art=config.p2_super_art,
            start_frame=current_frame,
        )
        schedule.extend(p2_schedule)

    return schedule


def estimate_fight_start_frame(config: FightStartConfig | None = None) -> int:
    """Return the last scheduled fight-start frame."""
    config = config or FightStartConfig()
    schedule = _build_schedule(config)
    return max((start_frame + duration_frames) for start_frame, _, duration_frames in schedule)


def build_fight_start_lua(config: FightStartConfig | None = None) -> str:
    """Build Lua that inserts coins, starts active players, and confirms selections."""
    config = config or FightStartConfig()
    schedule_lines = []

    for start_frame, token, duration_frames in _build_schedule(config):
        schedule_lines.append(
            f'add_step({start_frame}, "{token}", {duration_frames})'
        )

    return """
local fight_start_steps = { }
local fight_start_active = { }
local fight_start_missing = { }
local fight_start_resolved = { }
local fight_start_frame = 0

local function add_step(start_frame, token, duration_frames)
    table.insert(fight_start_steps, {
        start_frame = start_frame,
        end_frame = start_frame + duration_frames - 1,
        token = token,
        field = nil,
    })
end

local function find_field(token)
    local input_type, player = manager.machine.ioport:token_to_input_type(token)
    if input_type == nil then
        return nil
    end

    for _, port in pairs(manager.machine.ioport.ports) do
        for _, field in pairs(port.fields) do
            if field.type == input_type and field.player == player then
                return field
            end
        end
    end

    return nil
end

""" + "\n".join(schedule_lines) + """
fight_start_subscription = emu.add_machine_frame_notifier(function ()
    if manager.machine.paused or manager.machine.exit_pending then
        return
    end

    fight_start_frame = fight_start_frame + 1
    local next_active = { }

    for _, step in ipairs(fight_start_steps) do
        if fight_start_frame >= step.start_frame and fight_start_frame <= step.end_frame then
            if not step.field then
                step.field = find_field(step.token)
                if step.field then
                    if not fight_start_resolved[step.token] then
                        fight_start_resolved[step.token] = true
                        emu.print_info(
                            "fight_start: resolved " .. step.token ..
                            " on frame " .. tostring(fight_start_frame)
                        )
                    end
                elseif not fight_start_missing[step.token] then
                    fight_start_missing[step.token] = true
                    emu.print_error("fight_start: missing input token " .. step.token)
                end
            end

            if step.field then
                next_active[step.token] = step.field
                if not fight_start_active[step.token] then
                    emu.print_info(
                        "fight_start: press " .. step.token ..
                        " on frame " .. tostring(fight_start_frame)
                    )
                end
            end
        end
    end

    for _, field in pairs(next_active) do
        field:set_value(1)
    end

    for token, field in pairs(fight_start_active) do
        if not next_active[token] then
            emu.print_info(
                "fight_start: release " .. token ..
                " on frame " .. tostring(fight_start_frame)
            )
            field:clear_value()
        end
    end

    fight_start_active = next_active
end)
"""

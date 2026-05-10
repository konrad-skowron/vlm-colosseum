# VLM Colosseum

Experimental environment for comparing large language and vision-language models in a real-time fighting game setting. The project runs `Street Fighter III: 3rd Strike` in MAME, sends gameplay screenshots to models through OpenRouter, converts model responses into controller inputs, and logs match-level results for later analysis.

The project is being developed as infrastructure for a master's thesis about decision-making quality, state handling, latency, and tactical behavior of LLM/VLM agents in a fighting game.

## Current Flow

1. `main.py` starts MAME with generated Lua scripts.
2. Lua automates coin insertion, character selection, and Super Art selection.
3. Lua exposes a text-file bridge for player commands.
4. Python workers request fresh screenshots, send them to selected models, parse JSON input sequences, and write command files.
5. Lua reads command files and executes inputs inside MAME.
6. Lua reads selected `sfiii3n` memory addresses and writes `match_state.json`.
7. `main.py` runs `N` matches and appends summary rows to `captures/experiment_summary.csv`.

## Important Files

- `main.py`  
  Main experiment runner. Controls match count, AI player selection, screenshot mode, MAME window arguments, and summary logging.

- `llm_arena.py`  
  OpenRouter integration, prompt construction, model response parsing, command writing, match-state Lua generation, and Super Art context.

- `fight_starter.py`  
  Generates Lua for automated match setup and character/Super Art selection.

- `screenshot_loop.py`  
  Generates Lua for screenshots. Supports fixed-interval screenshots and on-demand screenshots requested before model calls.

- `mame_launcher.py`  
  Launches MAME from `C:\Emulators\mame` by default.

- `log_viewer.py`  
  Tkinter telemetry window with separate P1/P2 decision logs.

## Requirements

- Windows
- Python `3.10+`
- MAME installed in `C:\Emulators\mame`
- `sfiii3n` ROM available to MAME
- OpenRouter API key

The project currently uses only the Python standard library.

## Environment

Create `.env` in the repository root:

```env
OPENROUTER_API_KEY=your_key
OPENROUTER_MODEL_P1=openai/gpt-5.4-nano
OPENROUTER_MODEL_P2=google/gemini-2.5-flash
```

If only one AI player is enabled, only that player's model variable is required.

## Running

```powershell
python .\main.py
```

Main runtime flags are near the top of `main.py`:

```python
ENABLE_LLM_ARENA = True
ENABLE_LOG_WINDOW = True
CAPTURES_DIR = "captures"
EXPERIMENT_MATCH_COUNT = 1
MATCH_MAX_SECONDS = 230.0
AI_PLAYERS = (1, 2)
USE_ON_DEMAND_SCREENSHOTS = True
MAME_WINDOW_ARGS = []
LLM_ROUND_START_BUFFER_SECONDS = 12.0
LLM_SCREENSHOT_WARMUP_UPDATES = 4
```

Meaning:

- `EXPERIMENT_MATCH_COUNT`  
  Number of matches to run sequentially.

- `MATCH_MAX_SECONDS`  
  Safety timeout for one match after LLM workers start.

- `AI_PLAYERS`  
  Tuple of AI-controlled players. Use `(1, 2)` for AI vs AI, `(1,)` for P1 AI only, or `(2,)` for P2 AI only.

- `USE_ON_DEMAND_SCREENSHOTS`  
  If `True`, MAME takes a fresh screenshot only when a model worker is about to send a request. This is preferred for experiments because it reduces stale observations and unnecessary disk writes.

- `MAME_WINDOW_ARGS`  
  Optional extra MAME display arguments. Empty by default, so MAME uses its configured window behavior.

## AI Player Modes

`AI_PLAYERS` controls both the LLM workers and the automated fight-start sequence. The script always inserts two credits so a human can join later if needed.

- `(1, 2)` starts both players automatically and runs AI vs AI.
- `(1,)` starts only player 1 automatically and runs P1 AI in single-player mode.
- `(2,)` starts only player 2 automatically and runs P2 AI in single-player mode if the game accepts P2 start from credits.

If only one AI player is started, a human can still join manually through normal MAME controls by inserting/joining with the other player.

## Model Output

Models must return only JSON:

```json
{
  "steps": [
    { "tokens": ["DOWN"], "hold_frames": 3 },
    { "tokens": ["DOWN", "RIGHT"], "hold_frames": 3 },
    { "tokens": ["RIGHT", "HP"], "hold_frames": 5 }
  ]
}
```

Allowed tokens:

- `UP`
- `DOWN`
- `LEFT`
- `RIGHT`
- `LP`
- `MP`
- `HP`
- `LK`
- `MK`
- `HK`
- `NONE`

Rules enforced by the parser:

- maximum `16` steps;
- maximum `3` simultaneous tokens per step;
- `hold_frames` must be `1-60`;
- `LEFT` and `RIGHT` are physical joystick directions, not semantic forward/back.

The prompt also informs models about Super Art meter, selected Super Art commands, throws, overheads, EX moves, dash/backdash, parry, and charge moves.

## Runtime Artifacts

Each match gets its own directory:

```text
captures/match_001/
captures/match_002/
...
```

Important files:

- `latest_frame.png`  
  Latest screenshot for that match.

- `snapshot_request.txt`  
  Request file used by on-demand screenshot mode.

- `llm_moves_p1.txt`, `llm_moves_p2.txt`  
  IPC command files read by MAME Lua. These are not logs.

- `fight_log.csv`  
  Per-action model log with parsed action, latency, and hallucination flag.

- `match_state.json`  
  Lua-exported match state from MAME memory: round wins, HP values, `match_over`, and winner.

- `captures/experiment_summary.csv`  
  Batch-level summary with match status, result, model names, duration, and match directory.

## Match Result Detection

The project reads these `sfiii3n` memory addresses through MAME Lua:

```python
fighting  = 0x0200EE44
wins_p1   = 0x02011383
wins_p2   = 0x02011385
health_p1 = 0x02068D0B
health_p2 = 0x020691A3
```

The match is considered finished when either player reaches two round wins. HP is useful as telemetry, but final result should be based on `wins_p1/wins_p2`.

Known HP detail: full HP appears to be around `160`. After KO or round transitions, the game may expose `255`, which should be treated as a state/sentinel value, not real health.

## Current Limitations

- The project is currently Windows/MAME-path specific.
- True AI vs CPU/NPC is not implemented; current setup is local two-player mode.
- Memory addresses are specific to `sfiii3n` and may need adjustment for other ROM revisions.
- Super Art command data is prompt guidance, not a guarantee that a model will execute the move reliably.
- Python and MAME communicate through files. This is simple and debuggable but not a low-latency IPC design.

## Verification

Useful local checks:

```powershell
python -m py_compile main.py llm_arena.py screenshot_loop.py fight_starter.py mame_launcher.py log_viewer.py
```

For a safe smoke test, set:

```python
EXPERIMENT_MATCH_COUNT = 1
AI_PLAYERS = (1, 2)
USE_ON_DEMAND_SCREENSHOTS = True
```

# VLM Colosseum

This project is a test environment for analyzing the behavior of large language models and multimodal models in a fighting game setting. In its current form, the repository allows models to control characters in `Street Fighter III: 3rd Strike` running in MAME, observe the game state through screenshots, and generate input sequences that are then executed directly inside the emulator.

The project is being developed as experimental infrastructure for a master's thesis focused on analyzing the behavior of large language models in a real-time fighting game environment.

## Project Goal

The environment currently performs four main tasks:

1. launches MAME with the `sfiii3n` ROM,
2. automates the transition from the startup screens to an active match,
3. captures the current game state into a single rolling screenshot file,
4. sends that image to two LLM/VLM models, receives their move proposals, and executes them for player 1 and player 2.

An important design decision is that both input execution and screenshot generation happen inside MAME through Lua scripts. Because of that, the setup continues to work even when the emulator window is not focused or is minimized.

## Architecture

The most important files in the project are:

- [main.py](main.py)  
  Main entry point. It assembles the full pipeline: launches MAME, injects Lua scripts, starts the screenshot loop, initializes the fight, and optionally starts the LLM agents.

- [mame_launcher.py](mame_launcher.py)  
  Responsible for launching MAME with the required arguments.

- [fight_starter.py](fight_starter.py)  
  Generates a Lua script that automates match setup: coin insertion, 2-player mode selection, character selection, and Super Art selection.

- [screenshot_loop.py](screenshot_loop.py)  
  Generates a temporary Lua script that saves the latest game snapshot to `captures/latest_frame.png`.

- [llm_arena.py](llm_arena.py)  
  Handles OpenRouter integration. It encodes screenshots, sends them to the models, parses the JSON response, and writes input commands for both players.

- [log_viewer.py](log_viewer.py)  
  Provides a simple log window with separate columns for `P1` and `P2`.

## How It Works

The current execution flow is as follows:

1. `main.py` launches MAME together with a generated Lua script.
2. Lua inside MAME performs the automated fight-start sequence.
3. Lua saves the current game state every `0.5` seconds to `captures/latest_frame.png`.
4. Two independent Python workers send the most recent screenshot to two models through OpenRouter.
5. Each model returns a short JSON description of the next input sequence.
6. Python writes those commands to:
   - `captures/llm_moves_p1.txt`
   - `captures/llm_moves_p2.txt`
7. Lua inside MAME reads those files and executes the corresponding actions for both players.

The files `llm_moves_p1.txt` and `llm_moves_p2.txt` are not logs. They are a simple IPC mechanism between the Python process and MAME.

## Requirements

The current implementation assumes:

- Windows,
- Python `3.10+`,
- MAME installed in `C:\Emulators\mame`,
- the `sfiii3n` ROM available in that installation,
- an OpenRouter account and API key if LLM Arena mode is enabled.

## Configuration

### 1. Python

At the moment, the project does not require external Python dependencies declared in `pyproject.toml`. It relies primarily on the Python standard library.

### 2. MAME

The emulator path is defined in [mame_launcher.py](mame_launcher.py):

```python
MAME_DIR = Path(r"C:\Emulators\mame")
DEFAULT_ROM = "sfiii3n"
```

If MAME is installed elsewhere, or if a different ROM should be used, these values need to be changed.

### 3. `.env` File

If `ENABLE_LLM_ARENA = True`, the project expects a `.env` file in the repository root with at least these variables:

```env
OPENROUTER_API_KEY=your_key
OPENROUTER_MODEL_P1=openai/gpt-4.1-mini
OPENROUTER_MODEL_P2=google/gemini-2.5-flash
```

## Running the Project

Basic execution:

```powershell
python .\main.py
```

The most important runtime switches are defined in [main.py](main.py):

```python
ENABLE_LLM_ARENA = True
ENABLE_LOG_WINDOW = True
CAPTURES_DIR = "captures"
LLM_ROUND_START_BUFFER_SECONDS = 10.0
LLM_SCREENSHOT_WARMUP_UPDATES = 4
```

Meaning:

- `ENABLE_LLM_ARENA`  
  Enables or disables model-controlled fighting. If set to `False`, the project will still launch the game and run the automated fight-start logic, but without LLM workers.

- `ENABLE_LOG_WINDOW`  
  Opens a separate GUI window with dedicated log columns for both players.

- `LLM_ROUND_START_BUFFER_SECONDS`  
  Adds an extra delay after the automated fight-start sequence before the model workers begin making decisions.

- `LLM_SCREENSHOT_WARMUP_UPDATES`  
  Specifies how many additional screenshot updates the system waits for before sending the first requests to the models.

## Model Output Format

Models are expected to return JSON in the following form:

```json
{
  "steps": [
    { "tokens": ["RIGHT"], "hold_frames": 10 },
    { "tokens": ["HP"], "hold_frames": 4 }
  ],
  "summary": "short explanation of the decision"
}
```

Allowed input tokens:

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

`hold_frames` defines how many frames a given input should be held.

## Logs and Runtime Artifacts

During execution, the project generates several important artifacts:

- `captures/latest_frame.png`  
  The most recent screenshot of the match. Each new capture overwrites the previous one.

- `captures/llm_moves_p1.txt`  
  The current command file for player 1.

- `captures/llm_moves_p2.txt`  
  The current command file for player 2.

Model logs are displayed in a separate GUI window, split into `P1` and `P2` columns to make it easier to analyze both agents independently.

## Current Limitations

The current implementation has several important limitations:

- game-state observation is based entirely on screenshots, without direct access to game memory,
- decision quality depends heavily on the quality and timing of the captured image,
- Python ↔ MAME communication is implemented through text files, which is easy to debug but not the most sophisticated IPC approach,
- the setup is currently tightly coupled to Windows and a local MAME installation,
- some models or providers may occasionally reject an image with errors such as `Provided image is not valid`.

## Possible Future Work

Natural next steps for the project include:

1. adding systematic model benchmarking,
2. storing complete experiment logs in result files,
3. standardizing match scenarios and character setups,
4. automatically aggregating statistics such as win rate, reaction time, and decision types,
5. comparing language-only and multimodal models under controlled experimental conditions.

## License

The repository includes a [LICENSE](LICENSE) file. If the thesis project evolves further toward publication or broader release, it would also be worth clarifying the usage conditions for the ROM, the emulator, and the external model providers.

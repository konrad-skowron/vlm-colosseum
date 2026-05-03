from fight_starter import FightStartConfig, build_fight_start_lua
from mame_launcher import open_sfiii3n
from screenshot_loop import create_snapshot_loop


def main() -> None:
    fight_start = FightStartConfig(
        p1_character=None,
        p2_character=None,
        p1_super_art=1,
        p2_super_art=1,
    )
    snapshot_loop = create_snapshot_loop(
        output_dir="captures",
        interval_seconds=1.0,
        extra_lua=build_fight_start_lua(fight_start),
    )
    process = open_sfiii3n(
        extra_args=snapshot_loop.mame_args() + ["-skip_gameinfo"]
    )

    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
    finally:
        snapshot_loop.cleanup()


if __name__ == "__main__":
    main()

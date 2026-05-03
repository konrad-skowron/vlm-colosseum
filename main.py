from mame_launcher import open_sfiii3n
from screenshot_loop import create_snapshot_loop, delete_screenshots


def main() -> None:
    snapshot_loop = create_snapshot_loop(output_dir="captures", interval_seconds=1.0)
    process = open_sfiii3n(extra_args=snapshot_loop.mame_args())

    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
    finally:
        delete_screenshots(snapshot_loop.output_dir)
        snapshot_loop.cleanup()


if __name__ == "__main__":
    main()

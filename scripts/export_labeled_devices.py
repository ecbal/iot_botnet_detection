from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd


DATA_DIR = Path("archive-2")
OUTPUT_DIR = Path("data/labeled_devices")
COMBINED_OUTPUT_PATH = OUTPUT_DIR / "all_devices_labeled.csv"
CHUNKSIZE = 100_000


def discover_device_ids() -> list[int]:
    device_ids = set()
    for file_path in DATA_DIR.glob("*.csv"):
        first_part = file_path.stem.split(".", maxsplit=1)[0]
        if first_part.isdigit():
            device_ids.add(int(first_part))
    return sorted(device_ids)


def get_device_traffic_files(device_id: int) -> list[Path]:
    return sorted(DATA_DIR.glob(f"{device_id}.*.csv"))


def binary_label_for(file_path: Path) -> tuple[str, int]:
    traffic_type = file_path.stem.split(".")[1]
    if traffic_type == "benign":
        return "benign", 0
    return "attack", 1


def export_device_labeled(device_id: int) -> tuple[Path, int]:
    output_path = OUTPUT_DIR / f"device_{device_id}_labeled.csv"
    traffic_files = get_device_traffic_files(device_id)

    if not traffic_files:
        raise FileNotFoundError(f"No traffic CSV files found for device {device_id}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    total_rows = 0
    write_header = True

    for file_path in traffic_files:
        binary_label, binary_target = binary_label_for(file_path)

        for chunk in pd.read_csv(file_path, chunksize=CHUNKSIZE):
            metadata = pd.DataFrame(
                {
                    "binary_label": binary_label,
                    "binary_target": binary_target,
                    "source_file": file_path.name,
                },
                index=chunk.index,
            )
            labeled_chunk = pd.concat([chunk, metadata], axis=1)
            labeled_chunk.to_csv(
                output_path,
                mode="w" if write_header else "a",
                header=write_header,
                index=False,
            )

            total_rows += len(labeled_chunk)
            write_header = False

    return output_path, total_rows


def combine_labeled_devices(device_paths: list[Path]) -> tuple[Path, int]:
    if COMBINED_OUTPUT_PATH.exists():
        COMBINED_OUTPUT_PATH.unlink()

    total_rows = 0
    write_header = True

    for device_path in device_paths:
        for chunk in pd.read_csv(device_path, chunksize=CHUNKSIZE):
            chunk.to_csv(
                COMBINED_OUTPUT_PATH,
                mode="w" if write_header else "a",
                header=write_header,
                index=False,
            )
            total_rows += len(chunk)
            write_header = False

    return COMBINED_OUTPUT_PATH, total_rows


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Export per-device labeled CSV files and one combined labeled CSV."
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        type=int,
        default=None,
        help="Device ids to export. Defaults to all devices discovered in archive-2.",
    )
    parser.add_argument(
        "--skip-combine",
        action="store_true",
        help="Only export per-device labeled CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device_ids = sorted(args.devices) if args.devices else discover_device_ids()

    if not device_ids:
        raise FileNotFoundError(f"No device CSV files found in {DATA_DIR}")

    device_paths = []
    for device_id in device_ids:
        output_path, rows = export_device_labeled(device_id)
        device_paths.append(output_path)
        print(f"Exported: {output_path} rows={rows}")

    if not args.skip_combine:
        combined_path, combined_rows = combine_labeled_devices(device_paths)
        print(f"Combined: {combined_path} rows={combined_rows}")


if __name__ == "__main__":
    main()

from pathlib import Path

import pandas as pd


DATA_DIR = Path("archive-2")
OUTPUT_DIR = Path("data/labeled_devices")
OUTPUT_PATH = OUTPUT_DIR / "device_2_labeled.csv"
DEVICE_ID = 2
CHUNKSIZE = 100_000


def get_device_traffic_files() -> list[Path]:
    return sorted(DATA_DIR.glob(f"{DEVICE_ID}.*.csv"))


def binary_label_for(file_path: Path) -> tuple[str, int]:
    traffic_type = file_path.stem.split(".")[1]
    if traffic_type == "benign":
        return "benign", 0
    return "attack", 1


def export_device_2_labeled() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    total_rows = 0
    write_header = True

    for file_path in get_device_traffic_files():
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
                OUTPUT_PATH,
                mode="w" if write_header else "a",
                header=write_header,
                index=False,
            )

            total_rows += len(labeled_chunk)
            write_header = False

    return total_rows


def main() -> None:
    total_rows = export_device_2_labeled()
    print(f"Exported: {OUTPUT_PATH}")
    print(f"Rows: {total_rows}")


if __name__ == "__main__":
    main()

import os
import shutil
from pathlib import Path


def collect_csv_files(
    source_root: Path,
    target_root: Path,
    prefix: str = "noise_",
):
    """
    Collect all CSV files from source_root (recursively),
    copy them into target_root (flat),
    and add a prefix to all collected file names.

    Parameters
    ----------
    source_root : Path
        The folder to search for CSV files (searched recursively).
    target_root : Path
        The folder where collected CSV files will be copied.
    prefix : str, optional
        Prefix to add to each collected file name (default: 'noise_').
    """

    source_root = Path(source_root)
    target_root = Path(target_root)

    if not source_root.exists():
        print(f"Source folder does not exist: {source_root}")
        return

    target_root.mkdir(parents=True, exist_ok=True)

    print(f"Source folder: {source_root}")
    print(f"Target folder: {target_root}")
    print("-" * 40)

    count_found = 0
    count_copied = 0

    # Recursively walk through all subdirectories under source_root
    for root, dirs, files in os.walk(source_root):
        for file in files:
            if file.lower().endswith(".csv"):
                count_found += 1
                src_path = Path(root) / file

                # Build base target name with prefix
                original_stem = Path(file).stem
                original_suffix = Path(file).suffix  # should be '.csv'
                base_name = f"{prefix}{original_stem}{original_suffix}"
                dest_path = target_root / base_name

                # Avoid name conflicts: if the same name exists, append an index
                if dest_path.exists():
                    stem = dest_path.stem
                    suffix = dest_path.suffix
                    idx = 1
                    while True:
                        new_name = f"{stem}_{idx}{suffix}"
                        new_dest = target_root / new_name
                        if not new_dest.exists():
                            dest_path = new_dest
                            break
                        idx += 1

                # Copy the file while preserving metadata
                shutil.copy2(src_path, dest_path)
                count_copied += 1
                rel_path = src_path.relative_to(source_root)
                print(f"[{count_copied}] Copied: {rel_path} -> {dest_path.name}")

    print("-" * 40)
    print(f"Total CSV files found: {count_found}")
    print(f"Total CSV files copied: {count_copied}")
    print(f"All files collected in: {target_root}")


if __name__ == "__main__":
    # Example template usage:
    desktop = Path.home() / "Desktop"

    # You can customize these three variables without touching the function
    SOURCE_FOLDER = desktop / "output_noise"          # e.g. your noise CSV folder
    TARGET_FOLDER = desktop / "output_noise_csv"      # e.g. collected CSV folder
    FILENAME_PREFIX = "noise_"                        # e.g. 'sine_', 'square_', 'noise_', etc.

    collect_csv_files(
        source_root=SOURCE_FOLDER,
        target_root=TARGET_FOLDER,
        prefix=FILENAME_PREFIX,
    )

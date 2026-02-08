import os
import shutil
from pathlib import Path
import argparse


def collect_png_files(
        source_root: Path,
        target_root: Path,
        prefix: str = "",
):
    """
    Recursively collect all PNG files from source_root and copy them flat into target_root.

    Parameters
    ----------
    source_root : Path
        Source folder containing PNG files (searched recursively through subdirectories).
    target_root : Path
        Target folder where all PNG files will be collected (flat structure).
    prefix : str, optional
        Optional prefix to add to output filenames (default: empty).
    """

    source_root = Path(source_root)
    target_root = Path(target_root)

    if not source_root.exists():
        print(f"Source folder does not exist: {source_root}")
        return

    target_root.mkdir(parents=True, exist_ok=True)

    print(f"Source folder: {source_root}")
    print(f"Target folder: {target_root}")
    print("-" * 50)

    count_found = 0
    count_copied = 0
    file_counter = 1

    # Recursively search all PNG files
    for root, dirs, files in os.walk(source_root):
        current_path = Path(root)
        for file in files:
            if file.lower().endswith('.png'):
                count_found += 1
                source_file = current_path / file

                # Generate target filename
                if prefix and not file.startswith(prefix):
                    # Add prefix and sequential numbering if prefix specified
                    new_filename = f"{prefix}{file_counter:04d}.png"
                    file_counter += 1
                else:
                    # Keep original name or use existing prefix
                    new_filename = file

                # Handle filename conflicts
                target_file = target_root / new_filename
                counter = 1
                while target_file.exists():
                    name_parts = new_filename.rsplit('.', 1)
                    new_filename = f"{name_parts[0]}_{counter:02d}.{name_parts[1]}"
                    target_file = target_root / new_filename
                    counter += 1

                try:
                    # Copy file preserving metadata
                    shutil.copy2(source_file, target_file)
                    count_copied += 1
                    rel_path = source_file.relative_to(source_root)
                    print(f"[✓ {count_copied}] {rel_path} -> {new_filename}")
                except Exception as e:
                    print(f"[✗] Copy failed {source_file.name}: {e}")

    print("-" * 50)
    print(f"Search complete!")
    print(f"Found {count_found} PNG files in source")
    print(f"Successfully copied {count_copied} files to target")
    print(f"Target folder: {target_root}")

    # Show first 20 files in target folder
    if count_copied > 0:
        print("\nFiles in target folder:")
        png_files = sorted(target_root.glob("*.png"))
        for i, png_file in enumerate(png_files[:20], 1):
            print(f"  {i:3d}. {png_file.name}")
        if len(png_files) > 20:
            print(f"  ... and {len(png_files) - 20} more files")


def main():
    """Command line interface with argparse"""
    parser = argparse.ArgumentParser(
        description='Recursively collect PNG files from source folder to target folder'
    )
    parser.add_argument('source', nargs='?', default=None,
                        help='Source folder path (default: Desktop/output_sine_txt_files)')
    parser.add_argument('-o', '--output', default=None,
                        help='Target folder name (default: Desktop/SineWave)')
    parser.add_argument('-p', '--prefix', default='',
                        help='Prefix to add to output filenames (default: none)')

    args = parser.parse_args()

    # Default desktop paths if not specified
    desktop = Path.home() / "Desktop"
    if args.source:
        source_path = Path(args.source)
    else:
        source_path = desktop / "output_sine"  # Template default

    if args.output:
        target_path = Path(args.output)
    else:
        target_path = desktop / "SineWave"  # Template default

    collect_png_files(source_path, target_path, args.prefix)


if __name__ == "__main__":
    # Template usage - customize these three variables:
    desktop = Path.home() / "Desktop"

    SOURCE_FOLDER = desktop / "output_sine"  # Change this
    TARGET_FOLDER = desktop / "SineWave"  # Change this  

    collect_png_files(
        source_root=SOURCE_FOLDER,
        target_root=TARGET_FOLDER,
        prefix=FILENAME_PREFIX,
    )

"""
Fix Windows-saved HDF5 weights files for cross-platform Linux / Render deployment.

Ensures every group containing backslashes ('\\') is duplicated with standard
POSIX forward slashes ('/'), so Keras on Linux can load weights without KeyError / ValueError.
"""

import h5py
from pathlib import Path

def convert_h5_to_posix(file_path: Path):
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    print(f"Processing: {file_path}")
    with h5py.File(file_path, "a") as f:
        keys_to_convert = [k for k in list(f.keys()) if "\\" in k]
        print(f"  Found {len(keys_to_convert)} backslash groups.")

        for k in keys_to_convert:
            posix_path = k.replace("\\", "/")
            parts = posix_path.split("/")
            curr = f
            for part in parts[:-1]:
                if part not in curr:
                    curr = curr.create_group(part)
                else:
                    curr = curr[part]
            leaf = parts[-1]
            if leaf not in curr:
                curr[leaf] = f[k]
                # print(f"    Linked: {k} -> {posix_path}")

    print(f"  Successfully updated {file_path} with POSIX hierarchy.")


if __name__ == "__main__":
    base = Path(".")
    convert_h5_to_posix(base / "models" / "retrieval" / "model.weights.h5")
    convert_h5_to_posix(base / "models" / "ranking" / "wide_only" / "model.weights.h5")
    convert_h5_to_posix(base / "models" / "ranking" / "wide_and_deep" / "model.weights.h5")

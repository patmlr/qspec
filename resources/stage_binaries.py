#!/usr/bin/env python3
import platform
import shutil
import sys
from pathlib import Path

# Detect platform
os_name = platform.system()
arch = platform.machine().lower()

PLATFORM_MAP = {
    ("Windows", "amd64"): ("windows-x64/Release", "win_amd64"),
    ("Windows", "arm64"): ("windows-arm64/Release", "win_arm64"),
    ("Linux", "x86_64"): ("linux-x64", "manylinux_2_28_x86_64"),
    ("Linux", "aarch64"): ("linux-arm64", "manylinux_2_28_aarch64"),
    ("Darwin", "x86_64"): ("macosx-x64", "macosx_12_0_x86_64"),
    ("Darwin", "arm64"): ("macosx-arm64", "macosx_12_0_arm64"),
}

key = (os_name, arch)
if key not in PLATFORM_MAP:
    print(f"Warning: Unsupported platform {os_name} {arch}, skipping binary staging")
    sys.exit(0)

lib_folder, PLAT_NAME = PLATFORM_MAP[key]

src = Path("qspec/_cpp/qspec_cpp/build") / lib_folder
dst = Path("qspec/_cpp/bin")

# Clear old staged binaries
if dst.exists():
    shutil.rmtree(dst)
dst.mkdir(parents=True, exist_ok=True)

# Copy binaries if present
if not src.exists() or not any(src.iterdir()):
    print(f"Warning: No binaries found for {lib_folder}. Continuing without them.")
else:
    for f in src.iterdir():
        try:
            shutil.copy2(f, dst)
            print(f"Copied {f.name} from {src} to {dst}")
        except Exception as e:
            print(f"Warning: Failed to copy {f.name} from {src} to {dst}: {e}")

print(f"Staging complete for {lib_folder}")
print(f"Determined platform tag {PLAT_NAME}")
sys.exit(0)

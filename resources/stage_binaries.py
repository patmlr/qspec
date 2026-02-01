#!/usr/bin/env python3
import platform
import shutil
import struct
import sys
from pathlib import Path

# Detect OS
os_name = platform.system()
machine = platform.machine().lower()
arch = struct.calcsize("P") * 8

if os_name == "Darwin":
    if machine == "arm64" and arch == 64:
        target_folder = "macos-arm64"
    elif machine == "x86_64" and arch == 64:
        target_folder = "macos-x64"
    else:
        print(f"Unsupported macOS architecture: {machine} ({arch}-bit)")
        sys.exit(0)

elif os_name == "Windows":
    if machine == "amd64" and arch == 64:
        target_folder = "windows-x64/Release"
    elif machine == "arm64" and arch == 64:
        target_folder = "windows-arm64/Release"
    else:
        print(f"Unsupported Windows architecture: {machine} ({arch}-bit)")
        sys.exit(0)

elif os_name == "Linux":
    if machine == "aarch64" and arch == 64:
        target_folder = "linux-aarch64"
    elif machine == "x86_64" and arch == 64:
        target_folder = "linux-x64"
    else:
        print(f"Unsupported Linux architecture: {machine} ({arch}-bit)")
        sys.exit(0)

else:
    print(f"Unsupported OS: {os_name}")
    sys.exit(0)

src = Path("qspec/_cpp/qspec_cpp/build") / target_folder
dst = Path("qspec/_cpp/bin")

if not src.exists():
    raise FileNotFoundError(f"No prebuilt binaries found for {target_folder}")

# Clear previous staged binaries
if dst.exists():
    shutil.rmtree(dst)
dst.mkdir(parents=True, exist_ok=True)

# Copy only target binaries
for f in src.iterdir():
    shutil.copy2(f, dst)

print(f"Staged {list(dst.iterdir())} for {target_folder}")

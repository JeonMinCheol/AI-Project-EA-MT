import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")

for candidate in (BASE_DIR, SRC_DIR):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from eamt.translation.lora_train import main as cli_main


if __name__ == "__main__":
    raise SystemExit(cli_main())

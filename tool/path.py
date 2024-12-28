from pathlib import Path

MAIN_DIR = Path(__file__).absolute().parent
BASE_DIR = MAIN_DIR.parent
DOP_DIR = BASE_DIR / ".cache"

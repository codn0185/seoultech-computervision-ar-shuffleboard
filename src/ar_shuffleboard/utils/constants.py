from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "config"
USER_CONFIG_PATH = CONFIG_DIR / "config.cfg"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "config_default.cfg"

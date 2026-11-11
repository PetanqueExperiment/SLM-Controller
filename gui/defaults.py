from pathlib import Path

HOME: Path = Path.home()

PETANQUE_DIR: Path = HOME / 'petanque_dev'

SLM_DIR: Path = PETANQUE_DIR / 'SLM'

DEVICES_DIR: Path = SLM_DIR / 'devices'
HOLO_DIR: Path = SLM_DIR / 'holograms'
PROFILE_DIR: Path = SLM_DIR / 'profiles'
SETTINGS_DIR: Path = SLM_DIR / 'settings'

SETTINGS_FILE: Path = SETTINGS_DIR / 'settings.json'

DEFAULT_PROFILE: str = 'default'

STARTUP_LAST_USED: str = 'last_used'
STARTUP_MODES: dict[str, str] = {STARTUP_LAST_USED: 'Last used'}
DEFAULT_STARTUP_MODE: str = STARTUP_LAST_USED

# HOLO_STATIC_DIR: Path = HOLO_DIR / 'static'
# HOLO_STATIC: str = 'static'

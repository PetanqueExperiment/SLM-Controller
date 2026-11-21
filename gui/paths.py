from pathlib import Path

HOME: Path = Path.home()

PETANQUE_DIR: Path = HOME / 'petanque_dev'

SLM_DIR: Path = PETANQUE_DIR / 'SLM'

DEVICES_DIR: Path = SLM_DIR / 'devices'
LOGS_DIR: Path = SLM_DIR / 'logs'
HOLO_DIR: Path = SLM_DIR / 'holograms'
PROFILE_DIR: Path = SLM_DIR / 'profiles'
SETTINGS_DIR: Path = SLM_DIR / 'settings'
PARAMETERS_DIR: Path = SLM_DIR / 'parameters'

SETTINGS_FILE: Path = SETTINGS_DIR / 'settings.json'
DEVICES_FILE: Path = DEVICES_DIR / 'devices.json'
PARAMETERS_FILE: Path = PARAMETERS_DIR / 'parameters.json'

DEFAULT_PROFILE: str = 'default'

# Default values of the GUI, tracked in the repository (defaults/): profiles/<name>.json only override them, and
# parameters/parameters.json is created from them.
DEFAULTS_DIR: Path = Path(__file__).resolve().parent.parent / 'defaults'
PROFILE_TEMPLATE_DIR: Path = DEFAULTS_DIR / 'profiles'
PROFILE_TEMPLATE_FILE: Path = PROFILE_TEMPLATE_DIR / 'default.json'
PARAMETERS_TEMPLATE_FILE: Path = DEFAULTS_DIR / 'parameters' / 'parameters.json'
SIMULATED_SLM_DEFAULTS_FILE: Path = DEFAULTS_DIR / 'devices' / 'simulated_slm.json'  # also used by a real SLM until it reports its own

STARTUP_LAST_USED: str = 'last_used'
STARTUP_FIXED: str = 'fixed'
STARTUP_MODES: dict[str, str] = {STARTUP_LAST_USED: 'Last used', STARTUP_FIXED: 'Fixed profile'}
DEFAULT_STARTUP_MODE: str = STARTUP_LAST_USED

# HOLO_STATIC_DIR: Path = HOLO_DIR / 'static'
# HOLO_STATIC: str = 'static'

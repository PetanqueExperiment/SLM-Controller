from pathlib import Path

HOME: Path = Path.home()

PETANQUE_DIR: Path = HOME / 'petanque_dev'

SLM_DIR: Path = PETANQUE_DIR / 'SLM'

HOLO_DIR: Path = PETANQUE_DIR / 'holograms'
HOLO_STATIC_DIR: Path = HOLO_DIR / 'static'

HOLO_STATIC: str = 'static'

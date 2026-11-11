"""SLM Controller GUI package."""

__all__: list[str] = ["SLMGUI"]


def __getattr__(name: str) -> type:
    if name == "SLMGUI":
        from gui.main_window import SLMGUI

        return SLMGUI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

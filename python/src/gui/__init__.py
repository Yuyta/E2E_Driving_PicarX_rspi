__all__ = ["PicarxMainWindow"]


def __getattr__(name):
    if name == "PicarxMainWindow":
        from src.gui.app import PicarxMainWindow

        return PicarxMainWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

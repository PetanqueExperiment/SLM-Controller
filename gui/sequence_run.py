"""Dynamic-mode sequence run on the SLM (no Qt): load the frames, show frame 0, then advance one frame per trigger.

Frame 0 is shown when the sequence is armed. Each trigger shows the next frame; after the last frame the next
trigger shows frame 0 again. The run ends when Stop is requested, when no trigger arrives within the timeout
(scan finished or aborted), or on an SDK error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np

RUN_STOPPED: str = 'stopped'
RUN_NO_TRIGGER: str = 'no trigger'
RUN_ERROR: str = 'error'


class SequenceSLM(Protocol):
    sequence_length: int | None

    def loadSequence(self, frames: Sequence[np.ndarray]) -> None: ...

    def selectFrame(self, frame: int, *, wait_for_trigger: bool = ..., timeout_ms: int = ..., flip_immediate: bool = ..., output_pulse: bool = ...) -> bool: ...

    def lastErrorMessage(self) -> str: ...


@dataclass
class SequenceRunResult:
    reason: str  # RUN_STOPPED, RUN_NO_TRIGGER or RUN_ERROR
    triggers: int  # triggers that advanced the frame
    last_frame: int | None  # frame shown on the SLM at the end (counted in software); None if none was shown
    message: str = ''


def playSequence(slm: SequenceSLM, frames: Sequence[np.ndarray], *, timeout_ms: int, flip_immediate: bool, stop_requested: Callable[[], bool], on_armed: Callable[[], None] | None = None) -> SequenceRunResult:
    """Load ``frames`` on the controller, show frame 0, then show the next frame on every trigger until the run ends.

    Blocks; run it off the GUI thread. ``stop_requested`` is checked after every trigger wait.
    """
    try:
        slm.loadSequence(frames)
        if not slm.selectFrame(0, wait_for_trigger=False, flip_immediate=flip_immediate):
            return SequenceRunResult(RUN_ERROR, 0, None, f'Select_image(0) failed. {slm.lastErrorMessage()}'.strip())
    except Exception as e:
        return SequenceRunResult(RUN_ERROR, 0, None, f'{type(e).__name__}: {e}')
    if on_armed is not None:
        on_armed()

    n_frames = int(slm.sequence_length)
    shown, triggers = 0, 0
    while not stop_requested():
        following = (shown + 1) % n_frames
        try:
            flipped = slm.selectFrame(following, wait_for_trigger=True, timeout_ms=timeout_ms, flip_immediate=flip_immediate)
        except Exception as e:
            return SequenceRunResult(RUN_ERROR, triggers, shown, f'{type(e).__name__}: {e}')
        if not flipped:
            if stop_requested():
                break
            return SequenceRunResult(RUN_NO_TRIGGER, triggers, shown, slm.lastErrorMessage())
        shown, triggers = following, triggers + 1
    return SequenceRunResult(RUN_STOPPED, triggers, shown)

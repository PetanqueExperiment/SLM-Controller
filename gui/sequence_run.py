"""Dynamic-mode sequence run on the SLM (no Qt): load the frames, show frame 0, then advance one frame per trigger.

Frame 0 is shown when the sequence is armed. Each trigger shows the next frame; after the last frame the next
trigger shows frame 0 again: each wait selects the frame that the coming trigger shows. Until the first trigger the
run waits without a time limit. After it, the run ends when no trigger arrives within the timeout (scan finished or
aborted). Stop or an SDK error end it at any time.

The first trigger wait after the untriggered display of frame 0 completes at once without a trigger. An extra wait
on frame 0 (already displayed, so nothing changes) absorbs it and is not counted. If that wait takes
SPURIOUS_COMPLETION_MAX_MS or longer, it was a real trigger: it is counted, and the following frames stay aligned
(trigger k shows frame k mod n).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np

RUN_STOPPED: str = 'stopped'
RUN_NO_TRIGGER: str = 'no trigger'
RUN_ERROR: str = 'error'
LOGGED_WAITS: int = 5
SPURIOUS_COMPLETION_MAX_MS: float = 2.0


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


def playSequence(slm: SequenceSLM, frames: Sequence[np.ndarray], *, timeout_ms: int, flip_immediate: bool, stop_requested: Callable[[], bool], on_armed: Callable[[], None] | None = None, on_trigger: Callable[[int], None] | None = None, log: Callable[[str], None] | None = None) -> SequenceRunResult:
    """Load ``frames`` on the controller, show frame 0, then show the next frame on every trigger until the run ends.

    Blocks; run it off the GUI thread. ``stop_requested`` is checked after every trigger wait (before the first
    trigger, waits of ``timeout_ms`` are repeated). ``on_trigger(count)`` is called after every trigger. ``log``
    receives the outcome and duration of the first LOGGED_WAITS trigger waits.
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
    triggers, shown = 0, 0
    try:
        completed, elapsed_ms = _timedTriggerWait(slm, 0, timeout_ms, flip_immediate)
    except Exception as e:
        return SequenceRunResult(RUN_ERROR, 0, 0, f'{type(e).__name__}: {e}')
    if completed and elapsed_ms < SPURIOUS_COMPLETION_MAX_MS:
        if log is not None:
            log(f'Arm: first trigger wait completed after {elapsed_ms:.2f} ms without a trigger (ignored)')
    elif completed:
        triggers = 1
        if log is not None:
            log(f'Arm: the first trigger came {elapsed_ms:.1f} ms after arming, during the extra wait: it kept frame 1 (should have shown frame {1 % n_frames + 1}); later frames stay aligned')
        if on_trigger is not None:
            on_trigger(triggers)
    elif log is not None:
        log(f'Arm: no early completion of the first trigger wait ({elapsed_ms:.1f} ms, timeout)')

    waits = 0
    while not stop_requested():
        following = (triggers + 1) % n_frames
        try:
            flipped, elapsed_ms = _timedTriggerWait(slm, following, timeout_ms, flip_immediate)
        except Exception as e:
            return SequenceRunResult(RUN_ERROR, triggers, shown, f'{type(e).__name__}: {e}')
        waits += 1
        if log is not None and waits <= LOGGED_WAITS:
            log(f'Sequence wait {waits}: frame {following + 1} selected, {"trigger" if flipped else "no trigger"} after {elapsed_ms:.2f} ms')
        if not flipped:
            if stop_requested():
                break
            if triggers == 0:
                continue
            return SequenceRunResult(RUN_NO_TRIGGER, triggers, shown, slm.lastErrorMessage())
        triggers, shown = triggers + 1, following
        if on_trigger is not None:
            on_trigger(triggers)
    return SequenceRunResult(RUN_STOPPED, triggers, shown)


def _timedTriggerWait(slm: SequenceSLM, frame: int, timeout_ms: int, flip_immediate: bool) -> tuple[bool, float]:
    """Select ``frame`` to be shown on the next trigger; (trigger received, duration [ms])."""
    start_s = time.perf_counter()
    flipped = slm.selectFrame(frame, wait_for_trigger=True, timeout_ms=timeout_ms, flip_immediate=flip_immediate)
    return flipped, (time.perf_counter() - start_s) * 1000

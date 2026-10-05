"""
Light-weight state machine logger for Autopatcher attempts.
- Creates ONE date-stamped *session* folder the first time it is used.
- Creates one sub-folder per attempt -> session/attempt_<n>/.
- Writes one *.pickle* per state -> <n>_<state>_<started>.pickle.
- Writes one *.json* per state -> <n>_<state>_<started>.json.
- Uses Unix-epoch seconds for start/finish timestamps (floats).
"""

import functools
import json
import os
import pickle
import threading
import time
from typing import Dict


from patcherbot.controller import RequestedAbortException, RequestedSuccessException


class StateMachineLogger:
    # ---------- outcome codes ---------- #
    SUCCESS = 0
    FAILURE = 1
    ABORTED = 2

    # ---------- statetype codes ---------- #
    CLASSIC = 0
    MANUAL = 1
    AGENT = 2
    TRAINING = 3
    NOMODE = 4
    ADAPTIVE = 5


    # ---------- class-level session storage ---------- #
    _session_folder: str | None = None          # set on first use

    # ---------- constructor ---------- #
    def __init__(self,
                 base_path: str = "experiments/Data/state_recorder_data/",
                 attempt_id: int = 1):
        """
        Parameters
        ----------
        base_path  : str
            Root folder **without** the date stamp. A single session folder
            (YYYY_MM_DD-HH_MM) is created once under this root.
        attempt_id : int
            Numeric counter supplied by the caller (AutoPatcher).
        """
        # Initialise the session folder only once
        if StateMachineLogger._session_folder is None:
            date_stamp = time.strftime("%Y_%m_%d-%H_%M")
            StateMachineLogger._session_folder = os.path.join(base_path,
                                                              date_stamp)
            os.makedirs(StateMachineLogger._session_folder, exist_ok=True)

        self.session_folder = StateMachineLogger._session_folder
        self.attempt_id     = attempt_id
        self.attempt_path   = os.path.join(self.session_folder,
                                           f"attempt_{attempt_id}")
        os.makedirs(self.attempt_path, exist_ok=True)

        self._states: Dict[str, Dict] = {}
        self._lock = threading.Lock()           # thread-safe writes

    # ------------------------------------------------------------------ #
    # public API – called by the decorator
    # ------------------------------------------------------------------ #
    def start(self, state: str, system_mode: int | None = None) -> None:
        """
        Stamp the *started* time (first call only) and record the
        system-mode code supplied by the caller.

        Parameters
        ----------
        state        : str
            Name of the state being entered.
        system_mode  : int | None
            One of the CLASSIC / MANUAL / AGENT / TRAINING / ADAPTIVE /
            NOMODE codes.
            If None, NOMODE is stored.
        """
        if system_mode is None:
            system_mode = StateMachineLogger.NOMODE

        with self._lock:
            record = self._states.get(state)
            if record is None or record.get("finished") is not None:
                epoch_ts = time.time()                               # seconds
                self._states[state] = {
                    "started":     epoch_ts,
                    "finished":    None,
                    "outcome":     None,
                    "system_mode": system_mode
                }
            else:
                # State already active: refresh the mode in case it changed.
                record["system_mode"] = system_mode

    def finish(self, state: str, outcome: int) -> None:
        """Stamp *finished*, set outcome, and write the pickle and JSON files."""
        with self._lock:
            rec = self._states[state]                      # must exist
            rec["finished"] = time.time()
            rec["outcome"]  = outcome
            self._save(state, rec)
            # Drop finished records so repeat runs get fresh timestamps/files.
            del self._states[state]

    # ------------------------------------------------------------------ #
    # internal helper
    # ------------------------------------------------------------------ #
    def _save(self, state: str, record: Dict) -> None:
        # Defensive copy and validation
        def _coerce_timestamp(value):
            if value is None:
                return None
            if isinstance(value, (int, float)):
                return float(value)
            raise TypeError(f"Invalid timestamp value for {state!r}: {value!r}")

        rec = {
            "started":     _coerce_timestamp(record.get("started")),
            "finished":    _coerce_timestamp(record.get("finished")),
            "outcome":     int(record["outcome"]) if record.get("outcome") is not None else None,
            "system_mode": int(record["system_mode"]) if record.get("system_mode") is not None else None,
        }

        fname_suffix = "unknown"
        if rec["started"] is not None:
            fname_suffix = str(int(rec["started"] * 1_000))

        fname_base = f"{self.attempt_id}_{state}_{fname_suffix}"
        pickle_path = os.path.join(self.attempt_path, f"{fname_base}.pickle")
        with open(pickle_path, "wb") as f:
            pickle.dump(rec, f)

        json_path = os.path.join(self.attempt_path, f"{fname_base}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)



# ---------------------------------------------------------------------- #
# decorator – attach to each top-level sub-method
# ---------------------------------------------------------------------- #
def record_state(state_name: str):
    """
    Decorator to log state transitions with StateMachineLogger.

    >>> @record_state("gigaseal")
    >>> def gigaseal(self): ...
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            logger = self._get_state_recorder()    # lazy / singleton

            # --- map self.config.mode → numeric code ---
            mode_str = getattr(getattr(self, "config", None), "mode", None)
            mode_map = {
                "Classic": StateMachineLogger.CLASSIC,
                "Manual":  StateMachineLogger.MANUAL,
                "Agent":   StateMachineLogger.AGENT,
                "Training": StateMachineLogger.TRAINING,
                'Adaptive': StateMachineLogger.ADAPTIVE,
            }
            mode_code = mode_map.get(mode_str, StateMachineLogger.NOMODE)

            logger.start(state_name, mode_code)

            try:
                # inside wrapper(), in the try block after fn returns
                result   = fn(self, *args, **kwargs)
                aborted  = bool(getattr(self, "abort_requested", False))
                success  = bool(getattr(self, "success_requested", False))

                if aborted:
                    outcome = StateMachineLogger.ABORTED
                elif success:
                    outcome = StateMachineLogger.SUCCESS
                else:
                    # didn't abort and no explicit success flag -> treat as FAILURE
                    outcome = StateMachineLogger.FAILURE

                logger.finish(state_name, outcome)
                return result

            except RequestedSuccessException:
                logger.finish(state_name, StateMachineLogger.SUCCESS)
                raise

            except RequestedAbortException:
                logger.finish(state_name, StateMachineLogger.ABORTED)
                raise

            except Exception as exc:
                msg     = str(exc).lower()
                aborted = "abort" in msg
                success = "success" in msg
                if success:
                    outcome = StateMachineLogger.SUCCESS
                elif aborted:
                    outcome = StateMachineLogger.ABORTED
                else:
                    outcome = StateMachineLogger.FAILURE
                logger.finish(state_name, outcome)
                raise

            finally:
                # If NOT inside patch(), reset logger so the next call
                # becomes a fresh *attempt*.
                if not getattr(self, "_in_patch", False):
                    self._state_recorder = None
        return wrapper
    return decorator

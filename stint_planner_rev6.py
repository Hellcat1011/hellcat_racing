# stint_planner_rev6.py
# Endurance Kart — Live Timer Stint Planner (with Caution + Max Stint support)
#
# Key features:
# - Driver-first fairness allocation (equal TOTAL time per driver regardless of stint count)
# - Race Mode: read-only timing console, sidebar hidden, single EXIT button
# - Normal Mode: editable driver assignments in a form
# - Live updates with user-entered refresh interval (0–60s)
# - Small Reset popover w/ confirmation
# - Planning view (normal mode only): Target Totals by Driver w/ green/yellow/red status
# - CAUTION phase (start/end multiple times):
#     - Race countdown continues
#     - Stint timers and pit timer freeze (do NOT count toward drivers)
#     - Caution time is subtracted from total available drive time
#     - Shows total caution time accumulated + small caution indicator (no current caution duration)
# - Max stint time:
#     - Default 50 min (set to 0 to disable)
#     - Warn-only mode (default) and optional Hard-cap mode with infeasibility detection
#     - Live banner + row tint when max is exceeded

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_extras.stylable_container import stylable_container

st.set_page_config(page_title="Endurance Kart — Timer Planner", layout="wide")

# Ensure this exists before sidebar reads it
if "race_mode_state" not in st.session_state:
    st.session_state.race_mode_state = False


# ----------------------------
# Formatting helpers
# ----------------------------
def fmt_hms(seconds: float) -> str:
    s = int(round(seconds))
    neg = s < 0
    s = abs(s)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    out = f"{h:d}:{m:02d}:{sec:02d}"
    return f"-{out}" if neg else out


def fmt_mmss(seconds: Optional[float]) -> str:
    if seconds is None:
        return ""
    s = int(round(seconds))
    neg = s < 0
    s = abs(s)
    m = s // 60
    sec = s % 60
    out = f"{m:d}:{sec:02d}"
    return f"-{out}" if neg else out


# ----------------------------
# Fairness math
# ----------------------------
def redistribute_overshoot(need: Dict[str, float]) -> Dict[str, float]:
    """
    need in minutes; negative means a driver has exceeded fair share.
    Overshoot is removed evenly from remaining-positive drivers.
    """
    need = dict(need)
    while True:
        overshoot = sum(-v for v in need.values() if v < 0)
        if overshoot <= 1e-9:
            return {k: max(0.0, v) for k, v in need.items()}

        for k in list(need.keys()):
            if need[k] < 0:
                need[k] = 0.0

        positives = [k for k, v in need.items() if v > 0]
        if not positives:
            return {k: 0.0 for k in need.keys()}

        share = overshoot / len(positives)
        for k in positives:
            need[k] -= share


def compute_targets(
    race_duration_min: float,
    min_pit_stops: int,
    pit_seconds: float,
    drivers: List[str],
    caution_total_sec: float,
) -> tuple[float, float]:
    """
    Total driving time excludes:
      - minimum required pit time
      - accumulated caution time (completed cautions)
    """
    pit_min = pit_seconds / 60.0
    caution_min = caution_total_sec / 60.0

    total_drive = race_duration_min - (min_pit_stops * pit_min) - caution_min
    total_drive = max(0.0, total_drive)
    per_driver = total_drive / max(1, len(drivers))
    return total_drive, per_driver


def compute_target_stint_minutes_driver_first(
    stint_drivers: List[Optional[str]],
    drivers: List[str],
    per_driver_target_min: float,
    actual_driven_min_by_driver: Dict[str, float],
    remaining_required_stints: int,
    max_stint_min: float = 0.0,
    hard_cap: bool = False,
) -> tuple[List[float], Dict[str, float]]:
    """
    DRIVER-FIRST allocation:
    - Compute each driver's remaining NEED (fair share - actual)
    - Split each driver's need evenly across their remaining assigned stints
    - Unassigned stints (blank driver) share the remaining pool evenly

    Max stint:
      - Warn-only: caller can highlight targets > max
      - Hard-cap: each stint target is clamped to max_stint_min
                 Any leftover need that can't be allocated indicates infeasibility.

    Returns: (targets_minutes, remaining_need_minutes_by_driver_after_allocation)
    """
    need = {d: per_driver_target_min - actual_driven_min_by_driver.get(d, 0.0) for d in drivers}
    need = redistribute_overshoot(need)

    remaining_counts = {d: 0 for d in drivers}
    unassigned_count = 0
    for d in stint_drivers[:remaining_required_stints]:
        if isinstance(d, str) and d in remaining_counts:
            remaining_counts[d] += 1
        else:
            unassigned_count += 1

    targets: List[float] = []
    cap = max(0.0, float(max_stint_min))

    for i in range(remaining_required_stints):
        d = stint_drivers[i] if i < len(stint_drivers) else None

        if isinstance(d, str) and d in remaining_counts and remaining_counts[d] > 0:
            base = need[d] / remaining_counts[d] if remaining_counts[d] else 0.0
            planned = base

            if hard_cap and cap > 0:
                planned = min(planned, cap)

            targets.append(planned)

            # Only reduce need by what we planned (if capped, leftover stays)
            need[d] = max(0.0, need[d] - planned)
            remaining_counts[d] -= 1

        else:
            # Unassigned stint: split remaining pool evenly across unassigned stints
            remaining_drive = sum(need.values())
            planned = (remaining_drive / unassigned_count) if unassigned_count > 0 else 0.0

            if hard_cap and cap > 0:
                planned = min(planned, cap)

            targets.append(planned)

            # Reduce pool proportionally across drivers
            if remaining_drive > 1e-9 and planned > 0:
                for k in list(need.keys()):
                    if need[k] > 0:
                        need[k] = max(0.0, need[k] - planned * (need[k] / remaining_drive))

            unassigned_count = max(0, unassigned_count - 1)

    return targets, need

def apply_early_buffer_to_targets(
    targets_min: List[float],
    completed_count: int,
    num_stints: int,
    buffer_pool_min: float,
    buffer_first_n_stints: int,
    max_stint_min: float,
) -> tuple[List[float], float, float, float]:
    """
    Post-process target minutes (remaining stints only):

    - Add buffer minutes evenly to eligible early stints (global stint # <= buffer_first_n_stints),
      but only for stints that are still remaining (not completed).
    - Clamp adds so a target never exceeds max_stint_min (if max enabled).
    - Remove the same total minutes from later remaining stints (global stint # > buffer_first_n_stints),
      proportionally to their current target minutes, never going below 0.

    Returns: (new_targets_min, total_added, total_removed, removal_shortfall)
    """
    # Fast exits
    if buffer_pool_min <= 0 or buffer_first_n_stints <= 0 or not targets_min:
        return list(targets_min), 0.0, 0.0, 0.0

    cap = float(max_stint_min) if float(max_stint_min) > 0 else None

    # Determine which remaining stints are eligible for buffer (by global stint number)
    eligible: List[int] = []
    later: List[int] = []

    for j in range(len(targets_min)):
        global_idx = completed_count + j      # 0-based in race
        stint_no = global_idx + 1             # 1-based for humans

        if stint_no <= buffer_first_n_stints:
            eligible.append(j)
        else:
            later.append(j)

    if not eligible or not later:
        # If no later stints exist, we can't keep total drive time constant.
        return list(targets_min), 0.0, 0.0, float(buffer_pool_min)

    new_targets = list(targets_min)

    # Add buffer evenly across eligible stints (with optional max clamp)
    per = float(buffer_pool_min) / len(eligible)
    total_added = 0.0

    for j in eligible:
        before = float(new_targets[j])
        after = before + per

        if cap is not None:
            after = min(after, cap)

        after = max(0.0, after)
        new_targets[j] = after
        total_added += (after - before)

    if total_added <= 1e-9:
        return new_targets, 0.0, 0.0, 0.0

    # Remove total_added from later stints proportionally (water-filling)
    remaining_to_remove = total_added
    total_removed = 0.0

    for _ in range(4):
        if remaining_to_remove <= 1e-9:
            break

        pool = sum(new_targets[j] for j in later if new_targets[j] > 1e-9)
        if pool <= 1e-9:
            break

        removed_this_pass = 0.0
        for j in later:
            if new_targets[j] <= 1e-9:
                continue

            share = remaining_to_remove * (new_targets[j] / pool)
            before = new_targets[j]
            new_targets[j] = max(0.0, before - share)
            removed_this_pass += (before - new_targets[j])

        total_removed += removed_this_pass
        remaining_to_remove = max(0.0, total_added - total_removed)

    shortfall = max(0.0, total_added - total_removed)
    return new_targets, total_added, total_removed, shortfall



# ----------------------------
# Race state
# ----------------------------
@dataclass
class StintRecord:
    target_sec: float = 0.0
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None
    paused_sec: float = 0.0  # total frozen time during this stint (cautions)


def ensure_state(num_stints: int):
    if "phase" not in st.session_state:
        st.session_state.phase = "idle"  # idle | stint | pit | caution | finished
    if "race_start_ts" not in st.session_state:
        st.session_state.race_start_ts = None
    if "active_stint_idx" not in st.session_state:
        st.session_state.active_stint_idx = 0  # 0-based
    if "pit_start_ts" not in st.session_state:
        st.session_state.pit_start_ts = None

    if "stints" not in st.session_state or len(st.session_state.stints) != num_stints:
        st.session_state.stints = [StintRecord() for _ in range(num_stints)]
    if "driver_assignments" not in st.session_state or len(st.session_state.driver_assignments) != num_stints:
        st.session_state.driver_assignments = [""] * num_stints

    # Live update controls
    if "live_mode_user" not in st.session_state:
        st.session_state.live_mode_user = True
    if "auto_paused" not in st.session_state:
        st.session_state.auto_paused = False
    if "auto_pause_reason" not in st.session_state:
        st.session_state.auto_pause_reason = ""

    # Caution tracking
    if "caution_total_sec" not in st.session_state:
        st.session_state.caution_total_sec = 0.0  # completed cautions only
    if "caution_start_ts" not in st.session_state:
        st.session_state.caution_start_ts = None
    if "phase_before_caution" not in st.session_state:
        st.session_state.phase_before_caution = None

    # Pit pause tracking (freeze pit countdown during caution)
    if "pit_paused_sec" not in st.session_state:
        st.session_state.pit_paused_sec = 0.0


def reset_race(num_stints: int):
    st.session_state.phase = "idle"
    st.session_state.race_start_ts = None
    st.session_state.active_stint_idx = 0
    st.session_state.pit_start_ts = None
    st.session_state.pit_paused_sec = 0.0

    st.session_state.stints = [StintRecord() for _ in range(num_stints)]
    st.session_state.driver_assignments = [""] * num_stints

    st.session_state.auto_paused = False
    st.session_state.auto_pause_reason = ""

    st.session_state.caution_total_sec = 0.0
    st.session_state.caution_start_ts = None
    st.session_state.phase_before_caution = None


def effective_live_mode() -> bool:
    return bool(st.session_state.live_mode_user) and (not bool(st.session_state.auto_paused))


def stint_elapsed_sec(rec: StintRecord, now_ts: float) -> Optional[float]:
    if rec.start_ts is None:
        return None
    if rec.end_ts is None:
        return max(0.0, (now_ts - rec.start_ts) - rec.paused_sec)
    return max(0.0, (rec.end_ts - rec.start_ts) - rec.paused_sec)


# ----------------------------
# UI header
# ----------------------------
st.title("Endurance Kart — Live Timer Stint Planner")

DEFAULT_RACE_MIN = 480.0
DEFAULT_MIN_PIT_STOPS = 12
DEFAULT_MAX_STINT_MIN = 50.0

with st.sidebar:
    st.header("Race Setup")
    race_duration_min = st.number_input("Race duration (minutes)", min_value=1.0, value=DEFAULT_RACE_MIN, step=1.0)
    min_pit_stops = st.number_input("Minimum pit stops", min_value=0, value=DEFAULT_MIN_PIT_STOPS, step=1)
    pit_seconds = st.number_input("Pit stop timer (seconds)", min_value=0.0, value=120.0, step=1.0)

    max_stint_min = st.number_input(
        "Max stint time (minutes, set 0 to disable)",
        min_value=0.0,
        value=DEFAULT_MAX_STINT_MIN,
        step=1.0,
        help="If > 0, the app will warn (or cap) when targets exceed this maximum.",
    )

    max_mode = st.selectbox(
        "Max stint handling",
        ["Warn only", "Hard cap targets"],
        index=0,
        help="Warn only keeps fairness-optimal targets. Hard cap clamps targets to max (may be infeasible).",
    )

    st.divider()
    st.subheader("Strategy (optional)")

    buffer_pool_min = st.number_input(
        "Early stint buffer pool (minutes)",
        min_value=0.0,
        value=0.0,
        step=1.0,
        help="Total minutes to front-load into early stints (then removed from later stints). Set 0 to disable."
    )

    buffer_first_n_stints = st.number_input(
        "Apply buffer within first N stints",
        min_value=0,
        value=0,
        step=1,
        help="Example: N=3 means apply to stints 1–3 (only if they haven't been completed yet). Set 0 to disable."
    )


    st.divider()
    st.header("Drivers")
    driver_text = st.text_input("Driver names (comma-separated)", value="A, B, C")
    drivers = [d.strip() for d in driver_text.split(",") if d.strip()]

    st.divider()
    st.header("Live Updates")
    st.checkbox("Live mode (update timers)", value=st.session_state.get("live_mode_user", True), key="live_mode_user")
    refresh_sec = st.number_input(
        "Refresh interval (seconds, 0–60)",
        min_value=0.0,
        max_value=60.0,
        value=2.0,
        step=0.5,
        help="Set to 0 for no auto-refresh. Typical: 2–5 sec.",
    )

    st.divider()
    st.header("Display")
    if not st.session_state.race_mode_state:
        if st.button("ENTER RACE MODE", use_container_width=True):
            st.session_state.race_mode_state = True
            st.rerun()
    st.caption("Tip: press F11 in your browser for true fullscreen.")

if not drivers:
    st.warning("Add at least one driver name in the sidebar.")
    st.stop()

min_stints = int(min_pit_stops) + 1
num_stints = min_stints

ensure_state(num_stints)
now = time.time()

# ----------------------------
# Race Mode CSS (fullscreen-ish)
# ----------------------------
if st.session_state.race_mode_state:
    st.markdown(
        """
        <style>
        header {visibility: hidden;}
        footer {visibility: hidden;}
        [data-testid="stToolbar"] {display: none;}
        [data-testid="stSidebar"] {display: none;}
        .block-container {padding-top: 0.8rem; padding-bottom: 0.8rem; max-width: 100%;}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------
# Auto-refresh (only when it matters)
# - include caution so race countdown keeps updating
# ----------------------------
refresh_sec_clamped = max(0.0, min(float(refresh_sec), 60.0))
if effective_live_mode() and st.session_state.phase in ("stint", "pit", "caution") and refresh_sec_clamped > 0:
    st_autorefresh(interval=int(refresh_sec_clamped * 1000), key="tick")

# ----------------------------
# Auto-advance: PIT -> next STINT (not during caution)
# ----------------------------
if st.session_state.phase == "pit" and st.session_state.pit_start_ts is not None:
    elapsed_pit = (now - st.session_state.pit_start_ts) - float(st.session_state.pit_paused_sec)
    if elapsed_pit >= pit_seconds:
        next_idx = st.session_state.active_stint_idx + 1
        if next_idx >= num_stints:
            st.session_state.phase = "finished"
        else:
            st.session_state.active_stint_idx = next_idx
            st.session_state.phase = "stint"
            st.session_state.pit_start_ts = None
            st.session_state.pit_paused_sec = 0.0
            st.session_state.stints[next_idx].start_ts = now

# ----------------------------
# Compute targets (fairness)
# ----------------------------
total_drive_min, per_driver_target_min = compute_targets(
    race_duration_min=float(race_duration_min),
    min_pit_stops=int(min_pit_stops),
    pit_seconds=float(pit_seconds),
    drivers=drivers,
    caution_total_sec=float(st.session_state.caution_total_sec),
)

# Actual driven minutes per driver from completed stints
actual_min = {d: 0.0 for d in drivers}
for i, rec in enumerate(st.session_state.stints):
    if rec.start_ts and rec.end_ts:
        dur_min = stint_elapsed_sec(rec, now) / 60.0
        d = st.session_state.driver_assignments[i]
        if d in actual_min:
            actual_min[d] += max(0.0, dur_min)

completed_count = sum(1 for r in st.session_state.stints if r.end_ts is not None)
remaining_required = max(0, num_stints - completed_count)

remaining_drivers: List[Optional[str]] = []
for i in range(completed_count, num_stints):
    remaining_drivers.append(st.session_state.driver_assignments[i] if st.session_state.driver_assignments[i] else None)

hard_cap = (max_mode == "Hard cap targets")

target_min_list, leftover_need = compute_target_stint_minutes_driver_first(
    stint_drivers=remaining_drivers,
    drivers=drivers,
    per_driver_target_min=per_driver_target_min,
    actual_driven_min_by_driver=actual_min,
    remaining_required_stints=remaining_required,
    max_stint_min=float(max_stint_min),
    hard_cap=hard_cap,
)

target_min_list, buffer_added, buffer_removed, buffer_shortfall = apply_early_buffer_to_targets(
    targets_min=target_min_list,
    completed_count=completed_count,
    num_stints=num_stints,
    buffer_pool_min=float(buffer_pool_min),
    buffer_first_n_stints=int(buffer_first_n_stints),
    max_stint_min=float(max_stint_min),
)


for j in range(remaining_required):
    idx = completed_count + j
    if idx < num_stints:
        st.session_state.stints[idx].target_sec = float(target_min_list[j] * 60.0)

# Infeasibility warning when hard-capping targets
if hard_cap and float(max_stint_min) > 0:
    leftover_total = sum(leftover_need.values())
    if leftover_total > 0.25:  # minutes; small tolerance
        st.error(
            "Max stint cap makes the plan infeasible with current assignments. "
            f"About {leftover_total:.1f} minutes of drive time cannot be allocated without exceeding the max. "
            "In the real race, at least one stint will need to go longer, or you must add more stints / adjust assignments."
        )

# ----------------------------
# Top controls row
# ----------------------------
# Layout: [Start] [Pause/Resume + Reset + Caution] [Race Timer + indicators + exit]
col_start, col_mid, col_right = st.columns([1.6, 1.2, 1.2])

with col_start:
    # Disable stint button during pit, caution, finished
    button_disabled = (st.session_state.phase in ("pit", "caution", "finished"))
    label = "START NEXT STINT" if st.session_state.phase in ("idle", "stint") else (
        "CAUTION ACTIVE" if st.session_state.phase == "caution" else "PIT IN PROGRESS"
    )

    with stylable_container(
        key="btn_start_next_stint",
        css_styles="""
        button {
            background-color: #2ecc71 !important;
            color: black !important;
            height: 90px !important;
            font-size: 28px !important;
            font-weight: 900 !important;
            border-radius: 12px !important;
        }
        button:disabled {
            background-color: #95a5a6 !important;
            color: #2c3e50 !important;
        }
        """,
    ):
        if st.button(label, use_container_width=True, disabled=button_disabled):
            if st.session_state.phase == "idle":
                st.session_state.phase = "stint"
                st.session_state.race_start_ts = now
                st.session_state.active_stint_idx = 0
                st.session_state.stints[0].start_ts = now
            elif st.session_state.phase == "stint":
                idx = st.session_state.active_stint_idx
                st.session_state.stints[idx].end_ts = now
                if idx >= num_stints - 1:
                    st.session_state.phase = "finished"
                else:
                    st.session_state.phase = "pit"
                    st.session_state.pit_start_ts = now
                    st.session_state.pit_paused_sec = 0.0

with col_mid:
    st.metric("Minimum Stints", f"{num_stints}")
    st.metric("Target per Driver (min)", f"{per_driver_target_min:.2f}")

    # Pause/Resume (manual pause for live updates)
    if st.session_state.auto_paused:
        st.warning(f"Paused: {st.session_state.auto_pause_reason}")
        with stylable_container(
            key="btn_resume_live",
            css_styles="""
            button {
                background-color: #2ecc71 !important;
                color: black !important;
                font-weight: 900 !important;
                border-radius: 12px !important;
            }
            """,
        ):
            if st.button("RESUME LIVE UPDATES", use_container_width=True):
                st.session_state.auto_paused = False
                st.session_state.auto_pause_reason = ""
                st.rerun()
    else:
        with stylable_container(
            key="btn_pause_live",
            css_styles="""
            button {
                background-color: #e67e22 !important;
                color: white !important;
                font-weight: 900 !important;
                border-radius: 12px !important;
            }
            button:disabled {
                background-color: #95a5a6 !important;
                color: #2c3e50 !important;
            }
            """,
        ):
            if st.button("PAUSE LIVE UPDATES", use_container_width=True, disabled=not effective_live_mode()):
                st.session_state.auto_paused = True
                st.session_state.auto_pause_reason = "Manual pause"
                st.rerun()

    # CAUTION toggle (only after race starts, not finished)
    can_toggle_caution = (st.session_state.race_start_ts is not None) and (st.session_state.phase != "finished")
    caution_label = "END CAUTION" if st.session_state.phase == "caution" else "START CAUTION"

    caution_bg = "#3498db" if st.session_state.phase == "caution" else "#f1c40f"
    caution_fg = "white" if st.session_state.phase == "caution" else "black"

    with stylable_container(
        key="btn_caution_toggle",
        css_styles=f"""
        button {{
            background-color: {caution_bg} !important;
            color: {caution_fg} !important;
            font-weight: 900 !important;
            border-radius: 12px !important;
        }}
        button:disabled {{
            background-color: #95a5a6 !important;
            color: #2c3e50 !important;
        }}
        """,
    ):
        if st.button(caution_label, use_container_width=True, disabled=not can_toggle_caution):
            if st.session_state.phase != "caution":
                st.session_state.phase_before_caution = st.session_state.phase
                st.session_state.caution_start_ts = now
                st.session_state.phase = "caution"
            else:
                start_ts = st.session_state.caution_start_ts
                if start_ts is not None:
                    dur = max(0.0, now - start_ts)
                    st.session_state.caution_total_sec += dur

                    prev = st.session_state.phase_before_caution
                    if prev == "stint":
                        idx = st.session_state.active_stint_idx
                        if 0 <= idx < len(st.session_state.stints):
                            st.session_state.stints[idx].paused_sec += dur
                    elif prev == "pit":
                        st.session_state.pit_paused_sec += dur

                    st.session_state.phase = prev if prev in ("stint", "pit") else "stint"

                st.session_state.caution_start_ts = None
                st.session_state.phase_before_caution = None

            st.rerun()

    # Small reset control with confirmation
    reset_spacer, reset_col = st.columns([3, 1])
    with reset_col:
        with st.popover("Reset"):
            st.warning(
                "This will reset the race timers and log.\n\n"
                "All recorded stints, pit timing, and race timing info will be lost."
            )
            confirm = st.checkbox("I understand — reset everything")

            with stylable_container(
                key="btn_confirm_reset",
                css_styles="""
                button {
                    background-color: #e74c3c !important;
                    color: white !important;
                    font-weight: 900 !important;
                    border-radius: 12px !important;
                }
                button:disabled {
                    background-color: #95a5a6 !important;
                    color: #2c3e50 !important;
                }
                """,
            ):
                if st.button("CONFIRM RESET", disabled=not confirm, use_container_width=True):
                    reset_race(num_stints)
                    st.rerun()

with col_right:
    # Race countdown (keeps running during caution)
    if st.session_state.race_start_ts is None:
        race_remaining = float(race_duration_min) * 60.0
    else:
        race_elapsed = now - st.session_state.race_start_ts
        race_remaining = (float(race_duration_min) * 60.0) - race_elapsed

    st.markdown("### Race Countdown")
    st.markdown(
        f"<div style='font-size: 54px; font-weight: 950;'>{fmt_hms(race_remaining)}</div>",
        unsafe_allow_html=True,
    )

    # Max stint banner
    if float(max_stint_min) > 0 and st.session_state.phase == "stint":
        idx = st.session_state.active_stint_idx
        rec = st.session_state.stints[idx]
        dur_sec = stint_elapsed_sec(rec, now)
        if dur_sec is not None and (dur_sec / 60.0) > float(max_stint_min):
            st.error("⛔ MAX STINT TIME EXCEEDED")

    # Small caution indicator + accumulated caution time (no current duration)
    if st.session_state.phase == "caution":
        st.markdown("🟡 **CAUTION ACTIVE**")
    st.caption(f"Total caution time: {fmt_mmss(float(st.session_state.caution_total_sec))}")

    # Single EXIT button (Race Mode only)
    if st.session_state.race_mode_state:
        with stylable_container(
            key="btn_exit_race_mode",
            css_styles="""
            button {
                background-color: #bdc3c7 !important;
                color: black !important;
                font-weight: 900 !important;
                border-radius: 12px !important;
            }
            """,
        ):
            if st.button("EXIT RACE MODE (to edit drivers)", use_container_width=True):
                st.session_state.race_mode_state = False
                st.rerun()

if not st.session_state.race_mode_state:
    st.divider()
    c_phase, c_active = st.columns([1, 2])
    with c_phase:
        st.write(f"**Phase:** {st.session_state.phase.upper()}")
    with c_active:
        st.write(f"**Active stint row:** {st.session_state.active_stint_idx + 1} (of {num_stints})")


# ----------------------------
# Timing table (read-only + active row highlight)
# ----------------------------
rows = []
for i in range(num_stints):
    rec = st.session_state.stints[i]
    driver = st.session_state.driver_assignments[i]
    target_sec = float(rec.target_sec or 0.0)

    stint_duration_sec = None
    stint_remaining_sec = None
    pit_remaining_sec = None

    if not (rec.start_ts is None and rec.end_ts is None):
        dur = stint_elapsed_sec(rec, now)
        if dur is not None:
            stint_duration_sec = dur
            stint_remaining_sec = target_sec - dur

    # Pit countdown displayed on the stint row that just ended (the driver who pitted)
    # NOTE: During PIT, active_stint_idx is still the stint that just ended.
    if st.session_state.phase == "pit" and st.session_state.pit_start_ts is not None:
        if i == st.session_state.active_stint_idx:
            pit_elapsed = (now - st.session_state.pit_start_ts) - float(st.session_state.pit_paused_sec)
            pit_remaining_sec = pit_seconds - pit_elapsed


    pit_cell = "NA" if i == num_stints - 1 else (fmt_mmss(pit_remaining_sec) if pit_remaining_sec is not None else "")

    rows.append(
        {
            "Stint #": i + 1,
            "Driver": driver,
            "Target Stint Time": fmt_mmss(target_sec),
            "Stint Duration": fmt_mmss(stint_duration_sec),
            "Stint Time Remaining": fmt_mmss(stint_remaining_sec),
            "Pit Time Remaining": pit_cell,
        }
    )

timing_df = pd.DataFrame(rows)

# Determine which row is "active" for highlighting
if st.session_state.phase == "stint":
    highlight_idx = st.session_state.active_stint_idx
elif st.session_state.phase == "pit":
    # Highlight the row of the driver who just pitted
    highlight_idx = st.session_state.active_stint_idx
elif st.session_state.phase == "caution":
    prev = st.session_state.phase_before_caution
    if prev == "pit":
        highlight_idx = st.session_state.active_stint_idx
    else:
        highlight_idx = st.session_state.active_stint_idx
else:
    highlight_idx = st.session_state.active_stint_idx



def _highlight_row(row):
    styles = [""] * len(row)

    # Active row highlight
    if row.name == highlight_idx:
        styles = ["background-color: rgba(255, 235, 59, 0.25); font-weight: 800;"] * len(row)

    # Max exceed highlight: tint whole row red if stint duration > max
    if float(max_stint_min) > 0:
        dur_txt = row.get("Stint Duration", "")
        if isinstance(dur_txt, str) and ":" in dur_txt:
            try:
                m, s = dur_txt.split(":")
                dur_min = int(m) + int(s) / 60.0
                if dur_min > float(max_stint_min):
                    styles = [
                        (sty + " background-color: rgba(231, 76, 60, 0.18);") if sty else "background-color: rgba(231, 76, 60, 0.18);"
                        for sty in styles
                    ]
            except Exception:
                pass

    return styles


if st.session_state.race_mode_state:
    st.markdown("#### Timing Table (active row highlighted)")
else:
    st.subheader("Timing Table (active row highlighted)")

st.dataframe(
    timing_df.style.apply(_highlight_row, axis=1),
    use_container_width=True,
    hide_index=True,
)


# ----------------------------
# Planning + editing (NORMAL MODE ONLY)
# ----------------------------
if st.session_state.race_mode_state:
    st.info("Race Mode is read-only. Exit Race Mode to edit driver assignments.")
else:
    if float(buffer_pool_min) > 0 and int(buffer_first_n_stints) > 0:
        st.info(
            f"Early stint buffer enabled: requested {buffer_pool_min:.1f} min within first "
            f"{int(buffer_first_n_stints)} stints. "
            f"Applied {buffer_added:.1f} min, removed {buffer_removed:.1f} min from later stints."
        )
        if buffer_shortfall > 0.25:
            st.warning(
                f"Buffer could not be fully balanced (shortfall {buffer_shortfall:.1f} min). "
                "Likely too few later stints remain to subtract from."
            )

    # ---- Target Totals by Driver (Planning View) ----
    target_sec_by_driver = {d: 0.0 for d in drivers}
    unassigned_target_sec = 0.0

    for i in range(num_stints):
        d = st.session_state.driver_assignments[i]
        t = float(st.session_state.stints[i].target_sec or 0.0)
        if d in target_sec_by_driver:
            target_sec_by_driver[d] += t
        else:
            unassigned_target_sec += t

    green_thr_min = 1.0
    yellow_thr_min = 3.0

    plan_rows = []
    for d in drivers:
        fair_sec = per_driver_target_min * 60.0
        assigned_sec = target_sec_by_driver.get(d, 0.0)
        delta_sec = assigned_sec - fair_sec
        delta_min_abs = abs(delta_sec) / 60.0

        if delta_min_abs <= green_thr_min:
            status = "🟢"
        elif delta_min_abs <= yellow_thr_min:
            status = "🟡"
        else:
            status = "🔴"

        plan_rows.append(
            {
                "Driver": d,
                "Fair Target Total": fmt_mmss(fair_sec),
                "Assigned Target Total": fmt_mmss(assigned_sec),
                "Delta": fmt_mmss(delta_sec),
                "Status": status,
            }
        )

    plan_df = pd.DataFrame(plan_rows)

    st.subheader("Target Totals by Driver (planning)")
    st.caption(
        "Goal: all drivers have equal **Assigned Target Total** (regardless of stint count). "
        "Caution time is excluded from total available drive time."
    )
    st.caption(
        f"Max stint: **{int(max_stint_min)} min** ({'disabled' if float(max_stint_min) <= 0 else max_mode})."
    )

    st.dataframe(plan_df, use_container_width=True, hide_index=True)

    if unassigned_target_sec > 0.0:
        st.warning(
            f"Unassigned target time exists: about {fmt_mmss(unassigned_target_sec)}. "
            "Assign a driver to every stint to make totals meaningful."
        )

    # ---- Driver assignments editor (FORM) ----
    st.subheader("Driver Assignments (editable; save when done)")

    # Auto-pause while editing drivers (to avoid interrupts)
    if (
        st.session_state.phase in ("stint", "pit", "caution")
        and st.session_state.live_mode_user
        and not st.session_state.auto_paused
    ):
        st.session_state.auto_paused = True
        st.session_state.auto_pause_reason = "Editing drivers"

    assign_df = pd.DataFrame({"Stint #": list(range(1, num_stints + 1)), "Driver": st.session_state.driver_assignments})

    with st.form("driver_assign_form_normal", clear_on_submit=False):
        edited_assign = st.data_editor(
            assign_df,
            use_container_width=True,
            num_rows="fixed",
            disabled=["Stint #"],
            column_config={"Driver": st.column_config.SelectboxColumn(options=[""] + drivers)},
            key="driver_assign_editor_form_normal",
        )
        saved = st.form_submit_button("Save assignments")

    if saved:
        st.session_state.driver_assignments = edited_assign["Driver"].fillna("").tolist()
        st.success("Assignments saved. (Targets will recalc automatically.)")
        st.rerun()

    # ---- Fairness dashboard (execution view) ----
    with st.expander("Fairness dashboard (actual vs target)"):
        dash_rows = []
        for d in drivers:
            dash_rows.append(
                {
                    "Driver": d,
                    "Driven so far (min)": round(actual_min.get(d, 0.0), 2),
                    "Target total (min)": round(per_driver_target_min, 2),
                    "Over/Under (min)": round(actual_min.get(d, 0.0) - per_driver_target_min, 2),
                }
            )
        st.dataframe(pd.DataFrame(dash_rows), use_container_width=True, hide_index=True)

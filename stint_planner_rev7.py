# stint_planner_rev7.py
# Endurance Kart — Live Timer Stint Planner
#
# Rev7 changes:
# - REMOVED: CAUTION feature (button + phase + logic)
# - Max stint time (default 50 min; set 0 to disable), Warn-only or Hard-cap targets
# - Early Stint Buffer strategy (front-load minutes early, subtract later)
# - PIT timer displays on the SAME ROW as the driver who just pitted
# - Shared live race state via Supabase Postgres:
#     - user-entered Race ID
#     - one "Race Controller" device at a time
#     - automatic takeover if controller goes offline
#     - viewers can watch the same live race state
#
# Requirements:
#   streamlit, pandas, streamlit-autorefresh, streamlit-extras, psycopg2-binary
#
# Streamlit Secrets (Streamlit Cloud -> Settings -> Secrets):
#   SUPABASE_DB_URL = "postgresql://postgres:YOUR_PASSWORD@db.xxxxx.supabase.co:5432/postgres"

import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import psycopg2
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_extras.stylable_container import stylable_container
from urllib.parse import urlparse, unquote

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Endurance Kart — Timer Planner", layout="wide")

# Local-only UI state
if "race_mode_state" not in st.session_state:
    st.session_state.race_mode_state = False


# ----------------------------
# DB helpers (Supabase Postgres)
# ----------------------------
CONTROLLER_TIMEOUT_SEC = 15  # takeover allowed after this many seconds without heartbeat


@st.cache_resource
def get_db_conn():
    url = st.secrets["SUPABASE_DB_URL"].strip()

    # Parse postgresql://user:pass@host:port/dbname
    u = urlparse(url)
    if u.scheme not in ("postgres", "postgresql"):
        raise ValueError("SUPABASE_DB_URL must start with postgresql://")

    user = u.username
    password = unquote(u.password or "")  # supports URL-encoded passwords
    host = u.hostname
    port = u.port or 5432
    dbname = (u.path or "").lstrip("/") or "postgres"

    return psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port,
        sslmode="require",
    )



def db_ensure_race_row(race_id: str) -> None:
    conn = get_db_conn()
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into races (race_id, state_json)
            values (%s, %s)
            on conflict (race_id) do nothing
            """,
            (race_id, json.dumps({})),
        )
        conn.commit()


def db_get_race(race_id: str) -> Tuple[dict, Optional[str], Optional[datetime]]:
    db_ensure_race_row(race_id)
    conn = get_db_conn()
    with conn.cursor() as cur:
        cur.execute(
            "select state_json, controller_id, controller_heartbeat_at from races where race_id = %s",
            (race_id,),
        )
        row = cur.fetchone()
    if not row:
        return {}, None, None
    state_json, controller_id, controller_heartbeat_at = row
    return (state_json or {}), controller_id, controller_heartbeat_at


def db_save_state(race_id: str, state: dict) -> None:
    conn = get_db_conn()
    with conn.cursor() as cur:
        cur.execute(
            """
            update races
            set state_json = %s,
                updated_at = now()
            where race_id = %s
            """,
            (json.dumps(state), race_id),
        )
        conn.commit()


def db_take_control(race_id: str, client_id: str) -> None:
    conn = get_db_conn()
    with conn.cursor() as cur:
        cur.execute(
            """
            update races
            set controller_id = %s,
                controller_heartbeat_at = now()
            where race_id = %s
            """,
            (client_id, race_id),
        )
        conn.commit()


def db_heartbeat(race_id: str, client_id: str) -> None:
    conn = get_db_conn()
    with conn.cursor() as cur:
        cur.execute(
            """
            update races
            set controller_heartbeat_at = now()
            where race_id = %s and controller_id = %s
            """,
            (race_id, client_id),
        )
        conn.commit()


# Optional: event log (commented out — enable if you want)
# def db_log_event(race_id: str, event_type: str, payload: dict) -> None:
#     conn = get_db_conn()
#     with conn.cursor() as cur:
#         cur.execute(
#             "insert into race_events (race_id, type, payload_json) values (%s, %s, %s)",
#             (race_id, event_type, json.dumps(payload or {})),
#         )
#         conn.commit()


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
) -> tuple[float, float]:
    """
    Total driving time excludes minimum required pit time.
    (No caution handling in Rev7.)
    """
    pit_min = pit_seconds / 60.0
    total_drive = race_duration_min - (min_pit_stops * pit_min)
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
    - Unassigned stints share the remaining pool evenly

    Max stint:
      - Warn-only: caller can highlight targets > max
      - Hard-cap: each stint target is clamped to max_stint_min.
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

            need[d] = max(0.0, need[d] - planned)
            remaining_counts[d] -= 1

        else:
            remaining_drive = sum(need.values())
            planned = (remaining_drive / unassigned_count) if unassigned_count > 0 else 0.0

            if hard_cap and cap > 0:
                planned = min(planned, cap)

            targets.append(planned)

            if remaining_drive > 1e-9 and planned > 0:
                for k in list(need.keys()):
                    if need[k] > 0:
                        need[k] = max(0.0, need[k] - planned * (need[k] / remaining_drive))

            unassigned_count = max(0, unassigned_count - 1)

    return targets, need


def apply_early_buffer_to_targets(
    targets_min: List[float],
    completed_count: int,
    buffer_pool_min: float,
    buffer_first_n_stints: int,
    max_stint_min: float,
) -> tuple[List[float], float, float, float]:
    """
    Post-process remaining targets:

    - Add buffer minutes evenly to eligible early stints (global stint # <= buffer_first_n_stints),
      but only for stints that are still remaining.
    - Clamp adds so a target never exceeds max_stint_min (if max enabled).
    - Remove the same total minutes from later remaining stints (global stint # > buffer_first_n_stints),
      proportionally to their current target minutes, never going below 0.

    Returns: (new_targets_min, total_added, total_removed, removal_shortfall)
    """
    if buffer_pool_min <= 0 or buffer_first_n_stints <= 0 or not targets_min:
        return list(targets_min), 0.0, 0.0, 0.0

    cap = float(max_stint_min) if float(max_stint_min) > 0 else None

    eligible: List[int] = []
    later: List[int] = []

    for j in range(len(targets_min)):
        global_idx = completed_count + j  # 0-based
        stint_no = global_idx + 1         # 1-based

        if stint_no <= buffer_first_n_stints:
            eligible.append(j)
        else:
            later.append(j)

    if not eligible or not later:
        return list(targets_min), 0.0, 0.0, float(buffer_pool_min)

    new_targets = list(targets_min)

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
# Race state (persisted)
# ----------------------------
@dataclass
class StintRecord:
    target_sec: float = 0.0
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None


PERSIST_KEYS = [
    "phase",            # idle | stint | pit | finished
    "race_start_ts",
    "active_stint_idx",
    "pit_start_ts",
    "pit_paused_sec",   # reserved (no caution now), but keeps your pit-freeze pattern consistent
    "stints",
    "driver_assignments",
]


def ensure_state(num_stints: int):
    if "phase" not in st.session_state:
        st.session_state.phase = "idle"
    if "race_start_ts" not in st.session_state:
        st.session_state.race_start_ts = None
    if "active_stint_idx" not in st.session_state:
        st.session_state.active_stint_idx = 0
    if "pit_start_ts" not in st.session_state:
        st.session_state.pit_start_ts = None
    if "pit_paused_sec" not in st.session_state:
        st.session_state.pit_paused_sec = 0.0

    if "stints" not in st.session_state or len(st.session_state.stints) != num_stints:
        st.session_state.stints = [StintRecord() for _ in range(num_stints)]
    if "driver_assignments" not in st.session_state or len(st.session_state.driver_assignments) != num_stints:
        st.session_state.driver_assignments = [""] * num_stints

    # Local-only live update controls (not persisted)
    if "live_mode_user" not in st.session_state:
        st.session_state.live_mode_user = True
    if "auto_paused" not in st.session_state:
        st.session_state.auto_paused = False
    if "auto_pause_reason" not in st.session_state:
        st.session_state.auto_pause_reason = ""


def serialize_state() -> dict:
    data: dict = {}
    for k in PERSIST_KEYS:
        if k == "stints":
            data[k] = [rec.__dict__ for rec in st.session_state.stints]
        else:
            data[k] = st.session_state.get(k)
    return data


def restore_state(data: dict, num_stints: int) -> None:
    # Restore only known keys; keep local UI settings untouched
    for k in PERSIST_KEYS:
        if k not in data:
            continue
        if k == "stints":
            recs = data.get("stints") or []
            st.session_state.stints = [StintRecord(**r) for r in recs]
            # If lengths mismatch (race config changed), reinit safely
            if len(st.session_state.stints) != num_stints:
                st.session_state.stints = [StintRecord() for _ in range(num_stints)]
        else:
            st.session_state[k] = data.get(k)

    # Ensure mandatory arrays match current num_stints
    if len(st.session_state.stints) != num_stints:
        st.session_state.stints = [StintRecord() for _ in range(num_stints)]
    if len(st.session_state.driver_assignments) != num_stints:
        st.session_state.driver_assignments = [""] * num_stints


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


def effective_live_mode() -> bool:
    return bool(st.session_state.live_mode_user) and (not bool(st.session_state.auto_paused))


def stint_elapsed_sec(rec: StintRecord, now_ts: float) -> Optional[float]:
    if rec.start_ts is None:
        return None
    end = rec.end_ts if rec.end_ts is not None else now_ts
    return max(0.0, end - rec.start_ts)


# ----------------------------
# Title
# ----------------------------
st.title("Endurance Kart — Live Timer Stint Planner")

DEFAULT_RACE_MIN = 480.0
DEFAULT_MIN_PIT_STOPS = 12
DEFAULT_MAX_STINT_MIN = 50.0


# ----------------------------
# Sidebar: Shared race + configuration
# ----------------------------
with st.sidebar:
    st.header("Shared Race")

    race_id = st.text_input(
        "Race ID",
        value="hellcat-test",
        help="Everyone (controller + viewers) must use the same Race ID to see the same live race.",
    )

    if "client_id" not in st.session_state:
        st.session_state.client_id = str(uuid.uuid4())

    st.caption(f"Device ID: {st.session_state.client_id[:8]}…")

    st.divider()
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
        help="Total minutes to front-load into early stints (then removed from later stints). Set 0 to disable.",
    )
    buffer_first_n_stints = st.number_input(
        "Apply buffer within first N stints",
        min_value=0,
        value=0,
        step=1,
        help="Example: N=3 applies to stints 1–3 (only if those stints are not yet completed). Set 0 to disable.",
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

# ----------------------------
# Derived config
# ----------------------------
num_stints = int(min_pit_stops) + 1
ensure_state(num_stints)
now = time.time()

# ----------------------------
# Load shared state from DB (every run)
# - This ensures viewers always see the latest controller state.
# - It also enables failover: open on another laptop and continue.
# ----------------------------
# If Race ID changes, force local reset of cached load markers
if st.session_state.get("_last_race_id") != race_id:
    st.session_state._last_race_id = race_id

# Pull latest shared state + controller info from DB
db_state, db_controller_id, db_heartbeat_at = db_get_race(race_id)
restore_state(db_state, num_stints)

# Determine controller status
is_controller = (db_controller_id == st.session_state.client_id)
controller_alive = False
controller_age_sec = None
if db_heartbeat_at is not None:
    controller_age_sec = (datetime.now(timezone.utc) - db_heartbeat_at).total_seconds()
    controller_alive = controller_age_sec < CONTROLLER_TIMEOUT_SEC

# Controller UI + heartbeat
with st.sidebar:
    st.subheader("Controller")
    if is_controller:
        st.success("You are Race Controller")
        # Keep heartbeat fresh every rerun
        try:
            db_heartbeat(race_id, st.session_state.client_id)
        except Exception:
            st.warning("Heartbeat failed (DB connection issue).")
    else:
        if (db_controller_id is None) or (not controller_alive):
            if db_controller_id is None:
                st.info("No controller currently set.")
            else:
                st.warning(f"Controller offline (last seen ~{int(controller_age_sec)}s ago). Takeover available.")
            if st.button("TAKE CONTROL", use_container_width=True):
                db_take_control(race_id, st.session_state.client_id)
                st.rerun()
        else:
            st.warning("Race controlled by another device.")
            st.caption(f"Controller last seen ~{int(controller_age_sec)}s ago")

    # View-only hint
    if not is_controller:
        st.caption("View-only mode: race controls are disabled.")


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
# ----------------------------
refresh_sec_clamped = max(0.0, min(float(refresh_sec), 60.0))
if effective_live_mode() and st.session_state.phase in ("stint", "pit") and refresh_sec_clamped > 0:
    st_autorefresh(interval=int(refresh_sec_clamped * 1000), key="tick")


# ----------------------------
# Auto-advance: PIT -> next STINT
# IMPORTANT: Only controller should mutate/save shared state.
# ----------------------------
if is_controller and st.session_state.phase == "pit" and st.session_state.pit_start_ts is not None:
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

        db_save_state(race_id, serialize_state())
        st.rerun()


# ----------------------------
# Compute targets (fairness)
# ----------------------------
total_drive_min, per_driver_target_min = compute_targets(
    race_duration_min=float(race_duration_min),
    min_pit_stops=int(min_pit_stops),
    pit_seconds=float(pit_seconds),
    drivers=drivers,
)

# Actual driven minutes per driver from completed stints
actual_min = {d: 0.0 for d in drivers}
for i, rec in enumerate(st.session_state.stints):
    if rec.start_ts and rec.end_ts:
        dur_min = (stint_elapsed_sec(rec, now) or 0.0) / 60.0
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

# Apply early buffer strategy (targets only)
target_min_list, buffer_added, buffer_removed, buffer_shortfall = apply_early_buffer_to_targets(
    targets_min=target_min_list,
    completed_count=completed_count,
    buffer_pool_min=float(buffer_pool_min),
    buffer_first_n_stints=int(buffer_first_n_stints),
    max_stint_min=float(max_stint_min),
)

# Assign targets (remaining stints only)
for j in range(remaining_required):
    idx = completed_count + j
    if idx < num_stints:
        st.session_state.stints[idx].target_sec = float(target_min_list[j] * 60.0)

# Infeasibility warning when hard-capping targets
if hard_cap and float(max_stint_min) > 0:
    leftover_total = sum(leftover_need.values())
    if leftover_total > 0.25:
        st.error(
            "Max stint cap makes the plan infeasible with current assignments. "
            f"About {leftover_total:.1f} minutes of drive time cannot be allocated without exceeding the max. "
            "In reality, at least one stint will need to go longer, or you must add more stints / adjust assignments."
        )


# ----------------------------
# Top controls row
# ----------------------------
col_start, col_mid, col_right = st.columns([1.6, 1.2, 1.2])

with col_start:
    # Disable during PIT or finished; also disable for viewers
    button_disabled = (st.session_state.phase in ("pit", "finished")) or (not is_controller)

    label = "START NEXT STINT" if st.session_state.phase in ("idle", "stint") else "PIT IN PROGRESS"

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

            db_save_state(race_id, serialize_state())
            st.rerun()

with col_mid:
    st.metric("Minimum Stints", f"{num_stints}")
    st.metric("Target per Driver (min)", f"{per_driver_target_min:.2f}")

    # Pause/Resume local-only live updates
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

    # Small reset control with confirmation (controller only)
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
                if st.button(
                    "CONFIRM RESET",
                    disabled=(not confirm) or (not is_controller),
                    use_container_width=True,
                ):
                    reset_race(num_stints)
                    db_save_state(race_id, serialize_state())
                    st.rerun()

with col_right:
    # Race countdown
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

    # Exit race mode (local-only UI)
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
# - PIT timer shows on the SAME ROW as the stint that just ended
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

    # PIT countdown displayed on the stint row that just ended (driver who pitted)
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

# Highlight row:
# - STINT: current stint row
# - PIT: row of the driver who just pitted (same row where pit timer appears)
if st.session_state.phase == "stint":
    highlight_idx = st.session_state.active_stint_idx
elif st.session_state.phase == "pit":
    highlight_idx = st.session_state.active_stint_idx
else:
    highlight_idx = st.session_state.active_stint_idx


def _highlight_row(row):
    styles = [""] * len(row)

    if row.name == highlight_idx:
        styles = ["background-color: rgba(255, 235, 59, 0.25); font-weight: 800;"] * len(row)

    # Tint row red if stint duration > max
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
# Planning + editing
# - In shared-state mode, only the controller can edit driver assignments.
# ----------------------------
if st.session_state.race_mode_state:
    st.info("Race Mode is read-only. Exit Race Mode to view planning and edit assignments (controller only).")
else:
    # Strategy status
    if float(buffer_pool_min) > 0 and int(buffer_first_n_stints) > 0:
        st.info(
            f"Early stint buffer enabled: requested {float(buffer_pool_min):.1f} min within first {int(buffer_first_n_stints)} stints. "
            f"Applied {buffer_added:.1f} min, removed {buffer_removed:.1f} min from later stints."
        )
        if buffer_shortfall > 0.25:
            st.warning(
                f"Buffer could not be fully balanced (shortfall {buffer_shortfall:.1f} min). "
                "Likely too few later stints remain to subtract from, or too many early stints are already completed."
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

    st.subheader("Target Totals by Driver (planning)")
    st.caption(
        "Goal: all drivers have equal **Assigned Target Total** (regardless of stint count). "
        "Pit minimum time is excluded from total available drive time."
    )
    st.caption(f"Max stint: **{int(max_stint_min)} min** ({'disabled' if float(max_stint_min) <= 0 else max_mode}).")

    st.dataframe(pd.DataFrame(plan_rows), use_container_width=True, hide_index=True)

    if unassigned_target_sec > 0.0:
        st.warning(
            f"Unassigned target time exists: about {fmt_mmss(unassigned_target_sec)}. "
            "Assign a driver to every stint to make totals meaningful."
        )

    # ---- Driver assignments editor (controller only) ----
    st.subheader("Driver Assignments (controller only)")
    if not is_controller:
        st.info("Only the Race Controller can edit driver assignments.")
    else:
        # Auto-pause local live updates while editing (to avoid interrupts)
        if st.session_state.phase in ("stint", "pit") and st.session_state.live_mode_user and not st.session_state.auto_paused:
            st.session_state.auto_paused = True
            st.session_state.auto_pause_reason = "Editing drivers"

        assign_df = pd.DataFrame(
            {"Stint #": list(range(1, num_stints + 1)), "Driver": st.session_state.driver_assignments}
        )

        with st.form("driver_assign_form", clear_on_submit=False):
            edited_assign = st.data_editor(
                assign_df,
                use_container_width=True,
                num_rows="fixed",
                disabled=["Stint #"],
                column_config={"Driver": st.column_config.SelectboxColumn(options=[""] + drivers)},
                key="driver_assign_editor",
            )
            saved = st.form_submit_button("Save assignments")

        if saved:
            st.session_state.driver_assignments = edited_assign["Driver"].fillna("").tolist()
            db_save_state(race_id, serialize_state())
            st.success("Assignments saved (shared).")
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

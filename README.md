# Hellcat Racing – Endurance Stint Planner

A live, interactive GUI for planning and managing driver stints during a team endurance go-kart race.

This tool is designed to **prioritize equal total driving time per driver**, while remaining flexible and realistic during an actual race (pit stops, overruns, cautions, etc.).

A public demo version is available here:  
👉 **(https://hellcatracing-tjwltqdpri2gwgwye6m6k3.streamlit.app/)**

---

## What this tool does

- Calculates **fair target driving time per driver**
- Splits that time across assigned stints (even if stint counts differ)
- Tracks the race **live**, including:
  - Stint timers (count up + remaining time)
  - Pit stop timers
  - Race countdown timer
  - Competition cautions (yellow/red flags)
- Dynamically adjusts fairness as the race unfolds
- Provides a clear **race-wall-friendly interface**

Each user session is **independent** — feel free to experiment without worrying about affecting others.

---

## Key concepts

### Stints
- A stint is one continuous driving segment
- The number of stints is determined by:
Minimum required stints = minimum pit stops + 1

### Fairness
- The app targets **equal total drive time per driver**, not equal stint length
- Drivers with more stints get **shorter stints**
- Drivers with fewer stints get **longer stints**
- Overruns and underruns are redistributed dynamically

### Caution periods
- Cautions (yellow/red flags) can be started and ended manually
- Caution time:
- **Counts toward race time**
- **Does NOT count toward driver stint time**
- Is accumulated and subtracted from total available driving time
- Multiple cautions are fully supported

---

## How to use (quick guide)

### Before the race
1. Enter race parameters (duration, pit time, number of drivers, etc.)
2. Assign drivers to stints
3. Verify:
 - Target totals by driver
 - Fairness dashboard shows ~0 deltas

### During the race
- Click **START NEXT STINT** to begin
- Click again when the driver pits
- Use **START / END CAUTION** for yellow/red flags
- Let stints run long if needed — the app handles it
- Use **Race Mode** for a clean, race-wall view

### Editing during the race
- Driver assignments are locked in Race Mode
- Exit Race Mode to edit assignments safely

---

## What to test / give feedback on

If you’re trying this out, feedback is especially helpful on:
- Overall clarity and readability
- Whether timers behave as expected
- Whether the fairness logic “feels right”
- Anything that feels confusing, annoying, or risky on race day
- Features you wish existed (max stint time, alerts, layouts, etc.)

---

## Notes

- This is a **planning and management tool**, not official timing
- Intended to assist strategy and fairness, not replace race control
- Still evolving — feedback is welcome

---

## Future ideas (not yet implemented)

- Max stint time enforcement
- Shared live state for race-day viewing
- View-only vs controller modes
- Smoother real-time UI framework

---

Built for **Hellcat Racing** 🐱🔥  
Endurance racing is hard — the math doesn’t have to be.

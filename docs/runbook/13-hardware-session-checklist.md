# Runbook 13 — Hardware Session Checklist (Track B S6)

**Audience:** whoever is about to run the real SO-101 with a policy in the loop.
**Outcome:** the procedural half of the Track B safety stack — the part that is not
in code and therefore cannot be enforced by it.
**Closes:** master plan `plans/2026-08-02-master-consolidated-plan.md` blocker **S6**.
**Read first:** [`10-deploy-to-hardware.md`](10-deploy-to-hardware.md) §Safety layers.

> **This is a checklist, not a description.** S1/S2/S3/S4/S5/S7/C1 are enforced in
> code. S6 is not, and cannot be: no amount of Python knows whether your hand is
> inside the reach envelope. Everything below is a thing a human confirms out loud.

---

## Before power on

- [ ] **Workspace cleared** to the full reach envelope, not the intended path. Max
      planar reach is **~0.346 m** from the base (vault: SO-101 reach envelope).
      Assume the arm can reach anywhere in that radius, at any height, at any moment.
- [ ] **Nothing fragile, wet, or live** inside that radius. No mugs, no laptop, no
      loose cable that can be snagged and pulled.
- [ ] **The fall path is clear.** `disable_torque_on_disconnect` defaults **True**,
      so the arm goes **limp** on every disconnect and at the end of every episode —
      it drops from wherever it stopped, still gripping whatever it holds. Check what
      is underneath it, not just what is around it.
- [ ] **Power-disconnect method chosen and reachable**: the physical power strip or
      barrel jack, identified *before* starting, within arm's reach of the operator
      and **outside** the robot's reach envelope.
- [ ] **Operator position:** standing, facing the arm, hand at the power switch,
      outside the reach envelope. Not seated at the keyboard with the arm beside you.
- [ ] **Supervisor named** for any run longer than a single episode. One person drives
      the terminal, one person watches the arm. The driver does not also watch.

> [!warning] There is no software e-stop
> Ctrl-C cannot reliably interrupt a blocked serial read or write. The runner's own
> banner says so. **The only reliable stop is cutting power.** Every plan that
> depends on "I'll just hit Ctrl-C" is wrong.

## Before each checkpoint (mandatory dry-run)

- [ ] **Dry run first, every checkpoint, no exceptions.** Run without `--execute` and
      read the predicted actions. A checkpoint that produces garbage in dry-run will
      produce garbage at the motors.
- [ ] **Note this gap:** the `max_deg_per_s_ceiling` refusal only fires when
      `--execute` is set. **A passing dry run is not evidence the real run will
      start.** Check the clamp number separately (below).
- [ ] **Confirm the clamp** from the startup banner, which prints resolved °/step and
      °/s per joint. Verified ladder at 30 Hz (see plan S2):

      | clamp | arm deg/s | gripper deg/s | |
      |---|---|---|---|
      | 1.0 | 30 | 38.9 | first run on a new policy |
      | 2.0 | 60 | 77.9 | validated policy |
      | 2.311 | 69 | 89.6 | maximum the ceiling permits |
      | 3.0 | 90 | 116.8 | **refused** |

      The **gripper** is the binding joint, not any arm joint. Sizing the clamp from
      the arm gives the wrong answer.
- [ ] **Run in a real TTY.** `require_interactive()` refuses a non-tty run under
      `--execute`, deliberately. Do not defeat it with `script`/`pty` — that gets you
      unattended arm motion with none of the compensating controls.

## During the run

- [ ] Hand stays at the power switch. Watch the arm, not the terminal.
- [ ] **Stop immediately** on: a joint straining or buzzing without moving, a servo
      audibly hunting, any unexpected pause, or the watchdog printing a breach line.
- [ ] The joint-current watchdog trips at 80% of each servo's own
      `Protection_Current` (arm **248**, gripper **200** raw) or 10 °C below its
      thermal limit, sampled at 6 Hz. **It stops the run; it does not cut power.**

## After the run

- [ ] Confirm the arm is limp and resting before reaching in.
- [ ] Check servo temperatures. Idle baseline measured on this arm: **23–29 °C**
      against a ~70 °C firmware cutoff. Sustained operation is uncharacterised — no
      published source covers STS3215 under RL-exploration duty cycle, so treat a
      rising trend across episodes as the signal, not an absolute number.
- [ ] Record what happened while it is fresh. A scoreboard written the next morning
      is a scoreboard of what you remember, not what occurred.

---

## Known hazards not covered by code

| Hazard | Why code cannot catch it |
|---|---|
| Arm goes limp and falls on disconnect | By design; `disable_torque_on_disconnect=True` |
| No software e-stop | A blocked serial write ignores Ctrl-C |
| `connect()` can leave servos energised | Contained by `safe_connect` in `robot-data-runner`, but any other bring-up path still needs its own `Torque_Enable=0` sweep on failure |
| Dry run passes, real run refused | The rate ceiling only evaluates under `--execute` |
| `shoulder_pan` under-travels | Measured ~63% of a commanded move at 12.8 deg/s — friction, not clamp. Lowering the clamp will not help |

## Related
- [`10-deploy-to-hardware.md`](10-deploy-to-hardware.md) — the six code-enforced layers
- [`11-closed-loop-eval.md`](11-closed-loop-eval.md) — the eval loop this gates
- `plans/2026-08-02-master-consolidated-plan.md` — Track B, S1–S7
- `NEXT_STEPS.md` — open hardware hazards

# EXEC ENGINE — crash-proof step loop for ROADMAP PR-15..20

Design goal: an agent with ZERO context (fresh session, memory systems dead)
must be able to resume, execute, test, improve, report, and document — by
reading disk only. This file is the protocol; ROADMAP_PR15-20.md is the
catalog of sub-task specs; EXEC_STATE.json is machine state; EXEC_LOG.md is
the append-only journal. Git history is ground truth for what is DONE.

## RECOVERY PROTOCOL (a fresh agent starts HERE, nothing else)
  1. cd /home/riju279/Documents/code/AIRA/histogate
  2. Read this file fully.
  3. `uv run python scripts/next_step.py` → prints the next sub-task, its
     spec reference, exact commands, and PASS criterion.
  4. Execute that sub-task per ROADMAP_PR15-20.md spec (sub-task IDs match).
  5. Follow THE LOOP below. Repeat. Never skip, never batch rewrites of
     state outside the loop.
  If scripts/next_step.py itself is broken (it is tested; see T-N1..N4),
  recover manually: `git log --oneline -15` (committed sub-task IDs) +
  `cat docs/EXEC_STATE.json` + `tail -30 docs/EXEC_LOG.md`.

## THE LOOP (one iteration = one sub-task attempt)
  1. CALL: `uv run python scripts/next_step.py` → {id, phase, commands}.
  2. DO: run the sub-task's commands exactly as specified in ROADMAP.
     Code edits: diff-based (patch tool / ast-grep pre+post checks). New
     files: write_file allowed. Long jobs (>5 min): background.
  3. TEST: run the sub-task's PASS command.
  4a. PASS →
      - `git add -A && git commit -m "<ID> <one-line>"`
      - `uv run python scripts/next_step.py --done <ID> --note "<result>"`
        (this updates EXEC_STATE.json + appends EXEC_LOG.md + commits both)
      - REPORT: one paragraph to the user: what, measured numbers, next id.
  4b. FAIL →
      - Increment attempt counter: `uv run python scripts/next_step.py
        --fail <ID> --note "<exact error + diagnosis>"` (appends to LOG).
      - Diagnose from the error; change exactly ONE variable; retry loop
        from step 2. Max 3 attempts per sub-task.
      - After 3 failures: `--fail <ID> --note "PARKED: <reason>"` → state
        becomes "parked", next_step skips it, and the loop moves on.
        PARKED tasks are listed in every report until resolved by the user
        or a later sub-task unblocks them (do NOT silently retry parked
        tasks; a human decides).
  5. IMPROVE-IF-NEEDED: if a measurement lands within 10% of a gate
     threshold, the sub-task's spec may allow ONE refinement iteration
     (named in ROADMAP as "refine:"). Only where named. Otherwise accept
     the measured value and record it honestly (gate-must-fail law).

## STATE FILE SCHEMA (docs/EXEC_STATE.json)
  {
    "version": 1,
    "roadmap": "docs/ROADMAP_PR15-20.md",
    "current_pr": "PR-15",
    "subtasks": {
      "P15-S1": {"status": "in_progress|done|parked|todo",
                 "attempts": 0,
                 "last_note": "",
                 "commit": "<sha|null>",
                 "artifact": "<path proving PASS|null>"},
      ...
    },
    "parked": [],
    "updated": "<iso timestamp>"
  }
  Invariants (checked by `next_step.py --verify`):
    V1 every sub-task whose commit sha exists in git log ⇒ status done.
    V2 every done sub-task ⇒ artifact path exists on disk (or "docs").
    V3 attempts <= 3 unless parked.
    V4 no parked task is "current".
    V5 state ids ⊆ ROADMAP ids (no orphan ids, no missing ids).
  On invariant violation: --verify prints the exact repair command
  (e.g. `--done <ID> --commit <sha>` for a committed-but-todo task).

## JOURNAL FORMAT (docs/EXEC_LOG.md, append-only)
  `<iso> | <ID> | <status> | attempt <n> | <note>`
  One line per loop transition. Never edit history; corrections are new
  lines with status "correction".

## REPORTING + DOCUMENTATION (per sub-task, no exceptions)
  - REPORT: to the user, one paragraph, measured numbers only.
  - DOCUMENT: docs/RESULTS.md is REGENERATED (never hand-edited) by
    scripts/render_readme_numbers.py at every PR boundary (P15-S1, P16-S6,
    P17-S4, P18-S4, P19-S3). Numbers in prose docs = links to RESULTS.md.
  - JOURNAL: the --done/--fail note is the durable record.

## LAWS (from user; repeat here so no context is needed)
  L1 no subagents. L2 adversarial tests only (tests attack; can fail).
  L3 real data only for blemishes/noise/stain calibration; synthetic =
  stress-test only. L4 docs after measurement; rendered from artifacts.
  L5 gate must be able to fail. L6 long jobs in background.
  L7 diff-based code edits + ast-grep pre/post checks.
  L8 email deadline: Monday 10:00 IST, surya.prajapati@airamatrix.com,
  numbers must come from docs/RESULTS.md only.

## THE LOOP'S OWN TESTS (adversarial, tests/test_next_step.py)
  T-N1 stale-state: commit sha exists but sub-task todo → --verify FAILS
     and names the repair. (attacks: state drift vs git)
  T-N2 orphan-id: state has an id absent from ROADMAP → --verify FAILS.
     (attacks: rename/leftover rot)
  T-N3 done-no-artifact: done sub-task whose artifact file was deleted →
     --verify FAILS. (attacks: false completion)
  T-N4 parked-never-current: parked id returned as next step → FAIL.
     (attacks: silent retry of parked work)
  T-N5 next_step determinism: two runs, same output, given same state.
  These tests guard the engine the same way the engine guards the project.

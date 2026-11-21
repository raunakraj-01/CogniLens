# Coding Agent Rules (from Section 0 of PRD)

You are implementing the `cogload` project. Follow these core operating rules:

1. **Build in phase order** (Section 16). Do not start Phase N+1 until every acceptance criterion of Phase N passes.
2. **Everything must be runnable and testable without a webcam.** Cloud and CI environments have no camera. Every component must work from (a) a video file, (b) a recorded-signal replay, or (c) the synthetic signal generator (FR-SYN). Only the final live demo needs a real webcam.
3. **Do not invent scope.** If it is not in Sections 3 to 15, do not build it. Put ideas in `docs/IDEAS.md` instead.
4. **Make thresholds and constants configurable** in `config.yaml`. Never hard-code magic numbers in logic.
5. **Write tests with the code.** Each module ships with unit tests in `tests/`. A phase is not done without passing tests.
6. **Keep the ML honest.** No data leakage. Subject-wise splits only. No tuning on the held-out subject. See Section 11.
7. **Privacy by default.** Never write raw video frames or images to disk. See Section 13.2.
8. **Ask the humans** (flag in `docs/QUESTIONS.md`) rather than guessing when a decision in Section 20 is blocking. Tasks tagged **[HUMAN]** need real people (data collection, ethics approval) and cannot be done by you; prepare everything they need and stop.
9. At the end of each phase, update `docs/PROGRESS.md` (what was done, what is left, known issues).
10. Prefer boring, widely used libraries. Pin versions in `requirements.txt`.

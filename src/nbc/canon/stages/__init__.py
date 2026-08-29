"""The stages themselves, one module per transformation.

Every stage module presents the same two names to the pipeline: `NAME`, the string that appears in
the trace, and `run`, the `Stage(text, ctx) -> StageResult` function. `canon/pipeline.py` is the
only module in `src/` that **calls** either of them, and a test enforces that by resolving the
name a stage module is bound to and refusing an entry-point reference on it anywhere else: AD-4
says no caller may invoke a stage out of band or reorder the list at runtime.

Importing a stage module for its **declared data** is a different act and is admitted from an exact
list with a reason per entry. `corpus/dressings.py` is on it, because story 3.4 requires the
zero-width dressing to draw its character from `invisible.ZERO_WIDTH` rather than from a second
hand-list — the character sets are one entity with one owner, and that owner is the stage.

A stage may publish more than those two, and one does. AD-18 requires step 4's candidate-test
constants to be declared in `decode.py`, with their units, and Epic 4 writes those names and units
into `results.json`, so `decode.py` exports the declared block and the predicate that applies it.
It also publishes a **second pair** — `CEILING_NAME` and `run_at_ceiling` — because AD-6 gives step
4 a different behaviour once the recursion ceiling is reached: replace nothing, and report a
would-have-decoded candidate under a name distinct from an ordinary rejection. Which pair applies
is `canon/pipeline.py`'s decision, declared on the `PipelineStage` and read structurally rather
than by recognizing a stage's name.

What no stage exports is state: everything published here is frozen, and no stage reads a depth or
a ceiling — the one comparison that involves either, `depth >= ctx.ceiling`, is the runner's.

`NAME` lives beside the code that emits it so the name in the trace and the name in `PIPELINE`
have one source. The runner still compares them, because a stage stamping a name its step never
declared would otherwise misattribute a change silently.
"""

"""The stages themselves, one module per transformation.

Every stage module presents the same two names to the pipeline: `NAME`, the string that appears in
the trace, and `run`, the `Stage(text, ctx) -> StageResult` function. `canon/pipeline.py` is the
only module in `src/` that imports them, and a test enforces that: AD-4 says no caller may invoke a
stage out of band or reorder the list at runtime, and an import is how a caller would get the
chance.

A stage may publish more than those two, and one does. AD-18 requires step 4's candidate-test
constants to be declared in `decode.py`, with their units, and Epic 4 writes those names and units
into `results.json`, so `decode.py` exports the declared block and the predicate that applies it.
What no stage exports is state: everything published here is frozen, and the pipeline still reaches
a stage only through `NAME` and `run`.

`NAME` lives beside the code that emits it so the name in the trace and the name in `PIPELINE`
have one source. The runner still compares the two, because a stage stamping a name it did not
declare would otherwise misattribute a change silently.
"""

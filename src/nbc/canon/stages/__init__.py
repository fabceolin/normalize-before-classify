"""The stages themselves, one module per transformation.

A stage module declares exactly two public names: `NAME`, the string that appears in the trace,
and `run`, the `Stage(text, ctx) -> StageResult` function. `canon/pipeline.py` is the only module
in `src/` that imports them, and a test enforces that: AD-4 says no caller may invoke a stage out
of band or reorder the list at runtime, and an import is how a caller would get the chance.

`NAME` lives beside the code that emits it so the name in the trace and the name in `PIPELINE`
have one source. The runner still compares the two, because a stage stamping a name it did not
declare would otherwise misattribute a change silently.
"""
